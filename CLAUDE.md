# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An AI-powered Polymarket trading pipeline. It detects breaking news, classifies its
directional impact on niche prediction markets via Claude, sizes positions with
quarter-Kelly, and executes trades (dry-run by default).

## Setup & running

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add ANTHROPIC_API_KEY (required)
python cli.py verify         # checks deps, API keys, DB, market/news connectivity
```

Everything goes through `cli.py`:

```bash
python cli.py watch          # V2: event-driven pipeline (real-time news -> classify -> trade), runs forever
python cli.py watch --live   # same, with live trading enabled (DRY_RUN=false)
python cli.py run            # V1: synchronous RSS -> score -> trade pipeline
python cli.py dashboard       # Textual TUI (read-only; --legacy for V1 rich dashboard)
python cli.py backtest       # replay resolved markets through the V2 classifier
python cli.py calibrate       # classification accuracy report
python cli.py niche           # browse markets within the volume filter
python cli.py markets         # browse all active markets in target categories
python cli.py trades          # view trade log from trades.db
python cli.py stats           # latency + calibration + exposure stats
python cli.py scrape          # test the RSS/NewsAPI scraper only
```

Run the test suite with `python -m pytest tests/` (no network or API keys needed —
the DB is a tmp fixture and external calls are faked). It covers the trade risk
gates, position sizing, matching, calibration math, dedup eviction, and task
supervision. Most modules also have a `__main__` block that runs a small live
smoke test against real APIs — run them as packages from the repo root,
e.g. `python -m locus.core.classifier`, `python -m locus.core.matcher`,
`python -m locus.markets.market_watcher`.

## Package layout

`cli.py` at the root is the only entry point. Everything else lives under `locus/`:

```
locus/
  config.py     .env loading, thresholds, PROJECT_ROOT (anchors trades.db, chroma_db/, docs/)
  core/         pipeline (+ gate_trade risk gates), classifier, scorer (V1),
                matcher, market_index, edge, executor, export_status
  sources/      news_stream (Twitter/Telegram/RSS/NewsAPI), scraper (V1)
  markets/      gamma (Gamma API client + Market model), market_watcher
  memory/       __init__ (track record + lessons API), calibrator, logger (SQLite layer)
  backtest/     synthetic (replay classifier), real (real-data pilot, currently parked)
  ui/           tui (Textual dashboard), dashboard (legacy V1 rich dashboard)
```

Runtime artifacts (`trades.db`, `chroma_db/`, `docs/status.json`, `.env`) stay at the
project root, resolved via `config.PROJECT_ROOT` — run everything from the repo root.
Note `memory.py`'s content lives in `locus/memory/__init__.py`, so `from locus import
memory; memory.get_track_record()` works as before the package split.

## Architecture

Two pipelines share the same infrastructure (`locus/config.py`, `locus/memory/logger.py`,
`locus/markets/gamma.py`):

### V2 — event-driven (`locus/core/pipeline.py: PipelineV2`, the recommended path)

Everything is asyncio-based and runs as concurrent tasks under `run_pipeline_v2()`:

```
news_stream.NewsAggregator   -> news_queue
market_watcher.MarketWatcher -> tracked_markets (niche, volume-filtered)
PipelineV2._process_news:
    news_queue -> matcher.match_news_to_markets -> classifier.classify_async
                -> edge.detect_edge_v2 -> signal_queue
PipelineV2._execute_signals:
    signal_queue -> executor.execute_trade_async -> logger (SQLite)
```

- `sources/news_stream.py` aggregates Twitter v2 filtered stream, Telegram bot polling,
  NewsAPI, and an RSS fallback into one deduplicated queue of `NewsEvent`s. Each source
  is independently optional — disabled sources just don't emit (checked via `enabled`
  flags from `config` tokens).
- `markets/market_watcher.py` maintains `tracked_markets`: active markets in
  `MARKET_CATEGORIES` whose volume is between `MIN_VOLUME_USD` and `MAX_VOLUME_USD`
  (full paginated band scan). Refreshed every 5 min via Gamma API; live prices via
  Polymarket WebSocket with a polling fallback if the socket is down. After each
  refresh it syncs `core/market_index.py` — a persistent Chroma collection of markets
  embedded locally with sentence-transformers (all-MiniLM-L6-v2) in `chroma_db/`.
- `core/matcher.py` matches headlines to markets: fast keyword overlap (no API call)
  unioned with semantic hits from the market index under `EMBED_DISTANCE_THRESHOLD`
  (`match_news_to_markets_hybrid`; the match source — keyword/embedding/both — is
  logged per classification).
- `core/pipeline.py: gate_trade` applies trade-time risk gates after classification:
  headlines older than `MAX_NEWS_AGE_SECONDS` are classified but never traded
  (action `stale`), and one headline opens at most one position (action `capped`).
- `core/classifier.py` asks Claude a *classification* question — "does this news make the
  market MORE likely YES / NO / NOT RELEVANT", plus a 0-1 materiality score. This is
  the core philosophical difference from V1 (see README "What Changed From V1"). Each
  call injects a "Your track record" section (via `memory.get_track_record()` +
  `logger.get_recent_lessons()`) so Claude can see its historical accuracy and recent
  mistakes before classifying.
- `core/edge.py: detect_edge_v2` only signals when direction is non-neutral and the
  market price has room to move in that direction (skips YES if price > 0.85, skips NO
  if price < 0.15). The materiality floor is direction-specific and enforced downstream
  in `gate_trade` (so every classification is still logged/calibrated). `size_position`
  applies quarter-Kelly, capped at `MAX_BET_USD`.
- `core/pipeline.py: gate_trade` (continued) also applies calibration-driven materiality
  gates: bullish signals need materiality >= `MATERIALITY_THRESHOLD_BULLISH`, bearish
  need >= `MATERIALITY_THRESHOLD_BEARISH` (bearish accuracy is lower, so a higher bar;
  action `low_materiality`), and any signal at/above `HIGH_MATERIALITY_THRESHOLD` must be
  seen in the same direction from `MIN_CONFIRMING_SOURCES` distinct news sources within
  `CONFIRMATION_WINDOW_HOURS` or it is held (action `needs_confirmation`).
- `core/event_context.py` runs after a signal clears the gates: markets sharing a Gamma
  `event_id` (added to the `Market` dataclass, populated in `gamma.fetch_active_markets`)
  are sibling outcomes of one event. `get_event_exposure` enforces a per-event position
  cap (`MAX_POSITIONS_PER_EVENT`, action `event_exposure_block`); `find_best_outcome`
  inspects a categorical event (sibling YES prices sum to ~1.0) and, when a sibling's
  implied play (bullish on A ⇒ bearish on its siblings, and vice-versa) has more edge
  than the market the news named, the pipeline switches the trade to it
  (`build_switched_signal`). `event_id` is stored on the `trades`, `classifications`, and
  `positions` tables (migrated in `logger._migrate_event_columns`).
- `core/executor.py` enforces `DAILY_LOSS_LIMIT_USD` (checked via `logger.get_daily_pnl`),
  then either logs a `dry_run` row or places a live order via `py_clob_client`
  (optional dependency, commented out in requirements.txt — install separately).

### V1 — synchronous loop (`core/pipeline.py: run_pipeline`, preserved for backward compat)

`sources/scraper.scrape_all` (RSS + NewsAPI) -> `markets/gamma.fetch_active_markets` ->
`core/scorer.score_market` (Claude estimates a YES probability) -> `core/edge.detect_edge`
(compares to market price) -> `core/executor.execute_trade`. Used by `cli.py run` and
`ui/dashboard.py` (the legacy dashboard; `cli.py dashboard` launches the Textual TUI
in `ui/tui.py`, a read-only view over `trades.db`).

### Shared infrastructure

- `config.py` — loads `.env`, defines all thresholds/keys/categories and
  `PROJECT_ROOT`. `DRY_RUN`, `MAX_BET_USD`, `DAILY_LOSS_LIMIT_USD`, `EDGE_THRESHOLD`
  apply to both pipelines;
  `MAX_VOLUME_USD`/`MIN_VOLUME_USD`/`MATERIALITY_THRESHOLD_BULLISH`/
  `MATERIALITY_THRESHOLD_BEARISH`/`HIGH_MATERIALITY_THRESHOLD`/`SPEED_TARGET_SECONDS` are
  V2-only. `cli.py` mutates `config.DRY_RUN` / `config.MATERIALITY_THRESHOLD_BULLISH` +
  `config.MATERIALITY_THRESHOLD_BEARISH` (via `--threshold`) / `config.EDGE_THRESHOLD` at
  runtime based on CLI flags — modules must read these as `config.X` (not via
  `from locus.config import X`) to see overrides.
- `memory/logger.py` — SQLite (`trades.db`, WAL mode). `init_db()` runs at import time
  and auto-migrates newer columns (`_migrate_v2_columns`,
  `_migrate_classification_columns`). Tables: `trades`, `outcomes`, `pipeline_runs`,
  `news_events`, `calibration`, `lessons`, `classifications`.
- `markets/gamma.py` — Polymarket Gamma API client (`fetch_active_markets`, paginated
  with a volume cursor past Gamma's 10k offset cap), with a CLOB API fallback. Infers
  a `category` per market from question text/tags (`_infer_category`, word-boundary
  matching), used by `filter_by_categories` against `MARKET_CATEGORIES`.
- `memory/calibrator.py` — polls Gamma API for resolved markets referenced in `trades`,
  compares the classification's predicted direction to the actual price move, and
  writes to the `calibration` table. `cli.py calibrate` surfaces accuracy by source
  and by classification. When a classification turns out wrong, it calls
  `memory.record_lesson()` to generate and store a lesson.
- `memory/__init__.py` — the classifier's feedback loop (importable as
  `from locus import memory`). `get_track_record()` reads the `calibration` table and
  returns total resolved count, overall accuracy, and accuracy broken down by market
  category (via `gamma._infer_category` on the question text) and by news source.
  `record_lesson()` asks Claude for a 1-2 sentence explanation of why an incorrect
  classification was wrong and stores it in `lessons`. `core/classifier.py` pulls both
  into every classification prompt.
