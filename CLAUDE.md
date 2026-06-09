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
python cli.py dashboard       # rich live terminal dashboard (runs V1 scan loop)
python cli.py backtest       # replay resolved markets through the V2 classifier
python cli.py calibrate       # classification accuracy report
python cli.py niche           # browse markets within the volume filter
python cli.py markets         # browse all active markets in target categories
python cli.py trades          # view trade log from trades.db
python cli.py stats           # latency + calibration + exposure stats
python cli.py scrape          # test the RSS/NewsAPI scraper only
```

There are no automated tests. Each module also has a `__main__` block (e.g.
`python classifier.py`, `python matcher.py`, `python market_watcher.py`) that runs a
small live smoke test against real APIs — useful for exercising one module in
isolation.

## Architecture

Two pipelines share the same infrastructure (`config.py`, `logger.py`, `markets.py`):

### V2 — event-driven (`pipeline.py: PipelineV2`, the recommended path)

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

- `news_stream.py` aggregates Twitter v2 filtered stream, Telegram bot polling, and an
  RSS fallback into one deduplicated queue of `NewsEvent`s. Each source is independently
  optional — disabled sources just don't emit (checked via `enabled` flags from
  `config` tokens).
- `market_watcher.py` maintains `tracked_markets`: active markets in
  `MARKET_CATEGORIES` whose volume is between `MIN_VOLUME_USD` and `MAX_VOLUME_USD`.
  Refreshed every 5 min via Gamma API; live prices via Polymarket WebSocket with a
  polling fallback if the socket is down.
- `matcher.py` does fast keyword-overlap matching from market question to headline
  (no API call). `match_news_to_markets_broad` adds a category-keyword fallback.
- `classifier.py` asks Claude a *classification* question — "does this news make the
  market MORE likely YES / NO / NOT RELEVANT", plus a 0-1 materiality score. This is
  the core philosophical difference from V1 (see README "What Changed From V1").
- `edge.py: detect_edge_v2` only signals when direction is non-neutral, materiality
  clears `MATERIALITY_THRESHOLD`, and the market price has room to move in that
  direction (skips YES if price > 0.85, skips NO if price < 0.15). `size_position`
  applies quarter-Kelly, capped at `MAX_BET_USD`.
- `executor.py` enforces `DAILY_LOSS_LIMIT_USD` (checked via `logger.get_daily_pnl`),
  then either logs a `dry_run` row or places a live order via `py_clob_client`
  (optional dependency, commented out in requirements.txt — install separately).

### V1 — synchronous loop (`pipeline.py: run_pipeline`, preserved for backward compat)

`scraper.scrape_all` (RSS + NewsAPI) -> `markets.fetch_active_markets` ->
`scorer.score_market` (Claude estimates a YES probability) -> `edge.detect_edge`
(compares to market price) -> `executor.execute_trade`. Used by `cli.py run` and
`dashboard.py`.

### Shared infrastructure

- `config.py` — loads `.env`, defines all thresholds/keys/categories. `DRY_RUN`,
  `MAX_BET_USD`, `DAILY_LOSS_LIMIT_USD`, `EDGE_THRESHOLD` apply to both pipelines;
  `MAX_VOLUME_USD`/`MIN_VOLUME_USD`/`MATERIALITY_THRESHOLD`/`SPEED_TARGET_SECONDS` are
  V2-only. `cli.py` mutates `config.DRY_RUN` / `config.MATERIALITY_THRESHOLD` /
  `config.EDGE_THRESHOLD` at runtime based on CLI flags — modules must read these as
  `config.X` (not via `from config import X`) to see overrides.
- `logger.py` — SQLite (`trades.db`, WAL mode). `init_db()` runs at import time and
  auto-migrates V2 columns onto the `trades` table (`_migrate_v2_columns`). Tables:
  `trades`, `outcomes`, `pipeline_runs`, `news_events`, `calibration`.
- `markets.py` — Polymarket Gamma API client (`fetch_active_markets`), with a CLOB API
  fallback. Infers a `category` per market from question text/tags
  (`_infer_category`), used by `filter_by_categories` against `MARKET_CATEGORIES`.
- `calibrator.py` — polls Gamma API for resolved markets referenced in `trades`,
  compares the classification's predicted direction to the actual price move, and
  writes to the `calibration` table. `cli.py calibrate` surfaces accuracy by source
  and by classification.
