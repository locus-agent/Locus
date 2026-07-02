# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An AI-powered Polymarket trading pipeline. It detects breaking news, classifies its
directional impact on niche prediction markets via Claude, sizes positions with
half-Kelly (scaled by a dynamic recent-win-rate factor), and executes trades
(dry-run by default).

## Setup & running

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add ANTHROPIC_API_KEY (required)
python cli.py verify         # checks deps, API keys, DB, market/news connectivity
```

Everything goes through `cli.py`:

```bash
python cli.py watch          # event-driven pipeline (real-time news -> classify -> trade), runs forever
python cli.py watch --live   # same, with live trading enabled (DRY_RUN=false)
python cli.py dashboard       # Textual TUI (read-only)
python cli.py calibrate       # classification accuracy report
python cli.py calibration-report  # read-only closed-position analytics (win rates by bucket, Brier scores)
python cli.py niche           # browse markets within the volume filter
python cli.py markets         # browse all active markets in target categories
python cli.py trades          # view trade log from trades.db
python cli.py stats           # latency + calibration + exposure stats
python cli.py scrape          # test the RSS/NewsAPI scraper only
python cli.py close <id>      # manually close an open position by id
python cli.py reconcile-positions  # sync DB open positions with real on-chain state (--fix to apply)
python cli.py evolve          # manually evolve the classification prompt
python cli.py suggestion list # list pending threshold-adjustment suggestions
```

`cli.py watch` wraps `run_pipeline_v2()` in `run_with_watchdog` — a clean return or
KeyboardInterrupt stops, but any other crash (e.g. a loky semaphore-leak abort on
macOS) is logged and the pipeline is restarted, up to `max_restarts` times. This is the
outer whole-process net; `locus/supervisor.py` is the inner per-task restart layer.

`reconcile-positions` audits each `status='open'` position against its real on-chain
token balance (`executor.held_token_shares`): held > 0 is OK, held == 0 is a phantom
open (MISMATCH), and an unverifiable balance is UNKNOWN and never auto-closed. It also
batch-checks each position's Gamma market status (`fetch_markets_by_condition_ids`,
whose raw dicts have a `closed` flag but no `active` key): an open position on a
CLOSED market is flagged loudly (report key `market_closed`) but NEVER auto-closed —
funds may still be claimable at resolution. Default is report-only; `--fix` closes
only confirmed held==0 phantoms as `closed_reconciled` (`reconcile_positions` in
`locus/core/positions.py`).

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
  core/         pipeline (+ gate_trade risk gates), classifier,
                matcher, market_index, edge, executor, export_status
  sources/      news_stream (Twitter/Telegram/RSS/NewsAPI), scraper (RSS/NewsAPI fetch helpers)
  markets/      gamma (Gamma API client + Market model), market_watcher
  memory/       __init__ (track record + lessons API), calibrator, logger (SQLite layer)
  backtest/     real (real-data pilot, currently parked)
  ui/           tui (Textual dashboard)
```

Runtime artifacts (`trades.db`, `chroma_db/`, `docs/status.json`, `.env`) stay at the
project root, resolved via `config.PROJECT_ROOT` — run everything from the repo root.
Note `memory.py`'s content lives in `locus/memory/__init__.py`, so `from locus import
memory; memory.get_track_record()` works as before the package split.

## Architecture

The pipeline is built on shared infrastructure (`locus/config.py`, `locus/memory/logger.py`,
`locus/markets/gamma.py`):

### Event-driven pipeline (`locus/core/pipeline.py: PipelineV2`)

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
  headlines older than the source-aware freshness window are classified but never
  traded (action `stale`), and one headline opens at most one position (action
  `capped`). News age is measured from the item's publication time when known —
  `sources/scraper.py` parses each RSS `<pubDate>` with `dateutil` (plus explicit
  `strptime` fallbacks) into `NewsEvent.published_at` — and falls back to receipt time
  when no publication time is available (see `pipeline.news_age` + the
  `classifications.published_at` column). The window itself is per-source via
  `config.get_max_age_seconds(source, category, hours_to_resolution)`: real-time feeds
  (Twitter 3h, Truth Social 2h) get a tighter bound than slow aggregators (RSS 6h,
  NewsAPI 5h), unknown sources fall back to `MAX_NEWS_AGE_SECONDS_DEFAULT` (4h), and
  geopolitical / far-resolution markets widen to `MAX_NEWS_AGE_SECONDS_GEOPOLITICAL` (24h).
- `core/classifier.py` asks Claude a *classification* question — "does this news make the
  market MORE likely YES / NO / NOT RELEVANT", plus a 0-1 materiality score. This is
  the core philosophical difference from V1 (see README "What Changed From V1"). Each
  call injects a "Your track record" section (via `memory.get_track_record()` +
  `logger.get_recent_lessons()`) so Claude can see its historical accuracy and recent
  mistakes before classifying.
  Direct Evidence Rule: We deliberately sacrifice some recall to dramatically increase
  precision. Indirect/implied connections are the #1 source of losing trades.
- `core/edge.py: detect_edge_v2` only signals when direction is non-neutral and the
  market price has room to move in that direction (skips YES if price > 0.85, skips NO
  if price < 0.15). The materiality floor is direction-specific and enforced downstream
  in `gate_trade` (so every classification is still logged/calibrated). `size_position`
  applies half-Kelly (via `_base_kelly_size`, from Claude's confidence and the market
  odds), then scales it by a dynamic recent-win-rate factor, capped at `MAX_BET_USD`.
- `core/pipeline.py: gate_trade` (continued) also applies calibration-driven materiality
  gates via `get_materiality_threshold(direction, category)`: category wins first
  (`MIN_MATERIALITY_GEOPOLITICAL` 0.30, `MIN_MATERIALITY_SPORTS` 0.40), then direction —
  `MIN_MATERIALITY_BULLISH` (0.34) vs `MIN_MATERIALITY_BEARISH` (0.27), else
  `MIN_MATERIALITY_DEFAULT` (0.33); below the floor is action `low_materiality`. Any
  signal at/above `HIGH_MATERIALITY_THRESHOLD` (0.5) must be seen in the same direction
  from `MIN_CONFIRMING_SOURCES` distinct news sources within `CONFIRMATION_WINDOW_HOURS`
  or it is held (action `needs_confirmation`). (The legacy `MATERIALITY_THRESHOLD_BULLISH`
  / `_BEARISH` env vars are still honored as fallbacks for the new `MIN_MATERIALITY_*`
  floors, but are not `config` attributes.)
- `core/event_context.py` runs after a signal clears the gates: markets sharing a Gamma
  `event_id` (added to the `Market` dataclass, populated in `gamma.fetch_active_markets`)
  are sibling outcomes of one event. `get_event_exposure` enforces a per-event position
  cap (`MAX_POSITIONS_PER_EVENT`, action `event_exposure_block`); `find_best_outcome`
  inspects a categorical event (sibling YES prices sum to ~1.0) and, when a sibling's
  implied play (bullish on A ⇒ bearish on its siblings, and vice-versa) has more edge
  than the market the news named, the pipeline switches the trade to it
  (`build_switched_signal`). `event_id` is stored on the `trades`, `classifications`, and
  `positions` tables (migrated in `logger._migrate_event_columns`).
- `core/whale_tracker.py` + the pipeline's `whale_tracker` task shadow `WHALE_WALLETS`:
  every `WHALE_CHECK_INTERVAL_MINUTES` it polls Polymarket's public trades feed
  (`POLYMARKET_DATA_HOST`), and `find_missed_opportunities` flags whale BUYs on tracked
  markets we had no actionable classification on (skipping markets closing within
  `WHALE_MIN_HOURS_TO_CLOSE` and those on a `WHALE_COOLDOWN_HOURS` cooldown). Each gets one
  Claude `decide_investigation` call; on "investigate" the market runs the normal
  classify -> gates path and is logged with action `whale_triggered`, `edge_type='whale'`.
  Empty `WHALE_WALLETS` disables the task (clean return).
- Re-entry logic lives in `core/positions.py` (`check_reentry_opportunity`, with the
  pipeline calling `_check_reentry`) — there is no separate `core/reentry.py`. Every
  non-resolution close writes a row to `watched_closed_positions` (via `positions._close`
  -> `logger.watch_closed_position`), watched for `REENTRY_WATCH_HOURS` (72h). When a
  later classification clears the direction/materiality gates on a watched market,
  `check_reentry_opportunity` decides from the position's `exit_reason`: it must be on
  `config.REENTRY_ALLOWED_REASONS` and not on `config.REENTRY_BLOCKED_REASONS`, clear
  `REENTRY_MIN_MATERIALITY` (0.45), and stay under the per-market (`MAX_REENTRY_PER_MARKET`)
  and per-event (`REENTRY_MAX_PER_EVENT`) caps. A re-entry is sized down by
  `REENTRY_SIZE_FACTOR` (0.7x), labeled `edge_type='reentry'` / action `reentry_triggered`;
  a watched market that fails the bar is suppressed (action `reentry_blocked`).
- `core/performance.py: compute_circuit_breaker` is the final gate before any position
  opens (and is re-checked under the per-event lock in `_recheck_risk_gates`): when recent
  realized performance has deteriorated past the limits — 7-day drawdown > `CIRCUIT_BREAKER_DD`
  (0.20) or a known 7-day Sharpe < `CIRCUIT_BREAKER_SHARPE` (-1.0) — every would-be trade is
  held (action `circuit_breaker`) until it recovers. `CIRCUIT_BREAKER_ENABLED=false` disables
  it; the latest read also drives the dashboard status line.
- `core/executor.py` enforces `DAILY_SPEND_LIMIT_USD` (total notional deployed per day,
  not realized losses; checked via `logger.get_daily_pnl`, which counts `dry_run` rows too),
  then either logs a `dry_run` row or places a live order via `py_clob_client_v2`. A live
  BUY whose reconciled fill lands below `MIN_FILL_USD` (default $1, env-overridable) is a
  dust fill: the executor sells the dust back at the bid (best-effort — it may be under
  the exchange minimums) and records the trade as `dust_fill` instead of opening a
  managed position, so the headline reservation is released. Dry-run is unaffected. The
  mirror problem on the close side is handled by top-up-and-sell: when an EXPLICIT close
  (manual / hard stop / time-pressure — never re-eval decisions, half-closes, or
  spontaneously) can't place its SELL because the held shares are below the exchange
  minimums (`MIN_ORDER_SHARES` 5 AND `MIN_ORDER_USD` $1), the executor buys just enough
  extra shares to clear the minimums (only if the buy costs under `TOPUP_MAX_USD`,
  default $2; else the dust is left alone) and sells the combined holding; the top-up's
  cost/tokens are folded into the position's basis whatever happens next, so PnL
  conservation holds (`plan_topup_buy` / `_execute_topup`, gated by
  `positions.TOPUP_EXIT_REASONS`). Before any top-up BUY, a liquidity precheck
  (`bid_depth_within`) requires cumulative bid depth priced within
  `TOPUP_MAX_BID_SLIPPAGE_PCT` (default 20%) of the ask to cover the post-top-up
  holding — a zombie book whose bids all rest at dust prices (0.001-0.007 under a 0.05
  mark) logs `topup_skipped_no_bid_liquidity` and leaves the dust alone rather than
  buying more of an unsellable asset. A position's `actual_cost_usd` is maintained in
  lockstep with `amount_usd` (scaled by the sold fraction on partial/half closes,
  increased by a top-up's cost), so both always describe the real cost of the
  REMAINING holding. `performance.calibration_report` (surfaced as
  `cli.py calibration-report`) also reports Brier scores — overall, by category, and by
  prompt version (inferred by timestamp; the schema has no direct link) — using
  `trades.confidence` as the predicted win probability, falling back to the
  entry-implied probability for rows without it. The CLOB
  SDK is an optional dependency, commented out in `requirements.txt` — install it separately
  for live trading. `executor.create_clob_client` signs with `POLYMARKET_PRIVATE_KEY` and,
  for a funded deposit wallet, `POLYMARKET_FUNDER_ADDRESS` + `POLYMARKET_SIGNATURE_TYPE`
  (default 3 = POLY_1271 deposit wallet; omit the funder for a plain EOA wallet).

The read-only Textual dashboard (`cli.py dashboard`) lives in `ui/tui.py` and renders
`trades.db` / `docs/status.json` without scanning, classifying, or trading.

### Shared infrastructure

- `config.py` — loads `.env`, defines all thresholds/keys/categories and
  `PROJECT_ROOT`: `DRY_RUN`, `MAX_BET_USD`, `DAILY_SPEND_LIMIT_USD`, `EDGE_THRESHOLD`,
  `MAX_VOLUME_USD`/`MIN_VOLUME_USD`/`MIN_MATERIALITY_DEFAULT`/`MIN_MATERIALITY_BULLISH`/
  `MIN_MATERIALITY_BEARISH`/`MIN_MATERIALITY_GEOPOLITICAL`/`MIN_MATERIALITY_SPORTS`/
  `HIGH_MATERIALITY_THRESHOLD`/`SPEED_TARGET_SECONDS`, and more. `cli.py --threshold`
  mutates `config.DRY_RUN` / `config.MIN_MATERIALITY_DEFAULT` + `config.MIN_MATERIALITY_BULLISH`
  / `config.EDGE_THRESHOLD` at runtime based on CLI flags — modules must read these as
  `config.X` (not via `from locus.config import X`) to see overrides.
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
