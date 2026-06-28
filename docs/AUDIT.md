# Locus — Codebase Inventory & Audit

**Read-only audit. No code was modified.** Generated 2026-06-28 against `HEAD=bac5ef62`.

**Test suite at audit time:** `787 passed` (`python3 -m pytest tests/`, 65 test files, ~4.4s, no network).

**Scope note:** This is a pure inventory pass. The live trading agent (`watch --live`) was
running during the audit; `executor.py`, `positions.py`, and `pipeline.py` were read but not
touched. Every "candidate" below is a *suggestion only* — nothing here was executed.

**Method:** module import graph + per-symbol reference counting via `ast` + word-boundary
regex across `locus/`, `cli.py`, and `tests/` (counting `module.func()` dotted calls and
own-file internal calls, excluding each symbol's own definition body). Config usage scanned
separately, including reads *inside* `config.py` itself (e.g. `get_max_age_seconds`).

---

## 1. PROJECT STRUCTURE

Actual tree of `locus/` (14,734 LOC across 33 `.py` files). One-line purpose per module;
the `<- N` column is how many *other* files (prod + tests) import the module.

```
cli.py                          entry point — 13 subcommands (watch, dashboard, …, reconcile-positions)
locus/
  config.py                     .env loading, all thresholds/keys/categories, PROJECT_ROOT, get_max_age_seconds()
  supervisor.py            <-3  supervise(): restart a long-running asyncio task on crash
  core/
    pipeline.py            <-15 PipelineV2 orchestrator: news→match→classify→edge→gates→execute; gate_trade
    classifier.py          <-22 Claude direction/materiality/confidence classify; Haiku prefilter; CoV; edge-type
    matcher.py             <-2  headline→market matching (keyword overlap ∪ semantic index)
    market_index.py        <-4  persistent Chroma collection of markets (local MiniLM embeddings)
    edge.py                <-18 detect_edge_v2 → Signal; half-Kelly sizing scaled by dynamic win-rate factor
    executor.py            <-12 dry-run log or live CLOB order; held_token_shares; daily spend limit
    positions.py           <-22 open-position tracking, exits, re-entry, correlation/category gates, reconcile_positions
    event_context.py       <-2  per-event exposure cap + best-outcome switching (sibling outcomes)
    whale_tracker.py       <-2  shadow WHALE_WALLETS, investigate missed niche moves
    orderbook.py           <-2  live CLOB orderbook-imbalance gate
    performance.py         <-8  realized-PnL aggregation, live-readiness metrics, circuit breaker
    journal.py             <-2  daily Claude retrospective (21:00 UTC) + missed-opportunity sweep
    telegram_bot.py        <-5  Telegram notifications + interactive /portfolio bot
    export_status.py       <-8  writes docs/status.json (+ archives) for the Pages dashboard
  sources/
    news_stream.py         <-14 NewsAggregator: Twitter/Telegram/RSS/NewsAPI → deduped NewsEvent queue
    scraper.py             <-3  RSS + NewsAPI fetch helpers; dateutil <pubDate> fallback
  markets/
    gamma.py               <-53 Gamma API client, Market model, category inference (most-imported module)
    market_watcher.py      <-2  MarketWatcher: niche-market refresh (5min) + WebSocket prices + index sync
  memory/
    __init__.py            <-—  track record + lessons API (importable as `from locus import memory`)
    logger.py              <-17 SQLite store (trades.db): trades, classifications, calibration, positions, …
    calibrator.py          <-9  grade classifications once markets resolve; price-bucket analysis
    meta_evolver.py        <-3  weekly self-rewrite of the classification prompt
  ui/
    tui.py                 <-1  Textual read-only dashboard over trades.db / status.json
  backtest/
    real.py                <-0  real-data backtest pilot — PARKED
```

### Files not imported anywhere (production)

| File | Status | Evidence |
|---|---|---|
| `locus/backtest/real.py` | **Parked module, imported by 0 files** | No `import`/`from locus.backtest` in any prod or test file. It is a standalone script (`run_pilot()` at module bottom) flagged "parked" in README. Self-contained: pulls in `classifier`, `edge`, `gamma`, `news_stream`, `matcher.extract_keywords`. |

All other modules are imported by ≥1 production module. `__init__.py` package files are
import shims (counted separately).

---

## 2. DEAD CODE CANDIDATES

**Nothing here should be deleted by this audit.** Listed with confidence. "Zero-ref" = the
symbol name appears nowhere outside its own definition body, across `locus/` + `cli.py` +
`tests/`.

### 2a. Truly zero-reference functions (HIGH confidence — defined, never called)

| File:line | Symbol | Evidence |
|---|---|---|
| `locus/core/pipeline.py:85` | `news_age_seconds()` | `grep -n '\bnews_age_seconds\b'` → only the def. A thin wrapper around `news_age()`; callers use `news_age()` directly. **Newly orphaned** by the recent `published_at` work (its last test caller was rewritten). |
| `locus/markets/gamma.py:338` | `fetch_slug_by_condition_id()` | Only the def line matches anywhere. |
| `locus/memory/logger.py:839` | `get_prompt_version_count()` | Only the def line matches anywhere. |
| `locus/memory/logger.py:1142` | `log_run_start()` | Only the def. The `pipeline_runs` table is written nowhere live. |
| `locus/memory/logger.py:1154` | `log_run_end()` | Only the def. Pairs with the unused `log_run_start`. |

> Note: many other `logger.*` functions looked "dead" under a naive grep that excluded
> dotted access — that was a false positive. They are called as `logger.func()` and are
> live. The five above are confirmed against the corrected (dotted-call-aware) scan.

### 2b. Reachable only from tests (MEDIUM confidence — vestigial in production)

| File:line | Symbol | Evidence |
|---|---|---|
| `locus/core/edge.py:25` | `reset_winrate_cache()` | 0 production refs; 4 references, all in `tests/`. Legitimate as test infra (resets a module-level cache), but no prod caller. |
| `locus/memory/meta_evolver.py:210` | `prompt_is_valid()` | 0 production refs; 6 references, all in `tests/`. Production uses the sibling `validate_prompt()` (the list form) directly. `prompt_is_valid` is a thin boolean wrapper tested but never used by prod. |

### 2c. Old ensemble / multi_classifier — DORMANT, not dead (specifically flagged)

`locus/core/multi_classifier.py` is headed: *"DISABLED — Grok is turned off. This module is
kept only for future multi-LLM experiments."*

| Symbol | Reality | Confidence it's removable |
|---|---|---|
| `classify_ensemble()` | Reachable from `pipeline._classify_with_semaphores` **only** when `config.ENSEMBLE_ENABLED` is true (default **false** — env-gated, "parked"). Also referenced by `classifier.py`. | **LOW** — wired in, just flag-gated off. |
| `blend()`, `_classify_grok()` | Internal helpers called by `classify_ensemble()` (same file). Not independently dead — they're the body of the dormant path. | **LOW** — dormant, not dead. |
| `consensus_score()`, `is_low_consensus()` | **Live** — used by `pipeline.py`, `classifier.py`, `export_status.py`, `logger.py`. The low-consensus gate runs whenever an ensemble result carries a score. | n/a — keep. |

So the *whole Grok ensemble path* (`classify_ensemble`/`blend`/`_classify_grok` + the
`GROK_*` config + the `openai` dependency + `GROK_CLASSIFICATION_PROMPT` in `classifier.py`)
is a single dormant unit, gated by `ENSEMBLE_ENABLED=false`. It is *not* orphaned code — it
is conditionally-compiled-off. Treat removal as a feature decision, not a cleanup.

### 2d. `matcher` / `market_watcher` (specifically flagged) — CLEAN

No dead top-level symbols. Apparent "test-only" flags were artifacts of excluding own-file
references:
- `matcher`: `tokenize` (tests + internal), `extract_keywords` (used by `backtest/real.py`),
  `_scored_keyword_matches` (internal helper). All reachable. **If `backtest/real.py` is ever
  removed, `extract_keywords` becomes test-only** — worth re-checking then.
- `market_watcher`: `MarketSnapshot` is instantiated internally by `MarketWatcher`
  (`self.snapshots: dict[str, MarketSnapshot]`); `MarketWatcher` is used by `pipeline.py`. Clean.

---

## 3. CONFIG DRIFT

`locus/config.py` defines **157** top-level UPPERCASE settings. Scan counts `config.X` reads
(and reads inside `config.py` itself). Below are the only genuine anomalies; everything else
is read in ≥1 production path.

### 3a. Defined but never read anywhere (HIGH confidence dead config)

| config.py:line | Variable | Evidence / note |
|---|---|---|
| `:31` | `POLYMARKET_API_KEY` | Only the definition. Live CLOB auth uses `POLYMARKET_PRIVATE_KEY` + `POLYMARKET_FUNDER_ADDRESS` and `create_or_derive_api_key()` at runtime — these three explicit creds are vestigial. |
| `:32` | `POLYMARKET_API_SECRET` | Only the definition (same as above). |
| `:33` | `POLYMARKET_API_PASSPHRASE` | Only the definition (same as above). |
| `:478` | `MISSED_OPPORTUNITY_AUTO_ADJUST_ENABLED` | Only the definition. The auto-adjust path it was meant to gate is not wired; calibrator raises *suggestions* only. |

### 3b. Legacy / duplicate threshold naming (the `MATERIALITY_THRESHOLD_*` → `MIN_MATERIALITY_*` migration)

The old names survive **only as env-var fallbacks**, not as `config` attributes:

```python
# config.py:227-239
_legacy_bullish = os.getenv("MATERIALITY_THRESHOLD_BULLISH")   # env read, NOT a config attr
_legacy_bearish = os.getenv("MATERIALITY_THRESHOLD_BEARISH")
MIN_MATERIALITY_BULLISH = float(os.getenv("MIN_MATERIALITY_BULLISH",
                                _legacy_bullish if _legacy_bullish is not None else "0.34"))
MIN_MATERIALITY_BEARISH = float(os.getenv("MIN_MATERIALITY_BEARISH",
                                _legacy_bearish if _legacy_bearish is not None else "0.27"))
```

Implications:
- There is **no `config.MATERIALITY_THRESHOLD_BULLISH/BEARISH` attribute**. Any doc or code
  referencing `config.MATERIALITY_THRESHOLD_*` is stale (see §4). The live vars are
  `config.MIN_MATERIALITY_{DEFAULT,BULLISH,BEARISH,GEOPOLITICAL,SPORTS}` + `HIGH_MATERIALITY_THRESHOLD`.
- **Default semantics inverted vs. the old docs:** current defaults are bullish **0.34** /
  bearish **0.27** — bearish now has the *lower* bar. README still claims bearish needs a
  *higher* bar (0.4 vs 0.3). The numbers and the rationale are both out of date.

### 3c. False positives explicitly cleared (read internally; NOT drift)

These flagged as "0 prod refs" by a naive scan but are genuinely used:
- `MAX_NEWS_AGE_SECONDS_{DEFAULT,TWITTER,RSS,NEWSAPI,TELEGRAM,TRUTHSOCIAL,GEOPOLITICAL}` —
  all consumed by `config.get_max_age_seconds()` *inside* config.py.
- `SPORTS_MAX_EXPOSURE` — consumed at `config.py:634` building the `MAX_EXPOSURE_PER_CATEGORY` map.

### 3d. Config for the dormant ensemble (see §2c)

`GROK_API_KEY`, `GROK_MODEL`, `ENSEMBLE_ENABLED`, `ENSEMBLE_MIN_CONSENSUS` are read, but the
path they feed is off by default. Keep/remove together with the §2c decision.

---

## 4. README / DOCS MISMATCH

Comparing `README.md` (and `CLAUDE.md`) against actual code.

### 4a. README.md

| # | README claim | Reality |
|---|---|---|
| 1 | **Config table (L271-273):** `MATERIALITY_THRESHOLD_BULLISH=0.3`, `MATERIALITY_THRESHOLD_BEARISH=0.4`, "bearish … needs a higher bar". | Those attribute names don't exist (env-fallback only). Live: `MIN_MATERIALITY_BULLISH=0.34`, `MIN_MATERIALITY_BEARISH=0.27` — bearish bar is **lower**. Also missing `MIN_MATERIALITY_{DEFAULT,GEOPOLITICAL,SPORTS}`. |
| 2 | **All Commands table (L121-131)** lists 9 commands. | 13 are registered. **Missing: `close`, `reconcile-positions`, `evolve`, `suggestion`** (`list`/`review`). |
| 3 | **`.env` example (L57):** `POLYMARKET_API_KEY=… # Optional — live trading only`. | `POLYMARKET_API_KEY` is unused (§3a). Live trading actually needs `POLYMARKET_PRIVATE_KEY` (+ `POLYMARKET_FUNDER_ADDRESS`, `POLYMARKET_SIGNATURE_TYPE`). Setup section never mentions these. |
| 4 | **py-clob-client-v2 not mentioned.** Setup implies `pip install -r requirements.txt` is enough for live trading. | The CLOB SDK is **commented out** in `requirements.txt` ("install separately for live trading"). Live mode silently no-ops without it. Undocumented in README. |
| 5 | **Module Map (L203-227)** omits several live modules. | **Missing: `whale_tracker.py`, `event_context.py`, `telegram_bot.py`, `meta_evolver.py`, `multi_classifier.py`.** |
| 6 | **Risk gates described as 4** (freshness · headline cap · correlation · orderbook), L12/L235. | Actual gate chain also includes: `low_materiality`, `needs_confirmation` (multi-source), `category_limit`, `event_exposure_block`, `circuit_breaker`, `reentry_blocked`, CoV `already_priced_in`, `low_consensus`, and sports-specific gates. |
| 7 | **"daily loss limit"** (L185, L216, L237). | It is a daily **spend/notional** limit (`DAILY_SPEND_LIMIT_USD`, total notional deployed/day, counts dry-run rows), **not** a realized-loss limit. |
| 8 | **Sizing described as static half-Kelly** (L14/182/211/236/285). | Sizing **is** half-Kelly (✓, contra CLAUDE.md) — but README omits that it's scaled by a **dynamic recent-win-rate factor** (`edge.py: winrate_factor`, `KELLY_DYNAMIC_*`), plus expected-edge and vol-adjust factors. "Static half-Kelly" understates it. |
| 9 | **No mention of newer features.** | Missing from README: **`reconcile-positions`** command, **`published_at`/dateutil `<pubDate>` freshness**, **per-source freshness windows** (`MAX_NEWS_AGE_SECONDS_*`), **tiered Haiku→Sonnet classification**, **CoV novelty check**, **circuit breaker**, **re-entry logic**, **event-context outcome switching**, and the **`run_with_watchdog`** crash-restart wrapper in `cli.py` (README only mentions `supervisor.py`, which is the *inner* per-task layer — the *outer* whole-process watchdog is undocumented). |
| 10 | **Architecture diagram (L157-194)** routes news→matcher→classifier→edge→executor with no gate layer, and labels edge.py as the sizer only. | Accurate at a high level but elides the entire `gate_trade` + late-gate chain that lives in `pipeline.py`. |

### 4b. CLAUDE.md (project doc) — also drifted

| CLAUDE.md claim | Reality |
|---|---|
| "sizes positions with **quarter-Kelly**" (intro) and "`size_position` applies quarter-Kelly". | Code is **half-Kelly** (`edge.py: _base_kelly_size` — "we bet HALF Kelly"), then dynamic-win-rate-scaled. README is right here; CLAUDE.md is stale. |
| References **`core/reentry.py`** as a module. | **No such file.** Re-entry logic lives in `positions.py` (`check_reentry_opportunity`, `_check_reentry`) + `logger.py` watch tables. |
| "`cli.py` mutates `config.MATERIALITY_THRESHOLD_BULLISH` + `_BEARISH` (via `--threshold`)". | `cli.py:92-93` mutates `config.MIN_MATERIALITY_DEFAULT` + `MIN_MATERIALITY_BULLISH` (the `MATERIALITY_THRESHOLD_*` attrs don't exist). |
| Command list omits `reconcile-positions`, `evolve`, `suggestion`, `close`. | All four are registered subcommands. |

---

## 5. TEST COVERAGE GAPS

787 tests across 65 files. Coverage is strong on the trading-critical core
(`edge`, `gamma`, `pipeline`, `positions`, `classifier`, `executor`, `logger`,
`news_stream` all have deep, multi-file coverage). Gaps, worst first:

| Module | Coverage | Gap |
|---|---|---|
| `locus/backtest/real.py` (851 LOC) | **None** | Not imported by any test (the apparent "17" references were the common word "real"). Parked, so low priority — but it's the single largest untested file. |
| `locus/ui/tui.py` (338 LOC) | **None** | No test references the TUI at all. Read-only/low-risk, but `LocusTUI` rendering is entirely unexercised. |
| `cli.py` (739 LOC) | **Thin** | Only `run_with_watchdog` (test_watchdog) and an evolve path (test_meta_evolver) are touched. The 13 `cmd_*` handlers — including the new `cmd_reconcile_positions` — are untested at the CLI layer (the underlying `positions.reconcile_positions` *is* tested; the arg-parsing/printing wrapper is not). |
| `locus/sources/scraper.py` (284 LOC) | **Partial, no dedicated file** | Exercised indirectly via `test_freshness_windows.py` (`<pubDate>` parsing) and `test_truthsocial.py`. `scrape_newsapi*`, budget tracking, and `scrape_all` orchestration are untested. No `tests/test_scraper.py`. |
| `locus/core/telegram_bot.py` (344 LOC) | **Thin** (1 file) | `test_telegram_bot.py` exists but the 6 `notify_*` formatters and `start_bot_polling` have no direct callers in prod *or* tests (see §2a-style note) — notification rendering is largely unverified. |
| `meta_evolver`, `whale_tracker`, `event_context`, `market_watcher`, `market_index`, `journal`, `multi_classifier`, `supervisor` | **One dedicated file each** | Adequate but shallow relative to the core; each has a single `test_<module>.py`. Acceptable, flagged for awareness. |

---

## 6. SUGGESTED CLEANUP PLAN

Ordered, each step small and independently testable. **None executed.** Risk = chance of
breaking the running agent or hiding a latent caller.

> Guardrail for every step: the agent is live. Do not touch `executor.py`, `positions.py`,
> or `pipeline.py` *logic*; the only pipeline-adjacent change suggested below (step 2) is the
> removal of a confirmed-orphan wrapper, and even that should wait for a maintenance window.

| # | Step | Risk | How to verify |
|---|---|---|---|
| 1 | **Fix docs only** (README §4a + CLAUDE.md §4b): correct quarter→half-Kelly, `MATERIALITY_THRESHOLD_*`→`MIN_MATERIALITY_*` (and the inverted bearish-bar rationale), add the 4 missing commands, add `POLYMARKET_PRIVATE_KEY`/py-clob-client-v2 setup notes, add the missing modules to the map, remove the `core/reentry.py` reference. | **None** (docs only) | `grep` the corrected names; no code change. |
| 2 | **Remove the 5 zero-ref functions** (§2a) one commit each: `news_age_seconds`, `fetch_slug_by_condition_id`, `get_prompt_version_count`, `log_run_start`, `log_run_end`. | **Low** | `python3 -m pytest tests/` stays at 787; re-grep each name for zero refs immediately before deleting. Do `news_age_seconds` (pipeline.py) only in a maintenance window since the file is hot. |
| 3 | **Delete the 3 unused `POLYMARKET_API_*` config vars + `MISSED_OPPORTUNITY_AUTO_ADJUST_ENABLED`** (§3a). | **Low** | Full suite + `python cli.py verify`; confirm live auth still derives via `create_or_derive_api_key`. |
| 4 | **Decide on `prompt_is_valid` and `reset_winrate_cache`** (§2b): either keep as test infra (document intent) or inline. `reset_winrate_cache` is genuine test infra — likely keep. | **Low** | Suite stays green. |
| 5 | **Ensemble/Grok decision** (§2c/§3d): if Grok stays parked indefinitely, remove `multi_classifier`'s Grok path, `GROK_*` config, `GROK_CLASSIFICATION_PROMPT`, and the `openai` dependency **as one reviewed feature-removal** — *keeping* `consensus_score`/`is_low_consensus` (still live). Otherwise leave as-is. | **Medium** | Suite green; `grep ENSEMBLE_ENABLED`; confirm `pipeline._classify_with_semaphores` default branch untouched. This is a product call, not a cleanup. |
| 6 | **`backtest/real.py` decision** (§1/§5): confirm it's still a wanted pilot. If yes, add a smoke test; if no, move out of the import path. Note it's the only consumer of `matcher.extract_keywords`. | **Medium** | If removed, re-run §2d check on `extract_keywords`; suite green. |
| 7 | **Backfill the worst test gaps** (§5): a `tests/test_scraper.py` for NewsAPI budget/`scrape_all`, and a thin CLI smoke test for `cmd_reconcile_positions` arg parsing. Pure additions. | **None** (additive) | New tests pass; total count rises. |

---

*End of audit. No files other than this report were created or modified.*
