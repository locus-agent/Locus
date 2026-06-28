# Backtest Results

This document summarizes the most recent run of the real-data backtest pilot
(`locus/backtest/real.py`), which replays resolved Polymarket crypto markets through the
classification → edge → gating pipeline using historical headlines, to see what the
strategy *would* have done.

- **Run generated:** 2026-06-11
- **Status:** **Parked** (see *Methodology & Limitations*)
- **Resolved-market window scanned:** 2026-04-25 → 2026-06-11
- **Headlines replayed (date range):** 2026-05-24 → 2026-06-08
- **Source data:** `backtest_results.json` (not committed to the repo)

> ⚠️ **DISCLAIMER — backtest results are NOT indicative of live performance.**
> This is an offline replay over a small, crypto-only sample with known data gaps, and it
> does not model real fill costs (slippage, fees, latency). It is a diagnostic of the
> backtester, not a performance claim. Live trading is tracked separately via the running
> agent's dashboard (`docs/status.json` / the GitHub Pages site); refer to that for actual
> results.

---

## Headline result

**This run produced zero tradeable signals, so no trades were placed.**

The pipeline replayed **48 headline → market classification pairs** across **16 markets**.
Every one was gated out before becoming a signal (`signal: false` for all 48). With **0
signals and 0 trades**, trade-level performance metrics — win rate, PnL, Sharpe, best/worst
trade, accuracy-by-category — **cannot be computed from this data**, and none are reported
below. (The raw `summary` block lists `win_rate_pct: 0.0`, but that is an artifact of a
zero denominator, not a measured win rate.)

---

## Run summary

| Metric | Value |
|---|---|
| Status | Parked |
| Raw market candidates scanned | 1,333 |
| Pages / windows scanned | 493 pages, 10 windows |
| Markets selected for replay | 30 |
| Classification pairs replayed | 48 (across 16 markets) |
| **Signals generated** | **0** |
| **Trades executed** | **0** |
| Total PnL | n/a — no trades |
| Win rate | n/a — no trades |
| Sharpe ratio | n/a — no trades (cannot be computed honestly) |

---

## Why zero signals (diagnostics)

Two independent reasons, both visible in the data:

1. **No price room.** All 48 replayed pairs were on binary crypto price-target markets
   ("Will Bitcoin be above $X on date Y?") whose price was already at an extreme by the time
   the headline landed — **48 / 48** had `price_before > 0.85` or `< 0.15` (i.e. the market
   was effectively already decided). The edge detector skips YES above 0.85 and NO below
   0.15, so there was nowhere for a thesis to pay off. Only **2 / 48** pairs cleared the
   materiality threshold (0.40) at all, and both were bearish calls on a market priced
   ~0.001 — no downside left.
2. **News coverage gap.** Historical headlines were thin: **GDELT succeeded on 0 / 30**
   market groups and **NewsAPI on 16 / 30**, leaving **14 / 30** selected markets with zero
   usable headlines. The market scan reaches 30 days of resolved markets, but the free news
   sources don't go back that far.

---

## Classification replay breakdown

The replay log is the only per-record data available. Each record is a classification
attempt (it is **not** a trade). Materiality is Claude's 0–1 score of how much the news
should move the price.

| Direction | Pairs | Avg materiality | Max materiality |
|---|---:|---:|---:|
| Bullish | 12 | 0.067 | 0.15 |
| Bearish | 13 | 0.250 | 0.95 |
| Neutral | 23 | 0.029 | 0.05 |
| **All** | **48** | **0.098** | **0.95** |

- Signals from these 48 pairs: **0** (every record `signal: false`).
- Average `price_before` across replayed pairs: 0.438, but bimodal — clustered near 0 and 1
  (the extreme-price problem above), not the mid-book where edge is tradeable.

---

## Market selection

The 30 markets selected for replay (context only — not graded trades):

| Field | Value |
|---|---|
| Count | 30 |
| Asset label | crypto (all labeled `bitcoin`) |
| Volume (avg / min / max) | $421,172 / $135,639 / $497,249 |
| Close-time range | 2026-05-04 → 2026-06-08 |
| Actual resolution | 14 resolved YES, 16 resolved NO |

The resolution split is reported for completeness; because the model issued no signals on
these markets, it is **not** a model-accuracy figure.

---

## Metrics intentionally omitted (and why)

Each of these requires at least one graded trade; this run has none, so reporting a number
would be inventing one:

- **Win rate** — needs ≥1 closed trade with an outcome.
- **Total / average PnL** — needs realized trade PnL; no trades occurred.
- **Sharpe ratio** — needs a series of per-trade (or per-day) returns; none exist.
- **Best / worst trade** — no trades to rank.
- **Accuracy by category** — no graded directional calls (and the sample is single-category
  crypto anyway).

---

## Methodology & Limitations

- **Parked pilot.** The backtester (`locus/backtest/real.py`) is flagged **parked** in
  `docs/AUDIT.md` (§1, §5) — it is a standalone script, imported by no production code and
  covered by no tests. Treat its output as exploratory.
- **The recorded blocker (verbatim from the run):** *"Deep-history news coverage. The
  windowed market scan now reaches 30 days of resolved markets, but the news sources don't:
  GDELT proved unreliable (succeeded 0/30 queries) and NewsAPI's free tier only returns
  articles from the last ~30 days, so 14/30 selected markets had zero usable headlines.
  Until a reliable historical headline source is added, the market scan outruns the news
  data."*
- **No fill-cost modeling.** The replay records hold only a single `price_before` and the
  market's final resolution — there are **no fields for slippage, trading fees, or
  execution latency**. So even if signals had fired, the resulting numbers would be
  **optimistic**: they would assume instant, costless fills at the pre-news price.
- **Narrow, near-resolved sample.** Crypto-only, 30 markets, dominated by binary price-target
  markets that were already at price extremes — exactly the kind of market the live pipeline
  also tends to avoid. This is not a representative sample of the live niche universe.
- **Small N.** 48 classification pairs / 16 markets is far too small to draw strategy
  conclusions from even if signals had been generated.

Because of the above, the faster path to real numbers is **live calibration** on the running
agent, not this backtest line.
