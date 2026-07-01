# Locus — Trading-Logic Correctness Review

**Read-only review. No code was modified.** Generated 2026-07-02 against `HEAD=3803ea26`,
while the live agent was running. Scope: math correctness (edge/Kelly/PnL), gate order,
async/concurrency, and reconcile/partial-fill state consistency. Code style and structure
are out of scope (see `docs/AUDIT.md`).

Severity: **critical** = wrong money outcome or a defeated risk control on the live path;
**medium** = wrong numbers/state under realistic conditions, usually with a partial
mitigation; **nit** = real but low-impact or contrived.

---

## Summary

| # | Severity | Where | What |
|---|----------|-------|------|
| 1 | critical | `pipeline.py` | One-position-per-headline cap races across concurrent candidates — one headline can open several positions |
| 2 | critical | `positions.py:790` / `performance.py` / `logger.py` | Resolution closes get status `resolved`, which `LIKE 'closed_%'` filters miss — circuit breaker, Kelly win-rate, live-readiness are blind to resolution PnL |
| 3 | medium | `positions.py:744` | `_close` UPDATE has no `status='open'` guard — concurrent closers can double-realize PnL |
| 4 | medium | `positions.py:766-771` | `close_half` never reduces `token_count` — DB share count diverges from on-chain after every half close |
| 5 | medium | `positions.py:705-727` | Partial live SELL shrinks `token_count` but not `amount_usd`, and never realizes the sale proceeds |
| 6 | medium | `positions.py:268` / `performance.py:593` | Live PnL computed from the cached mid at open, not the actual fill price — systematically overstates live PnL |
| 7 | medium | `edge.py:341-346` | Zero/negative-Kelly signals are floored to `KELLY_MIN_BET_USD` — the system knowingly bets $2 on trades its own confidence says are −EV at these odds |
| 8 | medium | `pipeline.py:838-856` / `event_context.py` | Event switch bypasses the market-structure gates for the market it switches into |
| 9 | medium | `event_context.py:40-46` | `is_categorical` (sum≈1 ± 0.20) can misclassify positively-correlated siblings — the implied play then takes the wrong side |
| 10 | medium | `pipeline.py:343-353` / `logger.py:960` | Confirmation gate counts a cross-posted identical headline as an independent source, and has no materiality floor on confirming rows |
| 11 | medium | `positions.py:863-875` | `reconcile --fix` overwrites `realized_pnl_usd=0`, destroying prior `close_half` realizations |
| 12 | nit | `pipeline.py:1298` / `executor.py:23` | Daily-spend limit is check-then-act across different event locks — bounded overspend |
| 13 | nit | `edge.py:214` | Fee model `rate·p·(1−p)` under-charges vs. a `rate·min(p,1−p)` schedule by up to 2× |
| 14 | nit | `edge.py:223` / `pipeline.py:608` | Blocking momentum HTTP call (2s timeout) runs on the event loop |
| 15 | nit | `executor.py:673` | `takingAmount` fill-field fallback is USDC (not shares) on a SELL — unit confusion if the client ever reports fills that way |
| 16 | nit | `pipeline.py:1332-1338` | Sanity downgrade `executed → resting` happens after the trades row was written as `executed` |
| 17 | nit | `performance.py:215` | `closed_reconciled` phantoms inflate the drawdown bankroll seed with capital never deployed |
| 18 | nit | `pipeline.py:884-889` | Re-entry budget is consumed before the execute-time risk re-check — a lost race burns the single re-entry |
| 19 | nit | `pipeline.py:256-271` | A sports market matching geopolitical keywords gets the *lower* geo materiality floor (0.30 < 0.40) |

Areas verified clean are listed at the end — notably the NO-side PnL sign conventions,
the Kelly formula itself, and the event-level lock's check-then-act coverage.

---

## 1. Math correctness (edge.py, positions.py, performance.py)

### Finding 7 (medium) — negative-Kelly trades still bet $2
`edge.py:336-338` correctly yields a base size of 0 when the Kelly fraction
`(p·b − q)/b` is non-positive ("no edge at these odds", per the docstring). But
`size_position` / `size_position_enhanced` (`edge.py:341-346, 349-363`) then apply
`max(capped, config.KELLY_MIN_BET_USD)`, turning that 0 into a **$2 bet**. This is
reachable on the live path: e.g. bullish, price 0.80 (inside `BULLISH_MAX_PRICE` 0.82),
confidence 0.55 → Kelly is negative, yet materiality 0.6 gives edge 0.12 ≥
`EDGE_THRESHOLD` 0.10, so a signal fires and bets $2. The tests
(`tests/test_sizing.py:68-72`) assert this behavior, so it is a deliberate floor — but
the consequence stands: whenever Claude's own win-probability says the odds are
unfavorable, the system still takes a small systematically −EV position instead of
skipping. The floor makes sense against *dust-sizing* (tiny-but-positive Kelly); applying
it to *zero/negative* Kelly contradicts the `_base_kelly_size` docstring and Kelly logic.

### Finding 13 (nit) — fee model under-charges
`edge.py:214` models the per-share fee as `fee_rate · p · (1−p)`. Since
`p(1−p) = min(p,1−p) · max(p,1−p) ≤ min(p,1−p)`, this under-estimates a
`rate·min(p,1−p)`-style fee schedule (Polymarket's documented form) by a factor of
`max(p,1−p)` — up to 2× at p=0.5, exactly where the pipeline trades most. Fee rates are
non-zero for most traded categories (politics 0.04, crypto 0.07, sports 0.03), so a
fee-heavy market's net edge can clear `EDGE_THRESHOLD` when the true net edge would not.
If `p·(1−p)` is an intentional in-house model this is fine; if it was meant to mirror the
exchange fee formula, it is off by `max(p,1−p)`.

### Otherwise clean
- **Kelly formula** (`_base_kelly_size`): decimal odds `b` are correct for both sides
  (YES: `(1−p)/p`; NO: `p/(1−p)` buying at `1−yes`), `f = (p·b − q)/b` is the standard
  Kelly fraction, half-Kelly and the win-rate scaling do what the comments claim.
  `winrate_factor` verified against its documented key points (0.25→0.25, 0.50→0.625,
  0.75→1.0); `edge_factor` (0.1→1.0, ≥0.2→1.5) and `vol_adj_factor` (0.5→1.0, extremes
  floor 0.6) match their docstrings exactly.
- **Win-rate cache** fails to a neutral 0.5 without poisoning the cache. Correct.
- **Momentum boost** is confirmation-only (never penalizes contradicting drift), bounded
  at +0.05, fails open. Matches the comments. (But see finding 14 on where it runs.)

## 2. PnL sign conventions — verified clean, with one basis flaw

**NO-side sign convention is consistent everywhere.** `pnl_pct` (`positions.py:204-208`),
`position_pnl` (`performance.py:593-603`), and `position_shares` all map NO to the
complement price (`1 − yes`) for both entry and mark, so NO profits when the YES price
falls — verified for the stop-loss, take-profit, drawdown, time-pressure, near-certain
(YES ≥ 0.95 / NO: YES ≤ 0.05, both lock in *wins*), resolution-close, and
`compute_performance` unrealized paths. No sign errors found. `close_on_resolution`
receives the resolved `outcomePrices[0]` and the same formula grades a NO win
(YES→0 ⇒ NO side price → 1) correctly.

### Finding 6 (medium) — live PnL uses the wrong entry basis
`open_position` stores `entry_yes_price = market.yes_price` — the cached Gamma/watcher
mid at open (`positions.py:268-270`) — while a live BUY actually fills at the best ask,
which is above the mid. Every PnL figure (`position_pnl`, `pnl_pct`) then divides by the
too-low mid, **overstating realized and unrealized PnL on every live YES fill** (and
distorting the stop-loss / take-profit trigger points relative to real cost). The true
basis is recoverable — `actual_cost_usd` and `token_count` are both stored (fill price =
`actual_cost_usd / token_count`) — but nothing uses them for PnL; the codebase already
recognizes this exact mid-vs-ask discrepancy for *share counts* (`held_token_shares`
docstring, `executor.py:511-523`) without applying it to PnL. Overstated realized PnL
feeds the Kelly win-rate, circuit breaker, and dashboards.

## 3. Gate order & contradictions (pipeline.py)

Mapped order per candidate: prefilter → dedup cache → classify →
`detect_edge_v2` (direction, price-room, fee, momentum, `EDGE_THRESHOLD`, sizing) →
`gate_trade` (stale → capped → sports_disabled → sports_event_cap →
too_close_to_resolution → price_target → coinflip → low_materiality →
needs_confirmation) → low_consensus → re-entry → correlation → category exposure →
multi-source size adjust → orderbook ∥ CoV → event exposure/switch → circuit breaker →
consume headline → queue. Then at execute time, under the event lock:
correlation → category → event exposure → circuit breaker → daily spend.

For a **single** candidate the order is coherent: no gate reads state that a later gate
of the same candidate modifies, the circuit breaker correctly runs last, and the
deferred-consumption design for the headline cap is right *in intent*. The problems are
cross-candidate and cross-market:

### Finding 1 (critical) — the headline cap races and can be defeated on the mainline path
`gate_trade` **checks** `event.headline in traded_headlines` (`pipeline.py:277-278`) but
the headline is **consumed** only at the very end (`pipeline.py:935-940`), after multiple
`await` points: the re-entry check (`:637`), multi-source lookup (`:728`), the
orderbook+CoV gather (`:777`), `get_open_positions` (`:821`), and the circuit breaker
(`:864`). The comment at `pipeline.py:532-535` claims the gates "are synchronous and
never yield mid-check, so the one-position-per-headline cap … still holds under the
concurrent gather" — **this is false**: the check-then-act pair spans several awaits.

Crucially, all candidates in a batch *are the same headline* matched to N markets and run
concurrently (`asyncio.gather`, `pipeline.py:967-971`). Candidate A passes the `capped`
check, awaits the orderbook fetch; candidate B then runs its `gate_trade` against a
`traded_headlines` set A hasn't updated yet, passes too, and both consume and enqueue.
The execute-time re-check (`_recheck_risk_gates`) never looks at headlines, and the
event-exposure cap only saves the case where both markets share an `event_id`. The
correlation gate needs 3+ correlated positions or >$75 exposure to block, so two slip
through cleanly.

**Consequence in live trading:** one headline matching several correlated markets — the
exact scenario the cap exists for — can open up to `PARALLEL_BATCH_SIZE` positions,
multiplying exposure to a single news event. This isn't an edge case; it's the normal
shape of a strong headline.

### Finding 8 (medium) — event switch bypasses gates for the market it switches into
Every gate before the event-context step (`pipeline.py:838-856`) was evaluated against
the **original** market. When `find_best_outcome` recommends a sibling,
`build_switched_signal` opens the *new* market without re-running:
`too_close_to_resolution`, `price_target_market`, `coinflip_market`, the sports gates,
the category-exposure gate (sibling can be a different inferred category), the orderbook
imbalance gate (different token, different book), and CoV. `find_best_outcome` does
re-check price room, edge threshold, held-markets, and correlation — but nothing else.
The switched signal is also sized with plain `size_position` (no fee subtraction for the
sibling's `fee_rate`, no `vol_adj`/`edge_factor`), and its raw sibling edge is compared
against the primary's raw edge while the primary had to clear the *fee-netted* threshold —
an asymmetry that favors switching. **Consequence:** capital can land in a market class
(e.g. a sibling resolving in 2 hours, or with a one-sided book) that the gate chain would
have rejected outright.

### Finding 9 (medium) — the categorical heuristic can put the implied play on the wrong side
`is_categorical` (`event_context.py:40-46`) treats any ≥2 tracked event siblings whose
YES prices sum to 1.0 ± 0.20 as mutually exclusive, and then derives *opposite*-direction
implied plays (`:98-112`). Events whose markets are **positively correlated** — date
ladders ("…by March?" at 0.35, "…by June?" at 0.60, sum 0.95), threshold ladders, or an
event where only some outcomes fall inside the volume band — can pass the sum test while
not being mutually exclusive. Bullish news on "by March" would then buy **NO** on
"by June", which the same news makes *more* likely YES. **Consequence:** a switched trade
positioned directly against the news. The ±0.20 tolerance is generous enough for this to
occur in practice.

### Finding 10 (medium) — confirmation gate accepts non-independent "confirmation"
The high-materiality gate (`pipeline.py:343-353`) requires ≥2 distinct `news_source`
values via `get_confirming_sources` (`logger.py:960-973`). Two weaknesses:
1. When the *identical headline text* arrives from a second source, the dedup cache path
   logs a `cached` classification row under the second source's name
   (`pipeline.py:560-583`) — and that row counts as an independent confirming source.
   Cross-posting (the same wire story on RSS and NewsAPI) is exactly what happens with
   big obvious news, so the gate's core assumption (independent corroboration) is
   satisfiable by one story.
2. `get_confirming_sources` has no materiality floor — a 0.05-materiality directional
   row confirms a 0.9-materiality signal — unlike its sibling
   `has_multi_source_confirmation`, which requires ≥0.35.

**Consequence:** the gate protecting the worst-graded signal class (obvious,
already-priced news) is much weaker than designed.

### Finding 19 (nit) — geo keyword match lowers the sports floor
In `gate_trade`, a market flagged geopolitical with >7d to resolution gets
`category = "geopolitical"` (`pipeline.py:256-271`), which `get_materiality_threshold`
resolves to floor 0.30 — even if the market's real category is sports (floor 0.40).
Sports keywords like "Korea"/"Olympics" overlap the geo list, so a sports market can
slip under the deliberately-higher sports bar. The dedicated sports gates
(`sports_disabled`, event cap, resolution floor) still apply, limiting the impact.

## 4. Async / concurrency

### The event-level lock itself — verified clean
`_acquire_position_lock` / `_execute_with_lock` (`pipeline.py:1244-1372`) do cover the
full check-then-act sequence: the risk re-check, `execute_trade_async`, and
`positions.open_position` (committed) all run **inside** the lock, so the second
same-event signal's re-read of `get_open_positions` genuinely sees the first's row and
backs off. Lazy lock creation has no await between lookup and insert — safe. Markets
without an `event_id` fall back to a per-category lock, and 44 of 45 positions in the
live DB have an `event_id`, so lock coverage is real. **The specific race the lock was
built for (two same-event signals double-opening) is closed.**

### Finding 3 (medium) — double-close race: `_close` has no status guard
`_close`'s full-close UPDATE (`positions.py:744-749`) is `WHERE id=?` with no
`AND status='open'`, and none of its callers re-check status inside the same
transaction. Three closers run on independent threads with stale snapshots:
`update_and_manage` (every 30s, iterating a `get_open_positions()` snapshot taken before
the loop), `close_on_resolution` (calibration cycle), and `trigger_news_reeval` →
`reevaluate` (per-candidate executor threads). If two of them close the same position,
both add to `realized_pnl_usd` (`COALESCE(realized_pnl_usd,0)+?`), **double-counting the
realized PnL** and double-notifying. In live mode the second `_live_close` is usually
saved by the on-chain balance cap (held=0 → thin-book skip → close not recorded), but in
dry-run — and live when the balance fetch returns None — the double-realize goes
through. All downstream metrics (win rate, breaker, readiness) inherit the error.

### Finding 12 (nit) — daily-spend limit is check-then-act across locks
Both the pre-execute re-check (`pipeline.py:1298-1301`) and `execute_trade`
(`executor.py:23-25`) read `get_daily_pnl()` and then write the trades row later, with no
global lock — signals on *different* events execute concurrently, so N concurrent signals
can each pass while collectively exceeding `DAILY_SPEND_LIMIT_USD` by up to
(N−1)·`MAX_BET_USD`. Bounded and rare, but the limit is soft under concurrency.

### Finding 14 (nit) — blocking calls on the event loop
`detect_edge_v2` is called directly in the coroutine (`pipeline.py:608`), and with
`MOMENTUM_ENABLED` (default true) it makes a synchronous `httpx.get` with a 2s timeout
(`edge.py:132-141`) — plus `find_recent_classification` (`pipeline.py:560`) and
`check_correlation_risk`/`get_open_positions` (`:668-669`) are synchronous DB reads on
the loop. Not a correctness bug, but each candidate can stall the entire pipeline
(news intake, other batches) for up to ~2s — at odds with `SPEED_TARGET_SECONDS`.

## 5. State consistency (reconcile / partial fill / partial sell)

### Verified clean
- **Phantom-fill fix** (`reconcile_order`, `executor.py:693-722`): LIVE → always resting
  (order cancelled) regardless of echoed fill fields; unknown/not-found → resting;
  MATCHED requires a reported fill. Conservative and correct. The partial-BUY path sizes
  the position to `actual_cost_usd`/`actual_shares` and cancels the remainder — the
  open-side partial-fill accounting is consistent end-to-end
  (`_execute_live` → `open_position` notional/token_count).
- **`reconcile_positions` mid-way failure**: the report phase is read-only; the fix phase
  commits once after the loop inside `try/finally`, so an exception mid-way rolls back
  atomically on close — no partially-fixed state. UNKNOWN is never auto-closed. Correct.
- **`_live_close` outcome contract** ("closed"/"partial"/"failed") is honored by `_close`:
  failed and partial outcomes never record a close, keep `status='open'`, and realize $0.

### Finding 2 (critical) — `resolved` status escapes every `closed_%` filter
`close_on_resolution` closes positions with `status="resolved"` (`positions.py:790`),
but four safety/metrics readers filter on `status LIKE 'closed_%'` (or `'closed%'`):
- `compute_circuit_breaker` (`performance.py:197`) — **the breaker never sees
  resolution-realized losses**;
- `get_recent_closed_position_pnls` (`logger.py:923-924`) — the dynamic-Kelly win rate
  never sees them either, so a streak of resolution losses neither trips the breaker nor
  shrinks bet sizing;
- `compute_live_readiness` (`performance.py:76`) and `calibration_report`
  (`performance.py:408`) — the graduation gate and calibration tables exclude them.

Meanwhile `compute_performance`, `get_closed_positions`, and the journal use
`status != 'open'` and *do* include them — confirming the `closed_%` pattern is an
oversight, not intent. Today's DB has no `resolved` rows yet (all 41 closes were
pre-resolution), which is why this hasn't shown up — but riding to resolution is exactly
what happens when a market leaves the tracked volume band and stops being marked (no
stop-loss can fire without price marks), i.e. **the least-supervised losses are the ones
the breaker is blind to**. In live trading, a cluster of ride-to-zero resolutions would
leave the circuit breaker green and Kelly sizing at full recent-win-rate.

### Finding 4 (medium) — `close_half` leaves `token_count` stale
The `fraction < 1.0` success branch of `_close` (`positions.py:766-771`) halves
`amount_usd` but **never touches `token_count`**, even though `_live_close` sold
`position_shares(...) * fraction` real tokens and even returned `remaining_shares`
(discarded in this branch). After a live half-close the DB claims the full original share
count. The next full close derives its SELL size from that inflated `token_count`; it is
rescued only by the on-chain balance cap in `close_position_live` — which is
best-effort and returns None on any balance-fetch error, in which case the SELL is placed
for ~2× the holding and rejected ("not enough balance") → `close_failed` → position stuck
open until a cycle where the balance fetch works. `token_count` and reality diverge on
**every** live half-close.

### Finding 5 (medium) — partial SELL: proceeds never realized, `amount_usd` never reduced
The `outcome == "partial"` branch (`positions.py:705-727`) correctly shrinks
`token_count` to the on-chain remainder and keeps the position open — but it realizes
$0 and leaves `amount_usd` at the full stake, even though real USDC proceeds landed from
the partial fill. Consequences: (a) the sold chunk's PnL is eventually realized at
*whatever price the remainder later exits at*, not the price it actually sold for — if
the price moves between the partial and final sell, realized PnL is simply wrong;
(b) until then, `amount_usd` overstates remaining exposure to the correlation, category,
and event-exposure gates (conservative, but wrong); (c) if the market then *resolves*,
`close_on_resolution` realizes the full original `amount_usd` at the resolution price,
misstating PnL for the already-sold portion. `token_count` and `amount_usd` are
internally inconsistent for the whole life of the partially-closed position.

### Finding 11 (medium) — `reconcile --fix` destroys prior partial realizations
`_mark_reconciled_closed` (`positions.py:863-875`) sets `realized_pnl_usd=0` outright.
Every other close path preserves history via `COALESCE(realized_pnl_usd,0)+?`. A position
that had a `close_half` (realized, say, +$5) and later reconciles as a phantom loses that
recorded $5 from every PnL aggregate. The intent ("excluded from the win-rate
denominator") only needs the *increment* to be 0, not the stored total.

### Finding 15 (nit) — SELL fill-field unit hazard
`reconcile_order` reads the fill from the first present of `size_matched`, `filled_size`,
`matched_size`, `filledAmount`, `filled`, `takingAmount` (`executor.py:673-674`) and
treats it as **shares**. On a SELL, `takingAmount` is the USDC received, not shares. The
canonical `size_matched` is tried first so this is dormant, but if a client release ever
reports sell fills via `takingAmount`, `sold_shares`/`remaining_shares` in
`close_position_live` would be computed in the wrong unit.

### Finding 16 (nit) — sanity downgrade leaves a stale trades row
`_execute_with_lock`'s phantom-fill sanity check (`pipeline.py:1332-1338`) downgrades an
`executed`-with-no-cost result to `resting` **after** `_log_and_return` already wrote the
trades row as `executed`. No position opens (correct), but the trades row still counts
against the daily spend limit (conservative) and the calibrator will grade it as a trade
that never existed.

### Finding 17 (nit) — reconciled phantoms dilute the breaker's drawdown
`compute_circuit_breaker` seeds its equity curve with `sum(amount_usd)` over all
window closes, including `closed_reconciled` phantoms whose capital was never actually
deployed (and whose realized PnL is 0). Each phantom inflates the denominator of the
drawdown fraction, making the breaker *less* sensitive right after reconciliation finds
problems — mildly the wrong direction.

### Finding 18 (nit) — re-entry budget burned before the trade is certain
`_record_reentry` increments the watch row's `reentry_count` once the in-process gate
chain passes (`pipeline.py:884-889`), but the signal can still die in
`_recheck_risk_gates` or execution (resting/rejected). With `MAX_REENTRY_PER_MARKET=1`,
a lost race consumes the market's only re-entry without opening anything.

---

## Explicitly clean (checked, no issue found)

- **NO-position PnL signs** — consistent across `pnl_pct`, `position_pnl`,
  `position_shares`, hard exits, resolution closes, and unrealized marks (section 2).
- **Kelly math and its scaling factors** — formulas match both theory and their
  docstring key-points (section 1), aside from the min-bet floor semantics (finding 7).
- **Event-level lock** — the recheck→execute→open critical section is fully inside the
  lock; the same-event double-open race is genuinely closed (section 4).
- **Phantom-fill defense in `reconcile_order`** — LIVE/unknown/not-found all map to
  resting-with-cancel; partial BUY fills open positions sized to actual cost/shares.
- **`reconcile_positions` atomicity** — single commit after the fix loop; a mid-way
  exception rolls back cleanly; UNKNOWN is never auto-closed.
- **Freshness gating** — age measured at decision time (queue dwell counts), publication
  time preferred with a correct `latency_ms == -1` sentinel fallback; the whale path's
  synthetic events age correctly from `now`.
- **Daily spend accounting** — counts nominal notional including dry-run and excludes
  cancelled/resting orders; conservative in the right direction.
- **`_close` partial/failed outcomes** — never record a close, never realize PnL, keep
  the row open; the "recorded close on an unconfirmed sell" class of bug is fixed.
