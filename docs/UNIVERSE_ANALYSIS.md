# Market Universe Analysis — Volume Window & Resolution Gate

**Date:** 2026-07-03 (single-day snapshot, ~12:00 UTC)
**Scope:** READ-ONLY analysis. No code or config changes. Proposals only.

**Method.** Full active, orderbook-enabled universe fetched from Gamma
(`/markets`, `order=volume desc`) using volume-cursor pagination to hop
Gamma's offset-2000 wall: 5,142 unique markets, complete scan (no
truncation). Markets parsed with the pipeline's own `gamma._parse_market`,
categories inferred with the pipeline's own `gamma._infer_category`, so all
category counts reflect **what the pipeline itself would see**. Trade history
read from `trades.db` (41 closed positions, +$47.56 total realized);
per-position market volume refetched from Gamma by condition_id (resolved
markets require `closed=true` in the query — they are invisible to the
default query).

---

## Question 1 — What does the volume window hide?

### The universe by volume bucket

| Bucket | Count | Median spread | Politics | Crypto | Sports | Other | Healthy price (0.10–0.90) |
|---|---:|---:|---:|---:|---:|---:|---:|
| <$1k | 4,024 | 0.055 | 226 | 273 | 400 | 3,125 | 2,549 |
| **$1k–$500k (our window)** | **998** | **0.010** | **154** | **64** | **82** | **698** | **392** |
| $500k–$2M | 84 | 0.0015 | 8 | 6 | 29 | 41 | 11 |
| >$2M | 36 | 0.001 | 10 | — | 12 | 14 | 5 |

(Spread is Gamma's `spread` field, available for every market; "healthy
price" = YES price in [0.10, 0.90].)

The liquidity hypothesis is confirmed and is dramatic: median spread is
**0.010 in our window vs 0.0015 in $500k–$2M and 0.001 above $2M** — the
excluded high-volume markets are ~7–10× tighter than what we trade. (And the
<$1k tail we also exclude has a 0.055 median spread — the volume *floor* is
clearly correct.)

### Politics above $500k with healthy prices: only 2 today

Of 18 politics markets above $500k, only **2** have healthy prices, and both
are long-horizon:

| Market | Volume | YES | Spread | Resolves |
|---|---:|---:|---:|---|
| Will the Democrats win the 2028 US Presidential Election? | $830k | 0.585 | 0.01 | 2028-11 |
| Will Mélenchon win the 2027 French presidential election? | $822k | 0.115 | 0.01 | 2027-04 |

The other 16 sit outside 0.10–0.90 — big political markets are mostly
already-decided questions. So **the static cost of the ceiling is small
today**: ~2 tradeable politics markets vs 44 healthy-priced politics markets
inside the window.

### The real cost is dynamic: mid-flight eviction

The ceiling's bigger cost doesn't show in a static snapshot. The market
watcher re-applies the volume band every refresh, so a tracked market whose
volume grows past $500k drops out of `tracked_markets` exactly when news flow
(and volume) is hottest. Our own history shows this pattern — see Question 3:
our three largest wins were all on markets that have since grown past $500k.

---

## Question 2 — What does the resolution-time gate cost?

### Current config

- `MIN_HOURS_TO_RESOLUTION = 4` (default, applies to politics)
- `SPORTS_MIN_HOURS_TO_RESOLUTION = 4` (same value; sports-aware floor exists but is currently identical)

Note the floor is **4 hours**, not days. A market resolving in 1–7 days is
*never* touched by this gate. Any "we're missing short political markets"
effect from this gate can only come from the final 4 hours.

### Time-to-resolution distribution (snapshot)

Politics markets, bucketed by time to resolution:

| Bucket | In-window ($1k–$500k), all | In-window, healthy price | Above $500k |
|---|---:|---:|---:|
| <24h | 1 | 0 | 0 |
| 1–3 days | 0 | 0 | 0 |
| 3–7 days | 0 | 0 | 0 |
| >7 days | 124 | 44 | 18 |
| end_date in past (just resolved / stale) | 27 | 0 | 0 |
| unknown end_date | 2 | 0 | 0 |

**There is essentially no standing supply of day-scale politics markets.**
Short political markets (primaries, PMQ-style novelty markets) exist in
bursts around specific dates — the 27 past-due rows are largely the June 30
primary wave that just resolved. On July 3, between primary dates, the
day-scale shelf is empty. The gate is not what limits closed-trade
accumulation on most days; supply is.

### Gate fire history and the Colorado counterfactual

`too_close_to_resolution` has fired **28 times** total (vs 85 signals, 516
stale, 160 low_materiality — it is a minor gate). Of the 28: ~13 were crypto
coin-flip/price-target markets that other filters (`EXCLUDE_COINFLIP_MARKETS`,
`EXCLUDE_PRICE_TARGET_MARKETS`) now exclude anyway, 9 were one "Will Starmer
say Trump at PMQ" market, and ~6 were genuine primary-night politics
(Colorado ×2, Oklahoma ×2, South Dakota, Virginia).

The Colorado pair (gated 2026-07-01 13:18 UTC, inside the final 4h):

| Market | Gated at YES | Direction | Resolved | Counterfactual |
|---|---:|---|---|---|
| Will Victor Marx win the CO Gov Republican primary? | 0.565 | bullish | YES (0.982) | **+74% win** |
| Will Barbara Kirkmeyer win the CO Gov Republican primary? | 0.411 | bullish | NO (0.012) | **−100% loss** |

At equal sizing the pair nets **negative** (≈ −0.26× stake). This is exactly
the "news-already-priced" failure mode the gate exists for: when results are
arriving live, a 0.41/0.57 price is not mispricing headroom, it's the market
mid-update. One blocked winner and one blocked loser, on the same primary,
from the same news cycle — the gate did not obviously cost us money here.

Our own closed trades tell the same story (time-to-resolution at entry):

| TTR at entry | n | PnL | W/L |
|---|---:|---:|---:|
| >7 days | 25 | **+$74.53** | 11/6 |
| 3–7 days | 1 | +$0.78 | 1/0 |
| 1–3 days | 1 | $0.00 | 0/0 |
| <24h | 10 | **−$24.60** | 4/6 |
| unknown end_date | 4 | −$3.14 | 2/1 |

But the <24h losses are **entirely** crypto "Up or Down" coin-flips (8 of 10
rows, −$35.15 combined) that are now excluded by `EXCLUDE_COINFLIP_MARKETS`.
The two non-coinflip <24h entries (a Democratic Senate primary nominee, a
Trump-speech market) went **2/0 for +$7.53**. Sample of two — but no evidence
that short-horizon *politics* entries lose; short-horizon *coin-flips* did.

---

## Question 3 — Our 41 closed trades by volume bucket

Bucketed by **current** volume (positions don't store entry volume — see
caveats):

| Bucket (current volume) | n | PnL | W/L | Staked |
|---|---:|---:|---:|---:|
| $1k–$500k | 38 | **−$22.37** | 15/13 | $713 |
| $500k–$2M | 2 | **+$63.72** | 2/0 | $41 |
| >$2M | 1 | +$6.22 | 1/0 | $4 |

We never *entered* above $500k — the watcher's band filter makes that
structurally impossible. The three above-ceiling rows are markets that
**grew past $500k after we entered**:

- "Will Trump physically sign US x Iran deal?" — now $877k, **+$63.01** (our biggest win)
- "Will Trump agree to withdraw troops from the Iranian region?" — now $6.6M, +$6.22
- "Will Donald Trump publicly insult Benjamin Netanyahu…" — now $1.9M, +$0.70

Everything that stayed inside the window nets **−$22.37**; everything that
outgrew it nets **+$69.94**. Two readings, both partially true: (a) our best
trades are on markets attracting real volume — the kind the ceiling would
have hidden from us had the volume arrived a week earlier; (b) survivor
confound — markets where the thesis *happens* attract volume, so winners
mechanically migrate to higher buckets. Either way, a market near the
ceiling when news hits is at risk of being evicted (or never tracked) exactly
when it's most tradeable.

---

## Recommendations (proposals only — nothing implemented)

**1. Raise `MAX_VOLUME_USD` from $500k to $2M.**
Evidence: median spread 0.0015 in $500k–$2M vs 0.010 in-window (≈7× tighter
execution); all three of our largest realized wins were on markets that
outgrew the current ceiling; the ceiling also evicts hot markets mid-flight.
Cost is tiny: +84 tracked markets (+8% watcher load).
Main risk: high-volume markets are more efficient — professional flow prices
news in seconds, and our news→classify→execute latency may be too slow to
capture edge there. The classifier's edge was proven in the inefficient
niche; it may not transfer. Mitigation if adopted: keep position sizing
unchanged and let calibration data from the new band accumulate before
trusting it.

**2. Keep `MIN_VOLUME_USD` at $1,000.**
Evidence: the <$1k tail (4,024 markets) has a 0.055 median spread —
structurally untradeable. No change proposed; noted for completeness.

**3. Keep `MIN_HOURS_TO_RESOLUTION` at 4 for politics — do not lower it.**
Evidence: the gate has fired only 28 times ever, mostly on markets other
filters now exclude; the Colorado counterfactual pair nets negative at equal
sizing (one +74% winner, one −100% loser); a 0.41/0.57 price during live
results is the market mid-update, not headroom. The felt cost of "gated
primary markets" is real but the data doesn't show it was profitable to
take. The actual bottleneck for closed-trade accumulation is supply — there
are ~0 day-scale politics markets standing on a typical day; they appear in
bursts around primary dates.

**4. If faster closed-trade accumulation is the goal, target the 1–7-day
politics burst supply, not the gate.** Short primary markets are never
touched by the 4h floor. The reason we traded so few (2 of 41 entries in the
1–7-day band) is matching/supply, not gating. A cheap read-only follow-up:
during the next primary week, log how many 1–7-day politics markets enter
`tracked_markets` and how many produce classifications, to find where the
funnel actually loses them.

**5. Instrumentation: store market volume on `positions` at entry.**
This analysis had to proxy with *current* volume (see caveats). One column
(`entry_volume_usd`) would make the volume-bucket PnL split exact next time.

---

## Caveats (honest ones)

- **Single-day snapshot** (2026-07-03, a post-primary lull). The time-to-
  resolution distribution especially is bursty; a snapshot on primary week
  would look different. Bucket counts and spreads are more stable but still
  one day's reading.
- **Volume buckets for our history use *current* volume, not entry volume.**
  Entry volume isn't stored. All entries were necessarily in-window at entry
  time, so "above-$500k" rows mean "outgrew the ceiling," and the −$22.37 vs
  +$69.94 split partially reflects winners attracting volume (survivorship),
  not just the ceiling hiding good markets.
- **41 closed trades is a small sample**; the W/L and PnL splits by bucket
  and TTR band are directional at best. The 2/0 non-coinflip <24h record is
  two trades.
- **Category counts use the pipeline's keyword inference on question text
  only** — Gamma's `/markets` returns no `tags` field, so both this analysis
  and the live pipeline classify e.g. "Will X win the Colorado Governor
  Republican primary?" as *other* unless a keyword (election, senate, trump,
  …) appears. Politics is undercounted everywhere here, in the same way the
  live pipeline undercounts it — some of the 698 in-window "other" markets
  are really politics. That's consistent for comparisons within this
  document, but absolute politics counts are floors, not totals.
- **Spread is Gamma's top-of-book `spread` field**, not depth. High-volume
  markets are tighter at the top; this says nothing about how much size the
  book absorbs (though for our $4–$25 positions, top-of-book is what matters).
- The Colorado counterfactual assumes the gated signal would have executed
  at the logged classification price, with no slippage, on a primary night —
  optimistic for the winner, so the negative pair-net is if anything
  understated.
