"""Replay backtest over the live agent's LOGGED classifications.

Why this exists: ``real.py`` re-classifies headlines with Claude (needs an
API key and costs money) and, on its crypto price-target universe, produced
0 signals. Here the decision inputs are the real Claude outputs the agent
logged while running (direction / materiality / confidence, recovered from
the git history of docs/classifications.json), joined to public Polymarket
price history and resolutions. No LLM calls, no lookahead in the inputs.

For each logged directional call (chronological):
  1. entry YES mid = last hourly price at or before the call (<= 3h old)
  2. gates replicated: per-category/direction materiality floor, price-room
     guards, one open position per market, one trade per headline, daily
     spend cap (DAILY_SPEND_LIMIT_USD), MAX_BET_USD
  3. model decides side + stake:
       legacy     -> edge.detect_edge_legacy (original formula + sizing; the
                     dynamic win-rate factor uses the backtest's own last 20
                     closed trades)
       calibrated -> edge.detect_edge_calibrated with a WALK-FORWARD
                     calibrator: trained only on calls whose 24h label was
                     observable before the current call
  4. fill against the synthetic book (costs.BookModel) + taker fee
  5. exit: hard stop at STOP_LOSS_PCT on the held side, near-certain exit at
     NEAR_CERTAIN_THRESHOLD (both sold into the synthetic bids, taker fee),
     else resolution payout (1/0), else (unresolved) marked at the last
     price minus the half-spread.

NOT modeled (no data): historical books, Claude re-evaluation exits /
take-profit decisions, momentum boost, multi-source confirmation, correlation
and category-exposure gates, CoV novelty checks, passive maker entries.
"""
from __future__ import annotations

import bisect
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean, pstdev

from locus import config
from locus.backtest import replay_data as rd
from locus.backtest.costs import BookModel, buy_cost, sell_proceeds
from locus.core import edge
from locus.core.classifier import Classification
from locus.markets import gamma
from locus.memory import prob_calibration as pc
from locus.sources.news_stream import NewsEvent

HOUR = 3600


def _ts(s: str) -> int:
    return int(datetime.fromisoformat(s.replace(" ", "T")).replace(tzinfo=timezone.utc).timestamp())


class Path:
    def __init__(self, hist: list[dict]):
        self.t = [int(x["t"]) for x in hist]
        self.p = [float(x["p"]) for x in hist]

    def at_or_before(self, ts: int, max_age: int = 3 * HOUR) -> float | None:
        i = bisect.bisect_right(self.t, ts) - 1
        if i < 0 or ts - self.t[i] > max_age:
            return None
        return self.p[i]

    def at_or_after(self, ts: int, max_gap: int = 6 * HOUR) -> float | None:
        i = bisect.bisect_left(self.t, ts)
        if i >= len(self.t) or self.t[i] - ts > max_gap:
            return None
        return self.p[i]

    def after(self, ts: int):
        i = bisect.bisect_right(self.t, ts)
        return zip(self.t[i:], self.p[i:])


@dataclass
class Call:
    ts: int
    question: str
    headline: str
    direction: str
    materiality: float
    adj_materiality: float | None
    confidence: float
    category: str
    fee_rate: float
    yes_mid: float
    path: Path
    resolution: float | None
    label_ts: int
    label_y_yes: float | None  # YES price CALIBRATION_HORIZON_HOURS later


@dataclass
class Trade:
    ts: int
    question: str
    side: str
    stake: float
    shares: float
    entry_vwap: float
    cash_out: float
    exit_ts: int
    cash_in: float
    exit_kind: str
    p_cal: float | None
    confidence: float

    @property
    def pnl(self) -> float:
        return self.cash_in - self.cash_out


def build_calls(rows: list[dict], markets: dict, histories: dict,
                entry: str = "after", minute: dict | None = None,
                latency_s: int = 60) -> tuple[list[Call], dict]:
    """entry="after" (default, conservative): fill at the first hourly price
    AT/AFTER the logged call (<= 1h later) — any move the news already caused
    in the preceding hour is not available to us. entry="before": last price
    at/before the call (<= 3h old) — optimistic, can buy before the move.
    entry="latency" (most realistic): first MINUTE-level price at/after
    call + latency_s (<= 10 min later); exits still use the hourly path."""
    horizon = int(config.CALIBRATION_HORIZON_HOURS * HOUR)
    stats = defaultdict(int)
    calls = []
    paths: dict[str, Path] = {}
    minute_paths: dict[str, Path] = {}
    for r in rows:
        if r.get("direction") not in ("bullish", "bearish") or r.get("confidence") is None:
            continue
        stats["directional_logged"] += 1
        meta = markets.get(r["market_question"]) or {}
        tok = rd._yes_token(meta) if meta else None
        if not tok:
            stats["no_market_match"] += 1
            continue
        hist = histories.get(tok) or []
        if not hist:
            stats["no_price_history"] += 1
            continue
        path = paths.setdefault(tok, Path(hist))
        ts = _ts(r["time"])
        if entry == "latency":
            mp = minute_paths.get(tok)
            if mp is None and minute and minute.get(tok):
                mp = minute_paths.setdefault(tok, Path(minute[tok]))
                # exits walk the union of hourly + minute points, so a call
                # inside an hourly gap still has a post-entry path
                merged = {x["t"]: x["p"] for x in hist}
                merged.update({x["t"]: x["p"] for x in minute[tok]})
                path = paths[tok] = Path([{"t": t, "p": merged[t]} for t in sorted(merged)])
            elif mp is not None:
                path = paths[tok]
            mid = mp.at_or_after(ts + latency_s, max_gap=600) if mp else None
            if mid is not None:
                ts = mp.t[bisect.bisect_left(mp.t, ts + latency_s)]
        elif entry == "after":
            mid = path.at_or_after(ts, max_gap=HOUR)
            if mid is not None:
                ts = path.t[bisect.bisect_left(path.t, ts)]  # decision executes then
        else:
            mid = path.at_or_before(ts)
        if mid is None or not (0.0 < mid < 1.0):
            stats["no_entry_price"] += 1
            continue
        cat = gamma._infer_category(r["market_question"], [])
        calls.append(Call(
            ts=ts, question=r["market_question"], headline=r.get("headline") or "",
            direction=r["direction"], materiality=float(r.get("materiality") or 0.0),
            adj_materiality=r.get("adjusted_materiality"),
            confidence=float(r["confidence"]), category=cat,
            fee_rate=gamma._fee_rate_for_category(cat, r["market_question"]),
            yes_mid=mid, path=path, resolution=rd.resolution_yes(meta),
            label_ts=ts + horizon, label_y_yes=path.at_or_after(ts + horizon),
        ))
        stats["usable"] += 1
    calls.sort(key=lambda c: c.ts)
    return calls, dict(stats)


def _cal_row(c: Call) -> pc.CalRow | None:
    return pc.row_from_grade(c.direction, c.confidence, c.yes_mid, c.label_y_yes, c.category)


def _exit(c: Call, side: str, shares: float, entry_cost_per_share: float,
          book: BookModel) -> tuple[int, float, str]:
    """(exit_ts, cash_in, kind) under hard-stop / near-certain / resolution."""
    for t, p_yes in c.path.after(c.ts):
        side_p = p_yes if side == "YES" else 1.0 - p_yes
        pnl_pct = (side_p - entry_cost_per_share) / entry_cost_per_share * 100
        if pnl_pct <= config.STOP_LOSS_PCT:
            _, cash = sell_proceeds(book.side_book(p_yes, side), shares, c.fee_rate)
            return t, cash, "stop_loss"
        if side_p >= config.NEAR_CERTAIN_THRESHOLD:
            _, cash = sell_proceeds(book.side_book(p_yes, side), shares, c.fee_rate)
            return t, cash, "near_certain"
    if c.resolution is not None:
        won = (c.resolution == 1.0) == (side == "YES")
        return (c.path.t[-1] if c.path.t else c.ts), (shares if won else 0.0), "resolution"
    after_pts = list(c.path.after(c.ts))
    last = after_pts[-1][1] if after_pts else c.yes_mid  # never mark before entry
    side_last = last if side == "YES" else 1.0 - last
    t_last = after_pts[-1][0] if after_pts else c.ts
    return t_last, shares * max(side_last - book.half_spread, 0.0), "marked_unresolved"


@dataclass
class Result:
    model: str
    trades: list[Trade] = field(default_factory=list)
    skips: dict = field(default_factory=lambda: defaultdict(int))
    brier_pcal: list = field(default_factory=list)
    brier_conf: list = field(default_factory=list)


def run(calls: list[Call], model: str, book: BookModel | None = None,
        recal_every_hours: float = 6.0) -> Result:
    from locus.core import pipeline  # materiality floors

    book = book or BookModel()
    res = Result(model)
    open_until: dict[str, int] = {}
    traded_headlines: set[str] = set()
    spend_by_day: dict[str, float] = defaultdict(float)
    closed_pnls: list[tuple[int, float]] = []
    cal = pc.ProbCalibrator([], config.CALIBRATION_MIN_CELL, config.CALIBRATION_PRIOR_STRENGTH)
    cal_built_at = -10**12
    labelled = sorted((c for c in calls if c.label_y_yes is not None), key=lambda c: c.label_ts)
    li = 0
    train: list[pc.CalRow] = []

    orig_wr = edge.get_cached_winrate
    orig_mom = config.MOMENTUM_ENABLED
    config.MOMENTUM_ENABLED = False  # no minute-level history: not modeled
    try:
        for c in calls:
            # walk-forward: add labels observable by now
            while li < len(labelled) and labelled[li].label_ts <= c.ts:
                row = _cal_row(labelled[li])
                if row:
                    train.append(row)
                li += 1
            if c.ts - cal_built_at >= recal_every_hours * HOUR:
                cal = pc.ProbCalibrator(list(train), config.CALIBRATION_MIN_CELL,
                                        config.CALIBRATION_PRIOR_STRENGTH)
                cal_built_at = c.ts

            mat = c.adj_materiality if c.adj_materiality is not None else c.materiality
            if mat < pipeline.get_materiality_threshold(c.direction, c.category):
                res.skips["low_materiality"] += 1
                continue
            if open_until.get(c.question, -1) > c.ts:
                res.skips["position_open"] += 1
                continue
            if c.headline in traded_headlines:
                res.skips["headline_capped"] += 1
                continue

            market = gamma.Market(c.question, c.question, c.category, c.yes_mid,
                                  1 - c.yes_mid, 0, "", True, [], fee_rate=c.fee_rate)
            cls = Classification(direction=c.direction, materiality=c.materiality,
                                 confidence=c.confidence, reasoning="", latency_ms=0, model="logged")
            ev = NewsEvent(headline=c.headline, source="logged", url="",
                           received_at=datetime.fromtimestamp(c.ts, timezone.utc),
                           published_at=None, latency_ms=0)
            side = "YES" if c.direction == "bullish" else "NO"
            if model == "legacy":
                recent = [p for t, p in closed_pnls if t <= c.ts][-config.KELLY_WINRATE_LOOKBACK:]
                wr = (sum(1 for p in recent if p > 0) / len(recent)) if recent else 0.5
                edge.get_cached_winrate = lambda wr=wr: wr
                m = edge.detect_edge_legacy(market, cls, ev)
            else:
                m = edge.detect_edge_calibrated(
                    market, cls, ev, book=book.book(c.yes_mid),
                    win_prob=lambda conf, cat, sp: cal.predict(conf, cat, sp))
            if m is None or m.signal is None:
                res.skips["no_edge" if m is None else (m.skip_reason or "no_signal")] += 1
                continue
            stake = m.signal.bet_amount
            day = datetime.fromtimestamp(c.ts, timezone.utc).strftime("%Y-%m-%d")
            if spend_by_day[day] + stake > config.DAILY_SPEND_LIMIT_USD:
                res.skips["daily_limit"] += 1
                continue

            sbook = book.side_book(c.yes_mid, side)
            shares, vwap, cash_out = buy_cost(sbook, stake, c.fee_rate)
            if shares <= 0:
                res.skips["no_fill"] += 1
                continue
            exit_ts, cash_in, kind = _exit(c, side, shares, cash_out / shares, book)
            spend_by_day[day] += stake
            traded_headlines.add(c.headline)
            open_until[c.question] = exit_ts
            t = Trade(c.ts, c.question, side, stake, shares, vwap, cash_out, exit_ts,
                      cash_in, kind, m.p_cal, c.confidence)
            res.trades.append(t)
            closed_pnls.append((exit_ts, t.pnl))
            closed_pnls.sort()
            if kind == "resolution":
                y = 1.0 if t.cash_in > 0 else 0.0
                if m.p_cal is not None:
                    res.brier_pcal.append((m.p_cal - y) ** 2)
                res.brier_conf.append((c.confidence - y) ** 2)
    finally:
        edge.get_cached_winrate = orig_wr
        config.MOMENTUM_ENABLED = orig_mom
    return res


def metrics(res: Result) -> dict:
    tr = res.trades
    if not tr:
        return {"model": res.model, "trades": 0, "skips": dict(res.skips)}
    pnl = [t.pnl for t in tr]
    deployed = sum(t.cash_out for t in tr)
    by_exit = sorted(tr, key=lambda t: t.exit_ts)
    eq = peak = mdd = 0.0
    daily: dict[str, float] = defaultdict(float)
    for t in by_exit:
        eq += t.pnl
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
        daily[datetime.fromtimestamp(t.exit_ts, timezone.utc).strftime("%Y-%m-%d")] += t.pnl
    dvals = list(daily.values())
    sharpe = (mean(dvals) / pstdev(dvals) * math.sqrt(365)) if len(dvals) > 1 and pstdev(dvals) > 0 else None
    kinds = defaultdict(int)
    for t in tr:
        kinds[t.exit_kind] += 1
    return {
        "model": res.model,
        "trades": len(tr),
        "win_rate_pct": round(100 * sum(1 for p in pnl if p > 0) / len(tr), 1),
        "total_pnl_usd": round(sum(pnl), 2),
        "deployed_usd": round(deployed, 2),
        "roi_pct": round(100 * sum(pnl) / deployed, 2) if deployed else None,
        "avg_pnl_per_trade_usd": round(mean(pnl), 3),
        "max_drawdown_usd": round(mdd, 2),
        "daily_sharpe_annualized": round(sharpe, 2) if sharpe is not None else None,
        "exits": dict(kinds),
        "brier_raw_confidence": round(mean(res.brier_conf), 4) if res.brier_conf else None,
        "brier_p_cal": round(mean(res.brier_pcal), 4) if res.brier_pcal else None,
        "first_trade": datetime.fromtimestamp(tr[0].ts, timezone.utc).isoformat(),
        "last_trade": datetime.fromtimestamp(tr[-1].ts, timezone.utc).isoformat(),
        "skips": dict(res.skips),
    }


def calibration_quality(calls: list[Call]) -> dict:
    """Walk-forward check of the calibrator itself on every usable call with a
    24h label: mean squared error vs the realized side value y of
      - raw Claude confidence (taken as a probability),
      - the market price (side mid at the call),
      - p_cal (trained only on labels observable before the call)."""
    labelled = sorted((c for c in calls if c.label_y_yes is not None), key=lambda c: c.label_ts)
    li, train, cal, built = 0, [], None, -10**12
    se = {"raw_confidence": [], "market_price": [], "p_cal": []}
    excess = []
    for c in calls:
        row = _cal_row(c)
        if row is None:
            continue
        while li < len(labelled) and labelled[li].label_ts <= c.ts:
            r = _cal_row(labelled[li])
            if r:
                train.append(r)
            li += 1
        if c.ts - built >= 6 * HOUR:
            cal = pc.ProbCalibrator(list(train), config.CALIBRATION_MIN_CELL,
                                    config.CALIBRATION_PRIOR_STRENGTH)
            built = c.ts
        p, _, _ = cal.predict(row.confidence, row.category, row.side_price)
        se["raw_confidence"].append((row.confidence - row.y) ** 2)
        se["market_price"].append((row.side_price - row.y) ** 2)
        se["p_cal"].append((p - row.y) ** 2)
        excess.append(row.y - row.side_price)
    n = len(excess)
    return {
        "n_calls": n,
        "mse": {k: round(mean(v), 4) for k, v in se.items() if v},
        "mean_side_value_change_24h": round(mean(excess), 4) if excess else None,
        "share_calls_moving_our_way": round(sum(1 for e in excess if e > 0) / n, 3) if n else None,
    }


def load_inputs(entry: str = "after") -> tuple[list[Call], dict]:
    rows = rd.load_or_extract_classifications()
    qs = [r["market_question"] for r in rows
          if r.get("direction") in ("bullish", "bearish") and r.get("confidence") is not None]
    markets = rd.fetch_markets(qs)
    ranges: dict[str, tuple[int, int]] = {}
    for r in rows:
        if r.get("direction") not in ("bullish", "bearish") or r.get("confidence") is None:
            continue
        tok = rd._yes_token(markets.get(r["market_question"]) or {})
        if not tok:
            continue
        ts = _ts(r["time"])
        lo, hi = ranges.get(tok, (ts, ts))
        ranges[tok] = (min(lo, ts), max(hi, ts))
    import time as _t
    now = int(_t.time())
    ranges = {t: (lo - 2 * 86400, min(hi + 75 * 86400, now)) for t, (lo, hi) in ranges.items()}
    histories = rd.fetch_histories(ranges)
    minute = None
    if entry == "latency":
        pairs = []
        for r in rows:
            if r.get("direction") not in ("bullish", "bearish") or r.get("confidence") is None:
                continue
            tok = rd._yes_token(markets.get(r["market_question"]) or {})
            if tok and histories.get(tok):
                ts = _ts(r["time"])
                pairs.append((tok, ts - ts % 86400))
        minute = rd.fetch_minute_days(pairs)
    return build_calls(rows, markets, histories, entry=entry, minute=minute)


def main(models=("legacy", "calibrated"), books=None,
         entries=("latency", "after", "before")) -> dict:
    books = books or {"base": BookModel()}
    out = {"runs": [], "data": {}, "calibration_quality": {}}
    for entry in entries:
        calls, stats = load_inputs(entry)
        out["data"][entry] = stats
        out["calibration_quality"][entry] = calibration_quality(calls)
        for bname, bm in books.items():
            for model in models:
                r = metrics(run(calls, model, bm))
                r["book"] = {"name": bname, **bm.__dict__}
                r["entry"] = entry
                out["runs"].append(r)
    return out


if __name__ == "__main__":
    import sys
    res = main(books={
        "tight (0.5c half-spread, $80/level)": BookModel(0.005, 0.01, 80.0),
        "base (1c half-spread, $40/level)": BookModel(0.01, 0.01, 40.0),
        "wide (2c half-spread, $20/level)": BookModel(0.02, 0.01, 20.0),
    })
    path = rd.DATA_DIR / "replay_results.json"
    path.write_text(json.dumps(res, indent=2))
    json.dump(res, sys.stdout, indent=2)
