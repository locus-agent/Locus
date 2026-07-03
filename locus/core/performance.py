"""
Dry-run performance aggregation: realized PnL from resolved positions
(calibration exits), unrealized PnL from open positions at current prices.

Honest about its limits: fills are simulated at the last seen price with no
fees or slippage — the dashboard says so in small print.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import httpx
from dateutil import parser as date_parser

from locus import config
from locus.memory import logger

log = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"

# Live-readiness bar: thresholds a dry-run strategy should clear before
# risking real capital (from the polymarket-skills research). Each criterion
# carries its comparator so both the export and the dashboard agree on what
# "pass" means.
LIVE_READINESS_CRITERIA = [
    {"key": "closed_trades", "label": "Closed Trades", "unit": "count",
     "cmp": ">=", "threshold": 20, "threshold_display": "≥ 20"},
    {"key": "win_rate", "label": "Win Rate", "unit": "pct",
     "cmp": ">", "threshold": 55, "threshold_display": "> 55%"},
    {"key": "sharpe_ratio", "label": "Sharpe Ratio", "unit": "ratio",
     "cmp": ">", "threshold": 0.5, "threshold_display": "> 0.50"},
    {"key": "max_drawdown", "label": "Max Drawdown", "unit": "pct",
     "cmp": "<", "threshold": 15, "threshold_display": "< 15%"},
]


def _passes(cmp: str, value: float | None, threshold: float) -> bool | None:
    """Evaluate one criterion; None (N/A) when the metric isn't computable yet."""
    if value is None:
        return None
    if cmp == ">=":
        return value >= threshold
    if cmp == ">":
        return value > threshold
    if cmp == "<":
        return value < threshold
    return None


def compute_live_readiness() -> dict:
    """Track-record gate for graduating from dry-run to live trading.

    Reads only fully-closed positions — any terminal status with a stamped
    closed_at, including 'resolved' (resolution closes) and every 'closed_*';
    partial close_half realizations on still-open positions don't count as a
    trade.

    - closed_trades: how many closed positions exist, excluding break-even
      closes (realized PnL of exactly 0, or NULL) — those are non-events.
    - win_rate: % of those graded closes with positive realized PnL.
    - sharpe_ratio: annualized (×√365) ratio of mean to std of realized PnL
      grouped by close date; null until at least 3 distinct days of data. Below
      7 distinct days the metric label is suffixed "(preliminary)" to flag the
      thin sample; sharpe_days_count is exported so the dashboard can do the same.
    - max_drawdown: deepest peak-to-trough drop of the realized-PnL equity
      curve, as a % of the running peak. The curve is seeded with the capital
      deployed across closed positions so the percentage stays well-defined
      even while cumulative PnL is negative; null with no closed trades.
    """
    conn = logger._conn()
    closed = [
        dict(r) for r in conn.execute(
            "SELECT amount_usd, realized_pnl_usd, closed_at, opened_at FROM positions "
            "WHERE status != 'open' AND closed_at IS NOT NULL ORDER BY closed_at"
        ).fetchall()
    ]
    conn.close()

    # Display-only window: when PERFORMANCE_START_DATE is set, count only
    # positions opened on or after it (same filter as compute_performance), so
    # closed_trades / win_rate / sharpe / max_drawdown all scope to that window.
    since = config.PERFORMANCE_START_DATE
    if since:
        closed = [p for p in closed if (p.get("opened_at") or "") >= since]

    # Graded closes only: a break-even close (realized PnL exactly 0, or NULL) is
    # a non-event — neither a win nor a loss — so it's excluded from closed_trades,
    # win_rate, the Sharpe series, and the drawdown curve alike.
    closed = [p for p in closed if (p["realized_pnl_usd"] or 0) != 0]

    closed_trades = len(closed)

    win_rate = (
        round(sum(1 for p in closed if p["realized_pnl_usd"] > 0)
              / closed_trades * 100, 1)
        if closed_trades else None
    )

    # Sharpe over realized PnL grouped by close date.
    daily: dict[str, float] = defaultdict(float)
    for p in closed:
        day = (p["closed_at"] or "")[:10]
        if day:
            daily[day] += p["realized_pnl_usd"] or 0.0
    sharpe = None
    sharpe_days_count = len(daily)
    if sharpe_days_count >= 3:
        vals = list(daily.values())
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((x - mean) ** 2 for x in vals) / (len(vals) - 1))
        if std > 0:
            sharpe = round(mean / std * math.sqrt(365), 2)

    # Max drawdown of the realized-PnL equity curve, seeded with deployed capital.
    max_drawdown = None
    bankroll = sum(p["amount_usd"] or 0.0 for p in closed)
    if closed_trades and bankroll > 0:
        equity = peak = bankroll
        worst = 0.0
        for p in closed:
            equity += p["realized_pnl_usd"] or 0.0
            peak = max(peak, equity)
            if peak > 0:
                worst = max(worst, (peak - equity) / peak * 100)
        max_drawdown = round(worst, 1)

    values = {
        "closed_trades": closed_trades,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
    }

    # Below 7 distinct close-days the Sharpe is computable (>= 3 days) but the
    # sample is thin — flag it so the dashboard reads "Sharpe Ratio (preliminary)".
    sharpe_preliminary = sharpe is not None and sharpe_days_count < 7

    metrics = []
    for c in LIVE_READINESS_CRITERIA:
        value = values[c["key"]]
        # With zero closed trades nothing is real yet — mark every criterion
        # N/A (not FAIL) so the dashboard reads "accumulating data".
        passed = None if closed_trades == 0 else _passes(c["cmp"], value, c["threshold"])
        label = c["label"]
        if c["key"] == "sharpe_ratio" and sharpe_preliminary:
            label += " (preliminary)"
        metrics.append({
            "key": c["key"],
            "label": label,
            "unit": c["unit"],
            "value": value,
            "threshold_display": c["threshold_display"],
            "pass": passed,
        })

    criteria_met = sum(1 for m in metrics if m["pass"] is True)
    return {
        "ready": all(m["pass"] is True for m in metrics),
        "criteria_met": criteria_met,
        "criteria_total": len(metrics),
        "sharpe_days_count": sharpe_days_count,
        "metrics": metrics,
    }


def compute_circuit_breaker() -> dict:
    """Auto-pause signal: trips when recent realized performance deteriorates.

    Reads positions fully closed within the last 7 days — any terminal status
    with a stamped closed_at, including 'resolved' resolution closes (which a
    'closed_%' pattern used to miss, leaving the breaker blind to
    resolution-realized losses) — and computes:

    - max_drawdown_7d (key 'drawdown_7d'): deepest peak-to-trough drop of the
      realized-PnL equity curve, seeded with the capital deployed across those
      closes, as a *fraction* of the running peak (0-1). Seeding keeps the
      fraction well-defined even while cumulative PnL is negative.
    - sharpe_7d: mean/std of realized PnL grouped by close date (not annualized
      — the threshold is on the raw daily ratio). Null until at least 2 distinct
      close-days with non-zero variance, so a thin window never trips it.

    Trips when drawdown_7d > config.CIRCUIT_BREAKER_DD OR (sharpe_7d is known and)
    sharpe_7d < config.CIRCUIT_BREAKER_SHARPE. When config.CIRCUIT_BREAKER_ENABLED
    is false the metrics are still returned but triggered is always False
    (reason 'disabled').

    config.CIRCUIT_BREAKER_START_DATE, when set, adds a floor on closed_at so
    positions closed before that date (e.g. legacy losses from before a strategy
    change) are excluded from both the drawdown and Sharpe calculation.

    Returns {triggered: bool, reason: str, metrics: dict}.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")

    # Always within the rolling 7-day window; optionally also at/after a
    # configured start date so old closes can't keep the breaker tripped.
    clauses = ["status != 'open'", "closed_at >= ?"]
    params: list = [since]
    if config.CIRCUIT_BREAKER_START_DATE:
        clauses.append("closed_at >= ?")
        params.append(config.CIRCUIT_BREAKER_START_DATE)

    conn = logger._conn()
    closed = [
        dict(r) for r in conn.execute(
            "SELECT amount_usd, realized_pnl_usd, closed_at FROM positions "
            "WHERE " + " AND ".join(clauses) + " ORDER BY closed_at",
            params,
        ).fetchall()
    ]
    conn.close()

    # Max drawdown of the realized-PnL equity curve, seeded with deployed capital.
    drawdown_7d = 0.0
    bankroll = sum(p["amount_usd"] or 0.0 for p in closed)
    if closed and bankroll > 0:
        equity = peak = bankroll
        for p in closed:
            equity += p["realized_pnl_usd"] or 0.0
            peak = max(peak, equity)
            if peak > 0:
                drawdown_7d = max(drawdown_7d, (peak - equity) / peak)

    # Sharpe over realized PnL grouped by close date (raw daily ratio).
    daily: dict[str, float] = defaultdict(float)
    for p in closed:
        day = (p["closed_at"] or "")[:10]
        if day:
            daily[day] += p["realized_pnl_usd"] or 0.0
    sharpe_7d = None
    if len(daily) >= 2:
        vals = list(daily.values())
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((x - mean) ** 2 for x in vals) / (len(vals) - 1))
        if std > 0:
            sharpe_7d = round(mean / std, 2)

    metrics = {
        "drawdown_7d": round(drawdown_7d, 4),
        "sharpe_7d": sharpe_7d,
        "closed_trades_7d": len(closed),
    }

    if not config.CIRCUIT_BREAKER_ENABLED:
        return {"triggered": False, "reason": "disabled", "metrics": metrics}

    reasons = []
    if drawdown_7d > config.CIRCUIT_BREAKER_DD:
        reasons.append(
            f"7d drawdown {drawdown_7d * 100:.1f}% > {config.CIRCUIT_BREAKER_DD * 100:.0f}% limit"
        )
    if sharpe_7d is not None and sharpe_7d < config.CIRCUIT_BREAKER_SHARPE:
        reasons.append(
            f"7d Sharpe {sharpe_7d:.2f} < {config.CIRCUIT_BREAKER_SHARPE:.2f} limit"
        )

    return {
        "triggered": bool(reasons),
        "reason": "; ".join(reasons),
        "metrics": metrics,
    }


# --- calibration-report analytics (read-only) --------------------------------

# Predicted-materiality buckets for the calibration report. Lower edge is the
# trade floor (MIN_MATERIALITY_BEARISH 0.27); anything below that or NULL lands
# in a catch-all so no closed trade is silently dropped from the tables.
MATERIALITY_BUCKETS = [
    (0.27, 0.40, "[0.27-0.40)"),
    (0.40, 0.50, "[0.40-0.50)"),
    (0.50, 0.60, "[0.50-0.60)"),
    (0.60, 0.70, "[0.60-0.70)"),
    (0.70, float("inf"), "[0.70+]"),
]

# Category print order (matches MARKET_CATEGORIES); anything else after these.
CATEGORY_ORDER = ["politics", "crypto", "tech", "ai", "sports", "geopolitical", "other"]

# Entry-price (YES) buckets for longshot analysis. Cheap longshots (a 0.07 YES)
# are a known loss pattern; this surfaces whether they bleed systematically.
ENTRY_PRICE_BUCKETS = [
    (0.0, 0.15, "[<0.15 longshot]"),
    (0.15, 0.35, "[0.15-0.35]"),
    (0.35, 0.65, "[0.35-0.65]"),
    (0.65, 0.85, "[0.65-0.85]"),
    (0.85, float("inf"), "[>0.85]"),
]

# Entry-volume buckets: the market's USD volume when the position opened
# (positions.entry_volume_usd). Rows from before the column existed are NULL
# and reported as their own 'unknown (pre-column)' bucket, never dropped.
ENTRY_VOLUME_BUCKETS = [
    (0.0, 50_000, "[<$50k]"),
    (50_000, 200_000, "[$50k-$200k]"),
    (200_000, 500_000, "[$200k-$500k]"),
    (500_000, float("inf"), "[>$500k]"),
]

# Time-to-resolution buckets, in hours (end_date - opened_at).
TTR_BUCKETS = [
    (0.0, 24.0, "[<24h]"),
    (24.0, 168.0, "[1-7 days]"),
    (168.0, float("inf"), "[>7 days]"),
]

# How many trades to list in the best/worst leaderboards.
TOP_TRADES_N = 5

# Below this many closed trades the report is not statistically meaningful.
CALIBRATION_MIN_SAMPLE = 25

# Brier reference: always predicting 0.5 on true 50/50 outcomes scores 0.25.
BRIER_RANDOM_REFERENCE = 0.25

# Below this many scored trades a per-bucket Brier is directional at best.
BRIER_MIN_GROUP_N = 10


def _is_win(row: dict) -> bool:
    """A trade is a win iff its realized PnL is strictly positive. NULL/0 (a
    break-even or never-realized close) is not a win."""
    return (row.get("realized_pnl_usd") or 0.0) > 0


def _group_stats(rows: list[dict]) -> dict:
    """n / wins / win-rate% / total PnL / avg PnL for a group of closed positions.
    Empty group yields zero counts and None rates so callers render '—' not 0%."""
    n = len(rows)
    total = sum((r.get("realized_pnl_usd") or 0.0) for r in rows)
    wins = sum(1 for r in rows if _is_win(r))
    return {
        "n": n,
        "wins": wins,
        "win_rate": round(wins / n * 100, 1) if n else None,
        "total_pnl": round(total, 2),
        "avg_pnl": round(total / n, 2) if n else None,
    }


def _materiality_label(m: float | None) -> str:
    """Bucket label for a predicted materiality, gracefully handling NULL and
    sub-floor (legacy 0.25) values so they're surfaced, not dropped."""
    if m is None:
        return "unknown"
    if m < MATERIALITY_BUCKETS[0][0]:
        return "[<0.27]"
    for lo, hi, label in MATERIALITY_BUCKETS:
        if lo <= m < hi:
            return label
    return MATERIALITY_BUCKETS[-1][2]


def _range_label(v: float | None, buckets: list[tuple]) -> str:
    """Half-open [lo, hi) bucket label for a numeric value; NULL/out-of-range ->
    'unknown' so nothing is silently dropped."""
    if v is None:
        return "unknown"
    for lo, hi, label in buckets:
        if lo <= v < hi:
            return label
    return "unknown"


def _hours_to_resolution(opened_at, end_date) -> float | None:
    """Hours from a position's open to its market's resolution (end_date), or
    None when either timestamp is missing/unparseable. opened_at is stored as
    naive UTC and end_date as ISO-with-Z; both are flattened to naive UTC so the
    subtraction never raises on a tz mismatch."""
    if not opened_at or not str(end_date).strip():
        return None
    try:
        o = date_parser.parse(str(opened_at)).replace(tzinfo=None)
        e = date_parser.parse(str(end_date)).replace(tzinfo=None)
    except (ValueError, TypeError, OverflowError):
        return None
    return (e - o).total_seconds() / 3600.0


# --- Brier score (probability calibration) -----------------------------------
#
# Brier = mean((predicted win probability - realized outcome)^2) over closed
# positions whose outcome is knowable (realized PnL strictly > 0 is a win = 1,
# strictly < 0 a loss = 0; break-even/NULL closes are unknowable and excluded).
#
# Prediction proxy, per row:
#   1. trades.confidence — Claude's stored win-probability estimate (0.5-1.0,
#      the number Kelly sizing used). This IS a classification-implied
#      probability, so it's preferred whenever stored.
#   2. Fallback (confidence NULL, e.g. legacy rows): the ENTRY-IMPLIED
#      probability — the market's own price for our side at entry (entry
#      YES price for a YES position, 1 - price for NO). Note what that
#      measures: the market's calibration at our entry, not Claude's; rows
#      using it are counted separately so the report is honest about the mix.

def _brier_outcome(r: dict) -> float | None:
    """1.0 win / 0.0 loss from realized PnL; None when unknowable (break-even
    or never-realized close)."""
    pnl = r.get("realized_pnl_usd")
    if pnl is None or pnl == 0:
        return None
    return 1.0 if pnl > 0 else 0.0


def _brier_prediction(r: dict) -> tuple[float, str] | None:
    """(predicted win probability, proxy name) for a closed position, or None
    when neither proxy is available."""
    conf = r.get("confidence")
    if conf is not None:
        return float(conf), "confidence"
    entry = r.get("entry_yes_price")
    if entry is None:
        return None
    p = float(entry) if r.get("side") == "YES" else 1.0 - float(entry)
    return p, "entry_implied"


def _prompt_version_label(trade_created_at: str | None,
                          versions: list[dict]) -> str:
    """Prompt version in force when a trade was classified, inferred by time.

    No direct positions -> prompt_versions link exists in the schema (trades
    and classifications carry no version column), but the classifier loads the
    LATEST prompt_versions row at each call (classifier.get_active_prompt), so
    the version in force at trade time is the newest row created at/before the
    trade's created_at — 'v0 (baseline)' means the hardcoded pre-evolution
    prompt. Timestamps are SQLite 'YYYY-MM-DD HH:MM:SS' strings in UTC, so
    string comparison orders correctly."""
    if not trade_created_at:
        return "unknown (no linked trade)"
    active = None
    for v in versions:  # ascending by version; created_at is monotone with it
        if (v.get("created_at") or "") <= trade_created_at:
            active = v["version"]
    return f"v{active}" if active is not None else "v0 (baseline)"


def _brier_of(scored: list[dict]) -> dict:
    """n / Brier score for a group of scored rows (each {'pred', 'outcome'})."""
    n = len(scored)
    if not n:
        return {"n": 0, "brier": None}
    b = sum((s["pred"] - s["outcome"]) ** 2 for s in scored) / n
    return {"n": n, "brier": round(b, 4)}


def compute_brier_report(rows: list[dict], versions: list[dict]) -> dict:
    """Brier analysis over closed-position rows (see calibration_report's
    SELECT for the expected columns: realized_pnl_usd, confidence, side,
    entry_yes_price, category, trade_created_at).

    Returns {overall, proxy_counts, excluded, by_category, by_prompt_version,
    prompt_versions_known} — by_* entries are {label, n, brier}."""
    scored: list[dict] = []
    proxy_counts = {"confidence": 0, "entry_implied": 0}
    excluded = 0
    for r in rows:
        outcome = _brier_outcome(r)
        pred = _brier_prediction(r)
        if outcome is None or pred is None:
            excluded += 1
            continue
        p, proxy = pred
        proxy_counts[proxy] += 1
        scored.append({"row": r, "pred": p, "outcome": outcome})

    def _grouped_brier(key_fn, order: list[str] | None = None) -> list[dict]:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for s in scored:
            buckets[key_fn(s["row"])].append(s)
        labels = list(buckets.keys())
        if order:
            ranked = [l for l in order if l in buckets]
            ranked += sorted(l for l in labels if l not in order)
        else:
            ranked = sorted(labels)
        return [{"label": l, **_brier_of(buckets[l])} for l in ranked]

    by_category = _grouped_brier(
        lambda r: (r.get("category") or "").strip() or "unknown",
        order=CATEGORY_ORDER + ["unknown"],
    )
    version_order = [f"v{v['version']}" for v in versions]
    by_prompt_version = _grouped_brier(
        lambda r: _prompt_version_label(r.get("trade_created_at"), versions),
        order=["v0 (baseline)"] + version_order + ["unknown (no linked trade)"],
    )

    return {
        "overall": _brier_of(scored),
        "proxy_counts": proxy_counts,
        "excluded": excluded,
        "by_category": by_category,
        "by_prompt_version": by_prompt_version,
        "prompt_versions_known": bool(versions),
    }


def _grouped(rows: list[dict], key_fn, order: list[str] | None = None) -> list[dict]:
    """Group rows by key_fn(row) and return [{'label', **stats}], ordered by
    `order` first (only labels that occur) then any remaining labels sorted."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[key_fn(r)].append(r)
    labels = list(buckets.keys())
    if order:
        ranked = [l for l in order if l in buckets]
        ranked += sorted(l for l in labels if l not in order)
    else:
        ranked = sorted(labels)
    return [{"label": l, **_group_stats(buckets[l])} for l in ranked]


def calibration_report() -> dict:
    """Read-only calibration analysis of CLOSED positions (every terminal
    status with a stamped closed_at, including 'resolved') joined to their
    trade's classification data.

    Pure read: opens its own connection, runs a single SELECT, and never writes,
    migrates, or touches trading logic — safe to run while the live agent trades.

    Returns a structured dict (rendered by format_calibration_report):
      - summary: n, wins, win_rate, total_pnl, small_sample flag + threshold
      - by_materiality / by_category / by_direction / by_fill_quality /
        by_exit_reason / by_entry_price / by_time_to_resolution /
        by_entry_volume: each a list of
        {label, n, wins, win_rate, total_pnl, avg_pnl}
      - best_trades / worst_trades: up to TOP_TRADES_N {market_question, direction,
        entry_yes_price, realized_pnl_usd} sorted by realized PnL

    Missing/NULL fields are tolerated: materiality/direction/category/exit_reason
    and out-of-range entry price / unparseable resolution times fall back to an
    'unknown' bucket, and positions without actual_cost_usd are reported as a
    separate 'unknown' fill-quality group rather than misclassified.
    """
    conn = logger._conn()
    rows = [
        dict(r) for r in conn.execute(
            """SELECT p.status, p.category, p.side, p.entry_yes_price, p.amount_usd,
                      p.actual_cost_usd, p.realized_pnl_usd, p.exit_reason,
                      p.market_question, p.opened_at, p.end_date, p.entry_volume_usd,
                      t.materiality AS materiality, t.classification AS direction,
                      t.confidence AS confidence, t.created_at AS trade_created_at
               FROM positions p
               LEFT JOIN trades t ON p.trade_id = t.id
               WHERE p.status != 'open' AND p.closed_at IS NOT NULL"""
        ).fetchall()
    ]
    # Prompt versions for the Brier-by-prompt-version breakdown (also a pure
    # read; versions are matched to trades by timestamp — see
    # _prompt_version_label for why that's the only linkage the schema has).
    prompt_versions = [
        dict(r) for r in conn.execute(
            "SELECT version, created_at FROM prompt_versions ORDER BY version ASC"
        ).fetchall()
    ]
    conn.close()

    summary = _group_stats(rows)
    summary["small_sample"] = summary["n"] < CALIBRATION_MIN_SAMPLE
    summary["small_sample_threshold"] = CALIBRATION_MIN_SAMPLE

    by_materiality = _grouped(
        rows, lambda r: _materiality_label(r.get("materiality")),
        order=[b[2] for b in MATERIALITY_BUCKETS] + ["[<0.27]", "unknown"],
    )
    by_category = _grouped(
        rows, lambda r: (r.get("category") or "").strip() or "unknown",
        order=CATEGORY_ORDER + ["unknown"],
    )
    by_direction = _grouped(
        rows, lambda r: (r.get("direction") or "").strip() or "unknown",
        order=["bullish", "bearish", "neutral", "unknown"],
    )
    by_exit_reason = _grouped(
        rows, lambda r: (r.get("exit_reason") or "").strip() or "unknown",
    )

    # Fill quality: a fill that cost < 50% of the position's nominal size was a
    # thin/partial fill. actual_cost_usd is NULL for legacy/dry-run rows — those
    # can't be judged, so they're their own 'unknown' group, never lumped in.
    def _fill_label(r: dict) -> str:
        cost = r.get("actual_cost_usd")
        nominal = r.get("amount_usd") or 0.0
        if cost is None:
            return "unknown (no cost data)"
        if nominal > 0 and cost < 0.5 * nominal:
            return "partial fill (<50% nominal)"
        return "full fill (>=50% nominal)"

    by_fill_quality = _grouped(
        rows, _fill_label,
        order=["partial fill (<50% nominal)", "full fill (>=50% nominal)",
               "unknown (no cost data)"],
    )

    by_entry_price = _grouped(
        rows, lambda r: _range_label(r.get("entry_yes_price"), ENTRY_PRICE_BUCKETS),
        order=[b[2] for b in ENTRY_PRICE_BUCKETS] + ["unknown"],
    )
    by_time_to_resolution = _grouped(
        rows,
        lambda r: _range_label(
            _hours_to_resolution(r.get("opened_at"), r.get("end_date")), TTR_BUCKETS),
        order=[b[2] for b in TTR_BUCKETS] + ["unknown"],
    )

    # Entry volume: NULL means the row predates the entry_volume_usd column (or
    # the open-time volume was unknowable) — its own labeled group, never lumped
    # into a numeric bucket.
    def _entry_volume_label(r: dict) -> str:
        v = r.get("entry_volume_usd")
        if v is None:
            return "unknown (pre-column)"
        return _range_label(v, ENTRY_VOLUME_BUCKETS)

    by_entry_volume = _grouped(
        rows, _entry_volume_label,
        order=[b[2] for b in ENTRY_VOLUME_BUCKETS] + ["unknown (pre-column)", "unknown"],
    )

    # Best/worst leaderboards: sort by realized PnL (NULL treated as 0.0). Best is
    # highest-first, worst is lowest-first; with < 2*TOP_TRADES_N closed trades the
    # two lists naturally overlap, which is fine for a small book.
    def _pnl(r: dict) -> float:
        return r.get("realized_pnl_usd") or 0.0

    by_pnl = sorted(rows, key=_pnl)

    def _trade_entry(r: dict) -> dict:
        return {
            "market_question": r.get("market_question") or "(unknown market)",
            "direction": (r.get("direction") or "").strip() or "unknown",
            "entry_yes_price": r.get("entry_yes_price"),
            "realized_pnl_usd": round(_pnl(r), 2),
        }

    worst_trades = [_trade_entry(r) for r in by_pnl[:TOP_TRADES_N]]
    best_trades = [_trade_entry(r) for r in by_pnl[::-1][:TOP_TRADES_N]]

    return {
        "summary": summary,
        "by_materiality": by_materiality,
        "by_category": by_category,
        "by_direction": by_direction,
        "by_fill_quality": by_fill_quality,
        "by_exit_reason": by_exit_reason,
        "brier": compute_brier_report(rows, prompt_versions),
        "by_entry_price": by_entry_price,
        "by_time_to_resolution": by_time_to_resolution,
        "by_entry_volume": by_entry_volume,
        "best_trades": best_trades,
        "worst_trades": worst_trades,
    }


def _fmt_pct(v: float | None) -> str:
    return f"{v:.1f}%" if v is not None else "—"


def _fmt_usd(v: float | None) -> str:
    return f"${v:+,.2f}" if v is not None else "—"


def _render_table(title: str, entries: list[dict], label_header: str,
                  purpose: str | None = None) -> list[str]:
    """Render one section as a fixed-width plain-text table."""
    lines = ["", title]
    if purpose:
        lines.append(f"  Purpose: {purpose}")
    if not entries:
        lines.append("  (no data)")
        return lines
    lines.append(
        f"  {label_header:<28}{'N':>5}{'Win%':>9}{'TotalPnL':>14}{'AvgPnL':>13}"
    )
    lines.append("  " + "-" * 67)
    for e in entries:
        lines.append(
            f"  {e['label']:<28}{e['n']:>5}{_fmt_pct(e['win_rate']):>9}"
            f"{_fmt_usd(e['total_pnl']):>14}{_fmt_usd(e['avg_pnl']):>13}"
        )
    return lines


def _fmt_brier(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "—"


def _render_brier_table(title: str, entries: list[dict], label_header: str) -> list[str]:
    """Render one Brier breakdown as a fixed-width table with small-n flags."""
    lines = ["", f"  {title}"]
    if not entries:
        lines.append("  (no data)")
        return lines
    lines.append(f"  {label_header:<28}{'N':>5}{'Brier':>10}")
    lines.append("  " + "-" * 43)
    for e in entries:
        flag = "  (small n)" if 0 < e["n"] < BRIER_MIN_GROUP_N else ""
        lines.append(
            f"  {e['label']:<28}{e['n']:>5}{_fmt_brier(e['brier']):>10}{flag}"
        )
    return lines


def _render_brier_section(brier: dict) -> list[str]:
    """Render the Brier score section: overall + by category + by prompt
    version, with the prediction proxy and the version linkage spelled out."""
    pc = brier["proxy_counts"]
    lines = ["", "6. BRIER SCORE (probability calibration)"]
    lines.append(
        "  Prediction: Claude's stored win-probability estimate "
        f"('confidence', n={pc['confidence']}) when present;"
    )
    lines.append(
        "  else the entry-implied probability — entry price for YES-side, "
        f"1-price for NO-side (n={pc['entry_implied']})."
    )
    lines.append(
        "  Outcome: realized win (1) / loss (0). Break-even/unrealized closes "
        f"excluded (n={brier['excluded']})."
    )
    lines.append(
        f"  Reference: {BRIER_RANDOM_REFERENCE} = random guessing on a 50/50 "
        "market; lower is better."
    )
    overall = brier["overall"]
    lines.append("")
    lines.append(
        f"  Overall Brier score : {_fmt_brier(overall['brier'])} "
        f"(n={overall['n']})"
    )
    lines += _render_brier_table("BRIER SCORE BY CATEGORY",
                                 brier["by_category"], "Category")
    lines.append("")
    lines.append("  BRIER SCORE BY PROMPT VERSION")
    lines.append(
        "  Note: the schema has no direct position->prompt_version link; the "
        "version is inferred"
    )
    lines.append(
        "  by time (the newest prompt version created at/before each trade — "
        "the one the"
    )
    lines.append(
        "  classifier had loaded). 'v0 (baseline)' = the hardcoded "
        "pre-evolution prompt."
    )
    if not brier["prompt_versions_known"]:
        lines.append(
            "  No evolved prompt versions exist yet — every trade used the "
            "baseline prompt."
        )
    entries = brier["by_prompt_version"]
    if not entries:
        lines.append("  (no data)")
    else:
        lines.append(f"  {'Prompt version':<28}{'N':>5}{'Brier':>10}")
        lines.append("  " + "-" * 43)
        for e in entries:
            flag = "  (small n)" if 0 < e["n"] < BRIER_MIN_GROUP_N else ""
            lines.append(
                f"  {e['label']:<28}{e['n']:>5}{_fmt_brier(e['brier']):>10}{flag}"
            )
    return lines


def _render_trade_list(title: str, trades: list[dict]) -> list[str]:
    """Render a best/worst leaderboard as a fixed-width plain-text table."""
    lines = ["", title]
    if not trades:
        lines.append("  (no data)")
        return lines
    lines.append(
        f"  {'Market':<45}{'Direction':>10}{'Entry':>8}{'PnL':>13}"
    )
    lines.append("  " + "-" * 74)
    for t in trades:
        q = (t["market_question"] or "")[:44]
        entry = t["entry_yes_price"]
        entry_s = f"{entry:.3f}" if entry is not None else "—"
        lines.append(
            f"  {q:<45}{t['direction']:>10}{entry_s:>8}"
            f"{_fmt_usd(t['realized_pnl_usd']):>13}"
        )
    return lines


def format_calibration_report(report: dict) -> str:
    """Render calibration_report() output as a plain-text report (no rich markup)."""
    s = report["summary"]
    out: list[str] = []
    out.append("=" * 69)
    out.append("  CALIBRATION REPORT  (read-only — closed positions only)")
    out.append("=" * 69)
    out.append("")
    out.append(f"  Total closed trades : {s['n']}")
    out.append(f"  Overall win rate    : {_fmt_pct(s['win_rate'])} ({s['wins']}/{s['n']} wins)")
    out.append(f"  Total realized PnL  : {_fmt_usd(s['total_pnl'])}")
    if s["small_sample"]:
        out.append("")
        out.append("  " + "!" * 65)
        out.append(f"  !! SMALL SAMPLE WARNING: only {s['n']} closed trades "
                   f"(< {s['small_sample_threshold']}).")
        out.append("  !! Win rates and per-bucket PnL below are NOT statistically")
        out.append("  !! reliable — treat every number as directional, not conclusive.")
        out.append("  " + "!" * 65)

    out += _render_table(
        "1. CALIBRATION BY MATERIALITY BUCKET", report["by_materiality"],
        "Materiality",
        purpose="does higher predicted materiality mean a higher win rate?",
    )
    out += _render_table("2. BREAKDOWN BY CATEGORY", report["by_category"], "Category")
    out += _render_table("3. BREAKDOWN BY DIRECTION", report["by_direction"], "Direction")
    out += _render_table(
        "4. FILL QUALITY ANALYSIS", report["by_fill_quality"], "Fill group",
        purpose="are illiquid/partial-fill markets costing us money?",
    )
    out += _render_table("5. EXIT REASON BREAKDOWN", report["by_exit_reason"], "Exit reason")
    out += _render_brier_section(report["brier"])
    out += _render_table(
        "7. BREAKDOWN BY ENTRY PRICE BUCKET (longshot analysis)",
        report["by_entry_price"], "Entry price",
        purpose="are cheap longshots systematically losing money?",
    )
    out += _render_table(
        "8. BREAKDOWN BY TIME TO RESOLUTION", report["by_time_to_resolution"],
        "Time to resolve",
        purpose="end_date - opened_at; NULL end_date shown as 'unknown'.",
    )
    out += _render_table(
        "9. BREAKDOWN BY ENTRY VOLUME BUCKET", report["by_entry_volume"],
        "Entry volume",
        purpose="market USD volume at open; rows from before the column are "
                "'unknown (pre-column)'.",
    )
    out += _render_trade_list("10. TOP 5 BEST TRADES", report["best_trades"])
    out += _render_trade_list("    TOP 5 WORST TRADES", report["worst_trades"])
    out.append("")
    return "\n".join(out)


def position_pnl(side: str, entry_yes_price: float, yes_price_now: float, amount_usd: float) -> float:
    """PnL of a position opened with amount_usd, marked at yes_price_now.

    Shares bought = amount / entry side price; value now = shares x side
    price now. Entry prices are clamped away from zero (some early dry-run
    trades entered at 0.0015 — a divide-by-near-zero, not a real position).
    """
    entry = entry_yes_price if side == "YES" else 1.0 - entry_yes_price
    now = yes_price_now if side == "YES" else 1.0 - yes_price_now
    entry = min(max(entry, 1e-6), 1.0)
    return amount_usd * (now / entry - 1.0)


def position_pnl_basis(position: dict, yes_price_now: float, fraction: float = 1.0) -> float:
    """PnL of `fraction` of a position marked at yes_price_now, computed on the
    ACTUAL fill basis when the real filled share count is known.

    A live BUY fills at the ask, above the cached mid stored as
    entry_yes_price, so the legacy derivation (implied shares =
    amount / entry-side-price) over-counts the shares owned and overstates
    PnL. Live fills record the truth: token_count is the real share count and
    amount_usd is set to the real filled cost at open (see
    positions.open_position), so PnL = token_count x current side price -
    amount_usd. Dry-run/legacy rows (token_count NULL) keep the legacy math,
    where implied and simulated shares coincide.

    Keep consistent with positions.pnl_pct_basis (its % counterpart) so the
    dollars and percentages agree, and unrealized marks match what a close
    realizes."""
    side = position["side"]
    amount = position.get("amount_usd") or 0.0
    shares = position.get("token_count")
    if shares is not None and shares > 0:
        now = yes_price_now if side == "YES" else 1.0 - yes_price_now
        return (shares * now - amount) * fraction
    return position_pnl(side, position["entry_yes_price"], yes_price_now,
                        amount * fraction)


def position_shares(side: str, entry_yes_price: float, amount_usd: float,
                    token_count: float | None = None) -> float:
    """Outcome-token shares a position holds.

    When `token_count` (the real filled share count recorded at open) is known it
    is the source of truth — the BUY filled at the higher ask, so deriving shares
    from amount_usd / entry_yes_price over-counts what we actually own and a live
    SELL of that inflated count is rejected "not enough balance". Falls back to
    amount / entry side price (same clamp as position_pnl) for dry-run/legacy
    positions with no token_count."""
    if token_count is not None and token_count > 0:
        return token_count
    entry = entry_yes_price if side == "YES" else 1.0 - entry_yes_price
    entry = min(max(entry, 1e-6), 1.0)
    return amount_usd / entry


def _fetch_current_yes_price(condition_id: str) -> float | None:
    """Current YES price from Gamma — fallback when no live prices passed."""
    try:
        resp = httpx.get(
            f"{GAMMA_API}/markets", params={"condition_ids": condition_id}, timeout=10
        )
        items = resp.json()
        if isinstance(items, list) and items:
            import json
            prices = items[0].get("outcomePrices", "")
            prices = json.loads(prices) if isinstance(prices, str) else prices
            if prices:
                return float(prices[0])
    except Exception as e:
        log.debug(f"[performance] Price fetch failed for {condition_id[:16]}: {e}")
    return None


def compute_performance(current_prices: dict[str, float] | None = None) -> dict:
    """Aggregate the positions table into a performance summary.

    Realized PnL sums realized_pnl_usd over all positions (including
    partial close_half realizations on still-open ones); wins (realized PnL
    > 0) and losses (realized PnL < 0) count closed positions by their total
    realized PnL, while break-even closes (exactly 0, or NULL) are excluded
    from wins, losses, and closed_count alike. Unrealized marks open
    positions at current_prices when given, else the position's stored
    mark, else a Gamma fetch, else entry (zero contribution).
    """
    trades = logger.get_trades_for_performance()
    current_prices = dict(current_prices or {})

    conn = logger._conn()
    rows = [dict(r) for r in conn.execute("SELECT * FROM positions").fetchall()]
    conn.close()

    # Display-only window for the dashboard performance panel: when set, scope
    # every metric below to positions opened on or after PERFORMANCE_START_DATE
    # (and the trades behind them, so deployed/trades_total stay coherent). The
    # circuit breaker, calibration, and dynamic-Kelly win rate are unaffected —
    # they don't go through compute_performance.
    since = config.PERFORMANCE_START_DATE
    if since:
        rows = [p for p in rows if (p.get("opened_at") or "") >= since]
        kept_trade_ids = {p["trade_id"] for p in rows}
        trades = [t for t in trades if t["id"] in kept_trade_ids]

    deployed = sum(t["amount_usd"] for t in trades)
    realized = sum(p["realized_pnl_usd"] or 0.0 for p in rows)
    # Graded closes: a closed position with exactly zero (or NULL) realized PnL
    # is a break-even non-event — it counts as neither a win nor a loss, and is
    # excluded from the closed-trade total. WIN is strictly > 0, LOSS strictly < 0.
    graded = [p for p in rows
              if p["status"] != "open" and (p["realized_pnl_usd"] or 0) != 0]
    wins = sum(1 for p in graded if p["realized_pnl_usd"] > 0)
    losses = sum(1 for p in graded if p["realized_pnl_usd"] < 0)
    closed_count = len(graded)

    unrealized = 0.0
    open_rows = [p for p in rows if p["status"] == "open"]
    for p in open_rows:
        price_now = current_prices.get(p["condition_id"]) or p["current_yes_price"]
        if price_now is None:
            price_now = _fetch_current_yes_price(p["condition_id"])
        if price_now is not None:
            unrealized += position_pnl_basis(p, price_now)

    return {
        "trades_total": len(trades),
        "deployed_usd": round(deployed, 2),
        "open_count": len(open_rows),
        "closed_count": closed_count,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(wins / closed_count * 100, 1) if closed_count else None,
        "realized_pnl_usd": round(realized, 2),
        "unrealized_pnl_usd": round(unrealized, 2),
        "total_pnl_usd": round(realized + unrealized, 2),
    }
