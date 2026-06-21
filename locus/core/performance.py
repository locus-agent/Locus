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

    Reads only fully-closed positions (status LIKE 'closed_%'); partial
    close_half realizations on still-open positions don't count as a trade.

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
            "WHERE status LIKE 'closed_%' ORDER BY closed_at"
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

    Reads positions fully closed within the last 7 days (status LIKE 'closed_%')
    and computes:

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
    clauses = ["status LIKE 'closed_%'", "closed_at >= ?"]
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


def position_shares(side: str, entry_yes_price: float, amount_usd: float) -> float:
    """Outcome-token shares a position holds: amount / entry side price.

    Mirrors position_pnl's share math (same entry-price clamp) so a live exit
    sells exactly what the position is marked as holding."""
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
            unrealized += position_pnl(
                p["side"], p["entry_yes_price"], price_now, p["amount_usd"]
            )

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
