"""
Dry-run performance aggregation: realized PnL from resolved positions
(calibration exits), unrealized PnL from open positions at current prices.

Honest about its limits: fills are simulated at the last seen price with no
fees or slippage — the dashboard says so in small print.
"""
from __future__ import annotations

import logging

import httpx

from locus.memory import logger

log = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"


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
    partial close_half realizations on still-open ones); wins/losses count
    closed positions by their total realized PnL. Unrealized marks open
    positions at current_prices when given, else the position's stored
    mark, else a Gamma fetch, else entry (zero contribution).
    """
    trades = logger.get_trades_for_performance()
    current_prices = dict(current_prices or {})

    conn = logger._conn()
    rows = [dict(r) for r in conn.execute("SELECT * FROM positions").fetchall()]
    conn.close()

    deployed = sum(t["amount_usd"] for t in trades)
    realized = sum(p["realized_pnl_usd"] or 0.0 for p in rows)
    closed = [p for p in rows if p["status"] != "open"]
    wins = sum(1 for p in closed if (p["realized_pnl_usd"] or 0) > 0)
    losses = len(closed) - wins

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
        "closed_count": len(closed),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(wins / len(closed) * 100, 1) if closed else None,
        "realized_pnl_usd": round(realized, 2),
        "unrealized_pnl_usd": round(unrealized, 2),
    }
