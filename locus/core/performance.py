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
    """Aggregate dry-run/executed trades into a performance summary.

    current_prices: {condition_id: yes_price} from the live watcher when
    available; missing open-position prices are fetched from Gamma (a few
    HTTP calls at most), and positions still unpriced are marked at entry
    (zero unrealized contribution).
    """
    trades = logger.get_trades_for_performance()
    exits = logger.get_calibration_exits()
    current_prices = dict(current_prices or {})

    deployed = sum(t["amount_usd"] for t in trades)
    wins = losses = 0
    realized = 0.0
    unrealized = 0.0
    open_count = 0

    for t in trades:
        if t["id"] in exits:
            pnl = position_pnl(t["side"], t["market_price"], exits[t["id"]], t["amount_usd"])
            realized += pnl
            if pnl > 0:
                wins += 1
            else:
                losses += 1
        else:
            open_count += 1
            price_now = current_prices.get(t["market_id"])
            if price_now is None:
                price_now = _fetch_current_yes_price(t["market_id"])
                if price_now is not None:
                    current_prices[t["market_id"]] = price_now
            if price_now is not None:
                unrealized += position_pnl(t["side"], t["market_price"], price_now, t["amount_usd"])

    closed = wins + losses
    return {
        "trades_total": len(trades),
        "deployed_usd": round(deployed, 2),
        "open_count": open_count,
        "closed_count": closed,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(wins / closed * 100, 1) if closed else None,
        "realized_pnl_usd": round(realized, 2),
        "unrealized_pnl_usd": round(unrealized, 2),
    }
