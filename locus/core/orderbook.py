"""
Orderbook imbalance gate.

Before placing a trade we look at the live CLOB orderbook for the market's YES
token and measure how lopsided resting liquidity is. A strongly one-sided book
is flow we don't want to trade into: heavy resting asks (sell pressure) on YES
argues against buying YES, and heavy resting bids (buy pressure on YES) argues
against buying NO.

The imbalance score is (bid_volume - ask_volume) / (bid_volume + ask_volume)
over the top levels, ranging -1 (all asks) to +1 (all bids). The gate FAILS
CLOSED by default (config.ORDERBOOK_FAIL_OPEN=false): if the book is
unreachable we cannot price slippage or see opposing flow, so we don't trade.

The book is read from the public REST endpoint ``GET /book?token_id=`` with
httpx, so paper mode needs no py-clob-client SDK. This module also hosts the
depth-walking helpers used for slippage-aware edge (edge.py) and the exit
ladder (executor.py).
"""
from __future__ import annotations

import logging

import httpx

from locus import config

log = logging.getLogger(__name__)

# Levels summed per side, and the score past which flow is "strong".
ORDERBOOK_DEPTH = 5
IMBALANCE_THRESHOLD = 0.3


def compute_imbalance(bids, asks, depth: int = ORDERBOOK_DEPTH) -> float | None:
    """Imbalance score over the best `depth` levels of each side.

    bids/asks are iterables of (price, size). Best bids are the highest prices,
    best asks the lowest, so we sort before truncating — the CLOB does not
    guarantee ordering. Returns None when there's no resting volume at all.
    """
    top_bids = sorted(bids, key=lambda lvl: lvl[0], reverse=True)[:depth]
    top_asks = sorted(asks, key=lambda lvl: lvl[0])[:depth]
    bid_volume = sum(size for _, size in top_bids)
    ask_volume = sum(size for _, size in top_asks)
    total = bid_volume + ask_volume
    if total <= 0:
        return None
    return (bid_volume - ask_volume) / total


def orderbook_allows(side: str, imbalance_score: float | None,
                     threshold: float = IMBALANCE_THRESHOLD) -> bool:
    """Whether the book permits a trade on `side`.

    - YES is blocked by strong sell pressure: score < -threshold.
    - NO is blocked by strong YES-buy pressure: score > +threshold.
    - A None score (book unavailable/empty) blocks unless
      config.ORDERBOOK_FAIL_OPEN is set — fail closed by default.
    """
    if imbalance_score is None:
        return bool(config.ORDERBOOK_FAIL_OPEN)
    if side == "YES":
        return imbalance_score >= -threshold
    return imbalance_score <= threshold


def _clob_client():
    """Minimal read-only CLOB client (get_order_book is a public endpoint).
    Isolated so tests can patch it without importing py_clob_client_v2."""
    from py_clob_client_v2 import ClobClient
    return ClobClient(host=config.POLYMARKET_HOST, chain_id=137)


def _http_get_book(token_id: str) -> dict:
    """Raw public REST book. Isolated so tests can patch it (no network)."""
    resp = httpx.get(
        f"{config.POLYMARKET_HOST}/book",
        params={"token_id": token_id},
        timeout=config.ORDERBOOK_FETCH_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()


def _levels(raw) -> list[tuple[float, float]]:
    out = []
    for lvl in raw or []:
        try:
            if isinstance(lvl, dict):
                p, s = float(lvl["price"]), float(lvl["size"])
            else:
                p, s = float(getattr(lvl, "price")), float(getattr(lvl, "size"))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if 0 < p < 1 and s > 0:
            out.append((p, s))
    return out


def fetch_book(token_id: str | None) -> dict | None:
    """{'bids': [(p, s)...] best-first, 'asks': [(p, s)...] best-first} for a
    token, or None when unavailable. Never raises."""
    if not token_id:
        return None
    try:
        raw = _http_get_book(token_id)
    except Exception as e:
        log.debug(f"[orderbook] book fetch failed for {str(token_id)[:12]}: {e}")
        return None
    bids = sorted(_levels(raw.get("bids")), key=lambda l: l[0], reverse=True)
    asks = sorted(_levels(raw.get("asks")), key=lambda l: l[0])
    return {"bids": bids, "asks": asks}


def side_book(yes_book: dict | None, side: str) -> dict | None:
    """Book for the outcome we BUY. A binary market's NO book mirrors the YES
    book: NO asks = 1 - YES bids, NO bids = 1 - YES asks."""
    if yes_book is None:
        return None
    if side == "YES":
        return yes_book
    return {
        "bids": [(round(1 - p, 6), s) for p, s in yes_book.get("asks", [])],
        "asks": [(round(1 - p, 6), s) for p, s in yes_book.get("bids", [])],
    }


def walk_asks_for_usd(asks, usd: float) -> tuple[float, float, float]:
    """Buy up to `usd` notional walking asks best-first.
    Returns (shares, vwap, spent_usd); vwap 0 when nothing fillable."""
    shares = spent = 0.0
    for price, size in asks:
        if spent >= usd - 1e-9:
            break
        take = min(size, (usd - spent) / price)
        shares += take
        spent += take * price
    return shares, (spent / shares if shares > 0 else 0.0), spent


def walk_bids_for_shares(bids, shares: float, floor_price: float = 0.0
                         ) -> tuple[float, float]:
    """Sell up to `shares` walking bids best-first, never below floor_price.
    Returns (filled_shares, vwap)."""
    filled = proceeds = 0.0
    for price, size in bids:
        if price < floor_price - 1e-12 or filled >= shares - 1e-9:
            break
        take = min(size, shares - filled)
        filled += take
        proceeds += take * price
    return filled, (proceeds / filled if filled > 0 else 0.0)


def fetch_orderbook_imbalance(token_id: str | None,
                              depth: int = ORDERBOOK_DEPTH) -> float | None:
    """Live YES-token imbalance score, or None if the book is unavailable or
    empty (the gate then blocks unless ORDERBOOK_FAIL_OPEN)."""
    book = fetch_book(token_id)
    if book is None:
        return None
    return compute_imbalance(book["bids"], book["asks"], depth)
