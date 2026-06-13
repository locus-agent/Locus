"""
Orderbook imbalance gate.

Before placing a trade we look at the live CLOB orderbook for the market's YES
token and measure how lopsided resting liquidity is. A strongly one-sided book
is flow we don't want to trade into: heavy resting asks (sell pressure) on YES
argues against buying YES, and heavy resting bids (buy pressure on YES) argues
against buying NO.

The imbalance score is (bid_volume - ask_volume) / (bid_volume + ask_volume)
over the top levels, ranging -1 (all asks) to +1 (all bids). The CLOB call
fails open: if the book is unreachable we return None and let the trade through
rather than blocking on a data outage.
"""
from __future__ import annotations

import logging

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
    - A None score (book unavailable) always allows — fail open.
    """
    if imbalance_score is None:
        return True
    if side == "YES":
        return imbalance_score >= -threshold
    return imbalance_score <= threshold


def _clob_client():
    """Minimal read-only CLOB client (get_order_book is a public endpoint).
    Isolated so tests can patch it without importing py_clob_client."""
    from py_clob_client.client import ClobClient
    return ClobClient(host=config.POLYMARKET_HOST, chain_id=137)


def fetch_orderbook_imbalance(token_id: str | None,
                              depth: int = ORDERBOOK_DEPTH) -> float | None:
    """Live YES-token imbalance score, or None if unavailable (fail open)."""
    if not token_id:
        return None
    try:
        book = _clob_client().get_order_book(token_id)
        bids = [(float(b.price), float(b.size)) for b in (book.bids or [])]
        asks = [(float(a.price), float(a.size)) for a in (book.asks or [])]
        return compute_imbalance(bids, asks, depth)
    except Exception as e:
        log.debug(f"[orderbook] imbalance fetch failed for {str(token_id)[:12]}: {e}")
        return None
