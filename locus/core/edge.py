from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from locus import config
from locus import memory
from locus.memory import logger
from locus.markets.gamma import Market
from locus.core.classifier import Classification
from locus.sources.news_stream import NewsEvent

log = logging.getLogger(__name__)

# Cached recent win rate for dynamic Kelly sizing. Refreshed at most once per
# config.KELLY_WINRATE_CACHE_TTL seconds (the underlying closes change slowly,
# and size_position runs on every signal). Shape: {winrate, timestamp,
# sample_count}; None until first computed.
_winrate_cache: dict | None = None


def reset_winrate_cache() -> None:
    """Drop the cached win rate so the next size_position recomputes it."""
    global _winrate_cache
    _winrate_cache = None


def get_cached_winrate() -> float:
    """Recent realized win rate (0.0-1.0), cached for KELLY_WINRATE_CACHE_TTL.

    Defensive: any DB error falls back to 0.5 (neutral) without caching, so a
    transient failure never blocks sizing or poisons the cache."""
    global _winrate_cache
    now = time.monotonic()
    if (
        _winrate_cache is not None
        and now - _winrate_cache["timestamp"] < config.KELLY_WINRATE_CACHE_TTL
    ):
        return _winrate_cache["winrate"]

    try:
        pnls = logger.get_recent_closed_position_pnls(config.KELLY_WINRATE_LOOKBACK)
    except Exception as e:
        log.warning(f"[edge] Win-rate lookup failed, using 0.5: {e}")
        return 0.5

    winrate = memory.winrate_from_pnls(pnls)
    _winrate_cache = {"winrate": winrate, "timestamp": now, "sample_count": len(pnls)}
    log.info(
        f"[edge] Win-rate cache refreshed: {winrate:.0%} over {len(pnls)} closed positions"
    )
    return winrate


@dataclass
class Signal:
    market: Market
    claude_score: float
    market_price: float
    edge: float
    side: str  # "YES" or "NO"
    bet_amount: float
    reasoning: str
    headlines: str
    # V2 fields
    news_source: str = ""
    classification: str = ""
    materiality: float = 0.0
    confidence: float = 0.5  # P(predicted direction) used for Kelly sizing
    news_latency_ms: int = 0
    classification_latency_ms: int = 0
    total_latency_ms: int = 0
    edge_type: str = "news"  # set by the pipeline once a signal clears the gates


def detect_edge(
    market: Market,
    claude_score: float,
    reasoning: str = "",
    headlines: str = "",
) -> Signal | None:
    """V1: Compare Claude's confidence against market price."""
    market_price = market.yes_price
    edge = claude_score - market_price

    if abs(edge) < config.EDGE_THRESHOLD:
        return None

    if edge > 0:
        side = "YES"
        raw_edge = edge
    else:
        side = "NO"
        raw_edge = abs(edge)

    # claude_score is Claude's estimated YES probability; confidence in the
    # chosen side is that probability (YES) or its complement (NO).
    confidence = claude_score if side == "YES" else 1.0 - claude_score
    bet_amount = size_position(side, market_price, confidence)

    return Signal(
        market=market,
        claude_score=claude_score,
        market_price=market_price,
        edge=raw_edge,
        side=side,
        bet_amount=bet_amount,
        reasoning=reasoning,
        headlines=headlines,
    )


def detect_edge_v2(
    market: Market,
    classification: Classification,
    news_event: NewsEvent,
) -> Signal | None:
    """
    V2: Use classification direction + materiality instead of probability estimation.
    Only generates a signal when:
    - Direction is bullish or bearish (not neutral)
    - Market price has room to move in the predicted direction

    The materiality floor is direction-specific and is enforced downstream in
    pipeline.gate_trade (so every classification is still logged/calibrated);
    edge here only requires a non-neutral direction with EDGE_THRESHOLD of room.
    """
    if classification.direction == "neutral":
        return None

    market_price = market.yes_price

    # Symmetric price-room guards: skip when the market is already priced
    # near certainty in either direction. High side: little room to profit.
    # Low side: longshot territory where the edge formula below (which
    # rewards distance-to-travel) would otherwise systematically buy
    # lottery tickets on any directional whiff.
    if classification.direction == "bullish":
        side = "YES"
        if not 0.05 <= market_price <= 0.85:
            return None
        edge = classification.materiality * (1.0 - market_price)
    else:  # bearish
        side = "NO"
        if not 0.15 <= market_price <= 0.95:
            return None
        edge = classification.materiality * market_price

    if edge < config.EDGE_THRESHOLD:
        return None

    # Size on Claude's win-probability estimate (confidence), not materiality.
    bet_amount = size_position(side, market_price, classification.confidence)
    total_latency = news_event.latency_ms + classification.latency_ms

    return Signal(
        market=market,
        claude_score=classification.materiality,
        market_price=market_price,
        edge=edge,
        side=side,
        bet_amount=bet_amount,
        reasoning=classification.reasoning,
        headlines=news_event.headline,
        news_source=news_event.source,
        classification=classification.direction,
        materiality=classification.materiality,
        confidence=classification.confidence,
        news_latency_ms=news_event.latency_ms,
        classification_latency_ms=classification.latency_ms,
        total_latency_ms=total_latency,
    )


def winrate_factor(recent_wr: float) -> float:
    """Dynamic-sizing multiplier from recent win rate (clamped to the factor band).

    Linear map: win rate 0.25 -> KELLY_DYNAMIC_MIN_FACTOR, 0.75 ->
    KELLY_DYNAMIC_MAX_FACTOR. With the defaults this is
    factor = clamp(0.25 + (wr - 0.25) * (0.75 / 0.50), 0.25, 1.0):
    wr 0.25 -> 0.25, wr 0.50 -> 0.625, wr 0.75 -> 1.0.
    """
    min_f = config.KELLY_DYNAMIC_MIN_FACTOR
    max_f = config.KELLY_DYNAMIC_MAX_FACTOR
    # Slope maps the 0.25..0.75 win-rate span onto the min..max factor band.
    factor = min_f + (recent_wr - 0.25) * ((max_f - min_f) / 0.50)
    return min(max(factor, min_f), max_f)


def size_position(side: str, yes_price: float, confidence: float) -> float:
    """Half-Kelly position sizing from win probability and market odds, scaled
    by recent realized win rate.

    Kelly needs the actual win probability, so we size on `confidence` (Claude's
    P that the market resolves in the predicted direction), not materiality.

    Decimal-odds payoff b for a $1 stake at the current price:
      YES bought at yes_price pays (1 - yes_price)/yes_price on a win
      NO  bought at (1 - yes_price) pays yes_price/(1 - yes_price)
    With p = confidence, q = 1 - p, full Kelly fraction is (p*b - q)/b. We bet
    HALF Kelly of KELLY_BANKROLL_USD as the base size.

    The base is then multiplied by a recent-win-rate factor (smooth linear
    scaling — a cold streak shrinks bets, a hot streak restores them), capped at
    MAX_BET_USD, and floored at KELLY_MIN_BET_USD. A non-positive Kelly (no edge
    at these odds) yields a base of 0, which the floor lifts to the minimum bet.
    """
    price = min(max(yes_price, 1e-6), 1.0 - 1e-6)
    p = min(max(confidence, 0.0), 1.0)
    q = 1.0 - p

    if side == "YES":
        b = (1.0 - price) / price
    else:  # NO
        b = price / (1.0 - price)

    full_kelly = (p * b - q) / b if b > 0 else 0.0
    base_kelly_size = config.KELLY_BANKROLL_USD * (full_kelly / 2.0)

    # Dynamic adjustment: scale the base by recent realized win rate.
    factor = winrate_factor(get_cached_winrate())
    position_size = base_kelly_size * factor

    # Cap at the per-trade max, then floor at the minimum bet (so a genuine
    # signal is never sized down to dust by the factor or a thin Kelly).
    capped = min(round(position_size, 2), config.MAX_BET_USD)
    return max(capped, config.KELLY_MIN_BET_USD)
