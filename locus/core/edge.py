from __future__ import annotations

from dataclasses import dataclass

from locus import config
from locus.markets.gamma import Market
from locus.core.classifier import Classification
from locus.sources.news_stream import NewsEvent


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
        # Kelly for buying YES at p with believed probability q: (q-p)/(1-p)
        kelly = edge / max(1.0 - market_price, 0.01)
    else:
        side = "NO"
        raw_edge = abs(edge)
        # Mirrored for NO bought at (1-p): (p-q)/p
        kelly = raw_edge / max(market_price, 0.01)

    bet_amount = size_position(min(kelly, 1.0))

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
    - Materiality exceeds threshold
    - Market price has room to move in the predicted direction
    """
    if classification.direction == "neutral":
        return None

    if classification.materiality < config.MATERIALITY_THRESHOLD:
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

    bet_amount = size_position(classification.materiality)
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
        news_latency_ms=news_event.latency_ms,
        classification_latency_ms=classification.latency_ms,
        total_latency_ms=total_latency,
    )


def size_position(kelly_fraction: float) -> float:
    """Quarter-Kelly position sizing against KELLY_BANKROLL_USD, capped at
    MAX_BET_USD, floored at $1.

    kelly_fraction is the full-Kelly fraction of bankroll. For V2 signals
    that's the materiality m: modelling the news as a believed probability
    shift of m x (room to move), Kelly for buying YES at price p with
    believed q = p + m(1-p) is (q-p)/(1-p) = m, and symmetrically m for NO.
    So conviction maps directly to size: with the default $100 bankroll,
    m=0.4 bets $10, m=0.8 bets $20, m=1.0 hits the $25 cap.

    The previous formula (250 x edge vs a 0.10 edge floor) saturated the
    cap on every signal — sizing was a constant $25 in practice.
    """
    raw_size = config.KELLY_BANKROLL_USD * kelly_fraction * 0.25
    return min(max(round(raw_size, 2), 1.0), config.MAX_BET_USD)
