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


def size_position(side: str, yes_price: float, confidence: float) -> float:
    """Half-Kelly position sizing from win probability and market odds.

    Previously this scaled size by *materiality* — how much the news matters —
    which is not the probability of winning. A high-materiality, low-confidence
    call (big news, unclear direction) was sized like a sure thing. Kelly needs
    the actual win probability, so we use `confidence` (Claude's P that the
    market resolves in the predicted direction).

    Decimal-odds payoff b for a $1 stake at the current price:
      YES bought at yes_price pays (1 - yes_price)/yes_price on a win
      NO  bought at (1 - yes_price) pays yes_price/(1 - yes_price)
    With p = confidence, q = 1 - p, full Kelly fraction is (p*b - q)/b. We bet
    HALF Kelly of KELLY_BANKROLL_USD, capped at MAX_BET_USD and floored at $1.
    A non-positive Kelly (no edge at these odds) floors to the $1 minimum.
    """
    price = min(max(yes_price, 1e-6), 1.0 - 1e-6)
    p = min(max(confidence, 0.0), 1.0)
    q = 1.0 - p

    if side == "YES":
        b = (1.0 - price) / price
    else:  # NO
        b = price / (1.0 - price)

    full_kelly = (p * b - q) / b if b > 0 else 0.0
    raw_size = config.KELLY_BANKROLL_USD * (full_kelly / 2.0)
    return min(max(round(raw_size, 2), 1.0), config.MAX_BET_USD)
