"""
Event context awareness.

Polymarket groups related markets under one *event* (a shared Gamma `event_id`).
For a categorical event — "Who will win?" with one "Will X win?" market per
candidate — the outcomes are mutually exclusive and their YES prices sum to ~1.
News that moves one outcome implies the opposite move on its siblings: bullish
on "Will Hilton win?" is implicitly bearish on every other candidate.

This module lets the pipeline, once a signal clears all gates, look across the
whole event before committing capital:

- `get_event_markets` — the tracked markets that share an event_id.
- `find_best_outcome` — among the signal's own market and the implied plays on
  its siblings, the single highest-edge (market, side).
- `get_event_exposure` — open-position exposure on one event_id, for the
  per-event position cap (MAX_POSITIONS_PER_EVENT).
"""
from __future__ import annotations

from locus import config
from locus.markets.gamma import Market
from locus.core.edge import Signal, size_position
from locus.core import positions

# A set of event markets is treated as a mutually-exclusive (categorical) event
# when its YES prices sum to ~1.0 within this tolerance. Many-outcome events
# drift a little from exactly 1, so the band is generous.
CATEGORICAL_SUM_TOLERANCE = 0.20


def get_event_markets(event_id: str, tracked_markets: list[Market]) -> list[Market]:
    """All tracked markets belonging to the same event (including the one the
    signal fired on). Empty when event_id is unknown."""
    if not event_id:
        return []
    return [m for m in tracked_markets if getattr(m, "event_id", "") == event_id]


def is_categorical(event_markets: list[Market]) -> bool:
    """True when the event looks like a mutually-exclusive set of outcomes:
    at least two markets whose YES prices sum to ~1.0."""
    if len(event_markets) < 2:
        return False
    total = sum(m.yes_price for m in event_markets)
    return abs(total - 1.0) <= CATEGORICAL_SUM_TOLERANCE


def _has_room(side: str, yes_price: float) -> bool:
    """Symmetric price-room guards, mirroring edge.detect_edge_v2: skip a YES
    longshot / priced-in favourite, and the bearish mirror for NO. Uses the same
    configurable bands as detect_edge_v2 so a switched sibling outcome respects
    the exact price-room guard the primary signal did."""
    if side == "YES":
        return config.BULLISH_MIN_PRICE <= yes_price <= config.BULLISH_MAX_PRICE
    return config.BEARISH_MIN_PRICE <= yes_price <= config.BEARISH_MAX_PRICE


def _edge_for(side: str, yes_price: float, materiality: float) -> float:
    """Materiality-weighted distance-to-travel edge, same formula as
    detect_edge_v2 (YES rewards low prices, NO rewards high prices)."""
    if side == "YES":
        return materiality * (1.0 - yes_price)
    return materiality * yes_price


def find_best_outcome(
    signal: Signal,
    event_markets: list[Market],
    open_positions: list[dict],
) -> dict | None:
    """Given a signal on market A, find the highest-edge play across the event.

    Candidates are the direct signal plus, for a categorical event, the implied
    play on each sibling outcome: bullish on A implies bearish (NO) on every
    sibling, bearish on A implies bullish (YES). Candidates failing the price-
    room guard, the edge threshold, already held in an open position, or that
    would trip the correlation gate (HIGH topic-concentration risk against the
    open book) are dropped. Returns the best candidate as
    `{recommended_market, recommended_side, implied_edge, reason}`, or None if
    nothing clears the bar.

    The correlation filter mirrors the pipeline's correlation gate (which blocks
    'high' risk and only warns on 'medium'), so a switched sibling trade respects
    the same concentration limit the primary signal did — previously the switch
    bypassed that check.
    """
    direction = signal.classification
    materiality = signal.materiality
    a_id = signal.market.condition_id
    held = {p.get("condition_id") for p in (open_positions or [])}

    # (market, side, edge, reason)
    candidates: list[tuple[Market, str, float, str]] = [
        (signal.market, signal.side, signal.edge, "direct signal")
    ]

    if direction in ("bullish", "bearish") and is_categorical(event_markets):
        # Sibling outcomes move opposite to the primary in a categorical event.
        implied_side = "NO" if direction == "bullish" else "YES"
        implied_dir = "bearish" if implied_side == "NO" else "bullish"
        for m in event_markets:
            if m.condition_id == a_id:
                continue
            if not _has_room(implied_side, m.yes_price):
                continue
            edge = _edge_for(implied_side, m.yes_price, materiality)
            candidates.append((
                m, implied_side, edge,
                f"implied {implied_dir} on sibling outcome "
                f"(categorical event, {direction} on primary)",
            ))

    # Don't recommend a market we already hold a position in.
    candidates = [c for c in candidates if c[0].condition_id not in held]
    candidates = [c for c in candidates if c[2] >= config.EDGE_THRESHOLD]
    # Correlation gate: drop any candidate (including a switched sibling) whose
    # topic overlap with the open book is HIGH risk. The primary signal already
    # cleared this upstream, but a switch to a sibling outcome previously went
    # straight to open without re-checking — so verify each candidate here.
    candidates = [
        c for c in candidates
        if positions.check_correlation_risk(
            c[0].question, c[1], open_positions or []
        )["risk_level"] != "high"
    ]
    if not candidates:
        return None

    market, side, edge, reason = max(candidates, key=lambda c: c[2])
    return {
        "recommended_market": market,
        "recommended_side": side,
        "implied_edge": round(edge, 4),
        "reason": reason,
    }


def build_switched_signal(original: Signal, recommendation: dict) -> Signal:
    """A fresh Signal for a recommended sibling outcome, re-sized for the new
    market/side. News context (headline, source, latencies) carries over."""
    market = recommendation["recommended_market"]
    side = recommendation["recommended_side"]
    edge = recommendation["implied_edge"]
    confidence = original.confidence
    direction = "bullish" if side == "YES" else "bearish"
    return Signal(
        market=market,
        claude_score=original.materiality,
        market_price=market.yes_price,
        edge=edge,
        side=side,
        bet_amount=size_position(side, market.yes_price, confidence),
        reasoning=f"[event-switch] {recommendation['reason']}: {original.reasoning}",
        headlines=original.headlines,
        news_source=original.news_source,
        classification=direction,
        materiality=original.materiality,
        confidence=confidence,
        news_latency_ms=original.news_latency_ms,
        classification_latency_ms=original.classification_latency_ms,
        total_latency_ms=original.total_latency_ms,
        edge_type=original.edge_type,
    )


def get_event_exposure(event_id: str, open_positions: list[dict]) -> dict:
    """Open-position exposure on one event_id: count, total dollars, and the
    related positions. Backs the per-event position cap."""
    related = [
        p for p in (open_positions or [])
        if event_id and (p.get("event_id") or "") == event_id
    ]
    total = sum((p.get("amount_usd") or 0.0) for p in related)
    return {
        "event_id": event_id,
        "position_count": len(related),
        "total_exposure_usd": round(total, 2),
        "positions": related,
    }
