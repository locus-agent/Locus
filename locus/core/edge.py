from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

from locus import config
from locus import memory
from locus.memory import logger
from locus.markets.gamma import Market, get_token_id
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
        # Scope the win rate to the PERFORMANCE_START_DATE window (same filter as
        # the other performance metrics) so the Kelly factor reflects only
        # positions opened on or after it.
        pnls = logger.get_recent_closed_position_pnls(
            config.KELLY_WINRATE_LOOKBACK,
            since=config.PERFORMANCE_START_DATE or None,
        )
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
    # Materiality with the time-horizon penalty applied; the gates check against
    # this. None for signals built outside detect_edge_v2 (tests) — consumers
    # fall back to `materiality` then.
    adjusted_materiality: float | None = None
    confidence: float = 0.5  # P(predicted direction) used for Kelly sizing
    news_latency_ms: int = 0
    classification_latency_ms: int = 0
    total_latency_ms: int = 0
    edge_type: str = "news"  # set by the pipeline once a signal clears the gates
    # Enhanced-edge metrics that drove sizing (carried through for logging).
    expected_edge: float = 0.0
    vol_adj: float = 1.0
    # Per-share trading fee modeled at entry, and the raw edge net of it.
    fee_cost: float = 0.0
    net_edge: float = 0.0


@dataclass
class EdgeMetrics:
    """The enhanced-edge computation behind a V2 signal.

    edge            — raw directional edge (materiality * distance-to-travel)
    fee_cost        — modeled per-share trading fee (feeRate * p * (1 - p))
    net_edge        — raw edge net of the fee; gated against EDGE_THRESHOLD
    expected_edge   — edge discounted by Claude's win-probability (confidence):
                      materiality * confidence * distance
    vol_adj         — volatility adjustment that penalizes near-certain markets
    recommended_size— the position size sizing produced from the above
    signal          — the Signal built from this market/classification (None when
                      no edge cleared the guards; detect_edge_v2 returns None then)
    """
    edge: float
    expected_edge: float
    vol_adj: float
    recommended_size: float
    fee_cost: float = 0.0
    net_edge: float = 0.0
    signal: "Signal | None" = None
    # Why no signal was built despite the edge guards passing — currently only
    # "kelly_negative" (zero/negative Kelly at these odds: do not trade). The
    # pipeline logs it as the classification's action so the funnel shows it.
    skip_reason: str | None = None


def get_price_momentum(market: Market, lookback_minutes: int = 60) -> float | None:
    """Relative YES-price drift over the last `lookback_minutes`, from the
    Polymarket CLOB price-history API: (price_now - price_then) / price_then.

    Returns None — never raises — when the market has no YES token, the history
    endpoint is unreachable, or too few points come back, so detect_edge_v2 can
    treat a missing reading as "no momentum signal" and proceed without a boost.
    """
    token_id = get_token_id(market, "YES")
    if not token_id:
        return None
    now = int(time.time())
    start_ts = now - lookback_minutes * 60
    try:
        resp = httpx.get(
            f"{config.POLYMARKET_HOST}/prices-history",
            params={
                "market": token_id,
                "startTs": start_ts,
                "endTs": now,
                "fidelity": max(1, lookback_minutes // 12),
            },
            timeout=2.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # network/HTTP/parse — fail open
        log.info(f"[edge] Momentum price history unavailable ({type(e).__name__}: {e})")
        return None

    history = data.get("history") if isinstance(data, dict) else None
    if not history or len(history) < 2:
        return None
    try:
        price_then = float(history[0]["p"])
        price_now = float(history[-1]["p"])
    except (KeyError, ValueError, TypeError, IndexError):
        return None
    if price_then <= 0:
        return None
    return (price_now - price_then) / price_then


def detect_edge_v2(
    market: Market,
    classification: Classification,
    news_event: NewsEvent,
) -> EdgeMetrics | None:
    """
    V2: Use classification direction + materiality instead of probability estimation.
    Only generates a signal when:
    - Direction is bullish or bearish (not neutral)
    - Market price has room to move in the predicted direction

    Returns an EdgeMetrics (carrying the built Signal) rather than a bare
    Signal, so the expected-edge / volatility-adjustment inputs to sizing are
    visible to the caller for logging. Returns None when no edge clears the
    guards.

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
        if not (config.BULLISH_MIN_PRICE < market_price <= config.BULLISH_MAX_PRICE):
            log.info(
                f"[edge] Price guard blocked YES at {market_price:.2f} "
                f"(outside {config.BULLISH_MIN_PRICE:.2f}-{config.BULLISH_MAX_PRICE:.2f})"
            )
            return None
        edge = classification.materiality * (1.0 - market_price)
    else:  # bearish
        side = "NO"
        if not (config.BEARISH_MIN_PRICE < market_price <= config.BEARISH_MAX_PRICE):
            log.info(
                f"[edge] Price guard blocked NO at {market_price:.2f} "
                f"(outside {config.BEARISH_MIN_PRICE:.2f}-{config.BEARISH_MAX_PRICE:.2f})"
            )
            return None
        edge = classification.materiality * market_price

    # Subtract the modeled trading fee before gating: the fee is symmetric in
    # price (feeRate * p * (1 - p)), so it's the same whichever side we buy.
    # Geopolitics markets are fee-free (fee_rate 0). A market whose fee eats the
    # edge below EDGE_THRESHOLD never signals.
    fee_cost = market.fee_rate * market_price * (1.0 - market_price)
    net_edge = edge - fee_cost
    log.info(
        f"[edge] Fee-adjusted edge: raw={edge:.3f} fee={fee_cost:.3f} net={net_edge:.3f}"
    )

    # Momentum hybrid: when recent price drift agrees with our direction (YES &
    # rising, or NO & falling), nudge the edge up a touch. Bounded to +0.05 and
    # skipped entirely when price history is unavailable (get_price_momentum None).
    if config.MOMENTUM_ENABLED:
        momentum = get_price_momentum(market, config.MOMENTUM_LOOKBACK_MINUTES)
        if momentum is not None:
            confirms = (side == "YES" and momentum > 0) or (side == "NO" and momentum < 0)
            if confirms:
                momentum_boost = min(0.05, abs(momentum) * 0.5)
                net_edge += momentum_boost
                log.info(
                    f"[edge] Momentum boost: {momentum:+.1%} → edge boost +{momentum_boost:.3f}"
                )

    if net_edge < config.EDGE_THRESHOLD:
        return None

    # Discount the raw edge by Claude's win-probability (confidence) and the
    # market's room to move, then size on those. Sizing still uses confidence
    # for the Kelly fraction; edge_factor/vol_adj scale that base bet.
    expected_edge = edge * classification.confidence
    vol_adj = vol_adj_factor(market_price)
    bet_amount = size_position_enhanced(
        side, market_price, classification.confidence, expected_edge, vol_adj
    )
    if bet_amount <= 0:
        # Zero/negative Kelly: Claude's own win-probability says these odds
        # are -EV, so DO NOT TRADE — uniformly, in every mode. (Distinct from
        # a positive-but-tiny Kelly, which sizing floors to KELLY_MIN_BET_USD;
        # see size_position_enhanced.) Returned as an EdgeMetrics with no
        # signal and skip_reason="kelly_negative" so the pipeline can log the
        # skip as its own funnel action rather than a generic no-edge skip.
        log.info(
            f"[edge] Kelly-negative: {side} at {market_price:.2f} with "
            f"confidence {classification.confidence:.2f} — no trade"
        )
        return EdgeMetrics(
            edge=edge,
            expected_edge=expected_edge,
            vol_adj=vol_adj,
            recommended_size=0.0,
            fee_cost=fee_cost,
            net_edge=net_edge,
            signal=None,
            skip_reason="kelly_negative",
        )
    total_latency = news_event.latency_ms + classification.latency_ms

    signal = Signal(
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
        adjusted_materiality=classification.adjusted_materiality,
        confidence=classification.confidence,
        news_latency_ms=news_event.latency_ms,
        classification_latency_ms=classification.latency_ms,
        total_latency_ms=total_latency,
        expected_edge=expected_edge,
        vol_adj=vol_adj,
        fee_cost=fee_cost,
        net_edge=net_edge,
    )
    return EdgeMetrics(
        edge=edge,
        expected_edge=expected_edge,
        vol_adj=vol_adj,
        recommended_size=bet_amount,
        fee_cost=fee_cost,
        net_edge=net_edge,
        signal=signal,
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


def edge_factor(expected_edge: float) -> float:
    """Sizing boost from the expected edge: min(1.5, 0.5 + expected_edge * 5).

    A stronger edge sizes up, capped at 1.5x. Key points: expected_edge 0.1 ->
    1.0 (neutral), 0.2 -> 1.5 (cap reached), >= 0.2 stays at 1.5.
    """
    return min(1.5, 0.5 + expected_edge * 5)


def vol_adj_factor(yes_price: float) -> float:
    """Volatility adjustment: penalize near-certain markets where the price has
    little room to move. max(0.6, 1.0 - |price - 0.5| * 0.8).

    Price 0.5 -> 1.0 (full size); the extremes (0/1) floor at 0.6.
    """
    return max(0.6, 1.0 - abs(yes_price - 0.5) * 0.8)


def _base_kelly_size(side: str, yes_price: float, confidence: float) -> float:
    """Half-Kelly base bet (pre cap/floor), scaled by the recent-win-rate factor.

    Kelly needs the actual win probability, so we size on `confidence` (Claude's
    P that the market resolves in the predicted direction), not materiality.

    Decimal-odds payoff b for a $1 stake at the current price:
      YES bought at yes_price pays (1 - yes_price)/yes_price on a win
      NO  bought at (1 - yes_price) pays yes_price/(1 - yes_price)
    With p = confidence, q = 1 - p, full Kelly fraction is (p*b - q)/b. We bet
    HALF Kelly of KELLY_BANKROLL_USD, then scale by a recent-win-rate factor
    (a cold streak shrinks bets, a hot streak restores them). A non-positive
    Kelly (no edge at these odds) yields 0.
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
    return base_kelly_size * winrate_factor(get_cached_winrate())


def size_position(side: str, yes_price: float, confidence: float) -> float:
    """Half-Kelly position sizing (see _base_kelly_size), capped at MAX_BET_USD.

    POLICY: zero/negative Kelly means DO NOT TRADE — returns 0.0 (the model's
    own win-probability says the odds are -EV at this price). Only a
    positive-but-tiny Kelly (0 < bet < KELLY_MIN_BET_USD) is floored up to
    KELLY_MIN_BET_USD: there the model says +EV and merely sized the bet
    technically small, which is different from saying "no edge at these odds".
    """
    base = _base_kelly_size(side, yes_price, confidence)
    if base <= 0:
        return 0.0
    capped = min(round(base, 2), config.MAX_BET_USD)
    return max(capped, config.KELLY_MIN_BET_USD)


def size_position_enhanced(
    side: str,
    yes_price: float,
    confidence: float,
    expected_edge: float,
    vol_adj: float,
) -> float:
    """Enhanced sizing: the half-Kelly base (with the dynamic win-rate factor)
    multiplied by the expected-edge boost and the volatility adjustment, then
    capped at MAX_BET_USD.

    Same floor policy as size_position: a zero/negative Kelly base returns 0.0
    (DO NOT TRADE — the multipliers are always positive, so the base's sign is
    the Kelly verdict); a positive-but-tiny size floors up to KELLY_MIN_BET_USD.
    """
    base = _base_kelly_size(side, yes_price, confidence)
    if base <= 0:
        return 0.0
    size = base * edge_factor(expected_edge) * vol_adj
    capped = min(round(size, 2), config.MAX_BET_USD)
    return max(capped, config.KELLY_MIN_BET_USD)
