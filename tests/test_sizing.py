"""Half-Kelly position sizing from win probability (confidence), not materiality."""
import pytest

from locus import config
from locus.core.edge import size_position, detect_edge_v2
from locus.core.classifier import Classification
from locus.markets.gamma import Market
from locus.sources.news_stream import NewsEvent
from datetime import datetime, timezone


@pytest.fixture(autouse=True)
def deterministic_config(monkeypatch):
    monkeypatch.setattr(config, "KELLY_BANKROLL_USD", 100.0)
    monkeypatch.setattr(config, "MAX_BET_USD", 25.0)
    # detect_edge_v2 no longer enforces a materiality floor (that moved to
    # pipeline.gate_trade); only EDGE_THRESHOLD and price-room guards remain.
    monkeypatch.setattr(config, "EDGE_THRESHOLD", 0.10)
    # Pin the dynamic win-rate factor to 1.0 (wr 0.75) so these tests validate
    # the base Kelly math; dynamic scaling has its own test module.
    from locus.core import edge
    monkeypatch.setattr(edge, "get_cached_winrate", lambda: 0.75)
    monkeypatch.setattr(config, "KELLY_MIN_BET_USD", 2.0)


def _cls(direction, materiality, confidence=0.5):
    return Classification(direction=direction, materiality=materiality,
                          confidence=confidence, reasoning="", latency_ms=10,
                          model="test")


def _mkt(price):
    return Market("c1", "Will X happen?", "ai", price, round(1 - price, 4),
                  5000, "", True, [])


EVENT = NewsEvent(headline="h", source="rss", url="",
                  received_at=datetime.now(timezone.utc),
                  published_at=datetime.now(timezone.utc), latency_ms=0)


# --- Kelly formula math ---

def test_kelly_even_odds_yes():
    # price 0.5 -> b = 1; half-Kelly of $100 = 100 * (p - q) / 2.
    assert size_position("YES", 0.5, 0.6) == 10.0   # (0.6-0.4)/2 * 100
    assert size_position("YES", 0.5, 0.7) == 20.0   # (0.7-0.3)/2 * 100


def test_kelly_symmetric_at_even_odds():
    # At price 0.5 the YES and NO payoffs are identical, so size matches.
    assert size_position("NO", 0.5, 0.7) == 20.0
    assert size_position("YES", 0.5, 0.7) == size_position("NO", 0.5, 0.7)


def test_kelly_scales_with_confidence():
    sizes = [size_position("YES", 0.5, c) for c in (0.55, 0.65, 0.75)]
    assert sizes == [5.0, 15.0, 25.0]
    assert sizes == sorted(sizes) and len(set(sizes)) == 3


def test_kelly_caps_at_max_bet():
    # Very high confidence wants more than the cap allows.
    assert size_position("YES", 0.5, 0.95) == config.MAX_BET_USD


def test_kelly_floors_at_min_bet_when_no_edge():
    # Fair coin at fair odds -> zero Kelly -> floored to KELLY_MIN_BET_USD.
    assert size_position("YES", 0.5, 0.5) == config.KELLY_MIN_BET_USD
    # YES at 0.80 but only 70% confident: market implies 80%, you're below it,
    # so Kelly is negative (don't bet) -> floored to the min bet, never negative.
    assert size_position("YES", 0.8, 0.7) == config.KELLY_MIN_BET_USD


def test_kelly_favorable_odds_size_up():
    # Cheap YES (price 0.2) with the same confidence pays more on a win, so
    # Kelly stakes more than the even-odds case (here it hits the cap).
    assert size_position("YES", 0.2, 0.7) > size_position("YES", 0.5, 0.7)


def test_kelly_no_side_odds():
    # NO at price 0.8 -> buying NO at 0.20, b = 0.8/0.2 = 4.
    # full Kelly = (0.85*4 - 0.15)/4 = 0.8125; half * 100 = 40.625 -> capped.
    assert size_position("NO", 0.8, 0.85) == config.MAX_BET_USD
    # NO at price 0.2 -> buying NO at 0.80 (b=0.25); 70% confidence is below
    # the 80% implied, negative Kelly -> floored to the min bet.
    assert size_position("NO", 0.2, 0.7) == config.KELLY_MIN_BET_USD


# --- signal integration ---

def test_signal_bet_scales_with_confidence():
    # detect_edge_v2 returns an EdgeMetrics carrying the built Signal.
    low = detect_edge_v2(_mkt(0.5), _cls("bullish", 0.8, confidence=0.6), EVENT)
    high = detect_edge_v2(_mkt(0.5), _cls("bullish", 0.8, confidence=0.9), EVENT)
    assert low and high
    assert low.signal.bet_amount < high.signal.bet_amount
    assert high.signal.confidence == 0.9  # confidence flows onto the Signal


def test_high_materiality_low_confidence_is_sized_small():
    # The bug this fixes: big news (materiality 0.95) but unsure of direction
    # (confidence 0.55) must NOT size like a sure thing.
    metrics = detect_edge_v2(_mkt(0.5), _cls("bullish", 0.95, confidence=0.55), EVENT)
    assert metrics is not None
    assert metrics.signal.bet_amount < 10.0  # small, driven by the weak confidence


def test_price_room_guards_are_symmetric():
    strong_bull = _cls("bullish", 0.9)
    strong_bear = _cls("bearish", 0.9)
    # bullish: blocked at both extremes, allowed in the middle
    assert detect_edge_v2(_mkt(0.03), strong_bull, EVENT) is None  # longshot YES
    assert detect_edge_v2(_mkt(0.90), strong_bull, EVENT) is None  # no room
    assert detect_edge_v2(_mkt(0.50), strong_bull, EVENT) is not None
    # bearish mirrored
    assert detect_edge_v2(_mkt(0.97), strong_bear, EVENT) is None  # longshot NO
    assert detect_edge_v2(_mkt(0.10), strong_bear, EVENT) is None  # no room
    assert detect_edge_v2(_mkt(0.50), strong_bear, EVENT) is not None


def test_neutral_never_signals():
    assert detect_edge_v2(_mkt(0.5), _cls("neutral", 0.9), EVENT) is None


def test_edge_below_threshold_never_signals():
    # Very low materiality leaves edge under EDGE_THRESHOLD. (The materiality
    # *floor* itself is now enforced in pipeline.gate_trade, not here.)
    assert detect_edge_v2(_mkt(0.5), _cls("bullish", 0.05), EVENT) is None
