"""Position sizing must scale with conviction; price-room guards symmetric."""
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
    monkeypatch.setattr(config, "MATERIALITY_THRESHOLD", 0.4)
    monkeypatch.setattr(config, "EDGE_THRESHOLD", 0.10)


def _cls(direction, materiality):
    return Classification(direction=direction, materiality=materiality,
                          reasoning="", latency_ms=10, model="test")


def _mkt(price):
    return Market("c1", "Will X happen?", "ai", price, round(1 - price, 4),
                  5000, "", True, [])


EVENT = NewsEvent(headline="h", source="rss", url="",
                  received_at=datetime.now(timezone.utc),
                  published_at=datetime.now(timezone.utc), latency_ms=0)


def test_size_scales_with_conviction():
    sizes = [size_position(m) for m in (0.4, 0.6, 0.8, 1.0)]
    assert sizes == [10.0, 15.0, 20.0, 25.0]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes), "sizing must not be constant"


def test_size_cap_and_floor():
    assert size_position(5.0) == config.MAX_BET_USD
    assert size_position(0.001) == 1.0


def test_signal_bet_scales_with_materiality():
    low = detect_edge_v2(_mkt(0.5), _cls("bullish", 0.5), EVENT)
    high = detect_edge_v2(_mkt(0.5), _cls("bullish", 0.9), EVENT)
    assert low and high
    assert low.bet_amount < high.bet_amount


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


def test_neutral_and_low_materiality_never_signal():
    assert detect_edge_v2(_mkt(0.5), _cls("neutral", 0.9), EVENT) is None
    assert detect_edge_v2(_mkt(0.5), _cls("bullish", 0.2), EVENT) is None
