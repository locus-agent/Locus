"""Enhanced edge detection + sizing: EdgeMetrics, edge_factor / vol_adj, the
expected-edge-boosted Kelly size, and the model-free time-pressure hard exit."""
from datetime import datetime, timezone

import pytest

from locus import config
from locus.core import edge, positions
from locus.core.edge import detect_edge_v2, EdgeMetrics
from locus.core.classifier import Classification
from locus.markets.gamma import Market
from locus.sources.news_stream import NewsEvent


@pytest.fixture(autouse=True)
def deterministic_config(monkeypatch):
    monkeypatch.setattr(config, "KELLY_BANKROLL_USD", 100.0)
    monkeypatch.setattr(config, "MAX_BET_USD", 25.0)
    monkeypatch.setattr(config, "EDGE_THRESHOLD", 0.10)
    monkeypatch.setattr(config, "KELLY_MIN_BET_USD", 2.0)
    # Pin the price-room guards to their defaults so the boundary tests don't
    # read a tuned value out of a local .env.
    monkeypatch.setattr(config, "BULLISH_MIN_PRICE", 0.12)
    monkeypatch.setattr(config, "BULLISH_MAX_PRICE", 0.82)
    monkeypatch.setattr(config, "BEARISH_MIN_PRICE", 0.18)
    monkeypatch.setattr(config, "BEARISH_MAX_PRICE", 0.88)
    # Pin the dynamic win-rate factor to 1.0 (wr 0.75) so sizing reflects the
    # base Kelly * edge_factor * vol_adj, not the streak adjustment.
    monkeypatch.setattr(edge, "get_cached_winrate", lambda: 0.75)
    # Pin the hard-exit thresholds to their defaults.
    monkeypatch.setattr(config, "TIME_PRESSURE_HOURS", 4.0)
    monkeypatch.setattr(config, "TIME_PRESSURE_LOSS_PCT", -20.0)


def _cls(direction, materiality, confidence=0.5):
    return Classification(direction=direction, materiality=materiality,
                          confidence=confidence, reasoning="", latency_ms=10,
                          model="test")


def _mkt(price, category="ai", fee_rate=0.0):
    return Market("c1", "Will X happen?", category, price, round(1 - price, 4),
                  5000, "", True, [], fee_rate=fee_rate)


EVENT = NewsEvent(headline="h", source="rss", url="",
                  received_at=datetime.now(timezone.utc),
                  published_at=datetime.now(timezone.utc), latency_ms=0)


# --- EdgeMetrics calculation -------------------------------------------------

def test_edge_metrics_bullish_calculation():
    # price 0.7, bullish: edge = materiality * (1 - price) = 0.5 * 0.3 = 0.15.
    m = detect_edge_v2(_mkt(0.7), _cls("bullish", 0.5, confidence=0.6), EVENT)
    assert isinstance(m, EdgeMetrics)
    assert m.edge == pytest.approx(0.5 * (1 - 0.7))
    assert m.expected_edge == pytest.approx(m.edge * 0.6)            # raw_edge * confidence
    assert m.vol_adj == pytest.approx(1.0 - abs(0.7 - 0.5) * 0.8)    # 0.84
    # recommended_size is exactly the enhanced sizing of those inputs, and the
    # Signal carries both the size and the metrics that drove it.
    assert m.recommended_size == edge.size_position_enhanced(
        "YES", 0.7, 0.6, m.expected_edge, m.vol_adj)
    assert m.signal.bet_amount == m.recommended_size
    assert m.signal.expected_edge == pytest.approx(m.expected_edge)
    assert m.signal.vol_adj == pytest.approx(m.vol_adj)
    assert m.signal.side == "YES"


def test_edge_metrics_bearish_calculation():
    # price 0.6, bearish: edge = materiality * price = 0.5 * 0.6 = 0.30.
    m = detect_edge_v2(_mkt(0.6), _cls("bearish", 0.5, confidence=0.6), EVENT)
    assert m is not None
    assert m.edge == pytest.approx(0.5 * 0.6)
    assert m.expected_edge == pytest.approx(m.edge * 0.6)
    assert m.vol_adj == pytest.approx(1.0 - abs(0.6 - 0.5) * 0.8)    # 0.92
    assert m.signal.side == "NO"


def test_detect_edge_v2_returns_none_below_threshold():
    # Tiny materiality leaves edge under EDGE_THRESHOLD -> no metrics at all.
    assert detect_edge_v2(_mkt(0.5), _cls("bullish", 0.05), EVENT) is None
    assert detect_edge_v2(_mkt(0.5), _cls("neutral", 0.9), EVENT) is None


# --- fee-adjusted edge -------------------------------------------------------

def test_geopolitics_market_is_fee_free():
    from locus.markets import gamma
    assert gamma._fee_rate_for_category("geopolitics") == 0.0
    assert gamma._fee_rate_for_category("world") == 0.0
    # A geopolitics market carries no fee, so net_edge == raw_edge.
    m = detect_edge_v2(_mkt(0.5, category="geopolitics", fee_rate=0.0),
                       _cls("bullish", 0.6), EVENT)
    assert m.fee_cost == 0.0
    assert m.net_edge == pytest.approx(m.edge)


def test_geopolitical_politics_market_is_fee_free():
    from locus.markets import gamma
    # A politics-category market whose question is geopolitical -> fee-free,
    # overriding the 0.04 politics rate.
    assert gamma._fee_rate_for_category(
        "politics", "Will the US and Iran sign a nuclear deal?") == 0.0
    assert gamma._fee_rate_for_category(
        "politics", "Will Russia and Ukraine reach a ceasefire?") == 0.0
    # Plain (non-geopolitical) politics keeps the standard 0.04 rate.
    assert gamma._fee_rate_for_category(
        "politics", "Will the Democrats win the House?") == pytest.approx(0.04)


def test_crypto_fee_rate_and_cost():
    from locus.markets import gamma
    assert gamma._fee_rate_for_category("crypto") == pytest.approx(0.07)
    # crypto fee at price 0.50: 0.07 * 0.50 * 0.50 = 0.0175 per share.
    m = detect_edge_v2(_mkt(0.5, category="crypto", fee_rate=0.07),
                       _cls("bullish", 0.6), EVENT)
    assert m.fee_cost == pytest.approx(0.07 * 0.25)            # 0.0175


def test_net_edge_is_raw_minus_fee():
    # price 0.5, bullish, materiality 0.6 -> raw edge 0.30; crypto fee 0.0175.
    m = detect_edge_v2(_mkt(0.5, category="crypto", fee_rate=0.07),
                       _cls("bullish", 0.6), EVENT)
    assert m.edge == pytest.approx(0.6 * 0.5)                  # 0.30 raw
    assert m.net_edge == pytest.approx(m.edge - m.fee_cost)    # 0.2825
    # The built Signal carries the same fee figures through for logging.
    assert m.signal.fee_cost == pytest.approx(m.fee_cost)
    assert m.signal.net_edge == pytest.approx(m.net_edge)


def test_edge_rejected_when_fee_eats_it():
    # price 0.5, bullish, materiality 0.22 -> raw edge 0.11, just over the 0.10
    # threshold. With no fee it signals; a fat fee drags net under the floor.
    assert detect_edge_v2(_mkt(0.5, fee_rate=0.0), _cls("bullish", 0.22), EVENT) is not None
    # fee 0.10 * 0.5 * 0.5 = 0.025 -> net 0.085 < EDGE_THRESHOLD (0.10) -> rejected.
    assert detect_edge_v2(_mkt(0.5, fee_rate=0.10), _cls("bullish", 0.22), EVENT) is None


# --- momentum hybrid edge ----------------------------------------------------
# price 0.5, bullish, materiality 0.6 -> raw edge 0.30, no fee (test default).

def test_momentum_boost_added_when_direction_confirms(monkeypatch):
    monkeypatch.setattr(config, "MOMENTUM_ENABLED", True)
    monkeypatch.setattr(edge, "get_price_momentum", lambda *a, **k: 0.20)  # +20% YES drift
    m = detect_edge_v2(_mkt(0.5), _cls("bullish", 0.6), EVENT)
    # boost = min(0.05, abs(0.20) * 0.5 = 0.10) = 0.05.
    assert m.net_edge == pytest.approx(0.30 + 0.05)


def test_momentum_boost_capped_below_cap(monkeypatch):
    monkeypatch.setattr(config, "MOMENTUM_ENABLED", True)
    monkeypatch.setattr(edge, "get_price_momentum", lambda *a, **k: 0.04)  # +4% drift
    m = detect_edge_v2(_mkt(0.5), _cls("bullish", 0.6), EVENT)
    # boost = min(0.05, abs(0.04) * 0.5 = 0.02) = 0.02.
    assert m.net_edge == pytest.approx(0.30 + 0.02)


def test_momentum_no_boost_when_direction_opposes(monkeypatch):
    monkeypatch.setattr(config, "MOMENTUM_ENABLED", True)
    # Bullish signal but price is falling -> momentum does not confirm.
    monkeypatch.setattr(edge, "get_price_momentum", lambda *a, **k: -0.20)
    m = detect_edge_v2(_mkt(0.5), _cls("bullish", 0.6), EVENT)
    assert m.net_edge == pytest.approx(0.30)


def test_momentum_skipped_when_history_unavailable(monkeypatch):
    monkeypatch.setattr(config, "MOMENTUM_ENABLED", True)
    monkeypatch.setattr(edge, "get_price_momentum", lambda *a, **k: None)
    m = detect_edge_v2(_mkt(0.5), _cls("bullish", 0.6), EVENT)
    assert m.net_edge == pytest.approx(0.30)


def test_momentum_not_consulted_when_disabled(monkeypatch):
    # MOMENTUM_ENABLED is False (conftest default): get_price_momentum must not
    # even be called, so a raising stub proves the guard short-circuits.
    def _boom(*a, **k):
        raise AssertionError("get_price_momentum should not be called when disabled")
    monkeypatch.setattr(edge, "get_price_momentum", _boom)
    m = detect_edge_v2(_mkt(0.5), _cls("bullish", 0.6), EVENT)
    assert m.net_edge == pytest.approx(0.30)


# --- configurable price guards (BULLISH/BEARISH_MIN/MAX_PRICE) ---------------
# Materiality is high enough that the edge clears EDGE_THRESHOLD across the whole
# allowed band, so the guard is the only thing that can return None here.

def test_bullish_price_guard_boundaries():
    # Defaults: BULLISH_MIN_PRICE=0.12, BULLISH_MAX_PRICE=0.82. The min side is
    # EXCLUSIVE (strict <) so an entry exactly at the floor is rejected; the max
    # side stays inclusive.
    assert detect_edge_v2(_mkt(0.12), _cls("bullish", 0.9), EVENT) is None      # at min (excluded)
    assert detect_edge_v2(_mkt(0.13), _cls("bullish", 0.9), EVENT) is not None  # just above min
    assert detect_edge_v2(_mkt(0.11), _cls("bullish", 0.9), EVENT) is None      # below min
    assert detect_edge_v2(_mkt(0.82), _cls("bullish", 0.9), EVENT) is not None  # at max (included)
    assert detect_edge_v2(_mkt(0.83), _cls("bullish", 0.9), EVENT) is None      # above max


def test_bearish_price_guard_boundaries():
    # Defaults: BEARISH_MIN_PRICE=0.18, BEARISH_MAX_PRICE=0.88. Min side is
    # EXCLUSIVE (strict <), max side inclusive.
    assert detect_edge_v2(_mkt(0.18), _cls("bearish", 0.9), EVENT) is None      # at min (excluded)
    assert detect_edge_v2(_mkt(0.19), _cls("bearish", 0.9), EVENT) is not None  # just above min
    assert detect_edge_v2(_mkt(0.17), _cls("bearish", 0.9), EVENT) is None      # below min
    assert detect_edge_v2(_mkt(0.88), _cls("bearish", 0.9), EVENT) is not None  # at max (included)
    assert detect_edge_v2(_mkt(0.89), _cls("bearish", 0.9), EVENT) is None      # above max


def test_price_guards_are_configurable(monkeypatch):
    # An overridden band changes what passes: widen bullish to allow 0.10.
    assert detect_edge_v2(_mkt(0.10), _cls("bullish", 0.9), EVENT) is None
    monkeypatch.setattr(config, "BULLISH_MIN_PRICE", 0.05)
    assert detect_edge_v2(_mkt(0.10), _cls("bullish", 0.9), EVENT) is not None
    # Min side is exclusive: an entry exactly at the (overridden) floor is still rejected.
    assert detect_edge_v2(_mkt(0.05), _cls("bullish", 0.9), EVENT) is None


# --- edge_factor at the key points -------------------------------------------

def test_edge_factor_key_points():
    assert edge.edge_factor(0.1) == pytest.approx(1.0)   # 0.5 + 0.1*5
    assert edge.edge_factor(0.3) == pytest.approx(1.5)   # 0.5 + 1.5 -> capped
    assert edge.edge_factor(0.2) == pytest.approx(1.5)   # cap boundary exactly
    assert edge.edge_factor(0.0) == pytest.approx(0.5)   # floor of the linear term
    assert edge.edge_factor(0.05) == pytest.approx(0.75)


def test_edge_factor_is_capped_and_monotonic():
    assert edge.edge_factor(1.0) == pytest.approx(1.5)   # well past the cap
    factors = [edge.edge_factor(e) for e in (0.0, 0.05, 0.1, 0.15)]
    assert factors == sorted(factors)


# --- vol_adj formula ---------------------------------------------------------

def test_vol_adj_formula():
    assert edge.vol_adj_factor(0.5) == pytest.approx(1.0)              # no penalty at the middle
    assert edge.vol_adj_factor(0.85) == pytest.approx(1.0 - 0.35 * 0.8)  # 0.72
    assert edge.vol_adj_factor(0.15) == pytest.approx(1.0 - 0.35 * 0.8)  # symmetric, 0.72
    assert edge.vol_adj_factor(1.0) == pytest.approx(0.6)             # floored
    assert edge.vol_adj_factor(0.0) == pytest.approx(0.6)             # floored


# --- enhanced sizing ---------------------------------------------------------

def test_size_position_enhanced_applies_factors():
    # base half-Kelly at conf 0.7 even odds = (0.7-0.3)/2 * 100 = $20.
    assert edge._base_kelly_size("YES", 0.5, 0.7) == pytest.approx(20.0)
    # expected_edge 0.1 -> edge_factor 1.0, vol_adj 1.0 -> base unchanged.
    assert edge.size_position_enhanced("YES", 0.5, 0.7, 0.1, 1.0) == pytest.approx(20.0)
    # vol_adj 0.6 shrinks: 20 * 1.0 * 0.6 = $12.
    assert edge.size_position_enhanced("YES", 0.5, 0.7, 0.1, 0.6) == pytest.approx(12.0)
    # Strong edge (edge_factor 1.5) wants $30, capped at MAX_BET_USD.
    assert edge.size_position_enhanced("YES", 0.5, 0.7, 0.2, 1.0) == config.MAX_BET_USD


def test_size_position_enhanced_floors_at_min():
    # No Kelly edge (fair coin at fair odds) -> base 0 -> floored to the min bet
    # regardless of the multipliers.
    assert edge.size_position_enhanced("YES", 0.5, 0.5, 0.2, 1.0) == config.KELLY_MIN_BET_USD


# --- time-pressure hard exit -------------------------------------------------

NOW = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def test_hours_to_close_parses_and_handles_unknown():
    assert positions.hours_to_close("2026-06-15T15:00:00Z", NOW) == pytest.approx(3.0)
    assert positions.hours_to_close("2026-06-15T11:00:00Z", NOW) == pytest.approx(-1.0)
    assert positions.hours_to_close(None, NOW) is None
    assert positions.hours_to_close("", NOW) is None
    assert positions.hours_to_close("not-a-date", NOW) is None


def test_hard_exit_fires_on_deep_loser_near_close():
    pos = {"end_date": "2026-06-15T15:00:00Z"}  # 3h to close (< 4h)
    assert positions.check_hard_exit(pos, -25.0, NOW) == ("time_pressure", "time_pressure")


def test_hard_exit_held_when_conditions_unmet():
    near = {"end_date": "2026-06-15T15:00:00Z"}   # 3h to close
    far = {"end_date": "2026-06-16T12:00:00Z"}    # 24h to close
    # Loss too shallow.
    assert positions.check_hard_exit(near, -10.0, NOW) is None
    # Deep loss but plenty of time left.
    assert positions.check_hard_exit(far, -50.0, NOW) is None
    # Close time unknown -> never force-closed.
    assert positions.check_hard_exit({"end_date": None}, -50.0, NOW) is None


def test_hard_exit_boundaries_are_strict():
    near = {"end_date": "2026-06-15T15:00:00Z"}        # 3h to close
    exactly_4h = {"end_date": "2026-06-15T16:00:00Z"}  # exactly 4h to close
    # Exactly -20% is not < -20%.
    assert positions.check_hard_exit(near, -20.0, NOW) is None
    # Exactly 4h to close is not < 4h.
    assert positions.check_hard_exit(exactly_4h, -50.0, NOW) is None
