"""Dynamic Kelly sizing: win-rate factor, the floor, the TTL cache, and the
recent-win-rate query feeding it."""
import pytest

from locus import config
from locus import memory
from locus.core import edge
from locus.memory import logger


@pytest.fixture(autouse=True)
def _reset_and_pin(monkeypatch):
    """Reset the module cache and pin the sizing config to known defaults so
    tests don't depend on the developer's .env or on cross-test cache state."""
    edge.reset_winrate_cache()
    monkeypatch.setattr(config, "KELLY_BANKROLL_USD", 100.0)
    monkeypatch.setattr(config, "MAX_BET_USD", 25.0)
    monkeypatch.setattr(config, "KELLY_WINRATE_LOOKBACK", 20)
    monkeypatch.setattr(config, "KELLY_WINRATE_MIN_SAMPLES", 5)
    monkeypatch.setattr(config, "KELLY_DYNAMIC_MIN_FACTOR", 0.25)
    monkeypatch.setattr(config, "KELLY_DYNAMIC_MAX_FACTOR", 1.0)
    monkeypatch.setattr(config, "KELLY_MIN_BET_USD", 2.0)
    monkeypatch.setattr(config, "KELLY_WINRATE_CACHE_TTL", 1800.0)
    yield
    edge.reset_winrate_cache()


# --- linear interpolation at the key points ---

def test_factor_at_anchor_points():
    assert edge.winrate_factor(0.25) == pytest.approx(0.25)
    assert edge.winrate_factor(0.50) == pytest.approx(0.625)
    assert edge.winrate_factor(0.75) == pytest.approx(1.0)


def test_factor_is_clamped_outside_band():
    assert edge.winrate_factor(0.0) == pytest.approx(0.25)   # below low anchor
    assert edge.winrate_factor(0.10) == pytest.approx(0.25)
    assert edge.winrate_factor(0.90) == pytest.approx(1.0)   # above high anchor
    assert edge.winrate_factor(1.0) == pytest.approx(1.0)


def test_factor_monotonic_across_band():
    factors = [edge.winrate_factor(wr) for wr in (0.30, 0.40, 0.50, 0.60, 0.70)]
    assert factors == sorted(factors)
    assert len(set(factors)) == len(factors)


def test_size_scales_by_factor(monkeypatch):
    # Even odds, confidence 0.7 -> base half-Kelly = (0.7-0.3)/2*100 = $20.
    monkeypatch.setattr(edge, "get_cached_winrate", lambda: 0.75)
    assert edge.size_position("YES", 0.5, 0.7) == pytest.approx(20.0)   # factor 1.0
    monkeypatch.setattr(edge, "get_cached_winrate", lambda: 0.50)
    assert edge.size_position("YES", 0.5, 0.7) == pytest.approx(12.5)   # factor 0.625
    monkeypatch.setattr(edge, "get_cached_winrate", lambda: 0.25)
    assert edge.size_position("YES", 0.5, 0.7) == pytest.approx(5.0)    # factor 0.25


# --- floor enforcement ---

def test_floor_lifts_tiny_bet_to_min(monkeypatch):
    # Base $5 (conf 0.55 at even odds) * cold-streak factor 0.25 = $1.25 -> $2.
    monkeypatch.setattr(edge, "get_cached_winrate", lambda: 0.25)
    assert edge.size_position("YES", 0.5, 0.55) == pytest.approx(2.0)


def test_floor_applies_when_no_edge(monkeypatch):
    monkeypatch.setattr(edge, "get_cached_winrate", lambda: 0.75)
    assert edge.size_position("YES", 0.5, 0.5) == config.KELLY_MIN_BET_USD


def test_floor_respects_override(monkeypatch):
    monkeypatch.setattr(config, "KELLY_MIN_BET_USD", 5.0)
    monkeypatch.setattr(edge, "get_cached_winrate", lambda: 0.25)
    # Base $5 * 0.25 = $1.25, floored up to the overridden $5.
    assert edge.size_position("YES", 0.5, 0.55) == pytest.approx(5.0)


# --- cache TTL ---

def test_cache_holds_within_ttl_then_refreshes(monkeypatch):
    state = {"pnls": [10.0] * 8 + [-1.0] * 2}  # 8 wins / 10 -> winrate 0.8
    monkeypatch.setattr(logger, "get_recent_closed_position_pnls", lambda n: state["pnls"])
    clock = {"t": 1000.0}
    monkeypatch.setattr(edge.time, "monotonic", lambda: clock["t"])

    assert edge.get_cached_winrate() == pytest.approx(0.8)

    # Underlying data flips, but within the TTL the cached value is returned.
    state["pnls"] = [-1.0] * 10  # would be winrate 0.0
    clock["t"] = 1000.0 + 1799.0
    assert edge.get_cached_winrate() == pytest.approx(0.8)

    # Past the TTL it recomputes from fresh data.
    clock["t"] = 1000.0 + 1801.0
    assert edge.get_cached_winrate() == pytest.approx(0.0)


def test_cache_stores_sample_count(monkeypatch):
    monkeypatch.setattr(logger, "get_recent_closed_position_pnls",
                        lambda n: [10.0, 10.0, 10.0, -1.0, -1.0, -1.0])
    monkeypatch.setattr(edge.time, "monotonic", lambda: 500.0)
    edge.get_cached_winrate()
    assert edge._winrate_cache["sample_count"] == 6
    assert edge._winrate_cache["winrate"] == pytest.approx(0.5)


def test_cache_failure_falls_back_to_half_without_caching(monkeypatch):
    def boom(n):
        raise RuntimeError("db down")
    monkeypatch.setattr(logger, "get_recent_closed_position_pnls", boom)
    assert edge.get_cached_winrate() == 0.5
    assert edge._winrate_cache is None  # a failure is not cached


# --- get_recent_winrate: fallback and computation ---

def test_winrate_fallback_with_too_few_samples(monkeypatch):
    monkeypatch.setattr(logger, "get_recent_closed_position_pnls",
                        lambda n: [10.0, 10.0, -1.0])  # only 3 < min 5
    assert memory.get_recent_winrate(20) == 0.5


def test_winrate_computed_with_enough_samples(monkeypatch):
    monkeypatch.setattr(logger, "get_recent_closed_position_pnls",
                        lambda n: [10.0, 10.0, 10.0, -1.0, -1.0])  # 3/5
    assert memory.get_recent_winrate(20) == pytest.approx(0.6)


def test_winrate_min_samples_override(monkeypatch):
    monkeypatch.setattr(config, "KELLY_WINRATE_MIN_SAMPLES", 3)
    monkeypatch.setattr(logger, "get_recent_closed_position_pnls",
                        lambda n: [10.0, 10.0, -1.0])  # 2/3 now meets the bar
    assert memory.get_recent_winrate(20) == pytest.approx(2 / 3)


# --- end to end through the positions table ---

def _close(tmp_db, mid, pnl, closed_at):
    from locus.core import positions
    from locus.markets.gamma import Market

    trade_id = tmp_db.log_trade(
        market_id=mid, market_question="Q?", claude_score=0.7, market_price=0.5,
        edge=0.2, side="YES", amount_usd=25.0, status="dry_run",
        classification="bullish", materiality=0.7,
    )
    positions.open_position(trade_id, Market(mid, "Q?", "ai", 0.5, 0.5, 5000, "", True, []),
                            "YES", 25.0)
    conn = tmp_db._conn()
    conn.execute(
        "UPDATE positions SET status='closed_resolution', realized_pnl_usd=?, "
        "closed_at=? WHERE trade_id=?",
        (pnl, closed_at, trade_id),
    )
    conn.commit()
    conn.close()


def test_get_recent_winrate_end_to_end(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "KELLY_WINRATE_MIN_SAMPLES", 5)
    # Fewer than 5 closes -> fallback 0.5.
    _close(tmp_db, "m1", +5.0, "2026-06-10 10:00:00")
    _close(tmp_db, "m2", -5.0, "2026-06-10 11:00:00")
    assert memory.get_recent_winrate(20) == 0.5

    # Reach 6 closes total: 4 winners / 6 -> 0.667.
    _close(tmp_db, "m3", +5.0, "2026-06-11 10:00:00")
    _close(tmp_db, "m4", +5.0, "2026-06-11 11:00:00")
    _close(tmp_db, "m5", +5.0, "2026-06-11 12:00:00")
    _close(tmp_db, "m6", -5.0, "2026-06-11 13:00:00")
    assert memory.get_recent_winrate(20) == pytest.approx(4 / 6)


def test_get_recent_winrate_respects_lookback_window(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "KELLY_WINRATE_MIN_SAMPLES", 5)
    # 5 recent losers, then 5 older winners. With lookback 5, only the recent
    # (newest closed_at) losers count -> winrate 0.0.
    for i in range(5):
        _close(tmp_db, f"old{i}", +5.0, f"2026-06-10 1{i}:00:00")
    for i in range(5):
        _close(tmp_db, f"new{i}", -5.0, f"2026-06-12 1{i}:00:00")
    assert memory.get_recent_winrate(5) == pytest.approx(0.0)
    # The full window sees all ten -> 5/10.
    assert memory.get_recent_winrate(10) == pytest.approx(0.5)
