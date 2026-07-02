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
    # Default the performance window off so tests don't depend on the developer's
    # .env. The dedicated filter tests below override it.
    monkeypatch.setattr(config, "PERFORMANCE_START_DATE", "")
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


def test_no_trade_when_no_edge(monkeypatch):
    # POLICY: zero Kelly (fair coin at fair odds) is no-trade (0.0), not a
    # floored bet — the floor only lifts positive-but-tiny sizes (see above).
    monkeypatch.setattr(edge, "get_cached_winrate", lambda: 0.75)
    assert edge.size_position("YES", 0.5, 0.5) == 0.0


def test_floor_respects_override(monkeypatch):
    monkeypatch.setattr(config, "KELLY_MIN_BET_USD", 5.0)
    monkeypatch.setattr(edge, "get_cached_winrate", lambda: 0.25)
    # Base $5 * 0.25 = $1.25, floored up to the overridden $5.
    assert edge.size_position("YES", 0.5, 0.55) == pytest.approx(5.0)


# --- cache TTL ---

def test_cache_holds_within_ttl_then_refreshes(monkeypatch):
    state = {"pnls": [10.0] * 8 + [-1.0] * 2}  # 8 wins / 10 -> winrate 0.8
    monkeypatch.setattr(logger, "get_recent_closed_position_pnls",
                        lambda n, since=None: state["pnls"])
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
                        lambda n, since=None: [10.0, 10.0, 10.0, -1.0, -1.0, -1.0])
    monkeypatch.setattr(edge.time, "monotonic", lambda: 500.0)
    edge.get_cached_winrate()
    assert edge._winrate_cache["sample_count"] == 6
    assert edge._winrate_cache["winrate"] == pytest.approx(0.5)


def test_cache_failure_falls_back_to_half_without_caching(monkeypatch):
    def boom(n, since=None):
        raise RuntimeError("db down")
    monkeypatch.setattr(logger, "get_recent_closed_position_pnls", boom)
    assert edge.get_cached_winrate() == 0.5
    assert edge._winrate_cache is None  # a failure is not cached


# --- get_recent_winrate: fallback and computation ---

def test_winrate_fallback_with_too_few_samples(monkeypatch):
    monkeypatch.setattr(logger, "get_recent_closed_position_pnls",
                        lambda n, since=None: [10.0, 10.0, -1.0])  # only 3 < min 5
    assert memory.get_recent_winrate(20) == 0.5


def test_winrate_computed_with_enough_samples(monkeypatch):
    monkeypatch.setattr(logger, "get_recent_closed_position_pnls",
                        lambda n, since=None: [10.0, 10.0, 10.0, -1.0, -1.0])  # 3/5
    assert memory.get_recent_winrate(20) == pytest.approx(0.6)


def test_winrate_min_samples_override(monkeypatch):
    monkeypatch.setattr(config, "KELLY_WINRATE_MIN_SAMPLES", 3)
    monkeypatch.setattr(logger, "get_recent_closed_position_pnls",
                        lambda n, since=None: [10.0, 10.0, -1.0])  # 2/3 now meets the bar
    assert memory.get_recent_winrate(20) == pytest.approx(2 / 3)


# --- end to end through the positions table ---

def _close(tmp_db, mid, pnl, closed_at, question="Q?"):
    from locus.core import positions
    from locus.markets.gamma import Market

    trade_id = tmp_db.log_trade(
        market_id=mid, market_question=question, claude_score=0.7, market_price=0.5,
        edge=0.2, side="YES", amount_usd=25.0, status="dry_run",
        classification="bullish", materiality=0.7,
    )
    positions.open_position(trade_id, Market(mid, question, "ai", 0.5, 0.5, 5000, "", True, []),
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


# --- coin-flip positions are excluded from the win-rate input ---

_COINFLIP_Q = "Bitcoin Up or Down on June 29?"


def _seed_coinflip_era(tmp_db):
    # Four coin-flip losers (the old 25%-win-rate era) followed by four closes
    # on normal markets (three winners, one loser).
    for i in range(4):
        _close(tmp_db, f"cf{i}", -5.0, f"2026-06-10 1{i}:00:00", question=_COINFLIP_Q)
    for i, pnl in enumerate((+5.0, +5.0, +5.0, -5.0)):
        _close(tmp_db, f"real{i}", pnl, f"2026-06-12 1{i}:00:00")


def test_recent_pnls_exclude_coinflip_closes(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "EXCLUDE_COINFLIP_MARKETS", True)
    monkeypatch.setattr(config, "COINFLIP_PATTERNS", ["up or down"])
    _seed_coinflip_era(tmp_db)
    # Only the four normal-market closes remain; the coin-flip losers are gone.
    pnls = logger.get_recent_closed_position_pnls(20)
    assert len(pnls) == 4
    assert sum(1 for p in pnls if p > 0) == 3


def test_coinflip_exclusion_happens_before_limit(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "EXCLUDE_COINFLIP_MARKETS", True)
    monkeypatch.setattr(config, "COINFLIP_PATTERNS", ["up or down"])
    # Newest four closes are coin-flips: they must not consume window slots and
    # shadow the older tradeable-market closes.
    for i, pnl in enumerate((+5.0, +5.0, +5.0, -5.0)):
        _close(tmp_db, f"real{i}", pnl, f"2026-06-10 1{i}:00:00")
    for i in range(4):
        _close(tmp_db, f"cf{i}", -5.0, f"2026-06-12 1{i}:00:00", question=_COINFLIP_Q)
    pnls = logger.get_recent_closed_position_pnls(4)
    assert len(pnls) == 4
    assert sum(1 for p in pnls if p > 0) == 3


def test_coinflip_closes_count_when_exclusion_disabled(tmp_db, monkeypatch):
    # If the agent can trade coin-flips again, their closes belong in the estimate.
    monkeypatch.setattr(config, "EXCLUDE_COINFLIP_MARKETS", False)
    monkeypatch.setattr(config, "COINFLIP_PATTERNS", ["up or down"])
    _seed_coinflip_era(tmp_db)
    pnls = logger.get_recent_closed_position_pnls(20)
    assert len(pnls) == 8
    assert sum(1 for p in pnls if p > 0) == 3


def test_recent_winrate_ignores_coinflip_era(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "KELLY_WINRATE_MIN_SAMPLES", 4)
    monkeypatch.setattr(config, "EXCLUDE_COINFLIP_MARKETS", True)
    monkeypatch.setattr(config, "COINFLIP_PATTERNS", ["up or down"])
    _seed_coinflip_era(tmp_db)
    # 3 winners / 4 tradeable closes, not 3 / 8 with the coin-flip losers.
    assert memory.get_recent_winrate(20) == pytest.approx(0.75)


# --- PERFORMANCE_START_DATE window on the win rate ---

_KW_OLD = "2026-06-10 09:00:00"
_KW_NEW = "2026-06-14 09:00:00"
_KW_CUTOFF = "2026-06-14"


def _close_with_opened(tmp_db, mid, pnl, opened_at, closed_at):
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
        "opened_at=?, closed_at=? WHERE trade_id=?",
        (pnl, opened_at, closed_at, trade_id),
    )
    conn.commit()
    conn.close()


def _seed_old_and_new(tmp_db):
    # Old: three winners. New: two winners + three losers.
    for i in range(3):
        _close_with_opened(tmp_db, f"old{i}", +5.0, _KW_OLD, f"2026-06-10 1{i}:00:00")
    _close_with_opened(tmp_db, "new0", +5.0, _KW_NEW, "2026-06-14 10:00:00")
    _close_with_opened(tmp_db, "new1", +5.0, _KW_NEW, "2026-06-14 11:00:00")
    for i, mid in enumerate(("new2", "new3", "new4")):
        _close_with_opened(tmp_db, mid, -5.0, _KW_NEW, f"2026-06-14 1{2 + i}:00:00")


def test_recent_pnls_filtered_by_since(tmp_db):
    _seed_old_and_new(tmp_db)
    # No filter -> all eight closes.
    assert len(logger.get_recent_closed_position_pnls(20)) == 8
    # Scoped to the cutoff -> only the five new closes (two winners).
    new = logger.get_recent_closed_position_pnls(20, since=_KW_CUTOFF)
    assert len(new) == 5
    assert sum(1 for p in new if p > 0) == 2


def test_recent_pnls_exclude_zero_pnl_closes(tmp_db):
    # A break-even close ($0.00) is a non-event: it must not land in the win-rate
    # denominator. $0.01 is a win, $-0.01 is a loss; only those two are returned.
    _close(tmp_db, "win", 0.01, "2026-06-14 10:00:00")
    _close(tmp_db, "loss", -0.01, "2026-06-14 11:00:00")
    _close(tmp_db, "even", 0.0, "2026-06-14 12:00:00")
    pnls = logger.get_recent_closed_position_pnls(20)
    assert len(pnls) == 2
    assert 0.0 not in pnls
    # win rate over the two graded closes is exactly 50%.
    assert sum(1 for p in pnls if p > 0) == 1


def test_recent_winrate_respects_performance_start_date(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "KELLY_WINRATE_MIN_SAMPLES", 5)
    _seed_old_and_new(tmp_db)
    # Full history: 5 winners / 8 -> 0.625.
    monkeypatch.setattr(config, "PERFORMANCE_START_DATE", "")
    assert memory.get_recent_winrate(20) == pytest.approx(0.625)
    # New window only: 2 winners / 5 -> 0.4.
    monkeypatch.setattr(config, "PERFORMANCE_START_DATE", _KW_CUTOFF)
    assert memory.get_recent_winrate(20) == pytest.approx(0.4)


def test_kelly_winrate_respects_performance_start_date(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "KELLY_WINRATE_MIN_SAMPLES", 5)
    monkeypatch.setattr(edge.time, "monotonic", lambda: 1000.0)
    _seed_old_and_new(tmp_db)

    # The cached Kelly win rate scopes to the same window: full history 0.625...
    monkeypatch.setattr(config, "PERFORMANCE_START_DATE", "")
    edge.reset_winrate_cache()
    assert edge.get_cached_winrate() == pytest.approx(0.625)

    # ...vs only positions opened after the cutoff: 0.4.
    monkeypatch.setattr(config, "PERFORMANCE_START_DATE", _KW_CUTOFF)
    edge.reset_winrate_cache()
    assert edge.get_cached_winrate() == pytest.approx(0.4)
