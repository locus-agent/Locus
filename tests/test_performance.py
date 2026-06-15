"""Performance aggregation: position PnL math, realized/unrealized rollups."""
from datetime import datetime, timedelta, timezone

import pytest

from locus import config
from locus.core import performance
from locus.core.performance import (
    position_pnl,
    compute_performance,
    compute_circuit_breaker,
    compute_live_readiness,
)


@pytest.fixture(autouse=True)
def _neutralize_breaker_start_date(monkeypatch):
    """Default the circuit-breaker and performance-panel start-date filters off
    so tests don't depend on the developer's local .env. Tests that exercise a
    filter override it."""
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_START_DATE", "")
    monkeypatch.setattr(config, "PERFORMANCE_START_DATE", "")


def test_yes_position_pnl():
    # $25 of YES at 0.50 = 50 shares; resolves YES (1.0) -> worth $50, pnl +$25
    assert position_pnl("YES", 0.50, 1.0, 25.0) == pytest.approx(25.0)
    # resolves NO -> worthless, pnl -$25
    assert position_pnl("YES", 0.50, 0.0, 25.0) == pytest.approx(-25.0)
    # marked midway: 0.60 -> 50 shares x 0.60 = $30, pnl +$5
    assert position_pnl("YES", 0.50, 0.60, 25.0) == pytest.approx(5.0)


def test_no_position_pnl():
    # $25 of NO at yes=0.80 (NO costs 0.20) = 125 shares
    # resolves NO (yes=0.0): NO worth 1.0 -> $125, pnl +$100
    assert position_pnl("NO", 0.80, 0.0, 25.0) == pytest.approx(100.0)
    # resolves YES: NO worthless -> pnl -$25
    assert position_pnl("NO", 0.80, 1.0, 25.0) == pytest.approx(-25.0)


def test_extreme_entry_is_clamped_not_infinite():
    pnl = position_pnl("YES", 0.0, 1.0, 25.0)
    assert pnl > 0 and pnl != float("inf")


def _position(tmp_db, mid, side, price, amount=25.0):
    from locus.core import positions
    from locus.markets.gamma import Market

    trade_id = tmp_db.log_trade(
        market_id=mid, market_question="Q?", claude_score=0.7,
        market_price=price, edge=0.2, side=side, amount_usd=amount,
        status="dry_run", classification="bullish", materiality=0.7,
    )
    mkt = Market(mid, "Q?", "ai", price, round(1 - price, 4), 5000, "", True, [])
    positions.open_position(trade_id, mkt, side, amount)
    return trade_id


def test_compute_performance_rollup(tmp_db, monkeypatch):
    from locus.core import positions

    # closed winner: YES @0.50 resolved 1.0 -> +25
    t1 = _position(tmp_db, "m1", "YES", 0.50)
    positions.close_on_resolution(t1, 1.0)
    # closed loser: YES @0.50 resolved 0.0 -> -25
    t2 = _position(tmp_db, "m2", "YES", 0.50)
    positions.close_on_resolution(t2, 0.0)
    # open position: YES @0.40, now 0.50 -> +6.25 unrealized
    _position(tmp_db, "m3", "YES", 0.40)
    # open position with no available price: marked at entry, contributes 0
    _position(tmp_db, "m4", "YES", 0.40)
    # clear the stored entry marks so the fallback chain is exercised
    conn = tmp_db._conn()
    conn.execute("UPDATE positions SET current_yes_price=NULL WHERE status='open'")
    conn.commit(); conn.close()

    monkeypatch.setattr(performance, "_fetch_current_yes_price", lambda cid: None)
    perf = compute_performance(current_prices={"m3": 0.50})

    assert perf["trades_total"] == 4
    assert perf["deployed_usd"] == 100.0
    assert perf["wins"] == 1 and perf["losses"] == 1
    assert perf["win_rate_pct"] == 50.0
    assert perf["closed_count"] == 2 and perf["open_count"] == 2
    assert perf["realized_pnl_usd"] == pytest.approx(0.0)
    assert perf["unrealized_pnl_usd"] == pytest.approx(6.25)


def test_half_closed_realization_counts_in_realized(tmp_db, monkeypatch):
    from locus.core import positions

    _position(tmp_db, "m5", "YES", 0.50)
    pos = positions.get_open_positions()[0]
    conn = tmp_db._conn()
    positions._close(conn, pos, 0.80, "", "", fraction=0.5)  # realize half at +60%
    conn.commit(); conn.close()

    monkeypatch.setattr(performance, "_fetch_current_yes_price", lambda cid: None)
    perf = compute_performance(current_prices={"m5": 0.80})
    assert perf["open_count"] == 1 and perf["closed_count"] == 0
    assert perf["realized_pnl_usd"] == pytest.approx(7.5)
    assert perf["unrealized_pnl_usd"] == pytest.approx(7.5)  # remaining $12.50 at +60%


def test_empty_db_yields_zeroes(tmp_db):
    perf = compute_performance(current_prices={})
    assert perf["trades_total"] == 0
    assert perf["win_rate_pct"] is None
    assert perf["realized_pnl_usd"] == 0.0
    assert perf["unrealized_pnl_usd"] == 0.0


# --- Circuit breaker ---

def _closed_position(tmp_db, mid, amount, realized_pnl, closed_at):
    """Insert a fully-closed position with a chosen realized PnL and close time."""
    from locus.core import positions
    from locus.markets.gamma import Market

    trade_id = tmp_db.log_trade(
        market_id=mid, market_question="Q?", claude_score=0.7,
        market_price=0.5, edge=0.2, side="YES", amount_usd=amount,
        status="dry_run", classification="bullish", materiality=0.7,
    )
    mkt = Market(mid, "Q?", "ai", 0.5, 0.5, 5000, "", True, [])
    positions.open_position(trade_id, mkt, "YES", amount)
    conn = tmp_db._conn()
    conn.execute(
        "UPDATE positions SET status='closed_resolution', realized_pnl_usd=?, "
        "closed_at=? WHERE trade_id=?",
        (realized_pnl, closed_at, trade_id),
    )
    conn.commit()
    conn.close()
    return trade_id


def _days_ago(days, hours=0):
    return (datetime.now(timezone.utc) - timedelta(days=days, hours=hours)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def test_circuit_breaker_drawdown_trigger(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_ENABLED", True)
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_DD", 0.20)
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_SHARPE", -1.0)

    # All on the same day (one close-day -> Sharpe stays None, so only the
    # drawdown path can trip). Bankroll 75; equity 75 -> 115 -> 65 -> 55,
    # peak 115, trough 55 -> drawdown 52%.
    base = _days_ago(1)
    _closed_position(tmp_db, "m1", 25.0, +40.0, base[:11] + "10:00:00")
    _closed_position(tmp_db, "m2", 25.0, -50.0, base[:11] + "11:00:00")
    _closed_position(tmp_db, "m3", 25.0, -10.0, base[:11] + "12:00:00")

    cb = compute_circuit_breaker()
    assert cb["triggered"] is True
    assert "drawdown" in cb["reason"].lower()
    assert cb["metrics"]["drawdown_7d"] > 0.20
    assert cb["metrics"]["sharpe_7d"] is None


def test_circuit_breaker_sharpe_trigger(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_ENABLED", True)
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_DD", 0.20)
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_SHARPE", -1.0)

    # Three distinct close-days of mild losses on a large bankroll: drawdown
    # stays tiny (<20%) but daily PnL {-10,-12,-8} gives Sharpe -10/2 = -5.
    _closed_position(tmp_db, "m1", 1000.0, -10.0, _days_ago(3))
    _closed_position(tmp_db, "m2", 1000.0, -12.0, _days_ago(2))
    _closed_position(tmp_db, "m3", 1000.0, -8.0, _days_ago(1))

    cb = compute_circuit_breaker()
    assert cb["triggered"] is True
    assert "sharpe" in cb["reason"].lower()
    assert cb["metrics"]["sharpe_7d"] == pytest.approx(-5.0)
    assert cb["metrics"]["drawdown_7d"] < 0.20


def test_circuit_breaker_disabled(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_ENABLED", False)
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_DD", 0.20)
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_SHARPE", -1.0)

    # Same draw-down-tripping data as above, but the flag is off.
    base = _days_ago(1)
    _closed_position(tmp_db, "m1", 25.0, +40.0, base[:11] + "10:00:00")
    _closed_position(tmp_db, "m2", 25.0, -50.0, base[:11] + "11:00:00")
    _closed_position(tmp_db, "m3", 25.0, -10.0, base[:11] + "12:00:00")

    cb = compute_circuit_breaker()
    assert cb["triggered"] is False
    assert cb["reason"] == "disabled"
    # Metrics are still computed so the dashboard can show them.
    assert cb["metrics"]["drawdown_7d"] > 0.20


def test_circuit_breaker_normal_operation(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_ENABLED", True)
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_DD", 0.20)
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_SHARPE", -1.0)

    # Steady winners across three days: no drawdown, positive Sharpe.
    _closed_position(tmp_db, "m1", 100.0, +10.0, _days_ago(3))
    _closed_position(tmp_db, "m2", 100.0, +12.0, _days_ago(2))
    _closed_position(tmp_db, "m3", 100.0, +8.0, _days_ago(1))

    cb = compute_circuit_breaker()
    assert cb["triggered"] is False
    assert cb["reason"] == ""
    assert cb["metrics"]["drawdown_7d"] == pytest.approx(0.0)
    assert cb["metrics"]["sharpe_7d"] > 0


def test_circuit_breaker_empty_db(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_ENABLED", True)
    cb = compute_circuit_breaker()
    assert cb["triggered"] is False
    assert cb["metrics"]["drawdown_7d"] == 0.0
    assert cb["metrics"]["sharpe_7d"] is None
    assert cb["metrics"]["closed_trades_7d"] == 0


def _seed_old_loss_plus_recent_winners(tmp_db):
    """One big legacy loss 6 days ago (within the 7-day window) plus three
    recent winners. Included as a whole it trips the drawdown breaker; with the
    old loss excluded only the winners remain (no trip)."""
    _closed_position(tmp_db, "old", 25.0, -50.0, _days_ago(6))
    _closed_position(tmp_db, "m1", 10.0, +3.0, _days_ago(3))
    _closed_position(tmp_db, "m2", 10.0, +2.0, _days_ago(2))
    _closed_position(tmp_db, "m3", 10.0, +4.0, _days_ago(1))


def test_circuit_breaker_start_date_excludes_old_positions(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_ENABLED", True)
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_DD", 0.20)
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_SHARPE", -1.0)
    _seed_old_loss_plus_recent_winners(tmp_db)

    # Start date 4 days ago -> the 6-day-old loss is dropped, leaving winners.
    start = (datetime.now(timezone.utc) - timedelta(days=4)).strftime("%Y-%m-%d %H:%M:%S")
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_START_DATE", start)

    cb = compute_circuit_breaker()
    assert cb["metrics"]["closed_trades_7d"] == 3  # old loss excluded
    assert cb["metrics"]["drawdown_7d"] == pytest.approx(0.0)
    assert cb["triggered"] is False


def test_circuit_breaker_no_start_date_includes_all(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_ENABLED", True)
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_DD", 0.20)
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_SHARPE", -1.0)
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_START_DATE", "")  # default: count all
    _seed_old_loss_plus_recent_winners(tmp_db)

    cb = compute_circuit_breaker()
    assert cb["metrics"]["closed_trades_7d"] == 4  # all included
    assert cb["metrics"]["drawdown_7d"] > 0.20     # old loss drives the drawdown
    assert cb["triggered"] is True
    assert "drawdown" in cb["reason"].lower()


def test_circuit_breaker_start_date_accepts_date_only(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_ENABLED", True)
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_DD", 0.20)
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_SHARPE", -1.0)
    _seed_old_loss_plus_recent_winners(tmp_db)

    # Date-only floor (as documented in .env.example) compares lexicographically
    # against the full closed_at timestamps and still excludes the old loss.
    start = (datetime.now(timezone.utc) - timedelta(days=4)).strftime("%Y-%m-%d")
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_START_DATE", start)

    cb = compute_circuit_breaker()
    assert cb["metrics"]["closed_trades_7d"] == 3
    assert cb["triggered"] is False


# --- live readiness respects PERFORMANCE_START_DATE ------------------------

def _closed_with_opened(tmp_db, mid, amount, realized_pnl, opened_at, closed_at):
    """A fully-closed position with a chosen open AND close time."""
    from locus.core import positions
    from locus.markets.gamma import Market

    trade_id = tmp_db.log_trade(
        market_id=mid, market_question="Q?", claude_score=0.7,
        market_price=0.5, edge=0.2, side="YES", amount_usd=amount,
        status="dry_run", classification="bullish", materiality=0.7,
    )
    positions.open_position(trade_id, Market(mid, "Q?", "ai", 0.5, 0.5, 5000, "", True, []),
                            "YES", amount)
    conn = tmp_db._conn()
    conn.execute(
        "UPDATE positions SET status='closed_resolution', realized_pnl_usd=?, "
        "opened_at=?, closed_at=? WHERE trade_id=?",
        (realized_pnl, opened_at, closed_at, trade_id),
    )
    conn.commit()
    conn.close()


_LR_OLD = "2026-06-10 09:00:00"
_LR_NEW = "2026-06-14 09:00:00"
_LR_CUTOFF = "2026-06-14"


def _seed_old_and_new_closes(tmp_db):
    # Old: two winners. New: one winner, one loser.
    _closed_with_opened(tmp_db, "old1", 25.0, 10.0, _LR_OLD, "2026-06-10 12:00:00")
    _closed_with_opened(tmp_db, "old2", 25.0, 10.0, _LR_OLD, "2026-06-11 12:00:00")
    _closed_with_opened(tmp_db, "new1", 25.0, 10.0, _LR_NEW, "2026-06-14 12:00:00")
    _closed_with_opened(tmp_db, "new2", 25.0, -10.0, _LR_NEW, "2026-06-15 12:00:00")


def _readiness_value(result, key):
    return next(m["value"] for m in result["metrics"] if m["key"] == key)


def test_live_readiness_filtered_by_date(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "PERFORMANCE_START_DATE", _LR_CUTOFF)
    _seed_old_and_new_closes(tmp_db)
    r = compute_live_readiness()
    # Only the two NEW closes count.
    assert _readiness_value(r, "closed_trades") == 2
    assert _readiness_value(r, "win_rate") == pytest.approx(50.0)   # 1 of 2


def test_live_readiness_unfiltered_counts_all(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "PERFORMANCE_START_DATE", "")
    _seed_old_and_new_closes(tmp_db)
    r = compute_live_readiness()
    assert _readiness_value(r, "closed_trades") == 4
    assert _readiness_value(r, "win_rate") == pytest.approx(75.0)   # 3 of 4


def test_live_readiness_future_date_excludes_all(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "PERFORMANCE_START_DATE", "2030-01-01")
    _seed_old_and_new_closes(tmp_db)
    r = compute_live_readiness()
    # No closed trades in the window -> every metric scopes to empty (N/A).
    assert _readiness_value(r, "closed_trades") == 0
    assert _readiness_value(r, "win_rate") is None
    assert _readiness_value(r, "sharpe_ratio") is None
    assert _readiness_value(r, "max_drawdown") is None
    assert r["ready"] is False
    assert r["criteria_met"] == 0
    assert all(m["pass"] is None for m in r["metrics"])
