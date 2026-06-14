"""Performance aggregation: position PnL math, realized/unrealized rollups."""
from datetime import datetime, timedelta, timezone

import pytest

from locus import config
from locus.core import performance
from locus.core.performance import (
    position_pnl,
    compute_performance,
    compute_circuit_breaker,
)


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
