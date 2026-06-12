"""Performance aggregation: position PnL math, realized/unrealized rollups."""
import pytest

from locus.core import performance
from locus.core.performance import position_pnl, compute_performance


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


def _trade(tmp_db, mid, side, price, amount=25.0):
    return tmp_db.log_trade(
        market_id=mid, market_question="Q?", claude_score=0.7,
        market_price=price, edge=0.2, side=side, amount_usd=amount,
        status="dry_run", classification="bullish", materiality=0.7,
    )


def test_compute_performance_rollup(tmp_db, monkeypatch):
    # closed winner: YES @0.50 resolved 1.0 -> +25
    t1 = _trade(tmp_db, "m1", "YES", 0.50)
    tmp_db.log_calibration(trade_id=t1, classification="bullish", materiality=0.7,
                           entry_price=0.50, exit_price=1.0, actual_direction="bullish",
                           correct=True, resolved_at="2026-06-13T00:00:00")
    # closed loser: YES @0.50 resolved 0.0 -> -25
    t2 = _trade(tmp_db, "m2", "YES", 0.50)
    tmp_db.log_calibration(trade_id=t2, classification="bullish", materiality=0.7,
                           entry_price=0.50, exit_price=0.0, actual_direction="bearish",
                           correct=False, resolved_at="2026-06-13T00:00:00")
    # open position: YES @0.40, now 0.50 -> +6.25 unrealized
    _trade(tmp_db, "m3", "YES", 0.40)
    # open position with no available price: marked at entry, contributes 0
    _trade(tmp_db, "m4", "YES", 0.40)

    monkeypatch.setattr(performance, "_fetch_current_yes_price", lambda cid: None)
    perf = compute_performance(current_prices={"m3": 0.50})

    assert perf["trades_total"] == 4
    assert perf["deployed_usd"] == 100.0
    assert perf["wins"] == 1 and perf["losses"] == 1
    assert perf["win_rate_pct"] == 50.0
    assert perf["closed_count"] == 2 and perf["open_count"] == 2
    assert perf["realized_pnl_usd"] == pytest.approx(0.0)
    assert perf["unrealized_pnl_usd"] == pytest.approx(6.25)


def test_empty_db_yields_zeroes(tmp_db):
    perf = compute_performance(current_prices={})
    assert perf["trades_total"] == 0
    assert perf["win_rate_pct"] is None
    assert perf["realized_pnl_usd"] == 0.0
    assert perf["unrealized_pnl_usd"] == 0.0
