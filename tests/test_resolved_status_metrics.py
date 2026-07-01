"""'resolved' positions (resolution closes) must be visible to every risk/metrics
reader — the circuit breaker, the dynamic-Kelly win rate, live-readiness, and the
calibration report. Regression tests for LOGIC_REVIEW.md finding #2 (the old
status LIKE 'closed_%' filters silently excluded status='resolved')."""
from datetime import datetime, timezone

import pytest

from locus import config
from locus.core.performance import (
    compute_circuit_breaker,
    compute_live_readiness,
    calibration_report,
)


@pytest.fixture(autouse=True)
def _neutralize_start_dates(monkeypatch):
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_START_DATE", "")
    monkeypatch.setattr(config, "PERFORMANCE_START_DATE", "")


def _resolved_position(tmp_db, mid, side, entry_price, amount, exit_price):
    """Open a position and close it through the REAL resolution path
    (close_on_resolution -> status='resolved'), returning the trade id."""
    from locus.core import positions
    from locus.markets.gamma import Market

    trade_id = tmp_db.log_trade(
        market_id=mid, market_question=f"Will {mid} resolve YES?", claude_score=0.7,
        market_price=entry_price, edge=0.2, side=side, amount_usd=amount,
        status="dry_run", classification="bullish", materiality=0.7,
    )
    mkt = Market(mid, f"Will {mid} resolve YES?", "ai", entry_price,
                 round(1 - entry_price, 4), 5000, "", True, [])
    positions.open_position(trade_id, mkt, side, amount)
    positions.close_on_resolution(trade_id, exit_price)
    return trade_id


def test_resolution_close_sets_resolved_status(tmp_db):
    _resolved_position(tmp_db, "m1", "YES", 0.50, 25.0, 0.0)
    conn = tmp_db._conn()
    row = conn.execute("SELECT status, closed_at, realized_pnl_usd FROM positions").fetchone()
    conn.close()
    assert row["status"] == "resolved"
    assert row["closed_at"] is not None
    assert row["realized_pnl_usd"] == pytest.approx(-25.0)


def test_circuit_breaker_sees_resolution_losses(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_ENABLED", True)
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_DD", 0.20)
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_SHARPE", -1.0)

    # Two ride-to-zero resolution losses today: bankroll 50, equity 50 -> 0,
    # drawdown 100% — far past the 20% limit. Before the fix these rows were
    # invisible ('resolved' escaped LIKE 'closed_%') and the breaker stayed green.
    _resolved_position(tmp_db, "m1", "YES", 0.50, 25.0, 0.0)
    _resolved_position(tmp_db, "m2", "YES", 0.50, 25.0, 0.0)

    cb = compute_circuit_breaker()
    assert cb["metrics"]["closed_trades_7d"] == 2
    assert cb["metrics"]["drawdown_7d"] == pytest.approx(1.0)
    assert cb["triggered"] is True
    assert "drawdown" in cb["reason"]


def test_kelly_winrate_sees_resolution_closes(tmp_db):
    # One resolution win (+25), one resolution loss (-25).
    _resolved_position(tmp_db, "m1", "YES", 0.50, 25.0, 1.0)
    _resolved_position(tmp_db, "m2", "YES", 0.50, 25.0, 0.0)

    pnls = tmp_db.get_recent_closed_position_pnls(20)
    assert len(pnls) == 2
    assert sorted(round(p, 2) for p in pnls) == [-25.0, 25.0]


def test_live_readiness_counts_resolution_closes(tmp_db):
    _resolved_position(tmp_db, "m1", "YES", 0.50, 25.0, 1.0)
    _resolved_position(tmp_db, "m2", "YES", 0.50, 25.0, 0.0)

    readiness = compute_live_readiness()
    by_key = {m["key"]: m for m in readiness["metrics"]}
    assert by_key["closed_trades"]["value"] == 2
    assert by_key["win_rate"]["value"] == pytest.approx(50.0)


def test_calibration_report_includes_resolution_closes(tmp_db):
    _resolved_position(tmp_db, "m1", "YES", 0.50, 25.0, 1.0)
    _resolved_position(tmp_db, "m2", "YES", 0.50, 25.0, 0.0)

    report = calibration_report()
    assert report["summary"]["n"] == 2
    assert report["summary"]["wins"] == 1
    labels = {e["label"] for e in report["by_exit_reason"] if e["n"]}
    assert "resolution" in labels


def test_half_close_realizations_still_excluded(tmp_db):
    """close_half rows keep status='open' — the new status != 'open' filters
    must not start counting them as closed trades."""
    from locus.core import positions
    from locus.markets.gamma import Market

    trade_id = tmp_db.log_trade(
        market_id="m5", market_question="Q?", claude_score=0.7,
        market_price=0.50, edge=0.2, side="YES", amount_usd=25.0,
        status="dry_run", classification="bullish", materiality=0.7,
    )
    mkt = Market("m5", "Q?", "ai", 0.50, 0.50, 5000, "", True, [])
    positions.open_position(trade_id, mkt, "YES", 25.0)
    pos = positions.get_open_positions()[0]
    conn = tmp_db._conn()
    positions._close(conn, pos, 0.80, "", "", fraction=0.5)
    conn.commit(); conn.close()

    assert compute_circuit_breaker()["metrics"]["closed_trades_7d"] == 0
    assert tmp_db.get_recent_closed_position_pnls(20) == []
    assert calibration_report()["summary"]["n"] == 0
    by_key = {m["key"]: m for m in compute_live_readiness()["metrics"]}
    assert by_key["closed_trades"]["value"] == 0
