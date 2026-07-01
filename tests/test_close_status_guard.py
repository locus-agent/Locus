"""_close must realize PnL exactly once: the UPDATE is guarded with
`AND status='open'`, so a second closer working from a stale snapshot becomes a
no-op instead of double-adding realized_pnl_usd. Regression tests for
LOGIC_REVIEW.md finding #3 (double-close race)."""
import pytest

from locus.core import positions
from locus.markets.gamma import Market


def _open_position(tmp_db, mid="m1", side="YES", price=0.50, amount=25.0):
    trade_id = tmp_db.log_trade(
        market_id=mid, market_question=f"Will {mid} happen?", claude_score=0.7,
        market_price=price, edge=0.2, side=side, amount_usd=amount,
        status="dry_run", classification="bullish", materiality=0.7,
    )
    mkt = Market(mid, f"Will {mid} happen?", "ai", price, round(1 - price, 4),
                 5000, "", True, [])
    positions.open_position(trade_id, mkt, side, amount)
    return trade_id


def _db_position(tmp_db, trade_id):
    conn = tmp_db._conn()
    row = conn.execute(
        "SELECT * FROM positions WHERE trade_id=?", (trade_id,)
    ).fetchone()
    conn.close()
    return dict(row)


def test_second_full_close_is_a_noop(tmp_db):
    trade_id = _open_position(tmp_db)
    # Both closers read the position while it was still open (stale snapshots).
    snapshot_a = _db_position(tmp_db, trade_id)
    snapshot_b = _db_position(tmp_db, trade_id)

    conn = tmp_db._conn()
    realized_a = positions._close(conn, snapshot_a, 0.80, "closed_sl", "sl")
    conn.commit(); conn.close()
    assert realized_a == pytest.approx(15.0)  # $25 at 0.50 -> 0.80 = +60%

    # The loser of the race: same position, different price/reason.
    conn = tmp_db._conn()
    realized_b = positions._close(conn, snapshot_b, 1.0, "closed_tp", "tp_decision")
    conn.commit(); conn.close()
    assert realized_b == 0.0

    final = _db_position(tmp_db, trade_id)
    # PnL realized exactly once; the first close's status/exit stand.
    assert final["realized_pnl_usd"] == pytest.approx(15.0)
    assert final["status"] == "closed_sl"
    assert final["exit_reason"] == "sl"
    assert final["exit_yes_price"] == pytest.approx(0.80)


def test_half_close_after_full_close_is_a_noop(tmp_db):
    trade_id = _open_position(tmp_db)
    snapshot = _db_position(tmp_db, trade_id)

    conn = tmp_db._conn()
    realized = positions._close(conn, snapshot, 0.80, "closed_sl", "sl")
    conn.commit(); conn.close()
    assert realized == pytest.approx(15.0)

    # A concurrent close_half from a stale snapshot realizes nothing and must
    # not shrink amount_usd on the already-closed row.
    conn = tmp_db._conn()
    realized_half = positions._close(conn, snapshot, 0.90, "", "", fraction=0.5)
    conn.commit(); conn.close()
    assert realized_half == 0.0

    final = _db_position(tmp_db, trade_id)
    assert final["realized_pnl_usd"] == pytest.approx(15.0)
    assert final["amount_usd"] == pytest.approx(25.0)


def test_noop_close_does_not_watch_for_reentry(tmp_db):
    """The losing closer must not insert a second re-entry watch row (nor any
    row keyed to its own exit reason)."""
    trade_id = _open_position(tmp_db)
    snapshot_a = _db_position(tmp_db, trade_id)
    snapshot_b = _db_position(tmp_db, trade_id)

    conn = tmp_db._conn()
    positions._close(conn, snapshot_a, 0.80, "closed_tp", "tp_decision")
    positions._close(conn, snapshot_b, 0.80, "closed_sl", "sl")
    conn.commit()
    rows = conn.execute(
        "SELECT exit_reason FROM watched_closed_positions"
    ).fetchall()
    conn.close()
    assert [r["exit_reason"] for r in rows] == ["tp_decision"]


def test_resolution_after_manual_close_realizes_once(tmp_db):
    """End-to-end shape of the real race: a manual/rule close lands first, the
    calibrator's resolution close arrives later and must be a no-op."""
    trade_id = _open_position(tmp_db)
    snapshot = _db_position(tmp_db, trade_id)

    conn = tmp_db._conn()
    positions._close(conn, snapshot, 0.80, "closed_sl", "sl")
    conn.commit(); conn.close()

    positions.close_on_resolution(trade_id, 1.0)

    final = _db_position(tmp_db, trade_id)
    assert final["realized_pnl_usd"] == pytest.approx(15.0)
    assert final["status"] == "closed_sl"
