"""reconcile_positions: sync DB open positions against real on-chain token
balances. OK / mismatch / unknown classification, --fix closing confirmed
phantoms, and dry-run leaving the DB untouched."""
import pytest

from locus.core import positions, executor
from locus.markets.gamma import Market


def _mkt(condition_id: str) -> Market:
    return Market(condition_id, f"Will {condition_id} happen?", "ai",
                  0.50, 0.50, 5000, "", True, [], slug=f"slug-{condition_id}")


def _open(tmp_db, condition_id: str, side: str = "YES", amount: float = 25.0) -> dict:
    trade_id = tmp_db.log_trade(
        market_id=condition_id, market_question=f"Will {condition_id} happen?",
        claude_score=0.7, market_price=0.50, edge=0.2, side=side, amount_usd=amount,
        status="dry_run", classification="bullish", materiality=0.7,
    )
    positions.open_position(trade_id, _mkt(condition_id), side, amount,
                            headline="h", reasoning="r")


def _holdings(monkeypatch, balances: dict[str, float | None]):
    """Stub token resolution + on-chain balance lookup. `balances` maps a
    position's condition_id to its held share count (None = unverifiable)."""
    monkeypatch.setattr(
        executor, "_resolve_token_id",
        lambda client, condition_id, side: f"tok-{condition_id}",
    )
    monkeypatch.setattr(
        executor, "held_token_shares",
        lambda client, token_id: balances.get(token_id.removeprefix("tok-")),
    )


CLIENT = object()  # opaque sentinel — resolution/balance are stubbed


def _status(tmp_db, condition_id: str) -> str:
    conn = tmp_db._conn()
    try:
        return conn.execute(
            "SELECT status FROM positions WHERE condition_id=?", (condition_id,)
        ).fetchone()["status"]
    finally:
        conn.close()


def test_ok_position(tmp_db, monkeypatch):
    _open(tmp_db, "c_ok")
    _holdings(monkeypatch, {"c_ok": 60.0})

    report = positions.reconcile_positions(fix=False, client=CLIENT)

    assert report["ok"] == [positions.get_open_positions()[0]["id"]]
    assert report["mismatches"] == [] and report["unknown"] == []
    line = report["entries"][0]["line"]
    assert line.startswith("OK:") and "held=60 tokens — matches DB" in line


def test_mismatch_detected(tmp_db, monkeypatch):
    _open(tmp_db, "c_phantom")
    pid = positions.get_open_positions()[0]["id"]
    _holdings(monkeypatch, {"c_phantom": 0.0})

    report = positions.reconcile_positions(fix=False, client=CLIENT)

    assert report["mismatches"] == [pid]
    assert report["entries"][0]["line"] == (
        f"MISMATCH: ID={pid} DB says open but held=0 on Polymarket"
    )
    # Dry-run: still open in the DB.
    assert _status(tmp_db, "c_phantom") == "open"


def test_fix_closes_mismatch(tmp_db, monkeypatch):
    _open(tmp_db, "c_phantom")
    pid = positions.get_open_positions()[0]["id"]
    _holdings(monkeypatch, {"c_phantom": 0.0})

    report = positions.reconcile_positions(fix=True, client=CLIENT)

    assert report["fixed"] == [pid]
    # No longer an open position.
    assert positions.get_open_positions() == []
    conn = tmp_db._conn()
    try:
        row = conn.execute(
            "SELECT status, exit_reason, realized_pnl_usd, closed_at "
            "FROM positions WHERE id=?", (pid,)
        ).fetchone()
    finally:
        conn.close()
    assert row["status"] == "closed_reconciled"
    assert row["exit_reason"] == "reconcile_mismatch"
    # The reconcile close realizes nothing (a never-realized row stays NULL/0).
    assert (row["realized_pnl_usd"] or 0) == 0
    assert row["closed_at"] is not None


def test_fix_preserves_prior_partial_realizations(tmp_db, monkeypatch):
    # A position that realized +$5.00 from an earlier close_half, whose
    # remainder later turns out to be gone on-chain: the reconcile close must
    # PRESERVE the $5 — it is real money that was actually realized — and only
    # flip status/exit_reason/closed_at.
    _open(tmp_db, "c_half_then_gone")
    pid = positions.get_open_positions()[0]["id"]
    conn = tmp_db._conn()
    conn.execute("UPDATE positions SET realized_pnl_usd=5.0 WHERE id=?", (pid,))
    conn.commit()
    conn.close()
    _holdings(monkeypatch, {"c_half_then_gone": 0.0})

    report = positions.reconcile_positions(fix=True, client=CLIENT)

    assert report["fixed"] == [pid]
    conn = tmp_db._conn()
    try:
        row = conn.execute(
            "SELECT status, exit_reason, realized_pnl_usd FROM positions "
            "WHERE id=?", (pid,)
        ).fetchone()
    finally:
        conn.close()
    assert row["status"] == "closed_reconciled"
    assert row["exit_reason"] == "reconcile_mismatch"
    assert row["realized_pnl_usd"] == pytest.approx(5.0)  # not zeroed


def test_dry_run_does_not_change_db(tmp_db, monkeypatch):
    _open(tmp_db, "c_phantom")
    _holdings(monkeypatch, {"c_phantom": 0.0})

    report = positions.reconcile_positions(fix=False, client=CLIENT)

    assert report["fixed"] == []
    assert _status(tmp_db, "c_phantom") == "open"
    assert len(positions.get_open_positions()) == 1


def test_none_balance_is_unknown_not_closed(tmp_db, monkeypatch):
    # held_token_shares returns None (client lacks the call / errored): the
    # position is UNKNOWN and must never be auto-closed, even with --fix.
    _open(tmp_db, "c_unknown")
    pid = positions.get_open_positions()[0]["id"]
    _holdings(monkeypatch, {"c_unknown": None})

    report = positions.reconcile_positions(fix=True, client=CLIENT)

    assert report["unknown"] == [pid]
    assert report["mismatches"] == [] and report["fixed"] == []
    assert report["entries"][0]["line"] == f"UNKNOWN: could not verify ID={pid}"
    assert _status(tmp_db, "c_unknown") == "open"


def test_unresolved_token_is_unknown(tmp_db, monkeypatch):
    # Token can't be resolved (e.g. exotic outcome labels) -> UNKNOWN, not closed.
    _open(tmp_db, "c_notoken")
    monkeypatch.setattr(executor, "_resolve_token_id",
                        lambda client, condition_id, side: None)
    report = positions.reconcile_positions(fix=True, client=CLIENT)
    assert report["unknown"] and report["fixed"] == []
    assert _status(tmp_db, "c_notoken") == "open"


def test_mixed_book_only_mismatch_closed(tmp_db, monkeypatch):
    _open(tmp_db, "c_ok")
    _open(tmp_db, "c_phantom")
    _open(tmp_db, "c_unknown")
    _holdings(monkeypatch, {"c_ok": 12.0, "c_phantom": 0.0, "c_unknown": None})

    report = positions.reconcile_positions(fix=True, client=CLIENT)

    assert len(report["ok"]) == 1
    assert len(report["mismatches"]) == 1 and len(report["fixed"]) == 1
    assert len(report["unknown"]) == 1
    # Only the phantom closed; OK and UNKNOWN stay open.
    assert _status(tmp_db, "c_ok") == "open"
    assert _status(tmp_db, "c_phantom") == "closed_reconciled"
    assert _status(tmp_db, "c_unknown") == "open"


def test_no_open_positions(tmp_db, monkeypatch):
    # No client should be built when there's nothing to check.
    monkeypatch.setattr(executor, "create_clob_client",
                        lambda: (_ for _ in ()).throw(AssertionError("should not build")))
    report = positions.reconcile_positions(fix=True)
    assert report == {"entries": [], "ok": [], "mismatches": [], "unknown": [], "fixed": []}
