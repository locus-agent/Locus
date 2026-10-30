"""reconcile_positions: sync DB open positions against real on-chain token
balances. OK / mismatch / unknown classification, --fix closing confirmed
phantoms, dry-run leaving the DB untouched, and per-position Gamma market
status (OPEN / CLOSED / NOT FOUND — a CLOSED market is flagged, never
auto-closed)."""
import pytest

from locus.core import positions, executor
from locus.markets.gamma import Market


@pytest.fixture(autouse=True)
def _gamma_open_markets(monkeypatch):
    """Default every queried market to OPEN on Gamma (no network); tests that
    exercise closed/missing markets override with _gamma_states."""
    monkeypatch.setattr(
        positions.gamma, "fetch_markets_by_condition_ids",
        lambda ids: {cid: {"condition_id": cid, "closed": False} for cid in ids},
    )


def _gamma_states(monkeypatch, states: dict[str, bool | None]):
    """Stub the Gamma batch fetch. `states` maps condition_id -> closed flag;
    None means the market is absent from Gamma's response entirely."""
    monkeypatch.setattr(
        positions.gamma, "fetch_markets_by_condition_ids",
        lambda ids: {cid: {"condition_id": cid, "closed": bool(states[cid])}
                     for cid in ids if states.get(cid) is not None},
    )


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
    # $25 at 0.50 -> the DB expects 50 shares; the chain agrees.
    _open(tmp_db, "c_ok")
    _holdings(monkeypatch, {"c_ok": 50.0})

    report = positions.reconcile_positions(fix=False, client=CLIENT)

    assert report["ok"] == [positions.get_open_positions()[0]["id"]]
    assert report["mismatches"] == [] and report["unknown"] == []
    assert report["market_closed"] == []
    entry = report["entries"][0]
    assert entry["line"].startswith("OK:")
    assert "held=50 tokens — matches DB" in entry["line"]
    assert entry["expected"] == 50.0
    assert entry["market_state"] == "open"
    assert "market OPEN" in entry["line"]


def test_mismatch_detected(tmp_db, monkeypatch):
    _open(tmp_db, "c_phantom")
    pid = positions.get_open_positions()[0]["id"]
    _holdings(monkeypatch, {"c_phantom": 0.0})

    report = positions.reconcile_positions(fix=False, client=CLIENT)

    assert report["mismatches"] == [pid]
    assert report["entries"][0]["line"] == (
        f"MISMATCH: ID={pid} DB says open but held=0 on Polymarket — market OPEN"
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
    assert report["entries"][0]["line"] == (
        f"UNKNOWN: could not verify ID={pid} — market OPEN"
    )
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
    _holdings(monkeypatch, {"c_ok": 50.0, "c_phantom": 0.0, "c_unknown": None})

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
    assert report == {"entries": [], "ok": [], "mismatches": [], "unknown": [],
                      "market_closed": [], "fixed": []}


# --- Gamma market status (OPEN / CLOSED / NOT FOUND) --------------------------


def test_closed_market_flagged_but_never_auto_closed(tmp_db, monkeypatch):
    # Tokens still held (balance OK) but the market resolved out from under the
    # position: report it prominently as market_closed, and --fix must NOT
    # touch it — the tokens may still be claimable at resolution.
    _open(tmp_db, "c_closedmkt")
    pid = positions.get_open_positions()[0]["id"]
    _holdings(monkeypatch, {"c_closedmkt": 50.0})
    _gamma_states(monkeypatch, {"c_closedmkt": True})

    report = positions.reconcile_positions(fix=True, client=CLIENT)

    assert report["market_closed"] == [pid]
    assert report["fixed"] == []
    entry = report["entries"][0]
    assert entry["market_state"] == "closed"
    assert "market CLOSED — position may need resolution handling!" in entry["line"]
    # The token-balance check is unchanged: held > 0 still reads OK.
    assert report["ok"] == [pid]
    assert _status(tmp_db, "c_closedmkt") == "open"


def test_market_not_found_reported(tmp_db, monkeypatch):
    _open(tmp_db, "c_gone")
    _holdings(monkeypatch, {"c_gone": 50.0})
    _gamma_states(monkeypatch, {"c_gone": None})  # absent from Gamma's response

    report = positions.reconcile_positions(fix=False, client=CLIENT)

    entry = report["entries"][0]
    assert entry["market_state"] == "not_found"
    assert "market NOT FOUND in Gamma" in entry["line"]
    assert report["market_closed"] == []


def test_closed_market_phantom_still_auto_fixable(tmp_db, monkeypatch):
    # held==0 stays auto-fixable even on a closed market: we hold zero tokens,
    # so nothing is claimable — the row is a plain phantom either way.
    _open(tmp_db, "c_both")
    pid = positions.get_open_positions()[0]["id"]
    _holdings(monkeypatch, {"c_both": 0.0})
    _gamma_states(monkeypatch, {"c_both": True})

    report = positions.reconcile_positions(fix=True, client=CLIENT)

    assert report["fixed"] == [pid]
    assert report["market_closed"] == [pid]
    assert _status(tmp_db, "c_both") == "closed_reconciled"


def test_gamma_fetch_failure_reports_not_found(tmp_db, monkeypatch):
    # A Gamma outage must not break reconciliation: the balance check still
    # runs, and every market reads NOT FOUND (report-only, nothing closed).
    _open(tmp_db, "c_outage")
    _holdings(monkeypatch, {"c_outage": 50.0})
    monkeypatch.setattr(
        positions.gamma, "fetch_markets_by_condition_ids",
        lambda ids: (_ for _ in ()).throw(RuntimeError("gamma down")),
    )

    report = positions.reconcile_positions(fix=True, client=CLIENT)

    assert report["ok"] and report["fixed"] == []
    assert report["entries"][0]["market_state"] == "not_found"


# --- Relative tolerance (holding_matches): >1% or >0.5 tokens is a MISMATCH ---


def _set_token_count(tmp_db, condition_id: str, token_count: float) -> None:
    conn = tmp_db._conn()
    conn.execute("UPDATE positions SET token_count=? WHERE condition_id=?",
                 (token_count, condition_id))
    conn.commit()
    conn.close()


def test_partial_divergence_is_mismatch_but_never_auto_closed(tmp_db, monkeypatch):
    # The position-57 case: the DB believes 1.995 tokens are held, the chain
    # holds 0.005 — a 99.7% divergence must read MISMATCH, but --fix must NOT
    # close it (held > 0 is real exposure, only held==0 phantoms are fixable).
    _open(tmp_db, "c_lepen")
    pid = positions.get_open_positions()[0]["id"]
    _set_token_count(tmp_db, "c_lepen", 1.995)
    _holdings(monkeypatch, {"c_lepen": 0.005})

    report = positions.reconcile_positions(fix=True, client=CLIENT)

    assert report["mismatches"] == [pid]
    assert report["fixed"] == []
    entry = report["entries"][0]
    assert entry["state"] == "mismatch"
    assert entry["held"] == 0.005 and entry["expected"] == 1.995
    assert "NOT auto-closed" in entry["line"]
    assert "99.7% off" in entry["line"]
    assert _status(tmp_db, "c_lepen") == "open"


def test_rounding_dust_within_tolerance_is_ok(tmp_db, monkeypatch):
    # 34.9 held vs 35 expected: 0.1 tokens / 0.29% — CLOB size-snapping dust.
    _open(tmp_db, "c_dust")
    _set_token_count(tmp_db, "c_dust", 35.0)
    _holdings(monkeypatch, {"c_dust": 34.9})

    report = positions.reconcile_positions(fix=True, client=CLIENT)

    assert report["ok"] == [positions.get_open_positions()[0]["id"]]
    assert report["mismatches"] == [] and report["fixed"] == []


def test_excess_on_chain_is_also_mismatch(tmp_db, monkeypatch):
    # More on-chain than the DB knows about (untracked fill) diverges too.
    _open(tmp_db, "c_excess")
    pid = positions.get_open_positions()[0]["id"]
    _set_token_count(tmp_db, "c_excess", 50.0)
    _holdings(monkeypatch, {"c_excess": 55.0})

    report = positions.reconcile_positions(fix=True, client=CLIENT)

    assert report["mismatches"] == [pid] and report["fixed"] == []
    assert _status(tmp_db, "c_excess") == "open"


def test_holding_matches_bounds():
    # Within both bounds -> ok.
    assert positions.holding_matches(100.0, 99.6)      # 0.4 sh, 0.4%
    assert positions.holding_matches(35.0, 34.9)       # 0.1 sh, 0.29%
    # Absolute bound: > 0.5 tokens fails even when the % is tiny.
    assert not positions.holding_matches(100.0, 99.4)  # 0.6 sh, 0.6%
    # Relative bound: > 1% fails even when the share gap is below 0.5.
    assert not positions.holding_matches(10.0, 9.6)    # 0.4 sh, 4%
    assert not positions.holding_matches(1.995, 0.005)  # position 57
