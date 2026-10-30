"""backfill-realized: recompute closed positions' realized PnL from the Data
API activity feed (TRADE + REDEEM/MERGE/SPLIT). Matching by
conditionId+outcomeIndex within the life window (clipped at sibling
re-entries on the same market+side), the bought/sold/redeemed completeness
check, delta vs the already-recorded realized, --apply writing delta + audit
row only when complete (idempotent), dry-run writing nothing, and open
positions never touched by the main report (though previewable)."""
import pytest

from locus.core import backfill, positions
from locus.markets.gamma import Market


WALLET = "0xWALLET"


def _mkt(condition_id: str, yes_price: float = 0.50) -> Market:
    return Market(condition_id, f"Will {condition_id} happen?", "ai",
                  yes_price, 1 - yes_price, 5000, "", True, [],
                  slug=f"slug-{condition_id}")


def _open(tmp_db, condition_id: str, side: str = "YES", amount: float = 25.0,
          entry: float = 0.50) -> int:
    trade_id = tmp_db.log_trade(
        market_id=condition_id, market_question=f"Will {condition_id} happen?",
        claude_score=0.7, market_price=entry, edge=0.2, side=side,
        amount_usd=amount, status="executed", classification="bullish",
        materiality=0.7,
    )
    positions.open_position(trade_id, _mkt(condition_id, entry), side, amount,
                            headline="h", reasoning="r")
    return positions.get_open_positions()[0]["id"]


def _close_row(tmp_db, pid: int, status: str = "closed_reconciled",
               realized: float | None = 0.0,
               opened_at: str = "2026-07-01 10:00:00",
               closed_at: str | None = "2026-09-01 10:00:00") -> None:
    conn = tmp_db._conn()
    conn.execute(
        "UPDATE positions SET status=?, realized_pnl_usd=?, opened_at=?, "
        "closed_at=? WHERE id=?",
        (status, realized, opened_at, closed_at, pid),
    )
    conn.commit()
    conn.close()


def _row(tmp_db, pid: int) -> dict:
    conn = tmp_db._conn()
    try:
        return dict(conn.execute(
            "SELECT * FROM positions WHERE id=?", (pid,)).fetchone())
    finally:
        conn.close()


def _audit_rows(tmp_db, pid: int) -> list[dict]:
    conn = tmp_db._conn()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM exit_decisions WHERE position_id=? AND "
            "trigger='backfill'", (pid,)).fetchall()]
    finally:
        conn.close()


# Epochs inside / outside the [2026-07-01, 2026-09-01] life window above.
T_IN_1 = 1784800000.0   # 2026-07-23
T_IN_2 = 1786000000.0   # 2026-08-06
T_BEFORE = 1780000000.0  # 2026-05-29
T_AFTER = 1790000000.0   # 2026-09-21


def _trade(cid: str, side: str, idx: int, usdc: float, size: float,
           ts: float = T_IN_1, type_: str = "TRADE") -> dict:
    return {"type": type_, "conditionId": cid, "outcomeIndex": idx,
            "side": side, "usdcSize": usdc, "size": size, "timestamp": ts,
            "price": round(usdc / size, 4) if size else 0}


def _redeem(cid: str, idx: int, usdc: float, size: float, ts: float) -> dict:
    return {"type": "REDEEM", "conditionId": cid, "outcomeIndex": idx,
            "side": "", "usdcSize": usdc, "size": size, "timestamp": ts}


def _fetcher(rows):
    return lambda wallet: rows


def test_matching_filters_condition_outcome_window(tmp_db):
    pid = _open(tmp_db, "c1", side="NO", entry=0.64)
    _close_row(tmp_db, pid)
    activity = [
        _trade("c1", "BUY", 1, 34.56, 96.0),            # matches
        _trade("c1", "SELL", 1, 29.88, 96.0, T_IN_2),   # matches
        _trade("c1", "SELL", 0, 10.0, 10.0),            # wrong outcome (YES token)
        _trade("other", "SELL", 1, 10.0, 10.0),         # wrong market
        _trade("c1", "SELL", 1, 10.0, 10.0, T_BEFORE),  # before open
        _trade("c1", "SELL", 1, 10.0, 10.0, T_AFTER),   # after close
    ]
    report = backfill.backfill_realized(apply=False, wallet=WALLET,
                                        fetcher=_fetcher(activity))
    assert len(report["rows"]) == 1
    row = report["rows"][0]
    assert row["trades_matched"] == 2
    assert row["redeems_matched"] == 0
    assert row["buys_usd"] == pytest.approx(34.56)
    assert row["sells_usd"] == pytest.approx(29.88)
    assert row["redeemed_usd"] == pytest.approx(0.0)
    assert row["complete"]
    assert row["unexplained_shares"] == pytest.approx(0.0)
    assert row["realized_true"] == pytest.approx(-4.68)
    assert row["delta"] == pytest.approx(-4.68)
    assert row["writable"]


def test_redeem_counts_as_proceeds_and_shares(tmp_db):
    # Position-48 shape: 56.75 bought, 56 sold (drawdown close), 0.75 sh of
    # dust redeemed later at market resolution — a pure proceeds leg the old
    # TRADE-only accounting missed entirely.
    pid = _open(tmp_db, "c48", side="NO", entry=0.44)
    _close_row(tmp_db, pid, realized=-8.05,
              opened_at="2026-06-29 09:00:00",
              closed_at="2026-06-29 10:09:40")
    activity = [
        _trade("c48", "BUY", 1, 24.97, 56.75,
              backfill._epoch("2026-06-29 09:43:29")),
        _trade("c48", "SELL", 1, 16.52, 56.0,
              backfill._epoch("2026-06-29 10:09:41")),
        _redeem("c48", 1, 0.75, 0.75, T_IN_2),  # lags closed_at by hours/days
    ]
    report = backfill.backfill_realized(apply=False, wallet=WALLET,
                                        fetcher=_fetcher(activity))
    row = report["rows"][0]
    assert row["redeems_matched"] == 1
    assert row["redeemed_usd"] == pytest.approx(0.75)
    assert row["redeemed_shares"] == pytest.approx(0.75)
    assert row["complete"]
    assert row["unexplained_shares"] == pytest.approx(0.0, abs=1e-6)
    # realized_true = sells(16.52) + redeem(0.75) - buys(24.97) = -7.70
    assert row["realized_true"] == pytest.approx(-7.70, abs=0.01)


def test_incomplete_ledger_flagged_not_estimated(tmp_db):
    # Position-59 shape: the feed shows only 31.75 of 36.75 bought shares.
    # No more cost-estimate fallback — the row is simply flagged incomplete
    # with the signed share gap, and is never writable.
    pid = _open(tmp_db, "c59", side="YES", entry=0.68)
    _close_row(tmp_db, pid)
    activity = [
        _trade("c59", "BUY", 0, 21.59, 31.75),
        _trade("c59", "SELL", 0, 23.7624, 36.5, T_IN_1),
        _trade("c59", "SELL", 0, 0.1882, 0.25, T_IN_2),
    ]
    report = backfill.backfill_realized(apply=False, wallet=WALLET,
                                        fetcher=_fetcher(activity))
    row = report["rows"][0]
    assert not row["complete"]
    # bought 31.75 - sold 36.75 - redeemed 0 = -5.0 (missing a BUY)
    assert row["unexplained_shares"] == pytest.approx(-5.0)
    assert not row["writable"]
    # realized_true is still shown, raw, for reference (no fallback invented)
    assert row["realized_true"] == pytest.approx(23.9506 - 21.59, abs=0.01)


def test_delta_subtracts_already_recorded(tmp_db):
    # Position-54 case: -4.46 already realized by earlier partial closes; the
    # feed's full-history truth is -5.33 -> only the -0.87 gap is written.
    pid = _open(tmp_db, "c54", side="YES", entry=0.0565)
    _close_row(tmp_db, pid, realized=-4.46)
    activity = [
        _trade("c54", "BUY", 0, 11.93, 192.45),
        _trade("c54", "SELL", 0, 6.60, 192.5, T_IN_2),
    ]
    report = backfill.backfill_realized(apply=True, wallet=WALLET,
                                        fetcher=_fetcher(activity))
    row = report["rows"][0]
    assert row["complete"]
    assert row["realized_true"] == pytest.approx(-5.33, abs=0.01)
    assert row["delta"] == pytest.approx(-0.87, abs=0.01)
    assert _row(tmp_db, pid)["realized_pnl_usd"] == pytest.approx(-5.33, abs=0.01)


def test_apply_writes_delta_and_audit_row(tmp_db):
    pid = _open(tmp_db, "c1", side="NO", entry=0.64)
    _close_row(tmp_db, pid)
    activity = [
        _trade("c1", "BUY", 1, 34.56, 96.0),
        _trade("c1", "SELL", 1, 29.88, 96.0, T_IN_2),
    ]
    report = backfill.backfill_realized(apply=True, wallet=WALLET,
                                        fetcher=_fetcher(activity))
    assert report["applied"] == [pid]
    assert _row(tmp_db, pid)["realized_pnl_usd"] == pytest.approx(-4.68)
    audits = _audit_rows(tmp_db, pid)
    assert len(audits) == 1
    assert audits[0]["decision"] == "pnl_backfill"
    assert backfill.ACTIVITY_SOURCE in audits[0]["reasoning"]
    assert WALLET in audits[0]["reasoning"]
    assert "-4.68" in audits[0]["reasoning"]


def test_incomplete_never_applied_even_if_in_ids(tmp_db):
    pid = _open(tmp_db, "c59", side="YES", entry=0.68)
    _close_row(tmp_db, pid)
    activity = [
        _trade("c59", "BUY", 0, 21.59, 31.75),
        _trade("c59", "SELL", 0, 23.9506, 36.75, T_IN_1),
    ]
    report = backfill.backfill_realized(apply=True, ids=[pid], wallet=WALLET,
                                        fetcher=_fetcher(activity))
    assert report["applied"] == []
    assert (_row(tmp_db, pid)["realized_pnl_usd"] or 0.0) == pytest.approx(0.0)
    assert _audit_rows(tmp_db, pid) == []


def test_apply_is_idempotent(tmp_db):
    pid = _open(tmp_db, "c1", side="NO", entry=0.64)
    _close_row(tmp_db, pid)
    activity = [
        _trade("c1", "BUY", 1, 34.56, 96.0),
        _trade("c1", "SELL", 1, 29.88, 96.0, T_IN_2),
    ]
    backfill.backfill_realized(apply=True, wallet=WALLET,
                               fetcher=_fetcher(activity))
    report2 = backfill.backfill_realized(apply=True, wallet=WALLET,
                                         fetcher=_fetcher(activity))
    assert report2["applied"] == []
    assert _row(tmp_db, pid)["realized_pnl_usd"] == pytest.approx(-4.68)
    assert len(_audit_rows(tmp_db, pid)) == 1


def test_dry_run_writes_nothing(tmp_db):
    pid = _open(tmp_db, "c1", side="NO", entry=0.64)
    _close_row(tmp_db, pid)
    activity = [
        _trade("c1", "BUY", 1, 34.56, 96.0),
        _trade("c1", "SELL", 1, 29.88, 96.0, T_IN_2),
    ]
    report = backfill.backfill_realized(apply=False, wallet=WALLET,
                                        fetcher=_fetcher(activity))
    assert report["applied"] == []
    assert report["rows"][0]["writable"]  # pending, but not written
    assert (_row(tmp_db, pid)["realized_pnl_usd"] or 0.0) == pytest.approx(0.0)
    assert _audit_rows(tmp_db, pid) == []


def test_open_positions_never_touched_by_main_report(tmp_db):
    _open(tmp_db, "c_open", side="YES", entry=0.50)
    activity = [
        _trade("c_open", "SELL", 0, 10.0, 20.0),
    ]
    report = backfill.backfill_realized(apply=True, wallet=WALLET,
                                        fetcher=_fetcher(activity))
    assert report["rows"] == [] and report["applied"] == []
    pos = positions.get_open_positions()[0]
    assert pos["status"] == "open"
    assert (pos["realized_pnl_usd"] or 0.0) == pytest.approx(0.0)


def test_ids_filter_limits_writes_but_not_report(tmp_db):
    pid1 = _open(tmp_db, "c1", side="NO", entry=0.64)
    _close_row(tmp_db, pid1)
    pid2 = _open(tmp_db, "c2", side="NO", entry=0.50)
    _close_row(tmp_db, pid2)
    activity = [
        _trade("c1", "BUY", 1, 34.56, 96.0),
        _trade("c1", "SELL", 1, 29.88, 96.0, T_IN_2),
        _trade("c2", "BUY", 1, 10.0, 20.0),
        _trade("c2", "SELL", 1, 8.0, 20.0, T_IN_2),
    ]
    report = backfill.backfill_realized(apply=True, ids=[pid1], wallet=WALLET,
                                        fetcher=_fetcher(activity))
    assert report["applied"] == [pid1]
    assert report["skipped_ids"] == [pid2]
    assert len(report["rows"]) == 2  # both still reported
    assert _row(tmp_db, pid1)["realized_pnl_usd"] == pytest.approx(-4.68)
    assert (_row(tmp_db, pid2)["realized_pnl_usd"] or 0.0) == pytest.approx(0.0)


def test_tiny_delta_not_written(tmp_db):
    pid = _open(tmp_db, "c1", side="NO", entry=0.64)
    _close_row(tmp_db, pid, realized=-4.684)
    activity = [
        _trade("c1", "BUY", 1, 34.56, 96.0),
        _trade("c1", "SELL", 1, 29.88, 96.0, T_IN_2),
    ]
    report = backfill.backfill_realized(apply=True, wallet=WALLET,
                                        fetcher=_fetcher(activity))
    assert report["applied"] == []
    assert _row(tmp_db, pid)["realized_pnl_usd"] == pytest.approx(-4.684)


def test_no_wallet_raises(tmp_db, monkeypatch):
    from locus import config
    monkeypatch.setattr(config, "POLYMARKET_FUNDER_ADDRESS", "", raising=False)
    with pytest.raises(RuntimeError):
        backfill.backfill_realized(apply=False, fetcher=_fetcher([]))


def test_activity_type_counts():
    activity = [
        {"type": "TRADE"}, {"type": "TRADE"}, {"type": "REDEEM"},
    ]
    assert backfill.activity_type_counts(activity) == {"TRADE": 2, "REDEEM": 1}


# --- Sibling re-entries: a trade/redeem must attach to the right occupant ----

def test_sibling_reentry_trades_not_cross_attributed(tmp_db):
    # Same market+side opened twice (close, then re-enter): position 44/49's
    # shape. Each position's own BUY/SELL must not leak into the other's row.
    pid1 = _open(tmp_db, "c_dup", side="NO", entry=0.68)
    _close_row(tmp_db, pid1, realized=0.0,
              opened_at="2026-06-27 18:37:15",
              closed_at="2026-06-28 12:47:20")
    pid2 = _open(tmp_db, "c_dup", side="NO", entry=0.295)
    _close_row(tmp_db, pid2, realized=0.0,
              opened_at="2026-06-30 00:56:15",
              closed_at="2026-10-30 15:03:51")

    activity = [
        _trade("c_dup", "BUY", 1, 25.0, 36.8,
              ts=backfill._epoch("2026-06-27 18:40:00")),   # belongs to pid1
        _trade("c_dup", "SELL", 1, 25.0, 36.8,
              ts=backfill._epoch("2026-06-28 12:00:00")),   # belongs to pid1
        _trade("c_dup", "BUY", 1, 25.14, 35.0,
              ts=backfill._epoch("2026-06-30 01:00:00")),   # belongs to pid2
        _trade("c_dup", "SELL", 1, 18.90, 35.0,
              ts=backfill._epoch("2026-08-04 00:16:42")),   # belongs to pid2
    ]
    report = backfill.backfill_realized(apply=False, wallet=WALLET,
                                        fetcher=_fetcher(activity))
    rows_by_id = {r["position_id"]: r for r in report["rows"]}
    assert rows_by_id[pid1]["trades_matched"] == 2
    assert rows_by_id[pid1]["buys_usd"] == pytest.approx(25.0)
    assert rows_by_id[pid1]["sells_usd"] == pytest.approx(25.0)
    assert rows_by_id[pid2]["trades_matched"] == 2
    assert rows_by_id[pid2]["buys_usd"] == pytest.approx(25.14)
    assert rows_by_id[pid2]["sells_usd"] == pytest.approx(18.90)


def test_sibling_reentry_redeem_capped_at_next_open(tmp_db):
    # A REDEEM timestamped AFTER the next sibling opened must not attach to
    # the earlier, already-closed occupant of this market+side.
    pid1 = _open(tmp_db, "c_dup2", side="YES", entry=0.50)
    _close_row(tmp_db, pid1, realized=0.0,
              opened_at="2026-06-01 00:00:00",
              closed_at="2026-06-02 00:00:00")
    pid2 = _open(tmp_db, "c_dup2", side="YES", entry=0.50)
    _close_row(tmp_db, pid2, realized=0.0,
              opened_at="2026-06-03 00:00:00",
              closed_at="2026-06-04 00:00:00")

    # This redeem lands AFTER pid2's open -> must attribute to pid2, not pid1.
    late_redeem_ts = backfill._epoch("2026-06-05 00:00:00")
    activity = [_redeem("c_dup2", 0, 5.0, 5.0, late_redeem_ts)]

    report = backfill.backfill_realized(apply=False, wallet=WALLET,
                                        fetcher=_fetcher(activity))
    rows_by_id = {r["position_id"]: r for r in report["rows"]}
    assert pid1 not in rows_by_id
    assert rows_by_id[pid2]["redeemed_usd"] == pytest.approx(5.0)


# --- preview_reconcile_with_ledger: informational only, never writes --------

def test_preview_open_position_shows_hypothetical_total(tmp_db):
    pid = _open(tmp_db, "c57", side="YES", entry=0.266)
    conn = tmp_db._conn()
    conn.execute(
        "UPDATE positions SET realized_pnl_usd=-0.148, opened_at=? WHERE id=?",
        ("2026-07-09 16:55:31", pid),
    )
    conn.commit()
    conn.close()
    activity = [
        _trade("c57", "SELL", 0, 38.0669, 148.0, T_IN_1),
        _trade("c57", "SELL", 0, 0.5119, 1.99, T_IN_1),
    ]
    preview = backfill.preview_reconcile_with_ledger(
        pid, wallet=WALLET, fetcher=_fetcher(activity))

    assert preview["position_id"] == pid
    assert preview["writable"] is False  # preview never writes, regardless
    # No BUY in the feed at all -> wildly incomplete, flagged loudly.
    assert not preview["complete"]
    assert preview["realized_true"] == pytest.approx(38.5788, abs=0.01)
    assert preview["preview_total_realized"] == pytest.approx(
        preview["already_realized"] + preview["delta"])
    # Nothing was written — still open, unchanged realized.
    assert positions.get_open_positions()[0]["status"] == "open"
    assert positions.get_open_positions()[0]["realized_pnl_usd"] == pytest.approx(-0.148)


def test_preview_missing_position_returns_none(tmp_db):
    assert backfill.preview_reconcile_with_ledger(
        99999, wallet=WALLET, fetcher=_fetcher([])) is None
