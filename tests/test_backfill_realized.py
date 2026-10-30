"""backfill-realized: recompute closed positions' realized PnL from the Data
API activity feed. Matching by conditionId+outcomeIndex within the life
window, the missing-BUY fallback (feed gaps costed at entry), delta vs the
already-recorded realized, --apply writing delta + audit row (idempotent),
dry-run writing nothing, and open positions never touched."""
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
               closed_at: str = "2026-09-01 10:00:00") -> None:
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
        _trade("c1", "SELL", 1, 10.0, 10.0, T_IN_1, type_="REDEEM"),  # not TRADE
    ]
    report = backfill.backfill_realized(apply=False, wallet=WALLET,
                                        fetcher=_fetcher(activity))
    assert len(report["rows"]) == 1
    row = report["rows"][0]
    assert row["trades_matched"] == 2
    assert row["buys_usd"] == pytest.approx(34.56)
    assert row["sells_usd"] == pytest.approx(29.88)
    assert not row["incomplete_buy"]
    assert row["realized_true"] == pytest.approx(-4.68)
    assert row["delta"] == pytest.approx(-4.68)


def test_missing_buy_fallback_costed_at_entry(tmp_db):
    # The feed shows only 31.75 of 36.75 bought shares (position-59 case):
    # the 5 missing shares are costed at the YES entry price 0.68.
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
    assert row["incomplete_buy"]
    assert row["missing_buy_shares"] == pytest.approx(5.0)
    assert row["est_missing_cost_usd"] == pytest.approx(3.40)
    # 23.9506 - 21.59 - 3.40 = -1.0394
    assert row["realized_true"] == pytest.approx(-1.0394, abs=1e-4)


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
    # Second run: realized already matches the truth -> delta ~0, nothing written.
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


def test_open_positions_never_touched(tmp_db):
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
