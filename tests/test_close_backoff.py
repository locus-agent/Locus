"""Close-retry backoff: a close that failed on a non-transient book status
(skipped_thin_book / _empty_book / _wide_spread) suppresses further close
ATTEMPTS on that position for CLOSE_RETRY_BACKOFF_SECONDS, instead of
re-placing a doomed order every ~30s management cycle (position 54 logged
1,325 close_failed decisions in ~11.5h on a zombie book). The position stays
marked/monitored; a manual close always bypasses; transient failures
(close_failed) and error_* statuses keep the retry-every-cycle behavior."""
import pytest

from locus import config
from locus.core import executor, positions
from locus.markets.gamma import Market


MKT = Market("cond1", "Will X happen?", "ai", 0.50, 0.50, 5000, "", True, [])


def _open(tmp_db, token_count=40.0):
    trade_id = tmp_db.log_trade(
        market_id="cond1", market_question="Will X happen?", claude_score=0.7,
        market_price=0.50, edge=0.2, side="YES", amount_usd=25.0,
        status="executed", classification="bullish", materiality=0.7,
    )
    positions.open_position(trade_id, MKT, "YES", 25.0, token_count=token_count)
    return positions.get_open_positions()[0]


def _stub_close(monkeypatch, status):
    """close_position_live stub returning a fixed status, counting calls."""
    calls = []

    def fake_close(condition_id, side, shares, max_spread=None, allow_topup=False):
        calls.append(status)
        result = {"status": status, "order_id": None, "price": None, "shares": None}
        if status == "executed":
            result.update(order_id="OK-1", price=0.5, shares=shares,
                          sold_shares=shares, remaining_shares=0.0)
        return result

    monkeypatch.setattr(executor, "close_position_live", fake_close)
    return calls


def _close(tmp_db, pos, reason="time_pressure", status="closed_time",
           bypass_backoff=False):
    conn = tmp_db._conn()
    realized = positions._close(conn, pos, 0.50, status, reason,
                                bypass_backoff=bypass_backoff)
    conn.commit()
    conn.close()
    return realized


def _decisions(tmp_db, position_id):
    conn = tmp_db._conn()
    rows = [r["decision"] for r in conn.execute(
        "SELECT decision FROM exit_decisions WHERE position_id=?", (position_id,)
    ).fetchall()]
    conn.close()
    return rows


def test_thin_book_failure_arms_backoff_and_suppresses_next_attempt(
        tmp_db, monkeypatch, caplog):
    monkeypatch.setattr(config, "DRY_RUN", False)
    calls = _stub_close(monkeypatch, "skipped_thin_book")
    pos = _open(tmp_db)

    _close(tmp_db, pos)                       # attempt 1: fails, arms backoff
    assert calls == ["skipped_thin_book"]

    with caplog.at_level("WARNING", logger="locus.core.positions"):
        realized = _close(tmp_db, pos)        # attempt 2: suppressed
    assert realized == 0.0
    assert calls == ["skipped_thin_book"]     # no second order attempt
    assert "close suppressed by backoff" in caplog.text
    assert "retry in" in caplog.text
    # only the first attempt wrote a close_failed decision — no row pollution
    assert _decisions(tmp_db, pos["id"]) == ["close_failed"]


@pytest.mark.parametrize("status", ["skipped_empty_book", "skipped_wide_spread"])
def test_other_non_transient_statuses_also_arm_backoff(tmp_db, monkeypatch, status):
    monkeypatch.setattr(config, "DRY_RUN", False)
    calls = _stub_close(monkeypatch, status)
    pos = _open(tmp_db)
    _close(tmp_db, pos)
    _close(tmp_db, pos)
    assert calls == [status]  # second attempt suppressed


def test_backoff_expires_and_close_retries(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", False)
    calls = _stub_close(monkeypatch, "skipped_thin_book")
    pos = _open(tmp_db)

    _close(tmp_db, pos)                       # arms backoff
    # age the entry past the window (in-memory monotonic timestamp)
    positions._close_backoff[pos["id"]] -= config.CLOSE_RETRY_BACKOFF_SECONDS + 1

    _close(tmp_db, pos)                       # backoff expired -> real attempt
    assert calls == ["skipped_thin_book", "skipped_thin_book"]
    # the second failure re-armed the backoff
    assert pos["id"] in positions._close_backoff


def test_manual_close_bypasses_backoff(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", False)
    calls = _stub_close(monkeypatch, "skipped_thin_book")
    pos = _open(tmp_db)
    _close(tmp_db, pos)                       # arms backoff
    assert calls == ["skipped_thin_book"]

    # user-requested close goes straight to the exchange, mid-backoff
    _stub_close(monkeypatch, "executed")
    res = positions.close_manual(pos["id"])
    assert res is not None
    assert res["close_failed"] is False

    conn = tmp_db._conn()
    status = conn.execute(
        "SELECT status FROM positions WHERE id=?", (pos["id"],)
    ).fetchone()["status"]
    conn.close()
    assert status == "closed_manual"
    # the successful sell also cleared the backoff entry
    assert pos["id"] not in positions._close_backoff


@pytest.mark.parametrize("status", ["close_failed", "error_no_token"])
def test_transient_and_error_statuses_keep_retrying(tmp_db, monkeypatch, status):
    # A rejected/unconfirmed order or a genuine error is NOT a book verdict —
    # the next cycle retries exactly as before.
    monkeypatch.setattr(config, "DRY_RUN", False)
    calls = _stub_close(monkeypatch, status)
    pos = _open(tmp_db)
    _close(tmp_db, pos)
    _close(tmp_db, pos)
    assert calls == [status, status]          # both attempts hit the exchange
    assert pos["id"] not in positions._close_backoff


def test_success_clears_backoff(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", False)
    _stub_close(monkeypatch, "skipped_thin_book")
    pos = _open(tmp_db)
    _close(tmp_db, pos)
    assert pos["id"] in positions._close_backoff

    positions._close_backoff[pos["id"]] -= config.CLOSE_RETRY_BACKOFF_SECONDS + 1
    _stub_close(monkeypatch, "executed")
    _close(tmp_db, pos)
    assert pos["id"] not in positions._close_backoff


def test_dry_run_closes_never_suppressed(tmp_db, monkeypatch):
    # Backoff only guards live CLOB attempts; a dry-run close simulates the
    # fill locally and must always go through — even with a stale entry armed.
    pos = _open(tmp_db)
    positions._close_backoff[pos["id"]] = 10**12  # armed "now" (monotonic-ish)
    realized = _close(tmp_db, pos)  # DRY_RUN=True via conftest
    conn = tmp_db._conn()
    status = conn.execute(
        "SELECT status FROM positions WHERE id=?", (pos["id"],)
    ).fetchone()["status"]
    conn.close()
    assert status == "closed_time"
    del realized
