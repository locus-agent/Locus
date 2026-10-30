"""Exit ladder: FAK sells walked down the book under a slippage cap, retries,
partial-fill continuation, stop-loss ignores the spread guard."""
import types

import pytest

from locus import config
from locus.core import executor
from test_live_close import _install_fake_clob, _level


@pytest.fixture(autouse=True)
def ladder(monkeypatch):
    monkeypatch.setattr(config, "EXIT_LADDER_STEPS", 3)
    monkeypatch.setattr(config, "EXIT_MAX_SLIPPAGE_PCT", 9.0)
    monkeypatch.setattr(config, "STOP_LOSS_MAX_SLIPPAGE_PCT", 30.0)
    monkeypatch.setattr(config, "EXIT_RETRY_DELAY_SECONDS", 0.0)


def _scripted(monkeypatch, results):
    calls = []

    def fake(cid, side, shares, max_spread=None, allow_topup=False, max_slippage_pct=None):
        calls.append({"shares": shares, "slip": max_slippage_pct,
                      "max_spread": max_spread, "topup": allow_topup})
        return results.pop(0)

    monkeypatch.setattr(executor, "close_position_live", fake)
    return calls


def test_ladder_widens_slippage_until_filled(monkeypatch):
    calls = _scripted(monkeypatch, [
        {"status": "close_failed", "order_id": None, "price": None, "shares": None},
        {"status": "executed", "order_id": "B", "price": 0.55, "shares": 40.0,
         "sold_shares": 40.0, "remaining_shares": 0.0},
    ])
    res = executor.close_position_ladder("c", "YES", 40.0, allow_topup=True)
    assert [c["slip"] for c in calls] == [pytest.approx(3.0), pytest.approx(6.0)]
    assert calls[0]["topup"] is True and calls[1]["topup"] is False
    assert res["status"] == "executed" and res["sold_shares"] == 40.0
    assert res["price"] == pytest.approx(0.55)


def test_ladder_continues_partial_with_remainder(monkeypatch):
    calls = _scripted(monkeypatch, [
        {"status": "partial_close", "order_id": "A", "price": 0.60, "shares": 10.0,
         "sold_shares": 10.0, "remaining_shares": 30.0},
        {"status": "executed", "order_id": "B", "price": 0.50, "shares": 30.0,
         "sold_shares": 30.0, "remaining_shares": 0.0},
    ])
    res = executor.close_position_ladder("c", "YES", 40.0)
    assert calls[1]["shares"] == 30.0
    assert res["status"] == "executed"
    assert res["sold_shares"] == 40.0
    assert res["price"] == pytest.approx((10 * 0.60 + 30 * 0.50) / 40)


def test_ladder_reports_partial_when_book_exhausted(monkeypatch):
    _scripted(monkeypatch, [
        {"status": "partial_close", "order_id": "A", "price": 0.6, "shares": 10.0,
         "sold_shares": 10.0, "remaining_shares": 30.0},
        {"status": "skipped_thin_book", "order_id": None, "price": None, "shares": None},
        {"status": "skipped_thin_book", "order_id": None, "price": None, "shares": None},
    ])
    res = executor.close_position_ladder("c", "YES", 40.0)
    assert res["status"] == "partial_close"
    assert res["remaining_shares"] == 30.0 and res["sold_shares"] == 10.0


def test_stop_loss_uses_wider_cap_and_ignores_spread(monkeypatch):
    calls = _scripted(monkeypatch, [{"status": "close_failed"}] * 3)
    res = executor.close_position_ladder("c", "YES", 40.0, stop_loss=True)
    assert [c["slip"] for c in calls] == [pytest.approx(10), pytest.approx(20), pytest.approx(30)]
    assert all(c["max_spread"] == 1.0 for c in calls)
    assert res["status"] == "close_failed"


def test_error_stops_ladder(monkeypatch):
    calls = _scripted(monkeypatch, [{"status": "error_no_clob_client"}])
    assert executor.close_position_ladder("c", "YES", 5.0)["status"] == "error_no_clob_client"
    assert len(calls) == 1


def test_fak_order_priced_at_slippage_floor(monkeypatch):
    # No get_trades on this fake client -> _order_fill_price finds nothing,
    # so the result falls back to the book-walk estimate AND must say so.
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    book = types.SimpleNamespace(
        bids=[_level(0.60, 10), _level(0.57, 20), _level(0.40, 1000)],
        asks=[_level(0.62, 1000)])
    captured = {}
    _install_fake_clob(monkeypatch, captured, book)
    res = executor.close_position_live("cond1", "YES", 40.0, max_spread=0.05,
                                       max_slippage_pct=10.0)
    oa = captured["order_args"]
    assert captured["posted"][1] == "FAK"           # never rests on the book
    assert oa.price == pytest.approx(0.54)          # ceil(0.60 * 0.90) to tick
    assert oa.size == pytest.approx(30.0)           # depth inside the band only
    assert res["status"] == "partial_close"         # 30 of 40 sold
    # realized price estimated by walking the book, not the floor
    assert res["price"] == pytest.approx((10 * 0.60 + 20 * 0.57) / 30, abs=1e-4)
    assert res["price_estimated"] is True           # honestly flagged as an estimate


def test_fak_order_prefers_real_fill_price_over_book_walk(monkeypatch):
    # When the exchange's trade history has a real fill for this order, it
    # must win over the book-snapshot estimate (and NOT be flagged estimated)
    # — the whole point of task 3's "honest accounting" fix.
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    book = types.SimpleNamespace(
        bids=[_level(0.60, 10), _level(0.57, 20), _level(0.40, 1000)],
        asks=[_level(0.62, 1000)])
    captured = {}
    trades = [{"taker_order_id": "LIVE-123", "price": "0.585", "size": "30"}]
    _install_fake_clob(monkeypatch, captured, book, trades_result=trades)
    res = executor.close_position_live("cond1", "YES", 40.0, max_spread=0.05,
                                       max_slippage_pct=10.0)
    assert res["status"] == "partial_close"
    assert res["price"] == pytest.approx(0.585)     # real VWAP, not the book-walk estimate
    assert res["price_estimated"] is False


def test_ordinary_gtc_close_has_no_price_estimated_key(monkeypatch):
    # A plain (non-ladder) GTC close never set this key before task 3 and
    # must not start now — only floor_price (FAK/ladder) closes carry it.
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    book = types.SimpleNamespace(bids=[_level(0.60, 1000)], asks=[_level(0.62, 1000)])
    captured = {}
    _install_fake_clob(monkeypatch, captured, book)
    res = executor.close_position_live("cond1", "YES", 40.0, max_spread=0.05)
    assert res["status"] == "executed"
    assert "price_estimated" not in res


# --- FAK + balance confirmation: no hanging orders, works like GTC --------

def test_fak_unconfirmed_sell_confirmed_by_balance_no_hanging_order(monkeypatch):
    # A FAK order the ledger reports as an unknown/terminal non-fill status
    # (reconcile_order's conservative default) must still be recognized as
    # filled via the on-chain balance, exactly like a GTC close — and the
    # (already-dead) order is still best-effort cancelled, never left
    # resting untracked.
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    monkeypatch.setattr(config, "BALANCE_CONFIRM_WAIT_SECONDS", 0)
    book = types.SimpleNamespace(
        bids=[_level(0.60, 10), _level(0.57, 20), _level(0.40, 1000)],
        asks=[_level(0.62, 1000)])
    captured = {}
    _install_fake_clob(monkeypatch, captured, book,
                       fill_result={"status": "CANCELLED", "size_matched": "0"},
                       held_balance=40.0)
    held_values = iter([40.0, 0.0])  # initial cap read, then the confirm probe
    monkeypatch.setattr(executor, "held_token_shares", lambda client, token_id: next(held_values))

    res = executor.close_position_live("cond1", "YES", 40.0, max_spread=0.05,
                                       max_slippage_pct=10.0)

    assert res["status"] == "executed"
    assert res["confirmed_by"] == "balance"
    assert res["price_estimated"] is True  # no get_trades on this fake client
    # the dead FAK order was cancelled, never left resting untracked
    assert captured.get("cancelled")


def test_ladder_stops_on_balance_error_without_retrying(monkeypatch):
    # A close_failed carrying a "not enough balance/allowance" rejection is
    # not a book problem — retrying with a wider slippage band on the next
    # step fixes nothing. The ladder must stop after step 1, not burn all
    # EXIT_LADDER_STEPS on a doomed retry.
    calls = _scripted(monkeypatch, [
        {"status": "close_failed", "order_id": None, "price": None,
         "shares": None, "error": "not enough balance / allowance"},
    ])
    res = executor.close_position_ladder("c", "YES", 40.0)
    assert len(calls) == 1          # no second/third step attempted
    assert res["status"] == "close_failed"


def test_ladder_retries_plain_close_failed_not_a_balance_error(monkeypatch):
    # A rejection with no balance/allowance signature is still transient —
    # the ladder keeps retrying it exactly as before this fix.
    calls = _scripted(monkeypatch, [
        {"status": "close_failed", "order_id": None, "price": None,
         "shares": None, "error": "order rejected: bad signature"},
        {"status": "executed", "order_id": "B", "price": 0.55, "shares": 40.0,
         "sold_shares": 40.0, "remaining_shares": 0.0},
    ])
    res = executor.close_position_ladder("c", "YES", 40.0)
    assert len(calls) == 2          # retried into step 2
    assert res["status"] == "executed"


# --- price_estimated must survive the multi-step result rebuild -------------

def test_ladder_price_estimated_true_when_any_selling_step_estimated(monkeypatch):
    # Step 1 sells part of the position with a REAL fill price (no flag);
    # step 2 sells the rest but only via a book-walk estimate (flag True).
    # The aggregated multi-step result must still say price_estimated=True —
    # before this fix the rebuilt dict silently dropped it.
    _scripted(monkeypatch, [
        {"status": "partial_close", "order_id": "A", "price": 0.60, "shares": 10.0,
         "sold_shares": 10.0, "remaining_shares": 30.0},
        {"status": "executed", "order_id": "B", "price": 0.50, "shares": 30.0,
         "sold_shares": 30.0, "remaining_shares": 0.0, "price_estimated": True},
    ])
    res = executor.close_position_ladder("c", "YES", 40.0)
    assert res["status"] == "executed"
    assert res["price_estimated"] is True


def test_ladder_price_estimated_true_when_first_selling_step_estimated(monkeypatch):
    # Order shouldn't matter: an estimate on an EARLIER step must still
    # propagate even if the later (final) step had a real fill price.
    _scripted(monkeypatch, [
        {"status": "partial_close", "order_id": "A", "price": 0.60, "shares": 10.0,
         "sold_shares": 10.0, "remaining_shares": 30.0, "price_estimated": True},
        {"status": "executed", "order_id": "B", "price": 0.50, "shares": 30.0,
         "sold_shares": 30.0, "remaining_shares": 0.0},
    ])
    res = executor.close_position_ladder("c", "YES", 40.0)
    assert res["status"] == "executed"
    assert res["price_estimated"] is True


def test_ladder_no_price_estimated_key_when_all_steps_real(monkeypatch):
    # No step was ever estimated -> the aggregated result must NOT carry the
    # key at all, matching the single-step (EXIT_LADDER_STEPS=1) behavior.
    _scripted(monkeypatch, [
        {"status": "partial_close", "order_id": "A", "price": 0.60, "shares": 10.0,
         "sold_shares": 10.0, "remaining_shares": 30.0},
        {"status": "executed", "order_id": "B", "price": 0.50, "shares": 30.0,
         "sold_shares": 30.0, "remaining_shares": 0.0},
    ])
    res = executor.close_position_ladder("c", "YES", 40.0)
    assert res["status"] == "executed"
    assert "price_estimated" not in res


def test_ladder_price_estimated_false_flag_does_not_count_as_estimated(monkeypatch):
    # A step that explicitly reports price_estimated=False (a real fill, same
    # as omitting the key) must not flip the aggregate to True.
    _scripted(monkeypatch, [
        {"status": "partial_close", "order_id": "A", "price": 0.60, "shares": 10.0,
         "sold_shares": 10.0, "remaining_shares": 30.0, "price_estimated": False},
        {"status": "executed", "order_id": "B", "price": 0.50, "shares": 30.0,
         "sold_shares": 30.0, "remaining_shares": 0.0, "price_estimated": False},
    ])
    res = executor.close_position_ladder("c", "YES", 40.0)
    assert "price_estimated" not in res


def test_legacy_close_still_gtc(monkeypatch):
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    book = types.SimpleNamespace(bids=[_level(0.60, 1000)], asks=[_level(0.62, 1000)])
    captured = {}
    _install_fake_clob(monkeypatch, captured, book)
    executor.close_position_live("cond1", "YES", 40.0, max_spread=0.05)
    assert captured["posted"][1] == "GTC"


# --- Dispatch scope: positions._live_close uses the ladder ONLY for sl/time_pressure -

def _open_position(tmp_db, token_count=40.0):
    from locus.core import positions
    from locus.markets.gamma import Market

    MKT = Market("cond1", "Will X happen?", "ai", 0.50, 0.50, 5000, "", True, [])
    trade_id = tmp_db.log_trade(
        market_id="cond1", market_question="Will X happen?", claude_score=0.7,
        market_price=0.50, edge=0.2, side="YES", amount_usd=25.0,
        status="executed", classification="bullish", materiality=0.7,
    )
    positions.open_position(trade_id, MKT, "YES", 25.0, token_count=token_count)
    return positions.get_open_positions()[0]


def _close_via(monkeypatch, tmp_db, exit_reason, status="closed_x", fraction=1.0,
              force=False):
    """Drive positions._close for `exit_reason`, with both close_position_ladder
    and close_position_live stubbed so the dispatch (and ONLY the dispatch)
    can be observed."""
    from locus import config
    from locus.core import executor, positions

    monkeypatch.setattr(config, "DRY_RUN", False)
    ladder_calls = []
    live_calls = []

    def fake_ladder(*a, **k):
        ladder_calls.append((a, k))
        return {"status": "executed", "order_id": "L", "price": 0.5,
                "shares": 40.0, "sold_shares": 40.0, "remaining_shares": 0.0}

    def fake_live(*a, **k):
        live_calls.append((a, k))
        return {"status": "executed", "order_id": "G", "price": 0.5,
                "shares": 40.0, "sold_shares": 40.0, "remaining_shares": 0.0}

    monkeypatch.setattr(executor, "close_position_ladder", fake_ladder)
    monkeypatch.setattr(executor, "close_position_live", fake_live)

    pos = _open_position(tmp_db)
    conn = tmp_db._conn()
    positions._close(conn, pos, 0.50, status, exit_reason, fraction=fraction,
                     force=force)
    conn.commit()
    conn.close()
    return ladder_calls, live_calls


def test_stop_loss_uses_ladder(tmp_db, monkeypatch):
    ladder_calls, live_calls = _close_via(monkeypatch, tmp_db, "sl")
    assert len(ladder_calls) == 1 and len(live_calls) == 0


def test_time_pressure_uses_ladder(tmp_db, monkeypatch):
    ladder_calls, live_calls = _close_via(monkeypatch, tmp_db, "time_pressure")
    assert len(ladder_calls) == 1 and len(live_calls) == 0


@pytest.mark.parametrize("exit_reason", [
    "manual", "tp_decision", "news_decision", "drawdown_decision",
    "near_certain_yes_0.97", "near_certain_no_0.02",
])
def test_other_exit_reasons_use_plain_gtc(tmp_db, monkeypatch, exit_reason):
    # Take-profit, manual Close/Half/Force, news re-eval, drawdown re-eval,
    # near-certain locks — everything except sl/time_pressure keeps the
    # original GTC-at-best-bid path, unchanged from before the ladder existed.
    ladder_calls, live_calls = _close_via(monkeypatch, tmp_db, exit_reason)
    assert len(live_calls) == 1 and len(ladder_calls) == 0


def test_manual_force_close_uses_plain_gtc_not_ladder(tmp_db, monkeypatch):
    # Force close (operator two-step confirm) is always exit_reason="manual"
    # in practice — confirm force=True doesn't accidentally route to the
    # ladder, and max_spread=inf is still passed through to close_position_live.
    ladder_calls, live_calls = _close_via(monkeypatch, tmp_db, "manual", force=True)
    assert len(live_calls) == 1 and len(ladder_calls) == 0
    assert live_calls[0][1]["max_spread"] == float("inf")


def test_half_close_via_sl_trigger_still_uses_ladder(tmp_db, monkeypatch):
    # A fractional close (close_half-style) under exit_reason="sl" still
    # dispatches to the ladder — the scoping is purely by exit_reason, not
    # by fraction.
    ladder_calls, live_calls = _close_via(monkeypatch, tmp_db, "sl", fraction=0.5)
    assert len(ladder_calls) == 1 and len(live_calls) == 0


# --- End-to-end: positions._close reads and persists the aggregated flag ----

def test_close_persists_aggregated_price_estimated_from_real_ladder(tmp_db, monkeypatch):
    # Drive the REAL close_position_ladder (not stubbed) through
    # positions._close for exit_reason="sl". The book is exhausted after a
    # partial, estimated fill, so the ladder's aggregated result is itself
    # "partial_close" — the one outcome branch in _close that writes an
    # exit_decisions row with the "[price estimated ...]" annotation. Proves
    # the aggregated flag survives close_position_ladder ->
    # positions._live_close -> _close's DB write end to end.
    from locus import config
    from locus.core import executor, positions

    monkeypatch.setattr(config, "DRY_RUN", False)
    monkeypatch.setattr(config, "EXIT_LADDER_STEPS", 3)
    monkeypatch.setattr(config, "EXIT_RETRY_DELAY_SECONDS", 0.0)
    _scripted(monkeypatch, [
        {"status": "partial_close", "order_id": "A", "price": 0.60, "shares": 10.0,
         "sold_shares": 10.0, "remaining_shares": 30.0, "price_estimated": True},
        {"status": "skipped_thin_book", "order_id": None, "price": None, "shares": None},
        {"status": "skipped_thin_book", "order_id": None, "price": None, "shares": None},
    ])

    pos = _open_position(tmp_db)
    conn = tmp_db._conn()
    positions._close(conn, pos, 0.50, "closed_sl", "sl")
    conn.commit()

    row = conn.execute(
        "SELECT reasoning FROM exit_decisions WHERE position_id=? "
        "ORDER BY id DESC LIMIT 1", (pos["id"],),
    ).fetchone()
    status_row = conn.execute(
        "SELECT status, token_count FROM positions WHERE id=?", (pos["id"],),
    ).fetchone()
    conn.close()

    assert status_row["status"] == "open"          # only partially flattened
    assert status_row["token_count"] == pytest.approx(30.0)
    assert "price estimated" in row["reasoning"]


def test_close_no_estimated_annotation_when_all_ladder_steps_real(tmp_db, monkeypatch):
    from locus import config
    from locus.core import executor, positions

    monkeypatch.setattr(config, "DRY_RUN", False)
    monkeypatch.setattr(config, "EXIT_LADDER_STEPS", 3)
    monkeypatch.setattr(config, "EXIT_RETRY_DELAY_SECONDS", 0.0)
    _scripted(monkeypatch, [
        {"status": "partial_close", "order_id": "A", "price": 0.60, "shares": 10.0,
         "sold_shares": 10.0, "remaining_shares": 30.0},
        {"status": "skipped_thin_book", "order_id": None, "price": None, "shares": None},
        {"status": "skipped_thin_book", "order_id": None, "price": None, "shares": None},
    ])

    pos = _open_position(tmp_db)
    conn = tmp_db._conn()
    positions._close(conn, pos, 0.50, "closed_sl", "sl")
    conn.commit()

    row = conn.execute(
        "SELECT reasoning FROM exit_decisions WHERE position_id=? "
        "ORDER BY id DESC LIMIT 1", (pos["id"],),
    ).fetchone()
    conn.close()
    assert "price estimated" not in row["reasoning"]
