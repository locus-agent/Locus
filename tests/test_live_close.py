"""Live position close: _close places a real CLOB SELL when DRY_RUN=false and
records the Polymarket order_id; dry-run (and resolution) simulate the fill
without hitting the exchange. Plus the sell planner and bid-side book helper."""
import sys
import types

import pytest

from locus import config
from locus.core import executor, positions
from locus.markets.gamma import Market


MKT = Market("cond1", "Will X happen?", "ai", 0.50, 0.50, 5000, "", True, [],
             slug="will-x-happen")


def _level(p, s):
    return types.SimpleNamespace(price=str(p), size=str(s))


def _open(tmp_db, side="YES", entry=0.50, amount=25.0):
    trade_id = tmp_db.log_trade(
        market_id="cond1", market_question="Will X happen?", claude_score=0.7,
        market_price=entry, edge=0.2, side=side, amount_usd=amount,
        status="executed", classification="bullish", materiality=0.7,
    )
    positions.open_position(trade_id, MKT, side, amount)
    return positions.get_open_positions()[0]


# --- plan_live_sell: the mirror of plan_live_order, hitting the best bid -------

def test_sell_priced_at_best_bid_sized_in_shares():
    price, shares, status = executor.plan_live_sell(
        shares=50.0, best_bid=0.50, best_ask=0.52, bid_size_shares=1000.0,
        max_spread=0.05,
    )
    assert status == "ok"
    assert price == 0.50
    assert shares == 50.0


def test_sell_wide_spread_skipped():
    _, _, status = executor.plan_live_sell(50.0, 0.40, 0.50, 1000.0, max_spread=0.05)
    assert status == "skipped_wide_spread"


def test_sell_thin_book_skipped():
    # only 1 share of demand at the bid -> $0.50 notional, below the $1 floor
    _, _, status = executor.plan_live_sell(50.0, 0.50, 0.51, 1.0, max_spread=0.05)
    assert status == "skipped_thin_book"


def test_sell_partial_depth_downsizes():
    price, shares, status = executor.plan_live_sell(50.0, 0.50, 0.51, 20.0, max_spread=0.05)
    assert status == "ok"
    assert shares == 20.0  # wanted to sell 50, book bids for 20


def test_sell_empty_book_skipped():
    _, _, status = executor.plan_live_sell(50.0, None, None, None, max_spread=0.05)
    assert status == "skipped_empty_book"


def test_bid_levels_extraction():
    book = types.SimpleNamespace(
        bids=[_level(0.48, 100), _level(0.50, 30), _level(0.45, 50)],
        asks=[_level(0.55, 100), _level(0.52, 10), _level(0.60, 5)],
    )
    best_bid, best_ask, bid_size = executor._bid_levels(book)
    assert best_bid == 0.50
    assert best_ask == 0.52
    assert bid_size == 30.0  # depth at the best BID, not the best ask


def test_bid_levels_empty_book():
    assert executor._bid_levels(types.SimpleNamespace(bids=[], asks=[])) == (None, None, None)


# --- close_position_live: full path against a faked py_clob_client_v2 ---------

def _install_fake_v2_module(monkeypatch, clob_client_cls):
    """Inject a flat `py_clob_client_v2` module exposing every name the executor
    imports (ClobClient + OrderArgs/OrderType/Side/PartialCreateOrderOptions +
    BalanceAllowanceParams/AssetType), wired to the given fake client class."""
    class OrderArgs:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class PartialCreateOrderOptions:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    mod = types.ModuleType("py_clob_client_v2")
    mod.ClobClient = clob_client_cls
    mod.OrderArgs = OrderArgs
    mod.PartialCreateOrderOptions = PartialCreateOrderOptions
    mod.OrderType = types.SimpleNamespace(GTC="GTC", FOK="FOK", FAK="FAK", GTD="GTD")
    # v2 sides are an enum; the executor only ever uses the members, and the
    # tests assert against the string values, so SimpleNamespace is enough.
    mod.Side = types.SimpleNamespace(BUY="BUY", SELL="SELL")
    mod.BalanceAllowanceParams = lambda **kw: ("params", kw)
    mod.AssetType = types.SimpleNamespace(COLLATERAL="COLLATERAL", CONDITIONAL="CONDITIONAL")
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", mod)
    return mod


def _install_fake_clob(monkeypatch, captured, book):
    class FakeClobClient:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def create_or_derive_api_key(self):
            return "creds"

        def set_api_creds(self, creds):
            pass

        def get_market(self, condition_id):
            captured["get_market"] = condition_id
            return {"tokens": [
                {"token_id": "tok-yes", "outcome": "YES"},
                {"token_id": "tok-no", "outcome": "NO"},
            ]}

        def get_order_book(self, token_id):
            captured["book_token"] = token_id
            return book

        def get_tick_size(self, token_id):
            return "0.01"

        def get_neg_risk(self, token_id):
            return False

        def create_order(self, order_args, options=None):
            captured["order_args"] = order_args
            captured["options"] = options
            return "signed-order"

        def post_order(self, signed, order_type):
            captured["posted"] = (signed, order_type)
            return {"orderID": "LIVE-123"}

    _install_fake_v2_module(monkeypatch, FakeClobClient)


def test_close_position_live_places_sell_order(monkeypatch):
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    monkeypatch.setattr(config, "POLYMARKET_FUNDER_ADDRESS", "0xfunder")
    monkeypatch.setattr(config, "POLYMARKET_SIGNATURE_TYPE", 3)

    book = types.SimpleNamespace(bids=[_level(0.60, 1000)], asks=[_level(0.62, 1000)])
    captured = {}
    _install_fake_clob(monkeypatch, captured, book)

    result = executor.close_position_live("cond1", "YES", 40.0, max_spread=0.05)

    assert result["status"] == "executed"
    assert result["order_id"] == "LIVE-123"
    assert result["price"] == 0.60
    assert result["shares"] == 40.0
    # token resolved for the held side, book read for that token
    assert captured["get_market"] == "cond1"
    assert captured["book_token"] == "tok-yes"
    # the order itself is a SELL of that token at the best bid
    oa = captured["order_args"]
    assert oa.side == "SELL"
    assert oa.token_id == "tok-yes"
    assert oa.price == 0.60
    assert oa.size == 40.0
    # create_order options carry the CLOB-fetched tick size and neg_risk flag
    opts = captured["options"]
    assert opts.tick_size == "0.01"
    assert opts.neg_risk is False
    # built on the deposit wallet (POLY_1271)
    assert captured["init"]["funder"] == "0xfunder"
    assert captured["init"]["signature_type"] == 3


def _install_fake_balance_types(monkeypatch):
    """Inject py_clob_client_v2 with the balance params get_live_balance imports,
    so the success/error paths can run without the real dependency."""
    _install_fake_v2_module(monkeypatch, clob_client_cls=object)


def test_get_live_balance_divides_by_usdc_decimals(monkeypatch):
    _install_fake_balance_types(monkeypatch)

    class FakeClient:
        def get_balance_allowance(self, params):
            return {"balance": "245000000"}  # 245 USDC in 6-decimal base units

    monkeypatch.setattr(executor, "create_clob_client", lambda: FakeClient())
    assert executor.get_live_balance() == pytest.approx(245.0)


def test_get_live_balance_returns_none_on_error(monkeypatch):
    _install_fake_balance_types(monkeypatch)

    class FakeClient:
        def get_balance_allowance(self, params):
            raise RuntimeError("network down")

    monkeypatch.setattr(executor, "create_clob_client", lambda: FakeClient())
    assert executor.get_live_balance() is None


def test_close_position_live_skips_empty_book(monkeypatch):
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    monkeypatch.setattr(config, "POLYMARKET_FUNDER_ADDRESS", "")
    book = types.SimpleNamespace(bids=[], asks=[])
    _install_fake_clob(monkeypatch, {}, book)

    result = executor.close_position_live("cond1", "YES", 40.0, max_spread=0.05)
    assert result["status"] == "skipped_empty_book"
    assert result["order_id"] is None


# --- _close wiring: live places a sell + records order_id, dry-run does not ----

def test_live_close_places_order_and_records_id(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", False)
    calls = []

    def fake_close(condition_id, side, shares, max_spread=None):
        calls.append((condition_id, side, round(shares, 4)))
        return {"status": "executed", "order_id": "LIVE-999", "price": 0.6, "shares": shares}

    monkeypatch.setattr(executor, "close_position_live", fake_close)

    pos = _open(tmp_db, side="YES", entry=0.50, amount=25.0)
    res = positions.close_manual(pos["id"])

    assert res is not None
    # one sell, for the full held size: $25 / $0.50 = 50 shares of YES
    assert calls == [("cond1", "YES", 50.0)]

    conn = tmp_db._conn()
    row = conn.execute(
        "SELECT exit_order_id, status FROM positions WHERE id=?", (pos["id"],)
    ).fetchone()
    conn.close()
    assert row["exit_order_id"] == "LIVE-999"
    assert row["status"] == "closed_manual"


def test_dry_run_close_does_not_hit_exchange(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", True)
    called = []
    monkeypatch.setattr(
        executor, "close_position_live",
        lambda *a, **k: called.append(a) or {"status": "executed", "order_id": "X",
                                             "price": 0.6, "shares": 1},
    )

    pos = _open(tmp_db)
    res = positions.close_manual(pos["id"])

    assert res is not None
    assert called == []  # simulated fill, no CLOB order
    conn = tmp_db._conn()
    row = conn.execute(
        "SELECT exit_order_id FROM positions WHERE id=?", (pos["id"],)
    ).fetchone()
    conn.close()
    assert row["exit_order_id"] is None


def test_resolution_close_does_not_hit_exchange(tmp_db, monkeypatch):
    # Even live, a resolved market settles on-chain — nothing to sell.
    monkeypatch.setattr(config, "DRY_RUN", False)
    called = []
    monkeypatch.setattr(
        executor, "close_position_live",
        lambda *a, **k: called.append(a) or {"status": "executed", "order_id": "X",
                                             "price": 0.6, "shares": 1},
    )

    pos = _open(tmp_db)
    positions.close_on_resolution(pos["trade_id"], 1.0)

    assert called == []


# --- close_position_live: a rejected SELL surfaces close_failed ---------------

def _install_fake_clob_failing_order(monkeypatch, captured, book):
    """Like _install_fake_clob, but post_order raises — the order is rejected."""
    class FakeClobClient:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def create_or_derive_api_key(self):
            return "creds"

        def set_api_creds(self, creds):
            pass

        def get_market(self, condition_id):
            return {"tokens": [{"token_id": "tok-yes", "outcome": "YES"},
                               {"token_id": "tok-no", "outcome": "NO"}]}

        def get_order_book(self, token_id):
            return book

        def get_tick_size(self, token_id):
            return "0.01"

        def get_neg_risk(self, token_id):
            return False

        def create_order(self, order_args, options=None):
            return "signed-order"

        def post_order(self, signed, order_type):
            exc = RuntimeError("rejected")
            exc.error_msg = "not enough balance"
            raise exc

    _install_fake_v2_module(monkeypatch, FakeClobClient)


def test_close_position_live_order_rejection_returns_close_failed(monkeypatch):
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    monkeypatch.setattr(config, "POLYMARKET_FUNDER_ADDRESS", "0xfunder")
    monkeypatch.setattr(config, "POLYMARKET_SIGNATURE_TYPE", 3)

    book = types.SimpleNamespace(bids=[_level(0.60, 1000)], asks=[_level(0.62, 1000)])
    _install_fake_clob_failing_order(monkeypatch, {}, book)

    result = executor.close_position_live("cond1", "YES", 40.0, max_spread=0.05)
    assert result["status"] == "close_failed"
    assert result["order_id"] is None
    assert "not enough balance" in result["error"]


# --- the close path must NOT flatten the local position on an unconfirmed sell -

def test_failed_live_close_keeps_position_open_and_records_nothing(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", False)
    monkeypatch.setattr(
        executor, "close_position_live",
        lambda *a, **k: {"status": "close_failed", "order_id": None,
                         "price": None, "shares": None, "error": "rejected"},
    )

    pos = _open(tmp_db, side="YES", entry=0.50, amount=25.0)
    res = positions.close_manual(pos["id"])

    # close_manual reports the failure and the position is untouched/open
    assert res is not None
    assert res["close_failed"] is True
    assert res["realized"] == 0.0

    conn = tmp_db._conn()
    row = conn.execute(
        "SELECT status, exit_order_id, exit_reason, realized_pnl_usd "
        "FROM positions WHERE id=?", (pos["id"],)
    ).fetchone()
    # a close_failed tracking row was written for the attempt
    dec = conn.execute(
        "SELECT decision FROM exit_decisions WHERE position_id=?", (pos["id"],)
    ).fetchall()
    conn.close()

    assert row["status"] == "open"            # still held
    assert row["exit_order_id"] is None
    assert row["exit_reason"] is None
    assert row["realized_pnl_usd"] == 0
    # only the failed-attempt decision exists; no successful 'close' was logged
    decisions = [d["decision"] for d in dec]
    assert decisions == ["close_failed"]


def test_skipped_book_also_keeps_position_open(tmp_db, monkeypatch):
    # A book that can't support the sell is just as much "not flattened" as an
    # outright rejection — the position must stay open.
    monkeypatch.setattr(config, "DRY_RUN", False)
    monkeypatch.setattr(
        executor, "close_position_live",
        lambda *a, **k: {"status": "skipped_empty_book", "order_id": None,
                         "price": None, "shares": None},
    )

    pos = _open(tmp_db)
    res = positions.close_manual(pos["id"])
    assert res["close_failed"] is True

    conn = tmp_db._conn()
    status = conn.execute(
        "SELECT status FROM positions WHERE id=?", (pos["id"],)
    ).fetchone()["status"]
    conn.close()
    assert status == "open"


def test_stop_loss_does_not_close_on_failed_sell(tmp_db, monkeypatch):
    # The hard stop-loss path must also refuse to record a close when the live
    # SELL can't be confirmed (otherwise we hide live exposure on a loser).
    monkeypatch.setattr(config, "DRY_RUN", False)
    monkeypatch.setattr(
        executor, "close_position_live",
        lambda *a, **k: {"status": "close_failed", "order_id": None,
                         "price": None, "shares": None, "error": "rejected"},
    )

    pos = _open(tmp_db, side="YES", entry=0.50, amount=25.0)
    # Mark the position deep underwater (~ -98%) to trip the hard stop loss.
    positions.update_and_manage({pos["condition_id"]: 0.01})

    conn = tmp_db._conn()
    status = conn.execute(
        "SELECT status FROM positions WHERE id=?", (pos["id"],)
    ).fetchone()["status"]
    conn.close()
    assert status == "open"  # stop loss tripped but sell failed -> still held
