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


def _open(tmp_db, side="YES", entry=0.50, amount=25.0, token_count=None):
    trade_id = tmp_db.log_trade(
        market_id="cond1", market_question="Will X happen?", claude_score=0.7,
        market_price=entry, edge=0.2, side=side, amount_usd=amount,
        status="executed", classification="bullish", materiality=0.7,
    )
    positions.open_position(trade_id, MKT, side, amount, token_count=token_count)
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


def test_bid_levels_dict_book():
    # dict-shaped get_order_book response (with dict levels) — the AttributeError
    # regression: 'dict' object has no attribute 'bids'.
    book = {
        "bids": [{"price": "0.48", "size": "100"}, {"price": "0.50", "size": "30"}],
        "asks": [{"price": "0.55", "size": "100"}, {"price": "0.52", "size": "10"}],
    }
    best_bid, best_ask, bid_size = executor._bid_levels(book)
    assert best_bid == 0.50
    assert best_ask == 0.52
    assert bid_size == 30.0


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
    # cancel_order_safe builds an OrderPayload to cancel an unfilled SELL.
    mod.OrderPayload = lambda **kw: ("payload", kw)
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", mod)
    return mod


def _install_fake_clob(monkeypatch, captured, book,
                       outcomes=("YES", "NO"),
                       fill_result=None,
                       held_balance=None):
    """Fake CLOB client. `outcomes` sets the real outcome labels get_market
    reports (default YES/NO; pass ("Up","Down") to exercise the positional
    fallback). `fill_result` is what get_order returns on reconcile (default a
    fully MATCHED order, so the SELL confirms as executed) — pass a LIST to
    script several orders in sequence (e.g. a top-up BUY then the SELL), each
    reconcile consuming the next entry. `held_balance`, when
    given (in shares), makes get_balance_allowance report that on-chain holding
    so the close's balance-cap can be exercised; None (default) omits the method
    entirely, matching a client release that lacks it (no cap applied).

    Every created order is appended to captured["orders"] (in order), with
    captured["order_args"] still the most recent one for older tests."""
    if fill_result is None:
        fill_result = {"status": "MATCHED", "size_matched": "9999"}
    fill_queue = list(fill_result) if isinstance(fill_result, list) else None
    # The close path reconciles after ORDER_RECONCILE_WAIT_SECONDS — keep tests fast.
    monkeypatch.setattr(config, "ORDER_RECONCILE_WAIT_SECONDS", 0)

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
                {"token_id": "tok-yes", "outcome": outcomes[0]},
                {"token_id": "tok-no", "outcome": outcomes[1]},
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
            captured.setdefault("orders", []).append(order_args)
            captured["options"] = options
            return "signed-order"

        def post_order(self, signed, order_type):
            captured["posted"] = (signed, order_type)
            return {"orderID": "LIVE-123"}

        def get_order(self, order_id):
            captured["reconciled"] = order_id
            if fill_queue is not None:
                return fill_queue.pop(0)
            return fill_result

        def cancel_order(self, payload):
            captured.setdefault("cancelled", []).append(payload)

    if held_balance is not None:
        # 6-decimal base units, mirroring the real CLOB collateral/token balances.
        def get_balance_allowance(self, params):
            captured["balance_params"] = params
            return {"balance": str(int(held_balance * 1_000_000))}
        FakeClobClient.get_balance_allowance = get_balance_allowance

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


# --- token resolution: positional fallback for non-Yes/No outcome labels -------

def test_resolve_token_id_label_match():
    client = types.SimpleNamespace(get_market=lambda c: {"tokens": [
        {"token_id": "t-yes", "outcome": "Yes"}, {"token_id": "t-no", "outcome": "No"}]})
    assert executor._resolve_token_id(client, "cond1", "YES") == "t-yes"
    assert executor._resolve_token_id(client, "cond1", "NO") == "t-no"


def test_resolve_token_id_positional_fallback_up_down():
    # "Bitcoin Up or Down" labels its tokens Up/Down, not Yes/No. Our positional
    # YES/NO convention must still resolve: YES -> first token, NO -> second.
    client = types.SimpleNamespace(get_market=lambda c: {"tokens": [
        {"token_id": "t-up", "outcome": "Up"}, {"token_id": "t-down", "outcome": "Down"}]})
    assert executor._resolve_token_id(client, "cond1", "YES") == "t-up"
    assert executor._resolve_token_id(client, "cond1", "NO") == "t-down"


def test_resolve_token_id_none_when_no_tokens():
    client = types.SimpleNamespace(get_market=lambda c: {"tokens": []})
    assert executor._resolve_token_id(client, "cond1", "YES") is None


def test_close_position_live_resolves_up_down_market(monkeypatch):
    # Regression: the close on an Up/Down market used to fail error_no_token
    # because the side label "YES" matched no outcome — now it resolves
    # positionally and places the SELL.
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    monkeypatch.setattr(config, "POLYMARKET_FUNDER_ADDRESS", "0xfunder")
    monkeypatch.setattr(config, "POLYMARKET_SIGNATURE_TYPE", 3)

    book = types.SimpleNamespace(bids=[_level(0.60, 1000)], asks=[_level(0.62, 1000)])
    captured = {}
    _install_fake_clob(monkeypatch, captured, book, outcomes=("Up", "Down"))

    result = executor.close_position_live("cond1", "YES", 40.0, max_spread=0.05)

    assert result["status"] == "executed"
    assert result["order_id"] == "LIVE-123"
    # YES resolved to the first token (the "Up" side) and that token was sold
    assert captured["book_token"] == "tok-yes"
    assert captured["order_args"].token_id == "tok-yes"
    assert captured["order_args"].side == "SELL"


def test_close_position_live_resting_sell_is_close_failed(monkeypatch):
    # A posted SELL that reconciles as resting/unfilled did NOT flatten the
    # position — report close_failed and cancel the dangling order.
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    monkeypatch.setattr(config, "POLYMARKET_FUNDER_ADDRESS", "")

    book = types.SimpleNamespace(bids=[_level(0.60, 1000)], asks=[_level(0.62, 1000)])
    captured = {}
    _install_fake_clob(monkeypatch, captured, book,
                       fill_result={"status": "LIVE", "size_matched": "0"})

    result = executor.close_position_live("cond1", "YES", 40.0, max_spread=0.05)

    assert result["status"] == "close_failed"
    assert result["order_id"] is None
    assert "reconcile" in result["error"]
    # the unfilled SELL was cancelled, not left dangling on the book
    assert captured.get("cancelled")


# --- _close wiring: live places a sell + records order_id, dry-run does not ----

def test_live_close_places_order_and_records_id(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", False)
    calls = []

    def fake_close(condition_id, side, shares, max_spread=None, allow_topup=False):
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


def test_live_close_sells_recorded_token_count(tmp_db, monkeypatch):
    # When token_count was recorded at open (the real filled shares), the close
    # sells exactly that, NOT the inflated amount_usd / entry_yes_price (50 here).
    monkeypatch.setattr(config, "DRY_RUN", False)
    calls = []

    def fake_close(condition_id, side, shares, max_spread=None, allow_topup=False):
        calls.append((condition_id, side, round(shares, 4)))
        return {"status": "executed", "order_id": "LIVE-777", "price": 0.6, "shares": shares}

    monkeypatch.setattr(executor, "close_position_live", fake_close)

    # $25 at 0.50 would derive 50 shares, but only 42 actually filled.
    pos = _open(tmp_db, side="YES", entry=0.50, amount=25.0, token_count=42.0)
    res = positions.close_manual(pos["id"])

    assert res is not None
    assert calls == [("cond1", "YES", 42.0)]  # held count, not the derived 50


def test_open_position_stores_token_count(tmp_db):
    pos = _open(tmp_db, side="YES", entry=0.50, amount=25.0, token_count=42.0)
    conn = tmp_db._conn()
    stored = conn.execute(
        "SELECT token_count FROM positions WHERE id=?", (pos["id"],)
    ).fetchone()["token_count"]
    conn.close()
    assert stored == pytest.approx(42.0)


def test_half_close_sells_half_the_token_count(tmp_db, monkeypatch):
    # A close_half on a token_count position sells half the real holding.
    monkeypatch.setattr(config, "DRY_RUN", False)
    calls = []
    monkeypatch.setattr(
        executor, "close_position_live",
        lambda cid, side, shares, max_spread=None, allow_topup=False:
        calls.append(round(shares, 4)) or
        {"status": "executed", "order_id": "LIVE-H", "price": 0.6, "shares": shares},
    )

    pos = _open(tmp_db, side="YES", entry=0.50, amount=25.0, token_count=42.0)
    conn = tmp_db._conn()
    positions._close(conn, pos, 0.60, "", "", fraction=0.5)
    conn.commit()
    conn.close()

    assert calls == [21.0]  # half of the 42 held, not half of the derived 50


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


# --- balance cap: never request more tokens than we own on-chain ---------------

def test_close_caps_sell_to_held_token_balance(monkeypatch):
    # The share count from amount_usd / entry_yes_price over-counts what we own
    # (the BUY filled at the higher ask). The close must cap the SELL to the real
    # on-chain holding so the order isn't rejected "not enough balance".
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    monkeypatch.setattr(config, "POLYMARKET_FUNDER_ADDRESS", "0xfunder")
    monkeypatch.setattr(config, "POLYMARKET_SIGNATURE_TYPE", 3)

    book = types.SimpleNamespace(bids=[_level(0.60, 1000)], asks=[_level(0.62, 1000)])
    captured = {}
    # We try to sell 66.64 sh but only hold 60.0 on-chain.
    _install_fake_clob(monkeypatch, captured, book, held_balance=60.0)

    result = executor.close_position_live("cond1", "YES", 66.64, max_spread=0.05)

    assert result["status"] == "executed"
    # the SELL was placed for the held 60 shares, not the inflated 66.64
    assert captured["order_args"].size == 60.0
    assert result["shares"] == 60.0


def test_close_does_not_upsize_when_request_below_balance(monkeypatch):
    # The cap only ever shrinks the order: a half-close (small request) against a
    # large holding must keep the requested size, not balloon to the full balance.
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    monkeypatch.setattr(config, "POLYMARKET_FUNDER_ADDRESS", "0xfunder")
    monkeypatch.setattr(config, "POLYMARKET_SIGNATURE_TYPE", 3)

    book = types.SimpleNamespace(bids=[_level(0.60, 1000)], asks=[_level(0.62, 1000)])
    captured = {}
    _install_fake_clob(monkeypatch, captured, book, held_balance=100.0)

    result = executor.close_position_live("cond1", "YES", 30.0, max_spread=0.05)

    assert result["status"] == "executed"
    assert captured["order_args"].size == 30.0  # unchanged; well under the 100 held


def test_held_token_shares_none_on_missing_method(monkeypatch):
    # A client release without get_balance_allowance must not raise — the cap is
    # best-effort and simply doesn't apply (None).
    _install_fake_balance_types(monkeypatch)
    client = types.SimpleNamespace()  # no get_balance_allowance
    assert executor.held_token_shares(client, "tok-yes") is None


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


# --- partial close: a SELL that only fills part of the holding keeps it open ---

def test_close_position_live_partial_fill_is_partial_close(monkeypatch):
    # We hold 40 on-chain and try to flatten all 40, but the book only matches 10.
    # That is NOT a full close — report partial_close with the 30 still held so the
    # caller keeps the position open (the live phantom-tokens bug).
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    monkeypatch.setattr(config, "POLYMARKET_FUNDER_ADDRESS", "")

    book = types.SimpleNamespace(bids=[_level(0.60, 1000)], asks=[_level(0.62, 1000)])
    captured = {}
    _install_fake_clob(monkeypatch, captured, book, held_balance=40.0,
                       fill_result={"status": "MATCHED", "size_matched": "10"})

    result = executor.close_position_live("cond1", "YES", 40.0, max_spread=0.05)

    assert result["status"] == "partial_close"
    assert result["sold_shares"] == pytest.approx(10.0)
    assert result["remaining_shares"] == pytest.approx(30.0)  # 40 held - 10 sold


def test_close_position_live_dust_remainder_is_full_close(monkeypatch):
    # 39.7 of an intended 40 sh match — the 0.3 sh remainder is dust (< 1 share),
    # so the position is treated as fully flat, not held open forever on rounding.
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    monkeypatch.setattr(config, "POLYMARKET_FUNDER_ADDRESS", "")

    book = types.SimpleNamespace(bids=[_level(0.60, 1000)], asks=[_level(0.62, 1000)])
    captured = {}
    _install_fake_clob(monkeypatch, captured, book,
                       fill_result={"status": "MATCHED", "size_matched": "39.7"})

    result = executor.close_position_live("cond1", "YES", 40.0, max_spread=0.05)

    assert result["status"] == "executed"
    assert result["sold_shares"] == pytest.approx(39.7)


def test_partial_close_keeps_position_open_with_reduced_token_count(tmp_db, monkeypatch):
    # A partial_close from the executor must NOT mark the position closed: it
    # stays open with token_count shrunk to the real on-chain remainder, the
    # sold chunk's PnL realized at ITS fill price, and the cost basis reduced
    # by the sold fraction. Hand-computed: hold 40 sh costing $25; 10 sold
    # @ 0.60 -> sold fraction 0.25, realized = 10*0.60 - 25*0.25 = -$0.25;
    # remaining basis $18.75 over 30 sh.
    monkeypatch.setattr(config, "DRY_RUN", False)
    monkeypatch.setattr(
        executor, "close_position_live",
        lambda *a, **k: {"status": "partial_close", "order_id": "LIVE-P",
                         "price": 0.6, "shares": 10.0, "sold_shares": 10.0,
                         "remaining_shares": 30.0},
    )

    pos = _open(tmp_db, side="YES", entry=0.50, amount=25.0, token_count=40.0)
    res = positions.close_manual(pos["id"])

    # manual close surfaces that the flatten didn't complete
    assert res["close_failed"] is True

    conn = tmp_db._conn()
    row = conn.execute(
        "SELECT status, token_count, amount_usd, realized_pnl_usd "
        "FROM positions WHERE id=?",
        (pos["id"],),
    ).fetchone()
    decisions = [d["decision"] for d in conn.execute(
        "SELECT decision FROM exit_decisions WHERE position_id=?", (pos["id"],)
    ).fetchall()]
    conn.close()

    assert row["status"] == "open"                       # still held
    assert row["token_count"] == pytest.approx(30.0)     # shrunk to the remainder
    assert row["amount_usd"] == pytest.approx(18.75)     # basis less the sold 25%
    assert row["realized_pnl_usd"] == pytest.approx(-0.25)  # sold chunk at ITS fill
    assert "partial_close" in decisions                  # tracked as a partial


def test_full_close_marks_position_closed(tmp_db, monkeypatch):
    # The complement: an executed close that fully flattens marks the row closed.
    monkeypatch.setattr(config, "DRY_RUN", False)
    monkeypatch.setattr(
        executor, "close_position_live",
        lambda *a, **k: {"status": "executed", "order_id": "LIVE-OK", "price": 0.6,
                         "shares": 40.0, "sold_shares": 40.0, "remaining_shares": 0.0},
    )

    pos = _open(tmp_db, side="YES", entry=0.50, amount=25.0, token_count=40.0)
    res = positions.close_manual(pos["id"])

    assert res["close_failed"] is False
    conn = tmp_db._conn()
    status = conn.execute(
        "SELECT status FROM positions WHERE id=?", (pos["id"],)
    ).fetchone()["status"]
    conn.close()
    assert status == "closed_manual"


# --- top-up-and-sell: dust positions below the exchange minimums ---------------

def test_plan_topup_buy_sizes_over_both_minimums():
    # Position 54's shape: 3.25 sh at a 0.05 bid — below 5 shares AND $1.
    # Sellable target at the bid: max(5, ceil(1/0.05)) = 20 sh -> top-up 16.75,
    # but the BUY itself must clear the minimums at the 0.06 ask and snap UP to
    # the CLOB size step (0.5 sh at 0.06) -> 17.0 sh, est $1.02.
    buy, cost, status = executor.plan_topup_buy(3.25, best_bid=0.05, best_ask=0.06)
    assert status == "ok"
    assert buy == pytest.approx(17.0)
    assert cost == pytest.approx(1.02)


def test_plan_topup_buy_not_dust_when_holding_clears_minimums():
    # 10 sh worth $5 at the bid clears both minimums — the thin-book skip was
    # depth-driven, so no top-up.
    _, _, status = executor.plan_topup_buy(10.0, best_bid=0.50, best_ask=0.52)
    assert status == "not_dust"


def test_plan_topup_buy_respects_cost_cap():
    # 2 sh at a 0.45 ask: the smallest placeable BUY is 5 sh = $2.25, over the
    # $2.00 TOPUP_MAX_USD cap — leave the dust alone.
    _, _, status = executor.plan_topup_buy(2.0, best_bid=0.42, best_ask=0.45)
    assert status == "topup_too_expensive"


def _dust_book():
    # 3.25 sh held at bid 0.05 / ask 0.06 — unsellable (< 5 sh, < $1 notional).
    return types.SimpleNamespace(bids=[_level(0.05, 1000)], asks=[_level(0.06, 1000)])


def test_close_dust_tops_up_then_fully_sells(monkeypatch):
    # The full top-up-and-sell path: BUY 17 sh @ 0.06 ($1.02), then SELL the
    # combined 20.25 sh (snapped to 20.2) @ 0.05, remainder 0.05 sh is dust.
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    monkeypatch.setattr(config, "POLYMARKET_FUNDER_ADDRESS", "")
    captured = {}
    _install_fake_clob(monkeypatch, captured, _dust_book(), held_balance=3.25,
                       fill_result=[
                           {"status": "MATCHED", "size_matched": "17"},    # BUY
                           {"status": "MATCHED", "size_matched": "20.2"},  # SELL
                       ])

    result = executor.close_position_live("cond1", "YES", 3.25, max_spread=0.05,
                                          allow_topup=True)

    assert result["status"] == "executed"
    assert result["topup_shares"] == pytest.approx(17.0)
    assert result["topup_cost_usd"] == pytest.approx(1.02)
    assert result["sold_shares"] == pytest.approx(20.2)
    assert result["price"] == 0.05
    assert result["remaining_shares"] == pytest.approx(0.05)
    # exactly two orders: the top-up BUY at the ask, then the SELL at the bid
    orders = captured["orders"]
    assert [o.side for o in orders] == ["BUY", "SELL"]
    assert orders[0].price == pytest.approx(0.06)
    assert orders[0].size == pytest.approx(17.0)
    assert orders[1].price == pytest.approx(0.05)
    assert orders[1].size == pytest.approx(20.2)


def test_bid_depth_within_sums_only_levels_near_reference():
    # Bids at 0.05 and 0.045 are within 20% of a 0.05 reference (floor 0.04);
    # the 500 shares resting at 0.007 are dust and must not count.
    book = types.SimpleNamespace(
        bids=[_level(0.05, 10), _level(0.045, 20), _level(0.007, 500)],
        asks=[_level(0.06, 100)],
    )
    assert executor.bid_depth_within(book, 0.05, 20.0) == pytest.approx(30.0)
    # tighter slippage excludes the 0.045 level too (floor 0.0475)
    assert executor.bid_depth_within(book, 0.05, 5.0) == pytest.approx(10.0)


def test_close_dust_topup_skipped_on_zombie_book(monkeypatch, caplog):
    # The live position-54 failure: the book LOOKS bid-supported but almost all
    # depth rests at dust prices (0.001-0.007) far below the ~0.05 ask; only
    # 2 sh sit near the mark. Topping up would buy at 0.05 what can only be
    # sold at 0.007 — the precheck must refuse BEFORE any money moves.
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    monkeypatch.setattr(config, "POLYMARKET_FUNDER_ADDRESS", "")
    book = types.SimpleNamespace(
        bids=[_level(0.04, 2), _level(0.007, 500), _level(0.001, 5000)],
        asks=[_level(0.05, 1000)],
    )
    captured = {}
    _install_fake_clob(monkeypatch, captured, book, held_balance=3.25)

    with caplog.at_level("WARNING", logger="locus.core.executor"):
        result = executor.close_position_live("cond1", "YES", 3.25,
                                              max_spread=0.05, allow_topup=True)

    assert result["status"] == "skipped_thin_book"
    assert "topup_shares" not in result
    assert "orders" not in captured  # no BUY was ever placed
    assert "topup_skipped_no_bid_liquidity" in caplog.text


def test_close_dust_topup_proceeds_when_bid_depth_near_mark(monkeypatch):
    # Dust levels may exist below — what matters is that depth NEAR the ask
    # covers the post-top-up holding (25 sh at 0.05 >= the 20.25 target).
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    monkeypatch.setattr(config, "POLYMARKET_FUNDER_ADDRESS", "")
    book = types.SimpleNamespace(
        bids=[_level(0.05, 25), _level(0.001, 99999)],
        asks=[_level(0.06, 1000)],
    )
    captured = {}
    _install_fake_clob(monkeypatch, captured, book, held_balance=3.25,
                       fill_result=[
                           {"status": "MATCHED", "size_matched": "17"},    # BUY
                           {"status": "MATCHED", "size_matched": "20.2"},  # SELL
                       ])

    result = executor.close_position_live("cond1", "YES", 3.25, max_spread=0.05,
                                          allow_topup=True)

    assert result["status"] == "executed"
    assert result["topup_shares"] == pytest.approx(17.0)
    assert result["sold_shares"] == pytest.approx(20.2)
    assert [o.side for o in captured["orders"]] == ["BUY", "SELL"]


def test_close_dust_topup_too_expensive_leaves_dust(monkeypatch):
    # The smallest placeable BUY costs $2.25 (>= TOPUP_MAX_USD $2) — no order is
    # placed at all and the close skips exactly as before the feature.
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    monkeypatch.setattr(config, "POLYMARKET_FUNDER_ADDRESS", "")
    book = types.SimpleNamespace(bids=[_level(0.42, 1000)], asks=[_level(0.45, 1000)])
    captured = {}
    _install_fake_clob(monkeypatch, captured, book, held_balance=2.0)

    result = executor.close_position_live("cond1", "YES", 2.0, max_spread=0.05,
                                          allow_topup=True)

    assert result["status"] == "skipped_thin_book"
    assert "topup_shares" not in result
    assert "orders" not in captured  # no BUY, no SELL — nothing hit the exchange


def test_close_dust_without_allow_topup_never_buys(monkeypatch):
    # Trigger discipline: the same dust holding with allow_topup unset (the
    # default) skips exactly as today — no spontaneous buying.
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    monkeypatch.setattr(config, "POLYMARKET_FUNDER_ADDRESS", "")
    captured = {}
    _install_fake_clob(monkeypatch, captured, _dust_book(), held_balance=3.25)

    result = executor.close_position_live("cond1", "YES", 3.25, max_spread=0.05)

    assert result["status"] == "skipped_thin_book"
    assert "topup_shares" not in result
    assert "orders" not in captured


def test_topup_armed_only_for_explicit_close_reasons(tmp_db, monkeypatch):
    # _live_close arms allow_topup only for explicit close attempts (manual /
    # hard stop / time-pressure) on a full close of a position whose real
    # holding is known — never for Claude re-eval decisions or half-closes.
    monkeypatch.setattr(config, "DRY_RUN", False)
    seen = []

    def fake_close(condition_id, side, shares, max_spread=None, allow_topup=False):
        seen.append(allow_topup)
        return {"status": "executed", "order_id": "X", "price": 0.5,
                "shares": shares, "sold_shares": shares, "remaining_shares": 0.0}

    monkeypatch.setattr(executor, "close_position_live", fake_close)

    cases = [
        ("manual", 1.0, 40.0, True),
        ("sl", 1.0, 40.0, True),
        ("time_pressure", 1.0, 40.0, True),
        ("tp_decision", 1.0, 40.0, False),   # Claude decision, not explicit
        ("news_decision", 1.0, 40.0, False),
        ("manual", 0.5, 40.0, False),        # half-close never tops up
        ("manual", 1.0, None, False),        # no token_count -> holding unknown
    ]
    for reason, fraction, token_count, expected in cases:
        pos = _open(tmp_db, token_count=token_count)
        conn = tmp_db._conn()
        positions._close(conn, pos, 0.50, "closed_x", reason, fraction)
        conn.commit()
        conn.close()
        assert seen[-1] is expected, f"exit_reason={reason} fraction={fraction}"


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
