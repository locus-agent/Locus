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


# --- close_position_live: full path against a faked py_clob_client ------------

def _install_fake_clob(monkeypatch, captured, book):
    class FakeClobClient:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def create_or_derive_api_creds(self):
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

        def create_order(self, order_args):
            captured["order_args"] = order_args
            return "signed-order"

        def post_order(self, signed, order_type):
            captured["posted"] = (signed, order_type)
            return {"orderID": "LIVE-123"}

    client_mod = types.ModuleType("py_clob_client.client")
    client_mod.ClobClient = FakeClobClient

    class OrderArgs:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    types_mod = types.ModuleType("py_clob_client.clob_types")
    types_mod.OrderArgs = OrderArgs
    types_mod.OrderType = types.SimpleNamespace(GTC="GTC")

    pkg = types.ModuleType("py_clob_client")
    monkeypatch.setitem(sys.modules, "py_clob_client", pkg)
    monkeypatch.setitem(sys.modules, "py_clob_client.client", client_mod)
    monkeypatch.setitem(sys.modules, "py_clob_client.clob_types", types_mod)


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
    # built on the deposit wallet (POLY_1271)
    assert captured["init"]["funder"] == "0xfunder"
    assert captured["init"]["signature_type"] == 3


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
