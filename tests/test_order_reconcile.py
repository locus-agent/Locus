"""Order reconciliation after a GTC post: a posted order_id is not a fill.

_execute_live re-queries the exchange (client.get_order) after a short wait and
maps the real order state to a status:
  - MATCHED / filled_size > 0 -> "executed"
  - LIVE (resting, unfilled)  -> "resting"
  - missing / query error     -> "error_not_found"

py_clob_client_v2 is an optional dependency, so we inject fakes into sys.modules.
"""
import sys
import types

from locus import config
from locus.core import executor
from locus.markets.gamma import Market


def _signal(bet_amount: float = 25.0):
    market = Market(
        condition_id="0xabc",
        question="Will it rain tomorrow?",
        category="weather",
        yes_price=0.5,
        no_price=0.5,
        volume=10_000.0,
        end_date="2026-12-31",
        active=True,
        tokens=[],
    )
    return executor.Signal(
        market=market,
        claude_score=0.7,
        market_price=0.5,
        edge=0.2,
        side="YES",
        bet_amount=bet_amount,
        reasoning="test",
        headlines="test headline",
    )


def _install_fake_clob(monkeypatch, get_order_result):
    """Stub py_clob_client_v2 with a book deep enough to place a BUY, a post_order
    that returns an order_id, and a get_order returning `get_order_result`."""

    class FakeOrderArgs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeClobClient:
        def __init__(self, **kwargs):
            pass

        def create_or_derive_api_key(self):
            return "creds"

        def set_api_creds(self, creds):
            pass

        def get_market(self, condition_id):
            return {"tokens": [{"token_id": "tok-1", "outcome": "YES"},
                               {"token_id": "tok-no", "outcome": "NO"}]}

        def get_neg_risk(self, token_id):
            return False

        def get_order_book(self, token_id):
            # One fat ask level so plan_live_order clears the minimums.
            return types.SimpleNamespace(
                bids=[types.SimpleNamespace(price="0.49", size="500")],
                asks=[types.SimpleNamespace(price="0.50", size="500")],
            )

        def get_tick_size(self, token_id):
            return "0.01"

        def create_order(self, order_args, options=None):
            return "signed"

        def post_order(self, signed, order_type):
            return {"orderID": "order-123"}

        def get_order(self, order_id):
            return get_order_result

    mod = types.ModuleType("py_clob_client_v2")
    mod.ClobClient = FakeClobClient
    mod.OrderArgs = FakeOrderArgs
    mod.OrderType = types.SimpleNamespace(GTC="GTC", FOK="FOK", FAK="FAK", GTD="GTD")
    mod.Side = types.SimpleNamespace(BUY="BUY", SELL="SELL")
    mod.PartialCreateOrderOptions = FakeOrderArgs

    monkeypatch.setitem(sys.modules, "py_clob_client_v2", mod)


def _setup(monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", False)
    monkeypatch.setattr(config, "DAILY_SPEND_LIMIT_USD", 1000.0)
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xprivkey")
    monkeypatch.setattr(config, "POLYMARKET_FUNDER_ADDRESS", "")
    monkeypatch.setattr(config, "ORDER_RECONCILE_WAIT_SECONDS", 0)
    monkeypatch.setattr(executor, "get_token_id", lambda market, side: "tok-1")


def test_matched_order_is_executed(tmp_db, monkeypatch):
    _setup(monkeypatch)
    _install_fake_clob(monkeypatch, {"status": "MATCHED", "size_matched": "50"})
    result = executor.execute_trade(_signal())
    assert result["status"] == "executed"
    assert result["order_id"] == "order-123"


def test_filled_cost_from_making_amount():
    # makingAmount is the USDC paid in 6-decimal base units -> /1e6 dollars.
    assert executor._filled_cost_usd({"makingAmount": "21000000"}, 50.0, 0.5) == 21.0


def test_filled_cost_falls_back_to_notional_without_making_amount():
    # No makingAmount in the response -> planned notional (shares * price).
    assert executor._filled_cost_usd({"orderID": "x"}, 42.0, 0.5) == 21.0


def test_executed_buy_records_actual_cost(tmp_db, monkeypatch):
    _setup(monkeypatch)
    # post_order reports a $21 fill (makingAmount) on a nominal ~$25 bet.
    _install_fake_clob(monkeypatch, {"status": "MATCHED", "size_matched": "50"})
    import py_clob_client_v2 as client_mod
    monkeypatch.setattr(
        client_mod.ClobClient, "post_order",
        lambda self, signed, ot: {"orderID": "order-123", "makingAmount": "21000000"},
    )
    result = executor.execute_trade(_signal())
    assert result["status"] == "executed"
    assert result["actual_cost_usd"] == 21.0


def test_resting_buy_records_no_actual_cost(tmp_db, monkeypatch):
    # An unfilled (resting) order is cancelled and opens no position, so it
    # carries no real cost basis.
    _setup(monkeypatch)
    _install_fake_clob(monkeypatch, {"status": "LIVE", "size_matched": "0"})
    result = executor.execute_trade(_signal())
    assert result["status"] == "resting"
    assert result["actual_cost_usd"] is None


def test_filled_size_without_matched_status_is_executed(tmp_db, monkeypatch):
    _setup(monkeypatch)
    # Partially filled but still LIVE: any fill counts as executed.
    _install_fake_clob(monkeypatch, {"status": "LIVE", "size_matched": "10"})
    result = executor.execute_trade(_signal())
    assert result["status"] == "executed"


def test_resting_order_is_resting(tmp_db, monkeypatch):
    _setup(monkeypatch)
    _install_fake_clob(monkeypatch, {"status": "LIVE", "size_matched": "0"})
    result = executor.execute_trade(_signal())
    assert result["status"] == "resting"


def test_missing_order_is_error_not_found(tmp_db, monkeypatch):
    _setup(monkeypatch)
    _install_fake_clob(monkeypatch, None)
    result = executor.execute_trade(_signal())
    assert result["status"] == "error_not_found"


def test_buy_resolves_token_from_clob_not_gamma(tmp_db, monkeypatch):
    """The BUY path signs against the CLOB-resolved token_id (get_market), not
    the cached Gamma token; neg_risk is fetched and threaded into the options."""
    _setup(monkeypatch)
    # Gamma fallback would return a *different* token — it must not be used.
    monkeypatch.setattr(executor, "get_token_id", lambda market, side: "gamma-token")

    captured = {}

    class FakeOrderArgs:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class FakeOptions:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class FakeClobClient:
        def __init__(self, **kwargs):
            pass

        def create_or_derive_api_key(self):
            return "creds"

        def set_api_creds(self, creds):
            pass

        def get_market(self, condition_id):
            captured["get_market"] = condition_id
            return {"tokens": [{"token_id": "clob-token", "outcome": "YES"},
                               {"token_id": "clob-no", "outcome": "NO"}]}

        def get_neg_risk(self, token_id):
            captured["neg_risk_token"] = token_id
            return True

        def get_order_book(self, token_id):
            captured["book_token"] = token_id
            return types.SimpleNamespace(
                bids=[types.SimpleNamespace(price="0.49", size="500")],
                asks=[types.SimpleNamespace(price="0.50", size="500")],
            )

        def get_tick_size(self, token_id):
            return "0.01"

        def create_order(self, order_args, options=None):
            captured["order_args"] = order_args
            captured["options"] = options
            return "signed"

        def post_order(self, signed, order_type):
            return {"orderID": "order-123"}

        def get_order(self, order_id):
            return {"status": "MATCHED", "size_matched": "50"}

    mod = types.ModuleType("py_clob_client_v2")
    mod.ClobClient = FakeClobClient
    mod.OrderArgs = FakeOrderArgs
    mod.OrderType = types.SimpleNamespace(GTC="GTC")
    mod.Side = types.SimpleNamespace(BUY="BUY", SELL="SELL")
    mod.PartialCreateOrderOptions = FakeOptions
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", mod)

    result = executor.execute_trade(_signal())

    assert result["status"] == "executed"
    assert captured["get_market"] == "0xabc"        # CLOB consulted by condition_id
    assert captured["book_token"] == "clob-token"   # book read for the CLOB token
    assert captured["order_args"].token_id == "clob-token"  # signed against CLOB token
    assert captured["neg_risk_token"] == "clob-token"
    assert captured["options"].neg_risk is True     # neg_risk threaded into options
    assert captured["options"].tick_size == "0.01"


def test_get_order_raising_is_error_not_found(tmp_db, monkeypatch):
    _setup(monkeypatch)
    _install_fake_clob(monkeypatch, {"status": "MATCHED"})

    def boom(order_id):
        raise RuntimeError("network down")

    import py_clob_client_v2 as client_mod
    monkeypatch.setattr(client_mod.ClobClient, "get_order", lambda self, oid: boom(oid))
    result = executor.execute_trade(_signal())
    assert result["status"] == "error_not_found"


def test_lowercase_filled_status_is_executed(tmp_db, monkeypatch):
    # v2 may report lowercase 'filled'; status matching is case-insensitive.
    _setup(monkeypatch)
    _install_fake_clob(monkeypatch, {"status": "filled", "filledAmount": "0"})
    result = executor.execute_trade(_signal())
    assert result["status"] == "executed"


def test_camelcase_filled_amount_counts_as_fill(tmp_db, monkeypatch):
    # No matched status, but a camelCase filledAmount > 0 still means a fill.
    _setup(monkeypatch)
    _install_fake_clob(monkeypatch, {"status": "LIVE", "filledAmount": "12"})
    result = executor.execute_trade(_signal())
    assert result["status"] == "executed"


def test_reconcile_falls_back_to_open_orders_when_get_order_missing(tmp_db, monkeypatch):
    # A v2 client without get_order: reconcile scans get_open_orders by id.
    _setup(monkeypatch)
    _install_fake_clob(monkeypatch, None)
    import py_clob_client_v2 as client_mod
    monkeypatch.delattr(client_mod.ClobClient, "get_order", raising=False)
    monkeypatch.setattr(
        client_mod.ClobClient, "get_open_orders",
        lambda self, *a, **k: [{"orderID": "other", "status": "LIVE"},
                               {"orderID": "order-123", "status": "MATCHED"}],
        raising=False,
    )
    result = executor.execute_trade(_signal())
    assert result["status"] == "executed"


def test_resting_order_is_cancelled(tmp_db, monkeypatch):
    # An unfilled (resting) GTC buy must be cancelled via OrderPayload, not left
    # dangling on the book.
    _setup(monkeypatch)
    _install_fake_clob(monkeypatch, {"status": "LIVE", "size_matched": "0"})

    cancelled = []
    import py_clob_client_v2 as client_mod
    client_mod.OrderPayload = lambda orderID: ("payload", orderID)
    monkeypatch.setattr(
        client_mod.ClobClient, "cancel_order",
        lambda self, payload: cancelled.append(payload), raising=False,
    )

    result = executor.execute_trade(_signal())
    assert result["status"] == "resting"
    assert cancelled == [("payload", "order-123")]


def test_attribute_error_in_live_execution_is_logged(tmp_db, monkeypatch):
    # A v2 API mismatch (missing attribute) surfaces as error_AttributeError.
    _setup(monkeypatch)
    _install_fake_clob(monkeypatch, {"status": "MATCHED"})
    import py_clob_client_v2 as client_mod

    def boom(self, signed, order_type):
        raise AttributeError("'ClobClient' object has no attribute 'post_order'")

    monkeypatch.setattr(client_mod.ClobClient, "post_order", boom)
    result = executor.execute_trade(_signal())
    assert result["status"] == "error_AttributeError"
