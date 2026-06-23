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


def test_get_order_raising_is_error_not_found(tmp_db, monkeypatch):
    _setup(monkeypatch)
    _install_fake_clob(monkeypatch, {"status": "MATCHED"})

    def boom(order_id):
        raise RuntimeError("network down")

    import py_clob_client_v2 as client_mod
    monkeypatch.setattr(client_mod.ClobClient, "get_order", lambda self, oid: boom(oid))
    result = executor.execute_trade(_signal())
    assert result["status"] == "error_not_found"
