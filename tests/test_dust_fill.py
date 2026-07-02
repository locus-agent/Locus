"""MIN_FILL_USD dust-fill guard: a live BUY whose reconciled fill lands below
MIN_FILL_USD (e.g. a $2 order that partially fills for $0.23) must NOT open a
managed position — the overhead exceeds the value. The executor attempts to
sell the dust straight back at the bid (best-effort, via close_position_live)
and records the trade as status='dust_fill' so the funnel shows it. A fill at
or above the threshold opens normally, and dry-run is unaffected (no real
fills there).

py_clob_client_v2 is an optional dependency, so we inject fakes into
sys.modules (same pattern as test_order_reconcile).
"""
import sys
import types

import pytest

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

        def cancel_order(self, payload):
            pass

    mod = types.ModuleType("py_clob_client_v2")
    mod.ClobClient = FakeClobClient
    mod.OrderArgs = FakeOrderArgs
    mod.OrderType = types.SimpleNamespace(GTC="GTC", FOK="FOK", FAK="FAK", GTD="GTD")
    mod.Side = types.SimpleNamespace(BUY="BUY", SELL="SELL")
    mod.PartialCreateOrderOptions = FakeOrderArgs
    mod.OrderPayload = lambda orderID: ("payload", orderID)

    monkeypatch.setitem(sys.modules, "py_clob_client_v2", mod)


def _setup(monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", False)
    monkeypatch.setattr(config, "DAILY_SPEND_LIMIT_USD", 1000.0)
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xprivkey")
    monkeypatch.setattr(config, "POLYMARKET_FUNDER_ADDRESS", "")
    monkeypatch.setattr(config, "ORDER_RECONCILE_WAIT_SECONDS", 0)
    monkeypatch.setattr(config, "MIN_FILL_USD", 1.0)
    monkeypatch.setattr(executor, "get_token_id", lambda market, side: "tok-1")


def _stub_sellback(monkeypatch, result_status: str):
    """Replace close_position_live (the dust sell-back plumbing), recording
    each call as (condition_id, side, shares)."""
    calls = []

    def fake_close(condition_id, side, shares, max_spread=None):
        calls.append((condition_id, side, shares))
        return {"status": result_status, "order_id": "sell-1", "price": 0.49,
                "shares": shares}

    monkeypatch.setattr(executor, "close_position_live", fake_close)
    return calls


def _trade_status(tmp_db, market_id: str) -> str:
    conn = tmp_db._conn()
    try:
        return conn.execute(
            "SELECT status FROM trades WHERE market_id=?", (market_id,)
        ).fetchone()["status"]
    finally:
        conn.close()


def test_dust_fill_sells_back_and_records_dust_status(tmp_db, monkeypatch):
    _setup(monkeypatch)
    # 1 of 50 shares matched at $0.50 -> $0.50 filled < MIN_FILL_USD $1.00.
    _install_fake_clob(monkeypatch, {"status": "MATCHED", "size_matched": "1"})
    sellbacks = _stub_sellback(monkeypatch, "executed")

    result = executor.execute_trade(_signal())

    assert result["status"] == "dust_fill"
    assert result["actual_cost_usd"] == 0.5
    # The dust is sold back at the bid via the existing sell plumbing.
    assert sellbacks == [("0xabc", "YES", 1.0)]
    # The trade row records dust_fill so the funnel shows it.
    assert _trade_status(tmp_db, "0xabc") == "dust_fill"


def test_dust_fill_sell_back_failure_still_opens_nothing(tmp_db, monkeypatch):
    # The dust is usually below the exchange order minimums, so the sell-back
    # may be unplaceable (skipped_thin_book). The outcome must STILL be
    # dust_fill — never a managed position.
    _setup(monkeypatch)
    _install_fake_clob(monkeypatch, {"status": "MATCHED", "size_matched": "1"})
    sellbacks = _stub_sellback(monkeypatch, "skipped_thin_book")

    result = executor.execute_trade(_signal())

    assert result["status"] == "dust_fill"
    assert len(sellbacks) == 1
    assert _trade_status(tmp_db, "0xabc") == "dust_fill"


def test_fill_at_or_above_threshold_opens_normally(tmp_db, monkeypatch):
    _setup(monkeypatch)
    # 10 of 50 shares matched at $0.50 -> $5.00 >= MIN_FILL_USD: a normal
    # (partial) fill, no sell-back.
    _install_fake_clob(monkeypatch, {"status": "MATCHED", "size_matched": "10"})
    sellbacks = _stub_sellback(monkeypatch, "executed")

    result = executor.execute_trade(_signal())

    assert result["status"] == "executed"
    assert result["actual_cost_usd"] == 5.0
    assert sellbacks == []


def test_full_fill_unaffected(tmp_db, monkeypatch):
    _setup(monkeypatch)
    _install_fake_clob(monkeypatch, {"status": "MATCHED", "size_matched": "50"})
    sellbacks = _stub_sellback(monkeypatch, "executed")

    result = executor.execute_trade(_signal())

    assert result["status"] == "executed"
    assert result["actual_cost_usd"] == 25.0
    assert sellbacks == []


def test_dry_run_unaffected(tmp_db, monkeypatch):
    # Dry-run has no real fills, so the dust guard never applies: no CLOB
    # client is built and no sell-back is attempted.
    monkeypatch.setattr(config, "DRY_RUN", True)
    monkeypatch.setattr(config, "DAILY_SPEND_LIMIT_USD", 1000.0)
    monkeypatch.setattr(config, "MIN_FILL_USD", 1_000_000.0)  # would dust anything live
    sellbacks = _stub_sellback(monkeypatch, "executed")

    result = executor.execute_trade(_signal())

    assert result["status"] == "dry_run"
    assert sellbacks == []
    assert _trade_status(tmp_db, "0xabc") == "dry_run"
