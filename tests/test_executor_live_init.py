"""Live ClobClient initialization: the deposit wallet (POLY_1271) is wired in
via funder + signature_type=3. py_clob_client is an optional dependency, so we
inject fakes into sys.modules and assert the init kwargs."""
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


def _install_fake_clob(monkeypatch, captured):
    """Stub out py_clob_client so _execute_live can construct a client and we
    can capture the constructor kwargs. The fake returns an empty book so the
    flow stops early (status 'skipped_empty_book') before any real order."""

    class FakeClobClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def create_or_derive_api_creds(self):
            return "creds"

        def set_api_creds(self, creds):
            pass

        def get_order_book(self, token_id):
            return types.SimpleNamespace(bids=[], asks=[])

    client_mod = types.ModuleType("py_clob_client.client")
    client_mod.ClobClient = FakeClobClient

    types_mod = types.ModuleType("py_clob_client.clob_types")
    types_mod.OrderArgs = object
    types_mod.OrderType = types.SimpleNamespace(GTC="GTC")

    pkg = types.ModuleType("py_clob_client")

    monkeypatch.setitem(sys.modules, "py_clob_client", pkg)
    monkeypatch.setitem(sys.modules, "py_clob_client.client", client_mod)
    monkeypatch.setitem(sys.modules, "py_clob_client.clob_types", types_mod)


def test_client_initialized_with_deposit_wallet_signature(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", False)
    monkeypatch.setattr(config, "DAILY_SPEND_LIMIT_USD", 1000.0)
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xprivkey")
    monkeypatch.setattr(config, "POLYMARKET_FUNDER_ADDRESS", "0xdepositwallet")
    monkeypatch.setattr(config, "POLYMARKET_SIGNATURE_TYPE", 3)
    monkeypatch.setattr(executor, "get_token_id", lambda market, side: "tok-1")

    captured = {}
    _install_fake_clob(monkeypatch, captured)

    result = executor.execute_trade(_signal())

    # The empty fake book stops the flow after the client is built.
    assert result["status"] == "skipped_empty_book"
    assert captured["funder"] == "0xdepositwallet"
    assert captured["signature_type"] == 3
    assert captured["key"] == "0xprivkey"
    assert captured["chain_id"] == 137


def test_eoa_wallet_omits_funder_and_signature(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", False)
    monkeypatch.setattr(config, "DAILY_SPEND_LIMIT_USD", 1000.0)
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xprivkey")
    monkeypatch.setattr(config, "POLYMARKET_FUNDER_ADDRESS", "")
    monkeypatch.setattr(executor, "get_token_id", lambda market, side: "tok-1")

    captured = {}
    _install_fake_clob(monkeypatch, captured)

    result = executor.execute_trade(_signal())

    assert result["status"] == "skipped_empty_book"
    assert "funder" not in captured
    assert "signature_type" not in captured
