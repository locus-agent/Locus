"""onchain.py: read-only Polygon RPC balance fallback. Disabled when
POLYGON_RPC_URL is empty (conftest's autouse fixture pins this for the whole
suite), any failure returns None rather than raising, and the RPC URL/key is
never written out in full anywhere — including in failure log lines."""
import types

import pytest

from locus import config
from locus.core import onchain


OWNER = "0xadAeDBd606926aa399852caBec5028Ab99B9e7B9"
TOKEN_ID = 55764212211467781322980371912612507865974994976253196346176314491480419639168
SECRET_RPC_URL = "https://polygon-mainnet.g.alchemy.com/v2/SuperSecretAlchemyKey123"


def _hex_result(raw: int) -> dict:
    return {"jsonrpc": "2.0", "id": 1, "result": "0x" + format(raw, "x")}


def _fake_post(monkeypatch, responder):
    """responder(url, json) -> an object with .raise_for_status()/.json(),
    or raises to simulate a network error."""
    monkeypatch.setattr(onchain.httpx, "post", responder)


class _Resp:
    def __init__(self, body, status_ok=True):
        self._body = body
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP error")

    def json(self):
        return self._body


# --- mask_rpc_url: never leak the URL/key -------------------------------------

def test_mask_rpc_url_strips_path_and_query():
    masked = onchain.mask_rpc_url(SECRET_RPC_URL)
    assert "SuperSecretAlchemyKey123" not in masked
    assert masked == "https://polygon-mainnet.g.alchemy.com/***"


def test_mask_rpc_url_empty():
    assert onchain.mask_rpc_url("") == "(unset)"


def test_mask_rpc_url_unparseable():
    assert onchain.mask_rpc_url("not-a-url") == "(unparseable)/***"


# --- get_erc1155_balance: success / disabled / errors -------------------------

def test_disabled_when_url_not_set():
    # conftest's autouse fixture already pins config.POLYGON_RPC_URL to "";
    # confirm the module-level default (no explicit rpc_url) is a clean None.
    assert config.POLYGON_RPC_URL == ""
    assert onchain.get_erc1155_balance(OWNER, TOKEN_ID) is None


def test_disabled_via_explicit_empty_url():
    assert onchain.get_erc1155_balance(OWNER, TOKEN_ID, rpc_url="") is None


def test_success_returns_raw_balance(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        return _Resp(_hex_result(5067))

    _fake_post(monkeypatch, fake_post)

    raw = onchain.get_erc1155_balance(OWNER, TOKEN_ID, rpc_url=SECRET_RPC_URL)
    assert raw == 5067
    assert captured["url"] == SECRET_RPC_URL
    assert captured["payload"]["method"] == "eth_call"
    call_params = captured["payload"]["params"][0]
    assert call_params["to"] == onchain.CTF_CONTRACT_ADDRESS
    assert call_params["data"].startswith(onchain._BALANCE_OF_SELECTOR)


def test_zero_balance_is_a_valid_result(monkeypatch):
    _fake_post(monkeypatch, lambda url, json=None, timeout=None: _Resp(_hex_result(0)))
    assert onchain.get_erc1155_balance(OWNER, TOKEN_ID, rpc_url=SECRET_RPC_URL) == 0


def test_held_token_shares_onchain_divides_by_1e6(monkeypatch):
    _fake_post(monkeypatch, lambda url, json=None, timeout=None: _Resp(_hex_result(5067)))
    result = onchain.held_token_shares_onchain(OWNER, TOKEN_ID, rpc_url=SECRET_RPC_URL)
    assert result == pytest.approx(0.005067)


def test_held_token_shares_onchain_none_when_disabled():
    assert onchain.held_token_shares_onchain(OWNER, TOKEN_ID) is None


def test_network_error_retried_then_fails(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(1)
        raise ConnectionError("network down")

    _fake_post(monkeypatch, fake_post)

    result = onchain.get_erc1155_balance(
        OWNER, TOKEN_ID, rpc_url=SECRET_RPC_URL, retries=2
    )
    assert result is None
    assert len(calls) == 3  # initial attempt + 2 retries


def test_network_error_succeeds_on_retry(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(1)
        if len(calls) < 2:
            raise TimeoutError("timed out")
        return _Resp(_hex_result(42))

    _fake_post(monkeypatch, fake_post)

    result = onchain.get_erc1155_balance(
        OWNER, TOKEN_ID, rpc_url=SECRET_RPC_URL, retries=2
    )
    assert result == 42
    assert len(calls) == 2


def test_http_error_status_returns_none(monkeypatch):
    _fake_post(monkeypatch, lambda url, json=None, timeout=None:
              _Resp(None, status_ok=False))
    assert onchain.get_erc1155_balance(
        OWNER, TOKEN_ID, rpc_url=SECRET_RPC_URL, retries=0) is None


def test_json_rpc_error_field_returns_none(monkeypatch):
    body = {"jsonrpc": "2.0", "id": 1,
           "error": {"code": -32000, "message": "execution reverted"}}
    _fake_post(monkeypatch, lambda url, json=None, timeout=None: _Resp(body))
    assert onchain.get_erc1155_balance(OWNER, TOKEN_ID, rpc_url=SECRET_RPC_URL) is None


def test_missing_result_field_returns_none(monkeypatch):
    _fake_post(monkeypatch, lambda url, json=None, timeout=None:
              _Resp({"jsonrpc": "2.0", "id": 1}))
    assert onchain.get_erc1155_balance(OWNER, TOKEN_ID, rpc_url=SECRET_RPC_URL) is None


def test_unparseable_result_returns_none(monkeypatch):
    _fake_post(monkeypatch, lambda url, json=None, timeout=None:
              _Resp({"jsonrpc": "2.0", "id": 1, "result": "not-hex"}))
    assert onchain.get_erc1155_balance(OWNER, TOKEN_ID, rpc_url=SECRET_RPC_URL) is None


def test_bad_owner_address_returns_none():
    assert onchain.get_erc1155_balance(
        "not-an-address", TOKEN_ID, rpc_url=SECRET_RPC_URL) is None


# --- key/URL masking in logs ---------------------------------------------------

def test_failure_logs_never_contain_the_raw_url(monkeypatch, caplog):
    def fake_post(url, json=None, timeout=None):
        raise ConnectionError("network down")

    _fake_post(monkeypatch, fake_post)

    with caplog.at_level("WARNING", logger="locus.core.onchain"):
        onchain.get_erc1155_balance(OWNER, TOKEN_ID, rpc_url=SECRET_RPC_URL, retries=0)

    assert "SuperSecretAlchemyKey123" not in caplog.text
    assert "polygon-mainnet.g.alchemy.com/***" in caplog.text


def test_http_error_logs_never_contain_the_raw_url(monkeypatch, caplog):
    _fake_post(monkeypatch, lambda url, json=None, timeout=None:
              _Resp(None, status_ok=False))

    with caplog.at_level("WARNING", logger="locus.core.onchain"):
        onchain.get_erc1155_balance(OWNER, TOKEN_ID, rpc_url=SECRET_RPC_URL, retries=0)

    assert "SuperSecretAlchemyKey123" not in caplog.text


def test_json_rpc_error_logs_never_contain_the_raw_url(monkeypatch, caplog):
    body = {"jsonrpc": "2.0", "id": 1, "error": {"code": -1, "message": "boom"}}
    _fake_post(monkeypatch, lambda url, json=None, timeout=None: _Resp(body))

    with caplog.at_level("WARNING", logger="locus.core.onchain"):
        onchain.get_erc1155_balance(OWNER, TOKEN_ID, rpc_url=SECRET_RPC_URL)

    assert "SuperSecretAlchemyKey123" not in caplog.text
