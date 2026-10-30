"""Read-only on-chain balance fallback via a Polygon JSON-RPC endpoint.

Used ONLY as a fallback when Polymarket's own APIs (the CLOB client, the
public Data API) are unavailable or return nothing. Disabled entirely when
config.POLYGON_RPC_URL is empty (the default) — every caller behaves exactly
as it did before this module existed. Never signs, sends, or estimates gas
for anything; the only call made is a read-only `eth_call` against
`balanceOf`.

Contract: Conditional Tokens Framework (CTF), Polymarket's ERC1155 outcome-
token contract on Polygon mainnet (chain id 137/0x89):
    0x4D97DCd97eC945f40cF65F87097ACe5EA0476045
Source: https://docs.polymarket.com/resources/contracts (section
"Conditional Tokens"), cross-checked against the Polygonscan label
"Polymarket: Conditional Tokens" for the same address
(https://polygonscan.com/address/0x4d97dcd97ec945f40cf65f87097ace5ea0476045).
Not taken from memory — looked up this session specifically for this change.

balanceOf(address account, uint256 id) is the standard ERC1155 read function
(EIP-1155); its 4-byte selector 0x00fdd58e is keccak256("balanceOf(address,
uint256)")[:4] (verified by computing the hash, not recalled).

Owner address: for a Polymarket proxy/deposit wallet (the POLY_1271 /
signature_type=3 flow this bot uses — see executor.create_clob_client), the
ERC1155 balance is held by the FUNDER/proxy wallet
(config.POLYMARKET_FUNDER_ADDRESS), never by the signing EOA
(POLYMARKET_PRIVATE_KEY's own address). Confirmed empirically this session: a
direct eth_call for position 57's YES token returned 0.005067 for the funder
address (matching the CLOB-reported 0.005) and exactly 0 for the signing EOA.

Decimals: CTF outcome-token balances are scaled by 1e6 (matching USDC), same
as executor.held_token_shares' CLOB-side reading — confirmed by the same
eth_call above (raw 5067 / 1e6 = 0.005067).
"""
from __future__ import annotations

import logging
import re

import httpx

from locus import config

log = logging.getLogger(__name__)

CTF_CONTRACT_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# keccak256("balanceOf(address,uint256)")[:4] — the standard ERC1155 selector.
_BALANCE_OF_SELECTOR = "0x00fdd58e"

# CTF outcome tokens are scaled like USDC (6 decimals); see module docstring.
_DECIMALS_DIVISOR = 1_000_000


def mask_rpc_url(url: str) -> str:
    """`url` with any path/query (where an API key normally lives, e.g.
    Alchemy's /v2/<key>) stripped — only the scheme+host survive. Used in
    every log line so POLYGON_RPC_URL is never written out in full."""
    if not url:
        return "(unset)"
    match = re.match(r"^(https?://[^/?#]+)", url)
    host = match.group(1) if match else "(unparseable)"
    return f"{host}/***"


def _encode_balance_of(owner: str, token_id: int) -> str:
    """ABI-encode balanceOf(address,uint256) calldata for `owner`/`token_id`."""
    owner_hex = owner.lower().removeprefix("0x").rjust(64, "0")
    token_hex = format(int(token_id), "x").rjust(64, "0")
    return _BALANCE_OF_SELECTOR + owner_hex + token_hex


def get_erc1155_balance(owner: str, token_id: str | int,
                        rpc_url: str | None = None,
                        timeout: float | None = None,
                        retries: int | None = None) -> int | None:
    """Raw ERC1155 balance (base units, no decimal scaling) of `token_id` held
    by `owner` on the CTF contract, via a single read-only `eth_call`.

    Returns None on ANY failure: no RPC URL configured, malformed owner/
    token_id, a network/timeout error (retried up to `retries` times), a
    non-2xx HTTP response, or a JSON-RPC error field. Never raises."""
    rpc_url = rpc_url if rpc_url is not None else config.POLYGON_RPC_URL
    if not rpc_url:
        return None
    timeout = config.ONCHAIN_RPC_TIMEOUT_SECONDS if timeout is None else timeout
    retries = config.ONCHAIN_RPC_RETRIES if retries is None else retries

    try:
        data = _encode_balance_of(owner, token_id)
    except (ValueError, TypeError) as e:
        log.warning("[onchain] bad owner/token_id for balanceOf (owner=%s): %s",
                    owner, e)
        return None

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [{"to": CTF_CONTRACT_ADDRESS, "data": data}, "latest"],
    }
    masked = mask_rpc_url(rpc_url)

    last_error: str | None = None
    for attempt in range(max(retries, 0) + 1):
        try:
            resp = httpx.post(rpc_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            body = resp.json()
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < retries:
                log.info("[onchain] eth_call attempt %d/%d failed (%s), retrying "
                         "(rpc=%s)", attempt + 1, retries + 1, last_error, masked)
                continue
            log.warning("[onchain] eth_call failed after %d attempt(s): %s "
                       "(rpc=%s)", attempt + 1, last_error, masked)
            return None

        if isinstance(body, dict) and body.get("error"):
            log.warning("[onchain] eth_call JSON-RPC error: %s (rpc=%s)",
                       body["error"], masked)
            return None

        raw_hex = body.get("result") if isinstance(body, dict) else None
        if not raw_hex:
            log.warning("[onchain] eth_call returned no result field (rpc=%s)",
                       masked)
            return None
        try:
            return int(raw_hex, 16)
        except (ValueError, TypeError) as e:
            log.warning("[onchain] eth_call result not parseable as hex int "
                       "(%r): %s (rpc=%s)", raw_hex, e, masked)
            return None
    return None


def held_token_shares_onchain(owner: str, token_id: str | int,
                              rpc_url: str | None = None) -> float | None:
    """Real outcome-token shares held by `owner` for `token_id`, read directly
    from the CTF contract via Polygon RPC (no Polymarket API involved at
    all). None when the RPC fallback is disabled (no URL configured) or any
    call fails. Scaled by 1e6, matching executor.held_token_shares' CLOB-side
    reading (see module docstring for the empirical confirmation)."""
    raw = get_erc1155_balance(owner, token_id, rpc_url=rpc_url)
    if raw is None:
        return None
    return raw / _DECIMALS_DIVISOR
