from __future__ import annotations

import asyncio
import logging
import math
import time

from locus import config
from locus.memory import logger
from locus.core.edge import Signal
from locus.core import telegram_bot
from locus.markets.gamma import get_token_id

log = logging.getLogger(__name__)


def execute_trade(signal: Signal) -> dict:
    """Execute a trade on Polymarket or log a dry-run. Synchronous."""
    daily_spent = abs(logger.get_daily_pnl())
    if daily_spent + signal.bet_amount > config.DAILY_SPEND_LIMIT_USD:
        return _log_and_return(signal, status="rejected_daily_limit", order_id=None)

    if config.DRY_RUN:
        return _log_and_return(signal, status="dry_run", order_id=None)

    return _execute_live(signal)


async def execute_trade_async(signal: Signal) -> dict:
    """Async wrapper around execute_trade."""
    return await asyncio.get_event_loop().run_in_executor(None, execute_trade, signal)


def valid_size_step(price: float) -> float:
    """Smallest order size increment (in shares) that yields CLOB-valid amounts.

    Polymarket's CLOB has an undocumented rule: both scaled integer amounts of a
    signed order must be whole multiples of 10_000, where (for a BUY)
        maker_amount = price * size * 1e6   (USDC, 6 decimals)
        taker_amount =         size * 1e6   (outcome tokens, 6 decimals)
    Rounding size to 2 decimals is NOT enough — e.g. price=0.081 size=25.4 gives
    maker_amount 2_057_400, which is not a multiple of 10_000, so the exchange
    rejects the order body as "Invalid order inputs". The valid size step is
    price-dependent; we derive it from the GCD of the scaled price and 1e6.

    `price` must already be tick-aligned (multiples of 0.001/0.0001 etc.); the
    *10_000 scaling captures ticks down to 0.0001, the CLOB's finest."""
    p_int = int(round(price * 10_000))
    if p_int <= 0:
        return 0.01
    # Smallest size (in 1e-4-share units) making maker_amount a multiple of 1e6...
    maker_step = 1_000_000 // math.gcd(p_int, 1_000_000)
    # ...also forced to a multiple of 100 so taker_amount stays a multiple of 1e4.
    step_units = maker_step * 100 // math.gcd(maker_step, 100)
    return step_units / 10_000


def round_size_for_clob(price: float, size: float) -> float:
    """Round `size` DOWN to the nearest CLOB-valid increment at `price`.

    Always rounds down so the order never exceeds the requested size / the
    visible depth it was already capped to."""
    step = valid_size_step(price)
    if step <= 0:
        return round(size, 2)
    n = int((size + 1e-9) / step)
    return round(n * step, 4)


def plan_live_order(
    bet_usd: float,
    best_bid: float | None,
    best_ask: float | None,
    ask_size_shares: float | None,
    max_spread: float,
) -> tuple[float | None, float | None, str]:
    """Decide (price, size_in_shares, status) for a live BUY against the book.

    - No asks -> nothing to buy ("skipped_empty_book").
    - Spread wider than max_spread -> the cached-price "edge" is mostly
      spread; skip ("skipped_wide_spread").
    - Price at best ask (take liquidity, immediate fill), size in SHARES
      (the CLOB sizes orders in outcome tokens, not dollars), downsized to
      the visible depth at best ask and snapped to a CLOB-valid size step;
      below the exchange minimums -> "skipped_thin_book".
    """
    if best_ask is None or best_ask <= 0:
        return None, None, "skipped_empty_book"
    if best_bid is not None and (best_ask - best_bid) > max_spread:
        return None, None, "skipped_wide_spread"

    # shares = dollars / price (NOT dollars) — the CLOB sizes in outcome tokens.
    raw_shares = bet_usd / best_ask
    capped_shares = min(raw_shares, ask_size_shares) if ask_size_shares is not None else raw_shares
    shares = round_size_for_clob(best_ask, capped_shares)
    notional = shares * best_ask
    log.info(
        "[executor] BUY plan: bet=$%.2f price=%.4f -> raw %.2f sh, depth-capped %.2f sh, "
        "snapped %.2f sh (step=%.2f, notional=$%.2f)",
        bet_usd, best_ask, raw_shares, capped_shares, shares,
        valid_size_step(best_ask), notional,
    )
    if shares < config.MIN_ORDER_SHARES or notional < config.MIN_ORDER_USD:
        return None, None, "skipped_thin_book"
    return best_ask, shares, "ok"


def plan_live_sell(
    shares: float,
    best_bid: float | None,
    best_ask: float | None,
    bid_size_shares: float | None,
    max_spread: float,
) -> tuple[float | None, float | None, str]:
    """Decide (price, size_in_shares, status) for a live SELL into the book.

    The mirror image of plan_live_order: a close hits the best BID (take
    liquidity, immediate fill), sizing in SHARES and downsizing to the visible
    depth at best bid, then snapping to a CLOB-valid size step. Same guards as
    a buy:
    - No bids -> nothing to sell into ("skipped_empty_book").
    - Spread wider than max_spread -> skip ("skipped_wide_spread").
    - Below the exchange minimums after downsizing -> "skipped_thin_book".
    """
    if best_bid is None or best_bid <= 0:
        return None, None, "skipped_empty_book"
    if best_ask is not None and (best_ask - best_bid) > max_spread:
        return None, None, "skipped_wide_spread"

    capped_shares = min(shares, bid_size_shares) if bid_size_shares is not None else shares
    sell_shares = round_size_for_clob(best_bid, capped_shares)
    notional = sell_shares * best_bid
    log.info(
        "[executor] SELL plan: want %.2f sh, price=%.4f -> depth-capped %.2f sh, "
        "snapped %.2f sh (step=%.2f, notional=$%.2f)",
        shares, best_bid, capped_shares, sell_shares,
        valid_size_step(best_bid), notional,
    )
    if sell_shares < config.MIN_ORDER_SHARES or notional < config.MIN_ORDER_USD:
        return None, None, "skipped_thin_book"
    return best_bid, sell_shares, "ok"


def round_to_tick(price: float, tick_size) -> float:
    """Snap `price` to the nearest multiple of `tick_size`, clamped into the
    [tick, 1 - tick] band Polymarket's CLOB enforces (`price_valid`).

    Polymarket rejects orders whose price is not a multiple of the market's
    minimum tick (0.01 for most markets, 0.001 for tighter ones) with a
    ValidationException. Book levels are already tick-aligned, but cached or
    derived prices may not be, so we always normalise before sending an order.
    `tick_size` is whatever `client.get_tick_size()` returns (a str like
    "0.01") or a float; both are accepted."""
    tick = float(tick_size)
    if tick <= 0:
        return price
    # Decimal places implied by the tick, used to kill float dust like
    # 0.30000000000000004 after multiplying back up.
    tick_str = str(tick_size)
    decimals = len(tick_str.split(".")[1]) if "." in tick_str else 0
    rounded = round(round(price / tick) * tick, decimals)
    return max(tick, min(rounded, round(1 - tick, decimals)))


def _get_tick_size(client, token_id: str) -> str:
    """Market minimum tick via the CLOB client, defaulting to "0.01" on error.

    `get_tick_size` returns one of the string literals "0.1"/"0.01"/"0.001"/
    "0.0001". We keep it as the string: `round_to_tick` accepts a str fine, and
    v2's `PartialCreateOrderOptions(tick_size=...)` requires exactly this literal
    form (not a float)."""
    try:
        return str(client.get_tick_size(token_id))
    except Exception as e:
        log.warning("[executor] tick-size fetch failed for %s (%s); defaulting to 0.01",
                    token_id, e)
        return "0.01"


def _get_neg_risk(client, token_id: str) -> bool:
    """Whether `token_id` belongs to a negative-risk (multi-outcome) market,
    via the CLOB client; defaults to False on error.

    v2 signs neg-risk orders against a different exchange contract, so
    `PartialCreateOrderOptions(neg_risk=...)` must reflect the market or the
    order is rejected/mis-signed. `get_neg_risk` returns a bool."""
    try:
        return bool(client.get_neg_risk(token_id))
    except Exception as e:
        log.warning("[executor] neg_risk fetch failed for %s (%s); defaulting to False",
                    token_id, e)
        return False


def _diagnose_order_error(exc, token_id, price, size, side, tick_size=None) -> list[str]:
    """Log full detail on a failed live order and guess the likely cause(s).

    Polymarket surfaces order rejections as a ValidationException /
    PolyApiException carrying the HTTP status and the response body in
    `error_msg`. We log the exception, that body, and the exact order
    parameters, then heuristically flag the usual culprits (tick size,
    minimum order size, signature type) so the watch log says *why*."""
    body = getattr(exc, "error_msg", None)
    status = getattr(exc, "status_code", None)
    detail = body if body is not None else str(exc)
    log.error(
        "[executor] LIVE ORDER FAILED %s: side=%s token_id=%s price=%s size=%s "
        "tick_size=%s status_code=%s body=%s",
        type(exc).__name__, side, token_id, price, size, tick_size, status, detail,
    )
    low = str(detail).lower()
    reasons: list[str] = []
    if "tick" in low or ("price" in low and ("min" in low or "max" in low or "valid" in low)):
        reasons.append("tick_size")
    if "min" in low and ("size" in low or "order" in low or "amount" in low or "shares" in low):
        reasons.append("min_order_size")
    if ("signature" in low or "sig type" in low or "signature_type" in low
            or "unauthorized" in low or "not enough balance" in low
            or status in (401, 403)):
        reasons.append("signature_type")
    if reasons:
        log.error("[executor] likely cause(s): %s", ", ".join(reasons))
    else:
        log.error("[executor] cause not auto-classified; inspect body above")
    return reasons


def book_side(book, side: str) -> list:
    """Resting levels for one book side ('bids' or 'asks').

    py_clob_client_v2's `get_order_book` returns either an OrderBookSummary
    object (with `.bids`/`.asks` attributes) or a plain dict
    (`{'bids': [...], 'asks': [...]}`) depending on the release — tolerate both.
    Returns [] when the side is absent or empty."""
    if isinstance(book, dict):
        levels = book.get(side)
    else:
        levels = getattr(book, side, None)
    return levels or []


def book_level_price_size(level) -> tuple[float, float]:
    """(price, size) from a single book level, dict- or attribute-shaped."""
    if isinstance(level, dict):
        return float(level["price"]), float(level["size"])
    return float(level.price), float(level.size)


def _best_levels(book) -> tuple[float | None, float | None, float | None]:
    """(best_bid, best_ask, size_at_best_ask) from a CLOB OrderBookSummary."""
    bids = [book_level_price_size(b) for b in book_side(book, "bids")]
    asks = [book_level_price_size(a) for a in book_side(book, "asks")]
    best_bid = max(p for p, _ in bids) if bids else None
    best_ask_level = min(asks, key=lambda level: level[0]) if asks else None
    if best_ask_level is None:
        return best_bid, None, None
    return best_bid, best_ask_level[0], best_ask_level[1]


def _bid_levels(book) -> tuple[float | None, float | None, float | None]:
    """(best_bid, best_ask, size_at_best_bid) from a CLOB OrderBookSummary —
    the sell-side counterpart of _best_levels (which reports size at best ask)."""
    bids = [book_level_price_size(b) for b in book_side(book, "bids")]
    asks = [book_level_price_size(a) for a in book_side(book, "asks")]
    best_ask = min(p for p, _ in asks) if asks else None
    best_bid_level = max(bids, key=lambda level: level[0]) if bids else None
    if best_bid_level is None:
        return None, best_ask, None
    return best_bid_level[0], best_ask, best_bid_level[1]


def create_clob_client():
    """Build an authenticated Polymarket CLOB client.

    `key` is the SIGNING key (wallet private key). `funder` is the deposit
    wallet ADDRESS that holds the USDC, paired with its signature_type
    (3 = POLY_1271 deposit wallet). Omit both for a plain EOA wallet.
    Raises ImportError when py_clob_client_v2 (an optional dependency) is absent.
    """
    from py_clob_client_v2 import ClobClient

    client_kwargs = dict(
        host=config.POLYMARKET_HOST,
        key=config.POLYMARKET_PRIVATE_KEY,
        chain_id=137,
    )
    if config.POLYMARKET_FUNDER_ADDRESS:
        client_kwargs["funder"] = config.POLYMARKET_FUNDER_ADDRESS
        client_kwargs["signature_type"] = config.POLYMARKET_SIGNATURE_TYPE
    client = ClobClient(**client_kwargs)
    client.set_api_creds(client.create_or_derive_api_key())
    return client


def _resolve_token_id(client, condition_id: str, side: str) -> str | None:
    """Resolve the outcome token_id for `side` of a market via the CLOB client.

    Positions only carry a condition_id (no token list), so ask the exchange.
    Tolerates both dict and attribute-style market/token shapes.

    `side` is our positional YES/NO convention (YES = the first outcome, NO =
    the second), the same way gamma._parse_market labels its tokens and where
    yes_price = outcomePrices[0]. Many markets' *real* CLOB outcome labels are
    NOT "Yes"/"No" — "Up"/"Down", "Bitcoin"/"Ethereum", team names — so a strict
    label match finds nothing and the close silently fails with error_no_token
    (the symptom on "Bitcoin Up or Down"). We therefore try the label first,
    then fall back to position: YES -> first token, NO -> second. The BUY path
    never hit this because it falls back to the cached Gamma Market.tokens (which
    are always labeled "Yes"/"No"); a close has only a condition_id, so it must
    resolve robustly here."""
    market = client.get_market(condition_id)
    tokens = (market.get("tokens") if isinstance(market, dict)
              else getattr(market, "tokens", None)) or []

    def _tok(t):
        return (t.get("token_id") if isinstance(t, dict)
                else getattr(t, "token_id", None))

    def _outcome(t):
        return ((t.get("outcome") if isinstance(t, dict)
                 else getattr(t, "outcome", "")) or "")

    # 1) Exact label match — real Yes/No markets resolve here.
    for t in tokens:
        if _outcome(t).upper() == side.upper():
            tid = _tok(t)
            log.info("[executor] token resolved by label: side=%s outcome=%s token_id=%s",
                     side, _outcome(t), tid)
            return tid

    # 2) Positional fallback for the YES/NO convention on non-Yes/No markets.
    idx = {"YES": 0, "NO": 1}.get(side.upper())
    labels = [_outcome(t) for t in tokens]
    if idx is not None and len(tokens) > idx:
        tid = _tok(tokens[idx])
        log.warning(
            "[executor] no '%s' outcome among %s; resolving positionally to "
            "token[%d]=%s (YES=first / NO=second convention)",
            side, labels, idx, tid,
        )
        return tid

    log.error("[executor] could NOT resolve token for side=%s among outcomes=%s",
              side, labels)
    return None


def close_position_live(
    condition_id: str, side: str, shares: float, max_spread: float | None = None
) -> dict:
    """Flatten `shares` of an open `side` position with a real CLOB SELL.

    Mirrors _execute_live (buy) for the sell direction: build the client,
    resolve the token, read the live book, sell into the best bid (depth- and
    spread-guarded), and return {status, order_id, price, shares}. Status is
    'executed' ONLY when the SELL is confirmed/placed; a 'skipped_*' reason when
    the book can't support the sell; 'close_failed' (with an 'error' detail) when
    the order is rejected/errors; or 'error_*' on setup failure (incl. no
    py_clob_client). Only 'executed' means the position was actually flattened —
    the caller must NOT record a local close on any other status."""
    max_spread = config.LIVE_MAX_SPREAD if max_spread is None else max_spread
    log.info(
        "[executor] close_position_live START: condition_id=%s side=%s shares=%.4f "
        "max_spread=%.3f", condition_id, side, shares, max_spread,
    )
    try:
        from py_clob_client_v2 import OrderArgs, OrderType, Side, PartialCreateOrderOptions

        client = create_clob_client()
        log.info("[executor] close_position_live: CLOB client built; resolving token "
                 "for condition_id=%s side=%s", condition_id, side)
        token_id = _resolve_token_id(client, condition_id, side)
        if not token_id:
            log.error("[executor] close ABORTED: no token_id for side=%s of %s "
                      "(position stays open)", side, condition_id)
            return {"status": "error_no_token", "order_id": None, "price": None, "shares": None}

        # Cap to what we actually own on-chain. The requested `shares` is derived
        # from amount_usd / entry_yes_price, which over-counts (the BUY filled at
        # the higher ask), so an uncapped SELL is rejected "not enough balance".
        held = held_token_shares(client, token_id)
        if held is not None:
            log.info("[executor] on-chain token balance for token=%s: %.4f sh "
                     "(close requested %.4f sh)", token_id, held, shares)
            if shares > held:
                log.warning("[executor] capping SELL %.4f -> %.4f sh to on-chain "
                            "holding for token=%s", shares, held, token_id)
                shares = held

        book = client.get_order_book(token_id)
        best_bid, best_ask, bid_size = _bid_levels(book)
        log.info("[executor] close book for token=%s: best_bid=%s best_ask=%s bid_size=%s",
                 token_id, best_bid, best_ask, bid_size)
        price, sell_shares, status = plan_live_sell(
            shares, best_bid, best_ask, bid_size, max_spread
        )
        if status != "ok":
            log.warning("[executor] close SKIPPED: plan_live_sell -> %s (no SELL placed, "
                        "position stays open)", status)
            return {"status": status, "order_id": None, "price": None, "shares": None}

        tick_size = _get_tick_size(client, token_id)
        neg_risk = _get_neg_risk(client, token_id)
        price = round_to_tick(price, tick_size)
        log.info("[executor] close order ready: SELL token=%s price=%.4f size=%.4f "
                 "tick_size=%s neg_risk=%s", token_id, price, sell_shares, tick_size, neg_risk)

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=sell_shares,
            side=Side.SELL,
        )
        try:
            signed_order = client.create_order(
                order_args, PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk)
            )
            resp = client.post_order(signed_order, OrderType.GTC)
        except Exception as order_exc:
            _diagnose_order_error(order_exc, token_id, price, sell_shares, "SELL", tick_size)
            # The SELL was rejected — the position is still held on-chain. Report
            # close_failed so the caller keeps the local position open.
            return {"status": "close_failed", "order_id": None, "price": None,
                    "shares": None, "error": str(getattr(order_exc, "error_msg", None)
                                                 or order_exc)}
        order_id = resp.get("orderID", resp.get("id", "unknown"))
        log.info("[executor] SELL posted: order_id=%s; reconciling in %ss",
                 order_id, config.ORDER_RECONCILE_WAIT_SECONDS)

        # A posted GTC SELL can rest unfilled (thin/illiquid niche book) — a
        # returned order_id is NOT a fill. Mirror the BUY path: reconcile against
        # the exchange and only claim 'executed' on a confirmed fill, so we never
        # record a close for a sell that didn't actually flatten the position.
        # (A partial sell fill counts as executed and reconcile cancels the rest.)
        fill_status, _ = reconcile_order(client, order_id, sell_shares, price)
        if fill_status == "executed":
            log.info("[executor] close CONFIRMED: SELL %.4f sh token=%s @ %.4f (order %s)",
                     sell_shares, token_id, price, order_id)
            return {"status": "executed", "order_id": order_id, "price": price,
                    "shares": sell_shares}
        # Unconfirmed (resting/unknown): the position is NOT flattened. Cancel the
        # resting order so it can't fill later untracked, and report close_failed
        # so the caller keeps the local position open and retries next cycle.
        log.warning("[executor] close NOT confirmed (reconcile=%s) for order %s — "
                    "cancelling and reporting close_failed; position stays open",
                    fill_status, order_id)
        cancel_order_safe(client, order_id)
        return {"status": "close_failed", "order_id": None, "price": None, "shares": None,
                "error": f"SELL not confirmed (reconcile={fill_status})"}

    except ImportError:
        log.error("[executor] close FAILED: py_clob_client_v2 not installed")
        return {"status": "error_no_clob_client", "order_id": None, "price": None, "shares": None}
    except Exception as e:
        log.error("[executor] close FAILED with %s: %s", type(e).__name__, e, exc_info=True)
        # Any other failure (client build, token resolve, book fetch) means the
        # SELL never went through — surface close_failed, not a recorded close.
        return {"status": "close_failed", "order_id": None, "price": None,
                "shares": None, "error": f"{type(e).__name__}: {e}"}


def held_token_shares(client, token_id: str) -> float | None:
    """Real outcome-token (CTF ERC1155) balance held for `token_id`, in shares,
    or None on any error.

    A live BUY fills at the best ASK, but a position's share count is later
    re-derived as amount_usd / entry_yes_price — and entry_yes_price (the cached
    Gamma mid at open) is BELOW the ask we actually paid, so that quotient
    OVER-counts the tokens we own. Selling that inflated count makes the CLOB
    reject the close as "not enough balance / allowance" (e.g. balance 60.0,
    order 66.64), and the position never flattens. The close caps the SELL to
    this real holding so the order is always sellable. Best-effort: returns None
    (no cap) when the client lacks the call or it errors, leaving prior behaviour
    unchanged."""
    try:
        from py_clob_client_v2 import BalanceAllowanceParams, AssetType

        resp = client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=token_id)
        )
        raw = (resp.get("balance") if isinstance(resp, dict)
               else getattr(resp, "balance", None))
        if raw is None:
            return None
        return float(raw) / 1_000_000
    except Exception as e:
        log.warning("[executor] token balance fetch failed for %s (%s); SELL not "
                    "capped to on-chain holding", token_id, e)
        return None


def get_live_balance() -> float | None:
    """Real USDC collateral balance on Polymarket, in USD, or None on any error.

    The CLOB API reports balances in USDC base units (6 decimals), so the raw
    figure is divided by 1_000_000. Fails closed (returns None) when
    py_clob_client is absent or the call errors, so the Telegram balance view
    never crashes on a live-mode fetch."""
    try:
        from py_clob_client_v2 import BalanceAllowanceParams, AssetType

        client = create_clob_client()
        resp = client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        raw = (resp.get("balance") if isinstance(resp, dict)
               else getattr(resp, "balance", None))
        if raw is None:
            return None
        return float(raw) / 1_000_000
    except Exception as e:
        log.warning(f"[executor] live balance fetch failed: {e}")
        return None


def _field(obj, *names):
    """Read the first present field from a dict- or attribute-shaped object."""
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            return getattr(obj, name)
    return None


def _fetch_order(client, order_id: str):
    """Fetch a single order's state, tolerant of v2 client API differences.

    The v2 client exposes `get_order(order_id)`, but the method surface has
    shifted across releases. If `get_order` is missing we fall back to listing
    open orders (`get_open_orders`/`get_orders`) and matching by id — both id
    spellings ('id'/'orderID'/'order_id') are accepted. Returns the order
    object/dict, or None if nothing matches. Raises AttributeError only when the
    client offers no order-query method at all (so the caller can log it)."""
    getter = getattr(client, "get_order", None)
    if callable(getter):
        return getter(order_id)

    lister = (getattr(client, "get_open_orders", None)
              or getattr(client, "get_orders", None))
    if not callable(lister):
        raise AttributeError(
            "CLOB client exposes neither get_order nor get_open_orders/get_orders"
        )
    orders = lister() or []
    for o in orders:
        oid = _field(o, "id", "orderID", "order_id")
        if oid is not None and str(oid) == str(order_id):
            return o
    return None


def cancel_order_safe(client, order_id: str) -> bool:
    """Best-effort cancel of a resting order via the v2 cancel API.

    v2's `cancel_order` takes an `OrderPayload(orderID=...)`, not a bare string;
    passing a string raises (the payload is dataclass-typed). OrderPayload lives
    in `py_clob_client_v2.clob_types` (re-exported at the top level), so we try
    both. Returns True on a clean cancel, False on any failure (missing method,
    already gone, network) — callers treat cancellation as best-effort cleanup."""
    try:
        try:
            from py_clob_client_v2.clob_types import OrderPayload
        except ImportError:
            from py_clob_client_v2 import OrderPayload
        client.cancel_order(OrderPayload(orderID=order_id))
        log.info("[executor] cancelled resting order %s", order_id)
        return True
    except Exception as e:
        log.warning("[executor] cancel of order %s failed (%s): %s",
                    order_id, type(e).__name__, e)
        return False


def reconcile_order(client, order_id: str, total_shares: float | None = None,
                    price: float | None = None) -> tuple[str, float | None]:
    """Re-query a just-posted GTC order to learn its real fill, returning
    (status, filled_cost_usd).

    A GTC order can rest fully OR partially unfilled — post_order returning an
    order_id is not a fill, and the maker amount it echoes back is the *order*
    size, not what actually filled (the partial-fill bug: a $32.81 order that
    fills $4.41 would otherwise open a $32.81 position and leave the rest live).
    After ORDER_RECONCILE_WAIT_SECONDS we ask the exchange for the real matched
    size and map it (matching is case-insensitive; the v2 fill field has several
    spellings):
      - terminal MATCHED/FILLED status            -> ("executed", full filled cost)
      - 0 < matched < total, order still resting   -> cancel the unfilled remainder
        (so it can't fill later untracked) and report ("executed", partial cost)
      - matched == 0 (resting) / unknown status    -> cancel and ("resting", None)
      - order not found / query error              -> ("error_not_found", None)

    filled_cost_usd is the real notional that filled (filled_shares * price), so a
    partial fill opens a position sized to what we actually own — None when nothing
    filled, or when total_shares/price weren't supplied (legacy callers)."""
    time.sleep(config.ORDER_RECONCILE_WAIT_SECONDS)
    try:
        order = _fetch_order(client, order_id)
    except Exception as e:
        log.error("[executor] order reconcile failed for %s: %s", order_id, e)
        return "error_not_found", None
    if not order:
        log.error("[executor] order %s not found on reconcile", order_id)
        return "error_not_found", None

    status = (_field(order, "status") or "").upper()
    # v2 fill field varies: size_matched (snake) / filledAmount (camel) /
    # takingAmount (outcome tokens received on a BUY) / etc.
    filled_raw = _field(order, "size_matched", "filled_size", "matched_size",
                        "filledAmount", "filled", "takingAmount")
    try:
        filled = float(filled_raw) if filled_raw is not None else 0.0
    except (TypeError, ValueError):
        filled = 0.0

    def _cost(shares: float | None) -> float | None:
        if shares and shares > 0 and price:
            return round(shares * price, 2)
        return None

    # Terminal fill: the whole order matched. A MATCHED/FILLED status with no
    # size field reported means the full order size filled.
    if status in ("MATCHED", "FILLED"):
        filled_shares = filled if filled > 0 else (total_shares or 0.0)
        log.info("[executor] order %s reconciled: filled (status=%s, filled=%s sh)",
                 order_id, status, filled_shares)
        return "executed", _cost(filled_shares)

    # Partial fill still resting on the book: keep the filled part, cancel the rest.
    if filled > 0:
        if total_shares is not None and filled + 1e-9 < total_shares:
            log.warning(
                "[executor] order %s PARTIALLY filled: %.4f of %.4f sh — cancelling "
                "the unfilled remainder, opening on the filled part",
                order_id, filled, total_shares,
            )
            cancel_order_safe(client, order_id)
        else:
            log.info("[executor] order %s reconciled: filled %.4f sh", order_id, filled)
        return "executed", _cost(filled)

    # Nothing filled — resting or an unknown status. Cancel so it can't fill later
    # untracked; we open no position for it.
    if status == "LIVE":
        log.warning("[executor] order %s reconciled: resting/unfilled on the book",
                    order_id)
    else:
        log.warning("[executor] order %s reconciled: status=%s, no fill",
                    order_id, status or "?")
    cancel_order_safe(client, order_id)
    return "resting", None


def _execute_live(signal: Signal) -> dict:
    """Place a real order via Polymarket CLOB client, orderbook-aware."""
    try:
        from py_clob_client_v2 import OrderArgs, OrderType, Side, PartialCreateOrderOptions

        client = create_clob_client()

        # Resolve the token from the CLOB itself (get_market), not the cached
        # Gamma token list: the CLOB is the source of truth for the token_id we
        # sign against. Fall back to the Gamma token only if the CLOB has none.
        token_id = _resolve_token_id(client, signal.market.condition_id, signal.side)
        if not token_id:
            token_id = get_token_id(signal.market, signal.side)
        if not token_id:
            return _log_and_return(signal, status="error_no_token", order_id=None)

        # Check the live book instead of trusting the cached watcher price:
        # on thin niche books a stale price is a non-fill or a bad fill.
        book = client.get_order_book(token_id)
        best_bid, best_ask, ask_size = _best_levels(book)
        price, shares, status = plan_live_order(
            signal.bet_amount, best_bid, best_ask, ask_size, config.LIVE_MAX_SPREAD
        )
        if status != "ok":
            return _log_and_return(signal, status=status, order_id=None)

        # Polymarket rejects off-tick prices with a ValidationException; snap
        # the price onto the market's tick grid before signing. neg_risk picks
        # the right exchange contract for multi-outcome markets.
        tick_size = _get_tick_size(client, token_id)
        neg_risk = _get_neg_risk(client, token_id)
        price = round_to_tick(price, tick_size)

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=shares,
            side=Side.BUY,
        )

        try:
            signed_order = client.create_order(
                order_args, PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk)
            )
            resp = client.post_order(signed_order, OrderType.GTC)
        except Exception as order_exc:
            _diagnose_order_error(order_exc, token_id, price, shares, "BUY", tick_size)
            raise

        order_id = resp.get("orderID", resp.get("id", "unknown"))
        # A posted GTC order may rest fully OR partially unfilled — reconcile
        # against the exchange before claiming a fill. reconcile_order returns the
        # real filled cost (filled_shares * price): for a partial fill it cancels
        # the unfilled remainder and reports only what filled, so the position is
        # sized to what we actually own (not the nominal bet). An unfilled order is
        # cancelled and carries no cost basis (None).
        status, actual_cost = reconcile_order(client, order_id, shares, price)
        # Real filled share count = filled cost / fill price. Stored on the
        # position so a later live SELL flattens exactly what we own, instead of
        # re-deriving shares from amount_usd / entry_yes_price (which over-counts:
        # the BUY filled at the higher ask). None when nothing filled.
        actual_shares = (round(actual_cost / price, 4)
                         if actual_cost is not None and price else None)
        return _log_and_return(signal, status=status, order_id=order_id,
                               actual_cost_usd=actual_cost,
                               actual_shares=actual_shares)

    except ImportError:
        return _log_and_return(signal, status="error_no_clob_client", order_id=None)
    except AttributeError as e:
        # The v2 client surface differs from a method/attr we call above. Log the
        # full traceback so the exact missing attribute is visible in watch.log.
        log.error(
            "[executor] AttributeError in live execution (py_clob_client_v2 API "
            "mismatch): %s", e, exc_info=True,
        )
        return _log_and_return(signal, status="error_AttributeError", order_id=None)
    except Exception as e:
        return _log_and_return(signal, status=f"error_{type(e).__name__}", order_id=None)


def _log_and_return(signal: Signal, status: str, order_id: str | None,
                    actual_cost_usd: float | None = None,
                    actual_shares: float | None = None) -> dict:
    """Log trade to SQLite and return result dict.

    `actual_cost_usd` is the real USD filled on a live BUY (None for dry-run or
    unfilled orders); it flows into the open-position notification and the result
    dict so positions.open_position can store it as the PnL cost basis.
    `actual_shares` is the real filled token count (None for dry-run/unfilled),
    stored as the position's token_count so a later live SELL flattens exactly
    what we own."""
    trade_id = logger.log_trade(
        market_id=signal.market.condition_id,
        market_question=signal.market.question,
        claude_score=signal.claude_score,
        market_price=signal.market_price,
        edge=signal.edge,
        side=signal.side,
        amount_usd=signal.bet_amount,
        order_id=order_id,
        status=status,
        reasoning=signal.reasoning,
        headlines=signal.headlines,
        news_source=signal.news_source,
        classification=signal.classification,
        materiality=signal.materiality,
        news_latency_ms=signal.news_latency_ms,
        classification_latency_ms=signal.classification_latency_ms,
        total_latency_ms=signal.total_latency_ms,
        edge_type=signal.edge_type,
        confidence=signal.confidence,
        event_id=getattr(signal.market, "event_id", "") or None,
    )

    # Real-time notification on a position actually being taken (dry-run or live).
    if status in ("dry_run", "executed"):
        telegram_bot.notify_position_opened({
            "market_question": signal.market.question,
            "side": signal.side,
            "entry_yes_price": signal.market_price,
            "amount_usd": signal.bet_amount,
            "actual_cost_usd": actual_cost_usd,
            "edge": signal.edge,
            "confidence": signal.confidence,
        })

    return {
        "trade_id": trade_id,
        "market": signal.market.question,
        "side": signal.side,
        "amount": signal.bet_amount,
        "actual_cost_usd": actual_cost_usd,
        "actual_shares": actual_shares,
        "edge": signal.edge,
        "status": status,
        "order_id": order_id,
        "classification": signal.classification,
        "materiality": signal.materiality,
        "latency_ms": signal.total_latency_ms,
    }
