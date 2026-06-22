from __future__ import annotations

import asyncio
import logging

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
      the visible depth at best ask; under $1 notional -> "skipped_thin_book".
    """
    if best_ask is None or best_ask <= 0:
        return None, None, "skipped_empty_book"
    if best_bid is not None and (best_ask - best_bid) > max_spread:
        return None, None, "skipped_wide_spread"

    shares = bet_usd / best_ask
    if ask_size_shares is not None:
        shares = min(shares, ask_size_shares)
    if shares * best_ask < 1.0:
        return None, None, "skipped_thin_book"
    return best_ask, round(shares, 2), "ok"


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
    depth at best bid. Same guards as a buy:
    - No bids -> nothing to sell into ("skipped_empty_book").
    - Spread wider than max_spread -> skip ("skipped_wide_spread").
    - Under $1 notional after depth downsizing -> "skipped_thin_book".
    """
    if best_bid is None or best_bid <= 0:
        return None, None, "skipped_empty_book"
    if best_ask is not None and (best_ask - best_bid) > max_spread:
        return None, None, "skipped_wide_spread"

    sell_shares = shares
    if bid_size_shares is not None:
        sell_shares = min(sell_shares, bid_size_shares)
    if sell_shares * best_bid < 1.0:
        return None, None, "skipped_thin_book"
    return best_bid, round(sell_shares, 2), "ok"


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


def _get_tick_size(client, token_id: str) -> float:
    """Market minimum tick via the CLOB client, defaulting to 0.01 on error.

    `get_tick_size` returns a string ("0.01"/"0.001"); we keep it as a float
    for arithmetic but the original is fine to hand back to `round_to_tick`."""
    try:
        return float(client.get_tick_size(token_id))
    except Exception as e:
        log.warning("[executor] tick-size fetch failed for %s (%s); defaulting to 0.01",
                    token_id, e)
        return 0.01


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


def _best_levels(book) -> tuple[float | None, float | None, float | None]:
    """(best_bid, best_ask, size_at_best_ask) from a CLOB OrderBookSummary."""
    bids = [(float(b.price), float(b.size)) for b in (book.bids or [])]
    asks = [(float(a.price), float(a.size)) for a in (book.asks or [])]
    best_bid = max(p for p, _ in bids) if bids else None
    best_ask_level = min(asks, key=lambda level: level[0]) if asks else None
    if best_ask_level is None:
        return best_bid, None, None
    return best_bid, best_ask_level[0], best_ask_level[1]


def _bid_levels(book) -> tuple[float | None, float | None, float | None]:
    """(best_bid, best_ask, size_at_best_bid) from a CLOB OrderBookSummary —
    the sell-side counterpart of _best_levels (which reports size at best ask)."""
    bids = [(float(b.price), float(b.size)) for b in (book.bids or [])]
    asks = [(float(a.price), float(a.size)) for a in (book.asks or [])]
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
    Raises ImportError when py_clob_client (an optional dependency) is absent.
    """
    from py_clob_client.client import ClobClient

    client_kwargs = dict(
        host=config.POLYMARKET_HOST,
        key=config.POLYMARKET_PRIVATE_KEY,
        chain_id=137,
    )
    if config.POLYMARKET_FUNDER_ADDRESS:
        client_kwargs["funder"] = config.POLYMARKET_FUNDER_ADDRESS
        client_kwargs["signature_type"] = config.POLYMARKET_SIGNATURE_TYPE
    client = ClobClient(**client_kwargs)
    client.set_api_creds(client.create_or_derive_api_creds())
    return client


def _resolve_token_id(client, condition_id: str, side: str) -> str | None:
    """Resolve the outcome token_id for `side` of a market via the CLOB client.

    Positions only carry a condition_id (no token list), so ask the exchange.
    Tolerates both dict and attribute-style market/token shapes."""
    market = client.get_market(condition_id)
    tokens = (market.get("tokens") if isinstance(market, dict)
              else getattr(market, "tokens", None)) or []
    for t in tokens:
        outcome = (t.get("outcome") if isinstance(t, dict)
                   else getattr(t, "outcome", "")) or ""
        if outcome.upper() == side.upper():
            return (t.get("token_id") if isinstance(t, dict)
                    else getattr(t, "token_id", None))
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
    try:
        from py_clob_client.clob_types import OrderArgs, OrderType

        client = create_clob_client()
        token_id = _resolve_token_id(client, condition_id, side)
        if not token_id:
            return {"status": "error_no_token", "order_id": None, "price": None, "shares": None}

        book = client.get_order_book(token_id)
        best_bid, best_ask, bid_size = _bid_levels(book)
        price, sell_shares, status = plan_live_sell(
            shares, best_bid, best_ask, bid_size, max_spread
        )
        if status != "ok":
            return {"status": status, "order_id": None, "price": None, "shares": None}

        tick_size = _get_tick_size(client, token_id)
        price = round_to_tick(price, tick_size)

        order_args = OrderArgs(
            price=price,
            size=sell_shares,
            side="SELL",
            token_id=token_id,
        )
        try:
            signed_order = client.create_order(order_args)
            resp = client.post_order(signed_order, OrderType.GTC)
        except Exception as order_exc:
            _diagnose_order_error(order_exc, token_id, price, sell_shares, "SELL", tick_size)
            # The SELL was rejected — the position is still held on-chain. Report
            # close_failed so the caller keeps the local position open.
            return {"status": "close_failed", "order_id": None, "price": None,
                    "shares": None, "error": str(getattr(order_exc, "error_msg", None)
                                                 or order_exc)}
        order_id = resp.get("orderID", resp.get("id", "unknown"))
        return {"status": "executed", "order_id": order_id, "price": price, "shares": sell_shares}

    except ImportError:
        return {"status": "error_no_clob_client", "order_id": None, "price": None, "shares": None}
    except Exception as e:
        # Any other failure (client build, token resolve, book fetch) means the
        # SELL never went through — surface close_failed, not a recorded close.
        return {"status": "close_failed", "order_id": None, "price": None,
                "shares": None, "error": f"{type(e).__name__}: {e}"}


def get_live_balance() -> float | None:
    """Real USDC collateral balance on Polymarket, in USD, or None on any error.

    The CLOB API reports balances in USDC base units (6 decimals), so the raw
    figure is divided by 1_000_000. Fails closed (returns None) when
    py_clob_client is absent or the call errors, so the Telegram balance view
    never crashes on a live-mode fetch."""
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

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


def _execute_live(signal: Signal) -> dict:
    """Place a real order via Polymarket CLOB client, orderbook-aware."""
    try:
        from py_clob_client.clob_types import OrderArgs, OrderType

        client = create_clob_client()

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
        # the price onto the market's tick grid before signing.
        tick_size = _get_tick_size(client, token_id)
        price = round_to_tick(price, tick_size)

        order_args = OrderArgs(
            price=price,
            size=shares,
            side="BUY",
            token_id=token_id,
        )

        try:
            signed_order = client.create_order(order_args)
            resp = client.post_order(signed_order, OrderType.GTC)
        except Exception as order_exc:
            _diagnose_order_error(order_exc, token_id, price, shares, "BUY", tick_size)
            raise

        order_id = resp.get("orderID", resp.get("id", "unknown"))
        return _log_and_return(signal, status="executed", order_id=order_id)

    except ImportError:
        return _log_and_return(signal, status="error_no_clob_client", order_id=None)
    except Exception as e:
        return _log_and_return(signal, status=f"error_{type(e).__name__}", order_id=None)


def _log_and_return(signal: Signal, status: str, order_id: str | None) -> dict:
    """Log trade to SQLite and return result dict."""
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
            "edge": signal.edge,
            "confidence": signal.confidence,
        })

    return {
        "trade_id": trade_id,
        "market": signal.market.question,
        "side": signal.side,
        "amount": signal.bet_amount,
        "edge": signal.edge,
        "status": status,
        "order_id": order_id,
        "classification": signal.classification,
        "materiality": signal.materiality,
        "latency_ms": signal.total_latency_ms,
    }
