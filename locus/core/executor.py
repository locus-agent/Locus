from __future__ import annotations

import asyncio

from locus import config
from locus.memory import logger
from locus.core.edge import Signal
from locus.core import telegram_bot
from locus.markets.gamma import get_token_id


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
    'executed' on a placed order, a 'skipped_*' reason when the book can't
    support the sell, or 'error_*' on failure (incl. no py_clob_client)."""
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

        order_args = OrderArgs(
            price=price,
            size=sell_shares,
            side="SELL",
            token_id=token_id,
        )
        signed_order = client.create_order(order_args)
        resp = client.post_order(signed_order, OrderType.GTC)
        order_id = resp.get("orderID", resp.get("id", "unknown"))
        return {"status": "executed", "order_id": order_id, "price": price, "shares": sell_shares}

    except ImportError:
        return {"status": "error_no_clob_client", "order_id": None, "price": None, "shares": None}
    except Exception as e:
        return {"status": f"error_{type(e).__name__}", "order_id": None, "price": None, "shares": None}


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

        order_args = OrderArgs(
            price=price,
            size=shares,
            side="BUY",
            token_id=token_id,
        )

        signed_order = client.create_order(order_args)
        resp = client.post_order(signed_order, OrderType.GTC)

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
