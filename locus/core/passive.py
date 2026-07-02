"""
Passive limit-order entry (flag-gated subsystem, PASSIVE_LIMIT_ENABLED).

Every aggressive entry takes the ask and pays the spread — positions open at
-2..-5% instantly. Academic evidence (SSRN, $67B of Polymarket volume) shows
profit concentrates in liquidity PROVIDERS. For long-horizon markets there is
no reason to pay for speed: place a GTC limit BUY inside the spread (one tick
better than the best bid, capped one tick below the ask so it never crosses)
and wait to be filled as the maker.

Routing (executor.execute_trade): only when PASSIVE_LIMIT_ENABLED is true AND
the market resolves more than PASSIVE_MIN_HOURS_TO_RESOLUTION out. Everything
else — breaking-news / short-horizon markets, unknown end dates, dry-run —
keeps today's aggressive path byte-for-byte. Dry-run in particular never
routes here: its simulated instant fill at the cached price IS the limit-price
approximation absent a book, so flag-on dry-run behaves exactly like today.

CRITICAL divergence from the aggressive path: executor.reconcile_order CANCELS
resting orders (the phantom-position defense — a resting aggressive order is a
mistake). A passive order is SUPPOSED to rest, so placement never goes through
reconcile_order. Instead the order is persisted in the pending_orders table
and managed by check_pending_orders (called from the pipeline's management
cycle, and once at startup for crash recovery).

State machine (pending_orders.status; trades.status in parentheses):

    place GTC at bid + PASSIVE_PRICE_IMPROVE_TICKS  (trades: 'passive_pending';
      headline reservation HELD by the pipeline while the row is pending)
        |
    [pending] --- full fill seen ------------------> [filled]  (trades: 'executed')
        |            open the position exactly like an aggressive fill:
        |            actual_cost = filled_shares * limit, token_count = filled
        |            shares; reservation committed permanently.
        |
        +--- timeout (expires_at) -> cancel remainder:
        |        fill >= MIN_FILL_USD  -> [filled] (trades: 'executed'), as above
        |        0 < fill < MIN_FILL   -> [dust]   (trades: 'dust_fill');
        |                                 dust sold back best-effort; RELEASE
        |        no fill               -> [expired] (trades: 'passive_expired');
        |                                 RELEASE
        |
        +--- book ran away (best ask > limit * (1 + PASSIVE_CHASE_AWAY_PCT%))
        |        -> cancel remainder; same partial-fill split as timeout, with
        |           the no-fill case -> [chased_away] (trades:
        |           'passive_chased_away'); RELEASE
        |
        +--- order vanished from the CLOB (filled-and-purged, or cancelled
                 externally) -> resolve by on-chain token balance:
                 held > 0  -> treat as filled (late fill, e.g. while we were
                              down) -> [filled]
                 held == 0 -> [expired]; RELEASE
                 unverifiable -> stay [pending], retry next cycle

RELEASE means the row's headline is returned in the lifecycle summary's
released_headlines; the pipeline (which owns the in-memory reservation set)
discards it there, so a later candidate for the same headline may trade.

Crash safety: pending_orders lives in trades.db. On startup,
reconcile_on_startup() runs one lifecycle pass (fills that happened while we
were down open the position late with the real fill data; vanished orders
release) and returns the headlines of still-pending rows so the pipeline can
re-reserve them into its fresh in-memory set.

Known, accepted limitation: a resting passive order is not an open position,
so event/category exposure gates don't count it while it rests (it DOES count
against the daily spend limit via trades.status='passive_pending').
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from locus import config
from locus.memory import logger
from locus.markets.gamma import Market, get_token_id
from locus.core import executor, positions, telegram_bot

log = logging.getLogger(__name__)

# Fill-size comparison slack: matched sizes come back as strings of the snapped
# order size, so anything within this of the ordered shares is a full fill.
_FULL_FILL_EPS = 1e-4


def routes_passive(signal) -> bool:
    """Whether this signal should enter passively: flag on AND the market
    resolves further out than PASSIVE_MIN_HOURS_TO_RESOLUTION. Unknown end
    dates route aggressive (today's path) — never guess a horizon."""
    if not config.PASSIVE_LIMIT_ENABLED:
        return False
    hours = positions.hours_to_close(getattr(signal.market, "end_date", "") or "")
    return hours is not None and hours > config.PASSIVE_MIN_HOURS_TO_RESOLUTION


def _passive_limit_price(best_bid: float, best_ask: float | None,
                         tick: float) -> float:
    """The passive entry price: PASSIVE_PRICE_IMPROVE_TICKS better than the
    best bid, but never at/above the ask (that would cross the spread and
    take). On a one-tick-wide book this degrades to joining the best bid."""
    limit = best_bid + config.PASSIVE_PRICE_IMPROVE_TICKS * tick
    if best_ask is not None and limit >= best_ask:
        limit = best_ask - tick
    if limit < best_bid:
        limit = best_bid
    return limit


def place_passive_order(signal) -> dict:
    """Place the passive GTC entry and persist it as a pending order.

    Returns an executor-shaped result dict. Status 'passive_pending' means the
    order is resting and recorded; the pipeline keeps the headline reservation
    alive for it and must NOT open a position. Any placement failure returns
    the same statuses the aggressive path would (error_* / skipped_thin_book),
    which release the reservation as usual. A book with no bid at all falls
    back to the aggressive path — there is no maker price to improve on."""
    market = signal.market
    try:
        from py_clob_client_v2 import OrderArgs, OrderType, Side, PartialCreateOrderOptions

        client = executor.create_clob_client()
        token_id = (executor._resolve_token_id(client, market.condition_id, signal.side)
                    or get_token_id(market, signal.side))
        if not token_id:
            return executor._log_and_return(signal, status="error_no_token", order_id=None)

        book = client.get_order_book(token_id)
        best_bid, best_ask, _bid_size = executor._bid_levels(book)
        if best_bid is None or best_bid <= 0:
            log.info(
                "[passive] no bid to improve on for \"%s\" — falling back to "
                "the aggressive path", market.question[:40],
            )
            return executor._execute_live(signal)

        tick_str = executor._get_tick_size(client, token_id)
        tick = float(tick_str)
        limit = executor.round_to_tick(
            _passive_limit_price(best_bid, best_ask, tick), tick_str
        )
        shares = executor.round_size_for_clob(limit, signal.bet_amount / limit)
        if shares < config.MIN_ORDER_SHARES or shares * limit < config.MIN_ORDER_USD:
            log.info(
                "[passive] order below exchange minimums (%.2f sh @ %.4f) on "
                "\"%s\" — skipped", shares, limit, market.question[:40],
            )
            return executor._log_and_return(signal, status="skipped_thin_book", order_id=None)

        neg_risk = executor._get_neg_risk(client, token_id)
        order_args = OrderArgs(token_id=token_id, price=limit, size=shares, side=Side.BUY)
        try:
            signed = client.create_order(
                order_args, PartialCreateOrderOptions(tick_size=tick_str, neg_risk=neg_risk)
            )
            resp = client.post_order(signed, OrderType.GTC)
        except Exception as order_exc:
            executor._diagnose_order_error(order_exc, token_id, limit, shares, "BUY", tick_str)
            return executor._log_and_return(
                signal, status=f"error_{type(order_exc).__name__}", order_id=None
            )
        order_id = resp.get("orderID", resp.get("id", "unknown"))
    except ImportError:
        return executor._log_and_return(signal, status="error_no_clob_client", order_id=None)
    except Exception as e:
        log.error("[passive] placement failed with %s: %s", type(e).__name__, e,
                  exc_info=True)
        return executor._log_and_return(
            signal, status=f"error_{type(e).__name__}", order_id=None
        )

    # The order rests on the book ON PURPOSE — do NOT reconcile_order it (that
    # path cancels resting orders). Log the trade as passive_pending (counts
    # against the daily spend limit) and persist the pending row; the
    # management-cycle lifecycle takes it from here.
    result = executor._log_and_return(signal, status="passive_pending", order_id=order_id)
    entry_yes = limit if signal.side == "YES" else round(1.0 - limit, 4)
    expires_at = (
        datetime.now(timezone.utc)
        + timedelta(hours=config.PASSIVE_LIMIT_TIMEOUT_HOURS)
    ).isoformat()
    pending_id = logger.insert_pending_order(
        order_id=order_id,
        trade_id=result["trade_id"],
        condition_id=market.condition_id,
        market_question=market.question,
        slug=getattr(market, "slug", "") or None,
        side=signal.side,
        limit_price=limit,
        shares=shares,
        bet_amount=signal.bet_amount,
        entry_yes_price=entry_yes,
        headline=signal.headlines,
        reasoning=signal.reasoning,
        news_source=signal.news_source,
        event_id=getattr(market, "event_id", "") or None,
        category=getattr(market, "category", "") or None,
        end_date=getattr(market, "end_date", "") or None,
        expires_at=expires_at,
    )
    log.info(
        "[passive] resting entry placed: %s %.2f sh @ %.4f (bid was %.4f) on "
        "\"%s\" — order %s, pending row %s, expires %s",
        signal.side, shares, limit, best_bid, market.question[:40],
        order_id, pending_id, expires_at,
    )
    result["limit_price"] = limit
    result["shares"] = shares
    result["pending_id"] = pending_id
    return result


def _fill_size(order) -> float:
    """Matched size (shares) reported on a CLOB order, 0.0 when absent —
    the same field tolerance as executor.reconcile_order."""
    raw = executor._field(order, "size_matched", "filled_size", "matched_size",
                          "filledAmount", "filled", "takingAmount")
    try:
        return float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _market_from_row(row: dict) -> Market:
    """Rebuild a lightweight Market from a pending row, so a fill can open its
    position even when the market has left tracked_markets (or we restarted)."""
    entry_yes = row.get("entry_yes_price") or 0.5
    return Market(
        row["condition_id"], row.get("market_question") or "",
        row.get("category") or "", entry_yes, round(1.0 - entry_yes, 4),
        0.0, row.get("end_date") or "", True, [],
        slug=row.get("slug") or "", event_id=row.get("event_id") or "",
    )


def _open_from_fill(row: dict, filled_shares: float, summary: dict) -> None:
    """A pending order (fully or viably-partially) filled: open the position
    with the same accounting an aggressive fill gets — actual_cost_usd is the
    real filled notional, token_count the real share count — flip the trades
    row to 'executed', and resolve the pending row. The headline reservation
    is committed (NOT released)."""
    filled_shares = min(filled_shares, row["shares"])
    cost = round(filled_shares * row["limit_price"], 4)
    if not logger.resolve_pending_order(row["id"], "filled", cost, filled_shares):
        return  # a concurrent pass already resolved this row
    market = _market_from_row(row)
    positions.open_position(
        row["trade_id"], market, row["side"], row["bet_amount"],
        headline=row.get("headline") or "",
        reasoning=row.get("reasoning") or "",
        actual_cost_usd=cost, token_count=filled_shares,
    )
    logger.update_trade_status(row["trade_id"], "executed")
    telegram_bot.notify_position_opened({
        "market_question": market.question,
        "side": row["side"],
        "entry_yes_price": market.yes_price,
        "amount_usd": row["bet_amount"],
        "actual_cost_usd": cost,
        "edge": None,
        "confidence": None,
    })
    summary["filled"] += 1
    log.info(
        "[passive] FILLED: %s %.2f sh @ %.4f ($%.2f) on \"%s\" — position "
        "opened (order %s)", row["side"], filled_shares, row["limit_price"],
        cost, (row.get("market_question") or "")[:40], row["order_id"],
    )


def _release(row: dict, summary: dict) -> None:
    """Queue the row's headline for release by the pipeline (which owns the
    in-memory reservation set)."""
    headline = row.get("headline")
    if headline:
        summary["released_headlines"].append(headline)


def _finish_partial(row: dict, filled_shares: float, summary: dict,
                    cause: str) -> None:
    """Terminal handling after the remainder was cancelled (timeout or
    chase-away): a viable partial fill opens the position; a sub-MIN_FILL_USD
    fill is dust (sold back best-effort, like the aggressive dust guard);
    nothing filled resolves to `cause` and releases the headline."""
    cost = filled_shares * row["limit_price"]
    if filled_shares > 0 and cost >= config.MIN_FILL_USD:
        _open_from_fill(row, filled_shares, summary)
        return
    if filled_shares > 0:
        # Dust: too small to manage. Sell it back at the bid, best-effort —
        # the exchange minimums may make the sell unplaceable, in which case
        # the tokens stay held but unmanaged (same as the aggressive guard).
        if logger.resolve_pending_order(row["id"], "dust",
                                        round(cost, 4), filled_shares):
            sellback = executor.close_position_live(
                row["condition_id"], row["side"], filled_shares
            )
            log.warning(
                "[passive] dust fill at %s: %.4f sh ($%.2f) < MIN_FILL_USD on "
                "\"%s\" — sell-back %s", cause, filled_shares, cost,
                (row.get("market_question") or "")[:40], sellback.get("status"),
            )
            logger.update_trade_status(row["trade_id"], "dust_fill")
            summary["dust"] += 1
            _release(row, summary)
        return
    if logger.resolve_pending_order(row["id"], cause):
        logger.update_trade_status(row["trade_id"], f"passive_{cause}")
        summary[cause] += 1
        _release(row, summary)
        log.info(
            "[passive] passive_%s: order %s on \"%s\" cancelled unfilled — "
            "headline released", cause, row["order_id"],
            (row.get("market_question") or "")[:40],
        )


def _held_for_row(client, row: dict) -> float | None:
    """On-chain token balance for a pending row's outcome token — the source
    of truth for how much actually filled. None when unverifiable (token
    unresolvable, balance call unavailable, or any error)."""
    try:
        token_id = executor._resolve_token_id(client, row["condition_id"], row["side"])
        return executor.held_token_shares(client, token_id) if token_id else None
    except Exception as e:
        log.warning("[passive] balance check failed for row %s (%s)", row["id"], e)
        return None


def _resolve_vanished(client, row: dict, summary: dict) -> None:
    """The order is gone from the CLOB (not found): either it filled and was
    purged from the query surface, or something cancelled it externally.
    Resolve by the on-chain token balance — the source of truth."""
    held = _held_for_row(client, row)
    if held is None:
        log.warning(
            "[passive] order %s vanished and the token balance is "
            "unverifiable — leaving row %s pending, retrying next cycle",
            row["order_id"], row["id"],
        )
        return
    if held > _FULL_FILL_EPS:
        # Balance may include tokens from other holdings on this market; cap
        # to the ordered size (logged so the ambiguity is visible).
        log.warning(
            "[passive] order %s vanished with %.4f sh held on-chain — "
            "treating as filled (capped to the %.4f sh ordered)",
            row["order_id"], held, row["shares"],
        )
        _open_from_fill(row, held, summary)
        return
    if logger.resolve_pending_order(row["id"], "expired"):
        logger.update_trade_status(row["trade_id"], "passive_expired")
        summary["expired"] += 1
        _release(row, summary)
        log.warning(
            "[passive] order %s vanished with nothing held — released "
            "(row %s expired)", row["order_id"], row["id"],
        )


def _check_one(client, row: dict, now: datetime, summary: dict) -> None:
    """One lifecycle step for one pending row (see the module state machine)."""
    try:
        order = executor._fetch_order(client, row["order_id"])
    except Exception as e:
        log.warning("[passive] order query failed for %s (%s); retrying next "
                    "cycle", row["order_id"], e)
        return
    if not order:
        _resolve_vanished(client, row, summary)
        return

    status = str(executor._field(order, "status") or "").upper()
    filled = _fill_size(order)

    # Full fill: FILLED is terminal; MATCHED at (or within eps of) the ordered
    # size is complete too. A LIVE order is NEVER a fill regardless of any
    # echoed size field — resting orders echo order-sized amounts, the exact
    # phantom-fill class reconcile_order defends against; a real fill on a
    # LIVE-looking order is confirmed by on-chain balance at timeout instead.
    if status == "FILLED" or (status == "MATCHED"
                              and filled + _FULL_FILL_EPS >= row["shares"]):
        _open_from_fill(row, filled if filled > 0 else row["shares"], summary)
        return

    # Timeout: cancel whatever still rests, keep any viable partial fill.
    try:
        expires = datetime.fromisoformat(row["expires_at"])
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError, KeyError):
        expires = now  # unparseable expiry -> treat as expired (fail safe)
    if now >= expires:
        executor.cancel_order_safe(client, row["order_id"])
        _cancelled_with_fill(client, row, status, filled, summary,
                             cause="expired")
        return

    # Chase-away: the ask ran more than PASSIVE_CHASE_AWAY_PCT above our limit
    # — the entry thesis is stale; cancel rather than chase.
    if config.PASSIVE_CHASE_AWAY_PCT > 0:
        token_id = executor._resolve_token_id(client, row["condition_id"], row["side"])
        if token_id:
            try:
                book = client.get_order_book(token_id)
                _bid, best_ask, _sz = executor._bid_levels(book)
            except Exception as e:
                log.debug("[passive] chase-away book fetch failed (%s)", e)
                best_ask = None
            if (best_ask is not None and best_ask >
                    row["limit_price"] * (1.0 + config.PASSIVE_CHASE_AWAY_PCT / 100.0)):
                executor.cancel_order_safe(client, row["order_id"])
                _cancelled_with_fill(client, row, status, filled, summary,
                                     cause="chased_away")
                return
    # Still resting inside its window — leave pending.


def _cancelled_with_fill(client, row: dict, reported_status: str,
                         reported_filled: float, summary: dict,
                         cause: str) -> None:
    """After cancelling a pending order, establish how much REALLY filled and
    finish the row. Truth hierarchy:

      1. On-chain token balance — authoritative.
      2. The reported matched size, ONLY on a MATCHED order (a real match
         report). A LIVE order's fill fields can be echoes of the order size
         (the phantom-fill class reconcile_order defends against) and are
         never trusted.
      3. A zero report on an unverifiable balance is safe to take as zero.

    An unverifiable balance WITH a non-MATCHED non-zero report is left
    unresolved: the row stays pending, and next cycle — the order now
    cancelled/gone — the vanished-order path retries the balance until it can
    tell the truth. Never book a position (or a release) on an echo."""
    held = _held_for_row(client, row)
    if held is not None:
        filled_real = min(held, row["shares"])
    elif reported_status == "MATCHED" and reported_filled > 0:
        filled_real = min(reported_filled, row["shares"])
    elif reported_filled <= 0:
        filled_real = 0.0
    else:
        log.warning(
            "[passive] order %s cancelled at %s with an unverifiable partial "
            "fill report (%s %.4f sh, balance unknown) — row %s stays pending "
            "until the balance can confirm", row["order_id"], cause,
            reported_status, reported_filled, row["id"],
        )
        return
    _finish_partial(row, filled_real, summary, cause=cause)


def check_pending_orders(now: datetime | None = None, client=None) -> dict:
    """One lifecycle pass over every pending passive order. Called from the
    pipeline's management cycle (and once at startup). Returns a summary:
    {"checked", "filled", "expired", "chased_away", "dust",
     "released_headlines": [...]} — the caller must discard released_headlines
    from its reservation set. No pending rows -> a single cheap SELECT and out
    (flag-off installs never get past this line)."""
    summary = {"checked": 0, "filled": 0, "expired": 0, "chased_away": 0,
               "dust": 0, "released_headlines": []}
    rows = logger.get_pending_orders()
    if not rows:
        return summary
    now = now or datetime.now(timezone.utc)
    if client is None:
        try:
            client = executor.create_clob_client()
        except Exception as e:
            log.warning("[passive] CLOB client unavailable (%s); %d pending "
                        "order(s) unchecked this cycle", e, len(rows))
            return summary
    for row in rows:
        summary["checked"] += 1
        try:
            _check_one(client, row, now, summary)
        except Exception:
            log.exception("[passive] lifecycle check failed for pending row %s "
                          "(order %s); left pending", row["id"], row["order_id"])
    return summary


def pending_headlines() -> list[str]:
    """Headlines of still-pending passive orders — re-reserved at startup so a
    restart can't let a second candidate trade a headline whose order rests."""
    return [r["headline"] for r in logger.get_pending_orders() if r.get("headline")]


def reconcile_on_startup(client=None) -> dict:
    """Crash recovery, run once before the pipeline starts consuming news:
    one lifecycle pass over persisted pending rows (an order that filled while
    we were down opens its position late, with the real fill data; a vanished
    order with nothing held releases), then report the headlines of rows still
    pending so the pipeline re-reserves them into its fresh in-memory set."""
    summary = check_pending_orders(client=client)
    summary["reserved_headlines"] = pending_headlines()
    if summary["checked"]:
        log.info(
            "[passive] startup reconcile: %d checked, %d filled late, "
            "%d expired, %d chased away, %d dust; %d still pending "
            "(headlines re-reserved)",
            summary["checked"], summary["filled"], summary["expired"],
            summary["chased_away"], summary["dust"],
            len(summary["reserved_headlines"]),
        )
    return summary
