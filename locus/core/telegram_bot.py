"""
Telegram bot for real-time trading notifications and an interactive portfolio.

Two independent halves:

1. Notifications (sendMessage over the Telegram HTTP API via httpx). These are
   fire-and-forget, synchronous, and thread-safe, so they can be called straight
   from the pipeline's executor threads (executor, positions, journal). They are
   no-ops — never raising — when TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is unset.

2. Interactive command bot (python-telegram-bot Application long-polling) for
   /portfolio with inline [Close] / [Refresh] / [Balance] buttons, and
   /positions — the operator control view with per-position [Close] / [Half] /
   [Force close] buttons (Force is a two-step confirmation that previews the
   realized outcome at the current best bid before selling into it). Every
   command and button tap is auth-gated to TELEGRAM_CHAT_ID. Runs in a
   daemon thread with its own event loop, started from `cli.py watch`.

   NOTE: Telegram permits only one getUpdates long-poll per bot token. The news
   stream's TelegramMonitor already polls this token when TELEGRAM_CHANNEL_IDS is
   set, so interactive polling is skipped in that case (a second poller would
   409 both). Notifications work regardless. Use a dedicated bot token (or clear
   TELEGRAM_CHANNEL_IDS) to enable the interactive buttons.

python-telegram-bot is imported lazily inside the polling thread so this module
(and the notification path) stays importable even when the library is absent.
"""
from __future__ import annotations

import logging

import httpx

from locus import config

log = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org"

# Positions we've already sent a drawdown alert for, so a sinking position isn't
# re-alerted every 30s pipeline cycle. Bounded by the number of open positions.
_drawdown_alerted: set[int] = set()

_bot_thread = None  # the interactive-bot polling thread, once started


def _enabled() -> bool:
    """True when both the bot token and chat id are configured. Read from config
    at call time so env/test overrides are honored."""
    return bool(config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID)


def _send(text: str) -> bool:
    """Send a plain-text message to the configured chat. No-op (returns False)
    when disabled; never raises — logs and returns False on any failure."""
    if not _enabled():
        return False
    try:
        resp = httpx.post(
            f"{_API_BASE}/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        log.warning(f"[telegram] send failed: {e}")
        return False


def _q(position: dict) -> str:
    """Market question off a position/notification dict (a couple of key names)."""
    return position.get("market_question") or position.get("question") or ""


def _mode_badge() -> str:
    """Trading-mode badge for view headers. Read from config at call time so a
    runtime DRY_RUN override (cli.py watch --live) is reflected immediately."""
    return "🟢 LIVE" if not config.DRY_RUN else "🔵 DRY RUN"


# --- Notifications -----------------------------------------------------------

def notify_position_opened(position: dict) -> bool:
    """🟢 a new position was opened. `position` carries market_question, side,
    entry_yes_price, amount_usd, and (from the signal) edge + confidence."""
    price = position.get("entry_yes_price")
    price = position.get("price") if price is None else price
    amount = position.get("amount_usd", position.get("amount", 0.0)) or 0.0
    edge = position.get("edge", 0.0) or 0.0
    conf = position.get("confidence", position.get("conf", 0.0)) or 0.0
    # Live fills usually cost less than the nominal bet (size rounding/fees), so
    # show the real filled cost alongside the nominal when they differ.
    actual = position.get("actual_cost_usd")
    if actual and actual > 0 and abs(actual - amount) >= 0.01:
        amount_line = f"Amount: ${actual:.2f} (filled) | Nominal: ${amount:.2f}"
    else:
        amount_line = f"Amount: ${amount:.2f}"
    header = "🟢 LIVE POSITION" if not config.DRY_RUN else "🟢 NEW POSITION"
    text = (
        f"{header}\n"
        f"Market: {_q(position)}\n"
        f"Side: {position.get('side', '?')} | Entry: {price:.3f} | {amount_line}\n"
        f"Edge: {edge * 100:.1f}% | Conf: {conf * 100:.0f}%"
    )
    return _send(text)


def _shares(n: float) -> str:
    """Compact share-count display: 12.0 -> '12', 12.5 -> '12.5'."""
    return f"{n:g}"


def notify_position_closed(position: dict, pnl_pct: float, pnl_usd: float, reason: str) -> bool:
    """🔴 a position fully closed. Clears any drawdown-alert dedup for it.

    `pnl_pct` arrives from positions._close already as the realized return on
    the closed chunk's ACTUAL cost (the fill basis), so it is displayed as-is —
    rebasing it again via pnl_pct_on_cost would double-adjust.

    `position['realized_pnl_usd']` is the position's CUMULATIVE realized PnL
    including this final chunk (the same basis calibration-report reads). When
    earlier half-closes made the total differ from this chunk, both are shown
    so the final chunk's number doesn't read as the whole position's result."""
    _drawdown_alerted.discard(position.get("id"))
    total = position.get("realized_pnl_usd")
    lines = ["🔴 CLOSED", f"Market: {_q(position)}"]
    if total is not None and abs(total - pnl_usd) > 0.005:
        lines.append(f"This close: {pnl_usd:+.2f} ({pnl_pct:+.2f}%) | Reason: {reason}")
        lines.append(f"Position total: {total:+.2f} (incl. earlier partial sales)")
    else:
        lines.append(f"PnL: {pnl_usd:+.2f} ({pnl_pct:+.2f}%) | Reason: {reason}")
    return _send("\n".join(lines))


def notify_half_closed(position: dict, pnl_pct: float, pnl_usd: float) -> bool:
    """🟡 part of the position was realized, the rest left to ride.

    `pnl_usd`/`pnl_pct` are THIS chunk's realized return on the sold chunk's
    cost; `position` carries the post-close state — realized_pnl_usd (cumulative
    on this position so far), token_count and amount_usd (the remainder). A
    near-zero chunk is normal: Polymarket has no trading fees, so the spread is
    the only cost and it can eat the whole gain."""
    total = position.get("realized_pnl_usd")
    total = pnl_usd if total is None else total
    spread_note = " (spread)" if abs(pnl_usd) < 0.01 else ""
    tokens = position.get("token_count")
    amount = position.get("amount_usd") or 0.0
    if tokens:
        open_line = f"Still open: {_shares(tokens)} tokens (~${amount:.0f}) riding"
    else:
        open_line = f"Still open: ~${amount:.0f} riding"
    text = (
        "🟡 HALF CLOSED\n"
        f"Market: {_q(position)}\n"
        f"This sale: {pnl_usd:+.2f} ({pnl_pct:+.1f}%){spread_note}\n"
        f"Position total realized so far: {total:+.2f}\n"
        f"{open_line}"
    )
    return _send(text)


def notify_passive_filled(position: dict) -> bool:
    """🔵 a resting passive limit order filled — the position is now open.
    `position` carries market_question, side, price (the side-token limit
    price the fill happened at), token_count, and actual_cost_usd."""
    price = position.get("price")
    tokens = position.get("token_count") or 0.0
    cost = position.get("actual_cost_usd") or position.get("amount_usd") or 0.0
    price_part = f" | Entry: {price:.3f}" if price is not None else ""
    text = (
        "🔵 PASSIVE FILL\n"
        f"Market: {_q(position)}\n"
        f"Side: {position.get('side', '?')}{price_part}\n"
        f"Filled: {_shares(tokens)} tokens for ${cost:.2f}"
    )
    return _send(text)


def notify_passive_expired(market_question: str, filled_tokens: float,
                           reason: str = "expired") -> bool:
    """⚪ a resting passive limit order ended without becoming a (full)
    position: `reason` is 'expired' (timeout) or 'chased_away' (the ask ran
    away from our limit). A viable partial fill that DID open a position is
    announced by notify_passive_filled instead; this covers the nothing-opened
    endings, plus a defensive partial-fill wording if ever called with one."""
    header = f"⚪ PASSIVE {reason.replace('_', ' ').upper()}"
    if filled_tokens and filled_tokens > 0:
        detail = (f"Partial fill: {_shares(filled_tokens)} tokens "
                  "opened as a position")
    else:
        detail = "No fill — nothing opened"
    text = f"{header}\nMarket: {market_question}\n{detail}"
    return _send(text)


def notify_daily_summary(stats: dict) -> bool:
    """📊 end-of-day rollup. stats: opened, closed, realized, unrealized, win_rate."""
    win_rate = stats.get("win_rate")
    win_rate = 0 if win_rate is None else win_rate
    text = (
        "📊 DAILY SUMMARY\n"
        f"Trades: {stats.get('opened', 0)} opened, {stats.get('closed', 0)} closed\n"
        f"Realized PnL: {(stats.get('realized') or 0.0):+.2f}\n"
        f"Unrealized: {(stats.get('unrealized') or 0.0):+.2f}\n"
        f"Win Rate: {win_rate}%"
    )
    return _send(text)


def notify_drawdown_alert(position: dict, pnl_pct: float) -> bool:
    """⚠️ an open position is deep in the red. Sent at most once per position
    (until it closes) so the 30s cycle doesn't spam the same loser."""
    pid = position.get("id")
    if pid is not None:
        if pid in _drawdown_alerted:
            return False
        _drawdown_alerted.add(pid)
    text = (
        "⚠️ DRAWDOWN ALERT\n"
        f"Market: {_q(position)}\n"
        f"PnL: {pnl_pct:+.1f}% — consider closing"
    )
    return _send(text)


# --- Interactive command bot (/portfolio, /positions) ------------------------

def _authorized(update) -> bool:
    """Only the configured TELEGRAM_CHAT_ID may issue commands or tap buttons.
    Anything else (a stranger finding the bot, a group it was added to) is
    ignored — commands from it must never move money."""
    if not config.TELEGRAM_CHAT_ID:
        return False
    chat = getattr(update, "effective_chat", None)
    chat_id = getattr(chat, "id", None)
    if chat_id is None:
        return False
    if str(chat_id) != str(config.TELEGRAM_CHAT_ID):
        log.warning(f"[telegram] unauthorized chat {chat_id} rejected")
        return False
    return True


# Human-readable close-failure reasons (executor statuses -> operator text).
_CLOSE_FAILURE_TEXT = {
    "skipped_thin_book": "below exchange minimums / book too thin to sell into",
    "skipped_empty_book": "no bids at all (empty book)",
    "skipped_wide_spread": "spread too wide (protective gate — use Force close "
                           "to sell anyway)",
}

_TOPUP_SKIP_TEXT = {
    "no_bid_liquidity": "no real bid liquidity near the ask (zombie book)",
    "topup_too_expensive": "top-up cost would exceed the cap",
    "not_filled": "top-up buy did not fill",
    "order_error": "top-up buy was rejected",
    "no_book": "no book to price a top-up against",
    "not_dust": "holding already clears the minimums (bid depth is the blocker)",
}


def _failure_text(reason: dict | None) -> str:
    """Operator-facing explanation of why a manual close failed, from the
    failure_reason dict positions.close_manual returns."""
    if not reason:
        return "live SELL not confirmed"
    status = reason.get("status") or ""
    text = _CLOSE_FAILURE_TEXT.get(status, status or "live SELL not confirmed")
    if reason.get("error"):
        text += f" — {reason['error']}"
    skip = reason.get("topup_skipped")
    if skip:
        skip_text = _TOPUP_SKIP_TEXT.get(skip, skip)
        if skip == "topup_too_expensive":
            skip_text += f" (${config.TOPUP_MAX_USD:.2f})"
        text += (f"; top-up skipped: {skip_text} — position is stuck until "
                 "resolution")
    return text

def _build_portfolio():
    """(text, InlineKeyboardMarkup) for the portfolio view: each open position
    with live PnL and a [🔴 Close #id] button, then a [📈 Refresh] [💰 Balance]
    row. Imports telegram lazily — only called from the polling thread."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    from locus.core import positions

    open_pos = positions.get_open_positions()
    header = f"💼 PORTFOLIO {_mode_badge()}"
    if open_pos:
        lines = [header, ""]
        for p in open_pos:
            # Stored unrealized_pnl_pct is already marked on the actual fill
            # basis (positions.pnl_pct_basis), i.e. return-on-cost — display
            # it as-is; rebasing again would double-adjust.
            pct = p.get("unrealized_pnl_pct") or 0.0
            lines.append(f"#{p['id']} {p['side']} {p['market_question'][:40]} ({pct:+.1f}%)")
        text = "\n".join(lines)
    else:
        text = f"{header}\n\nNo open positions."

    rows = [
        [InlineKeyboardButton(f"🔴 Close #{p['id']}", callback_data=f"close:{p['id']}")]
        for p in open_pos
    ]
    rows.append([
        InlineKeyboardButton("📈 Refresh", callback_data="refresh"),
        InlineKeyboardButton("💰 Balance", callback_data="balance"),
    ])
    return text, InlineKeyboardMarkup(rows)


def _build_balance():
    """(text, InlineKeyboardMarkup) for the balance view: summary stats only —
    no per-position Close buttons — with [⬅️ Back to Portfolio] [🔄 Refresh]."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    from locus.core import positions
    from locus.core import executor
    from locus.core.performance import compute_performance

    perf = compute_performance()
    # Deployed capital and open count must reflect only positions still open
    # within the dashboard's performance window. perf['deployed_usd'] sums every
    # trade in the window — closed ones included — so it double-counts capital
    # already returned (the $261-vs-$100 bug). Recompute from the live open book
    # under the same PERFORMANCE_START_DATE filter the dashboard panel uses.
    since = config.PERFORMANCE_START_DATE or None
    open_pos = positions.get_open_positions(since=since)
    open_count = len(open_pos)

    if not config.DRY_RUN:
        # Live: the real USDC balance on Polymarket replaces the computed
        # deployed-capital figure (which only tracks simulated stakes).
        balance = executor.get_live_balance()
        balance_line = (
            f"Real Balance: ${balance:.2f}" if balance is not None
            else "Real Balance: unavailable"
        )
        text = (
            "💰 BALANCE 🟢 LIVE MODE\n"
            f"{balance_line}\n"
            f"Open: {open_count} | Closed: {perf['closed_count']}\n"
            f"Realized: {perf['realized_pnl_usd']:+.2f}\n"
            f"Unrealized: {perf['unrealized_pnl_usd']:+.2f}"
        )
    else:
        deployed = sum(p.get("amount_usd") or 0.0 for p in open_pos)
        text = (
            "💰 BALANCE 🔵 DRY RUN\n"
            f"Open: {open_count} | Closed: {perf['closed_count']}\n"
            f"Deployed: ${deployed:.2f}\n"
            f"Realized: {perf['realized_pnl_usd']:+.2f}\n"
            f"Unrealized: {perf['unrealized_pnl_usd']:+.2f}"
        )
    markup = InlineKeyboardMarkup([[
        InlineKeyboardButton("⬅️ Back to Portfolio", callback_data="portfolio"),
        InlineKeyboardButton("🔄 Refresh", callback_data="balance"),
    ]])
    return text, markup


def _position_metrics(p: dict) -> tuple[float, float, float, float]:
    """(tokens, cost_basis, current_value, pnl_pct) for an open-position row."""
    from locus.core import positions
    from locus.core.performance import position_shares

    tokens = position_shares(
        p["side"], p["entry_yes_price"], p.get("amount_usd") or 0.0,
        token_count=p.get("token_count"),
    )
    cost = positions.position_cost_basis(p)
    yes = p.get("current_yes_price")
    yes = p["entry_yes_price"] if yes is None else yes
    side_price = yes if p["side"] == "YES" else 1.0 - yes
    value = tokens * side_price
    pct = p.get("unrealized_pnl_pct") or 0.0
    return tokens, cost, value, pct


def _build_positions():
    """(text, InlineKeyboardMarkup) for the /positions operator view: every
    open position with tokens / cost basis / current value / unrealized PnL%,
    and a [Close] [Half] [Force] button row per position. Close runs the
    normal manual close path; Half realizes half (with the anti-dust
    escalation); Force is a two-step confirmed sell into whatever bid exists."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    from locus.core import positions

    open_pos = positions.get_open_positions()
    header = f"📋 POSITIONS {_mode_badge()}"
    rows = []
    if open_pos:
        lines = [header, ""]
        for p in open_pos:
            tokens, cost, value, pct = _position_metrics(p)
            lines.append(f"#{p['id']} {p['side']} {p['market_question'][:40]}")
            lines.append(
                f"    {_shares(round(tokens, 2))} tok | cost ${cost:.2f} | "
                f"value ${value:.2f} | {pct:+.1f}%"
            )
            rows.append([
                InlineKeyboardButton(f"🔴 Close #{p['id']}",
                                     callback_data=f"pclose:{p['id']}"),
                InlineKeyboardButton(f"🟡 Half #{p['id']}",
                                     callback_data=f"phalf:{p['id']}"),
                InlineKeyboardButton(f"⚠️ Force #{p['id']}",
                                     callback_data=f"pforce:{p['id']}"),
            ])
        text = "\n".join(lines)
    else:
        text = f"{header}\n\nNo open positions."
    rows.append([InlineKeyboardButton("🔄 Refresh", callback_data="positions")])
    return text, InlineKeyboardMarkup(rows)


def _build_positions_result(header: str):
    """(text, markup) for an action result: the outcome header above the
    freshly-refreshed positions list."""
    text, markup = _build_positions()
    return f"{header}\n\n{text}", markup


def _fetch_best_bid(position: dict) -> float | None:
    """Live best bid for the held side's token, or None when unavailable
    (dry-run, no CLOB client, book fetch failure). The force-close preview
    falls back to the marked price then."""
    from locus.core import executor
    try:
        client = executor.create_clob_client()
        token_id = executor._resolve_token_id(
            client, position["condition_id"], position["side"]
        )
        if not token_id:
            return None
        book = client.get_order_book(token_id)
        best_bid, _, _ = executor._bid_levels(book)
        return best_bid
    except Exception as e:
        log.warning(f"[telegram] best-bid fetch failed: {e}")
        return None


def _build_force_preview(pid: int):
    """(text, markup) for the force-close confirmation step: the realized
    outcome (proceeds vs cost, in $ and %) at the CURRENT best bid, with
    [Yes, force] / [Cancel]. Nothing is sold until the second tap."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    from locus.core import positions

    p = next((x for x in positions.get_open_positions() if x["id"] == pid), None)
    if p is None:
        return _build_positions_result(
            f"⚠️ Position #{pid} not found or already closed.")

    tokens, cost, _, _ = _position_metrics(p)
    yes = p.get("current_yes_price")
    yes = p["entry_yes_price"] if yes is None else yes
    mark = yes if p["side"] == "YES" else 1.0 - yes
    bid = _fetch_best_bid(p)
    bid_note = ""
    if bid is None:
        bid = mark
        bid_note = " (live bid unavailable — assuming the mark)"
    proceeds = tokens * bid
    realized = proceeds - cost
    realized_pct = realized / cost * 100.0 if cost > 0 else 0.0
    text = (
        f"⚠️ FORCE CLOSE #{pid} — {p['market_question'][:50]}\n"
        f"Best bid {bid:.3f} vs mark {mark:.3f}{bid_note}.\n"
        f"Selling {_shares(round(tokens, 2))} tokens returns ~${proceeds:.2f} "
        f"(you paid ${cost:.2f}).\n"
        f"This realizes {realized_pct:+.0f}% (${realized:+.2f}). Confirm?"
    )
    markup = InlineKeyboardMarkup([[
        InlineKeyboardButton("⚠️ Yes, force", callback_data=f"pforceyes:{pid}"),
        InlineKeyboardButton("Cancel", callback_data="positions"),
    ]])
    return text, markup


def _manual_action_header(pid: int, result: dict | None, label: str) -> str:
    """Outcome header for a manual Close/Half/Force action, surfacing the real
    failure reason (not a generic 'failed') and any half->full escalation."""
    if result is None:
        return f"⚠️ Position #{pid} not found or already closed."
    if result.get("close_failed"):
        return (f"❌ {label} #{pid} failed: "
                f"{_failure_text(result.get('failure_reason'))}")
    header = (
        f"✅ {label} #{pid} — {result['market_question'][:50]}\n"
        f"{result['side']} @ {result['price']:.3f} "
        f"({result['pnl_pct']:+.1f}%, ${result['realized']:+.2f})"
    )
    if result.get("escalated"):
        header += (f"\n(half escalated to FULL close: {result['escalated']})")
    return header


def _build_close_confirmation(header: str):
    """(text, InlineKeyboardMarkup) shown after a Close tap: a confirmation
    header above the freshly-refreshed portfolio list, with a [📊 Portfolio]
    button on top to dismiss the header and go back."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    text, portfolio_markup = _build_portfolio()
    text = f"{header}\n\n{text}"
    rows = [[InlineKeyboardButton("📊 Portfolio", callback_data="portfolio")]]
    rows.extend(portfolio_markup.inline_keyboard)
    return text, InlineKeyboardMarkup(rows)


async def _portfolio_cmd(update, context):
    """/portfolio (and /start): show the interactive portfolio."""
    if not _authorized(update):
        return
    text, markup = _build_portfolio()
    await update.message.reply_text(text, reply_markup=markup)


async def _positions_cmd(update, context):
    """/positions: the operator control view — every open position with
    per-position [Close] [Half] [Force] buttons."""
    if not _authorized(update):
        return
    text, markup = _build_positions()
    await update.message.reply_text(text, reply_markup=markup)


async def _button_cmd(update, context):
    """Inline-button taps: navigate (portfolio/refresh/balance/positions),
    close/half/force-close a position."""
    query = update.callback_query
    if not _authorized(update):
        return
    await query.answer()
    data = query.data or ""

    def _pid() -> int | None:
        try:
            return int(data.split(":", 1)[1])
        except (ValueError, IndexError):
            return None

    if data in ("portfolio", "refresh"):
        text, markup = _build_portfolio()
    elif data == "balance":
        text, markup = _build_balance()
    elif data == "positions":
        text, markup = _build_positions()
    elif data.startswith("close:"):
        from locus.core import positions
        pid = _pid()
        if pid is None:
            return
        result = positions.close_manual(pid)
        if result and not result.get("close_failed"):
            header = (
                f"✅ Closed #{pid} — {result['market_question'][:50]}\n"
                f"{result['side']} @ {result['price']:.3f} "
                f"({result['pnl_pct']:+.1f}%, ${result['realized']:+.2f})"
            )
        elif result:
            header = (f"❌ Close #{pid} failed: "
                      f"{_failure_text(result.get('failure_reason'))}")
        else:
            header = f"⚠️ Position #{pid} not found or already closed."
        # Auto-refresh: the confirmation embeds the updated portfolio list.
        text, markup = _build_close_confirmation(header)
    elif data.startswith("pclose:"):
        from locus.core import positions
        pid = _pid()
        if pid is None:
            return
        result = positions.close_manual(pid)
        text, markup = _build_positions_result(
            _manual_action_header(pid, result, "Closed"))
    elif data.startswith("phalf:"):
        from locus.core import positions
        pid = _pid()
        if pid is None:
            return
        result = positions.close_manual_half(pid)
        label = "Closed" if (result or {}).get("escalated") else "Half closed"
        text, markup = _build_positions_result(
            _manual_action_header(pid, result, label))
    elif data.startswith("pforce:"):
        # Step 1 of 2: preview only — show the realized outcome at the current
        # best bid. Nothing is sold until the [Yes, force] tap.
        pid = _pid()
        if pid is None:
            return
        text, markup = _build_force_preview(pid)
    elif data.startswith("pforceyes:"):
        # Step 2 of 2: the confirmed force close — bypasses the wide-spread
        # protective gate and sells into whatever bid exists (a sub-minimum
        # holding still tries top-up-and-sell first).
        from locus.core import positions
        pid = _pid()
        if pid is None:
            return
        result = positions.close_manual(pid, force=True)
        text, markup = _build_positions_result(
            _manual_action_header(pid, result, "Force closed"))
    else:
        return

    try:
        await query.edit_message_text(text, reply_markup=markup)
    except Exception as e:
        # Telegram rejects an edit that doesn't change anything (e.g. tapping
        # Refresh when nothing moved). That's a no-op, not an error.
        if "not modified" not in str(e).lower():
            raise


def _run_polling():
    """Thread target: build the Application and long-poll. Its own event loop."""
    import asyncio
    try:
        from telegram.ext import Application, CommandHandler, CallbackQueryHandler
    except Exception as e:  # library missing / import error — degrade gracefully
        log.warning(f"[telegram] python-telegram-bot unavailable, polling disabled: {e}")
        return
    try:
        asyncio.set_event_loop(asyncio.new_event_loop())
        app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
        app.add_handler(CommandHandler(["portfolio", "start"], _portfolio_cmd))
        app.add_handler(CommandHandler("positions", _positions_cmd))
        app.add_handler(CallbackQueryHandler(_button_cmd))
        log.info("[telegram] interactive bot polling started")
        # stop_signals=None: signal handlers can only be installed on the main
        # thread, and this runs in a daemon thread.
        app.run_polling(stop_signals=None, close_loop=False)
    except Exception as e:
        log.warning(f"[telegram] interactive bot polling stopped: {e}")


def start_bot_polling():
    """Start the interactive /portfolio bot in a daemon thread. No-op (returns
    None) when disabled, already running, or when the news stream's channel
    monitor already owns this token's getUpdates poll (see module docstring)."""
    global _bot_thread
    if not _enabled():
        log.info("[telegram] disabled (no token/chat id) — interactive bot not started")
        return None
    if _bot_thread is not None and _bot_thread.is_alive():
        return _bot_thread
    if config.TELEGRAM_CHANNEL_IDS:
        log.warning(
            "[telegram] TELEGRAM_CHANNEL_IDS is set — the news channel monitor owns "
            "this bot's getUpdates poll, so interactive /portfolio polling is disabled "
            "to avoid a 409 conflict. Notifications still work. Use a separate bot "
            "token (or clear TELEGRAM_CHANNEL_IDS) to enable the buttons."
        )
        return None

    import threading
    _bot_thread = threading.Thread(target=_run_polling, name="telegram-bot", daemon=True)
    _bot_thread.start()
    return _bot_thread
