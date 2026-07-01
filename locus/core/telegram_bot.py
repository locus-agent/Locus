"""
Telegram bot for real-time trading notifications and an interactive portfolio.

Two independent halves:

1. Notifications (sendMessage over the Telegram HTTP API via httpx). These are
   fire-and-forget, synchronous, and thread-safe, so they can be called straight
   from the pipeline's executor threads (executor, positions, journal). They are
   no-ops — never raising — when TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is unset.

2. Interactive command bot (python-telegram-bot Application long-polling) for
   /portfolio with inline [Close] / [Refresh] / [Balance] buttons. Runs in a
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


def notify_position_closed(position: dict, pnl_pct: float, pnl_usd: float, reason: str) -> bool:
    """🔴 a position fully closed. Clears any drawdown-alert dedup for it.

    `pnl_pct` arrives from positions._close already as the realized return on
    the closed chunk's ACTUAL cost (the fill basis), so it is displayed as-is —
    rebasing it again via pnl_pct_on_cost would double-adjust."""
    _drawdown_alerted.discard(position.get("id"))
    text = (
        "🔴 CLOSED\n"
        f"Market: {_q(position)}\n"
        f"PnL: {pnl_usd:+.2f} ({pnl_pct:+.2f}%) | Reason: {reason}"
    )
    return _send(text)


def notify_half_closed(position: dict, pnl_pct: float, pnl_usd: float) -> bool:
    """🟡 half the position was realized, the rest left to ride."""
    text = (
        "🟡 HALF CLOSED\n"
        f"Market: {_q(position)}\n"
        f"Locked: ${pnl_usd:.2f} ({pnl_pct:+.1f}%)"
    )
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


# --- Interactive command bot (/portfolio) ------------------------------------

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
    text, markup = _build_portfolio()
    await update.message.reply_text(text, reply_markup=markup)


async def _button_cmd(update, context):
    """Inline-button taps: navigate (portfolio/refresh/balance) or close a position."""
    query = update.callback_query
    await query.answer()
    data = query.data or ""

    if data in ("portfolio", "refresh"):
        text, markup = _build_portfolio()
    elif data == "balance":
        text, markup = _build_balance()
    elif data.startswith("close:"):
        from locus.core import positions
        try:
            pid = int(data.split(":", 1)[1])
        except (ValueError, IndexError):
            return
        result = positions.close_manual(pid)
        if result:
            header = (
                f"✅ Closed #{pid} — {result['market_question'][:50]}\n"
                f"{result['side']} @ {result['price']:.3f} "
                f"({result['pnl_pct']:+.1f}%, ${result['realized']:+.2f})"
            )
        else:
            header = f"⚠️ Position #{pid} not found or already closed."
        # Auto-refresh: the confirmation embeds the updated portfolio list.
        text, markup = _build_close_confirmation(header)
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
