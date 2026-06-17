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


# --- Notifications -----------------------------------------------------------

def notify_position_opened(position: dict) -> bool:
    """🟢 a new position was opened. `position` carries market_question, side,
    entry_yes_price, amount_usd, and (from the signal) edge + confidence."""
    price = position.get("entry_yes_price")
    price = position.get("price") if price is None else price
    amount = position.get("amount_usd", position.get("amount", 0.0)) or 0.0
    edge = position.get("edge", 0.0) or 0.0
    conf = position.get("confidence", position.get("conf", 0.0)) or 0.0
    text = (
        "🟢 NEW POSITION\n"
        f"Market: {_q(position)}\n"
        f"Side: {position.get('side', '?')} | Entry: {price:.3f} | Amount: ${amount:.2f}\n"
        f"Edge: {edge * 100:.1f}% | Conf: {conf * 100:.0f}%"
    )
    return _send(text)


def notify_position_closed(position: dict, pnl_pct: float, pnl_usd: float, reason: str) -> bool:
    """🔴 a position fully closed. Clears any drawdown-alert dedup for it."""
    _drawdown_alerted.discard(position.get("id"))
    text = (
        "🔴 CLOSED\n"
        f"Market: {_q(position)}\n"
        f"PnL: {pnl_usd:+.2f} ({pnl_pct:+.1f}%) | Reason: {reason}"
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
    """(text, InlineKeyboardMarkup) for the portfolio view: each open position,
    its live PnL, a [🔴 Close #id] button, plus [📈 Refresh] [💰 Balance].
    Imports telegram lazily — only called from the polling thread's handlers."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    from locus.core import positions

    open_pos = positions.get_open_positions()
    if open_pos:
        lines = ["💼 PORTFOLIO", ""]
        for p in open_pos:
            pct = p.get("unrealized_pnl_pct") or 0.0
            lines.append(f"#{p['id']} {p['side']} {p['market_question'][:40]} ({pct:+.1f}%)")
        text = "\n".join(lines)
    else:
        text = "💼 PORTFOLIO\n\nNo open positions."

    rows = [
        [InlineKeyboardButton(f"🔴 Close #{p['id']}", callback_data=f"close:{p['id']}")]
        for p in open_pos
    ]
    rows.append([
        InlineKeyboardButton("📈 Refresh", callback_data="refresh"),
        InlineKeyboardButton("💰 Balance", callback_data="balance"),
    ])
    return text, InlineKeyboardMarkup(rows)


def _balance_text() -> str:
    from locus.core.performance import compute_performance
    perf = compute_performance()
    return (
        "💰 BALANCE\n"
        f"Open: {perf['open_count']} | Closed: {perf['closed_count']}\n"
        f"Deployed: ${perf['deployed_usd']:.2f}\n"
        f"Realized: {perf['realized_pnl_usd']:+.2f}\n"
        f"Unrealized: {perf['unrealized_pnl_usd']:+.2f}"
    )


async def _portfolio_cmd(update, context):
    """/portfolio (and /start): show the interactive portfolio."""
    text, markup = _build_portfolio()
    await update.message.reply_text(text, reply_markup=markup)


async def _button_cmd(update, context):
    """Inline-button taps: close / refresh / balance."""
    query = update.callback_query
    await query.answer()
    data = query.data or ""

    if data == "refresh":
        text, markup = _build_portfolio()
        await query.edit_message_text(text, reply_markup=markup)
    elif data == "balance":
        _, markup = _build_portfolio()
        await query.edit_message_text(_balance_text(), reply_markup=markup)
    elif data.startswith("close:"):
        from locus.core import positions
        try:
            pid = int(data.split(":", 1)[1])
        except (ValueError, IndexError):
            return
        result = positions.close_manual(pid)
        if result:
            text = (
                f"✅ Closed #{pid}\n"
                f"{result['market_question'][:50]}\n"
                f"{result['side']} @ {result['price']:.3f} "
                f"({result['pnl_pct']:+.1f}%, ${result['realized']:+.2f})"
            )
        else:
            text = f"⚠️ Position #{pid} not found or already closed."
        _, markup = _build_portfolio()
        await query.edit_message_text(text, reply_markup=markup)


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
