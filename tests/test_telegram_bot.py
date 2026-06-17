"""Telegram bot: graceful no-op when unconfigured, exact message formats, and
the interactive portfolio / balance / close-and-refresh UX."""
import asyncio
import types

import pytest

from locus import config
from locus.core import telegram_bot
from locus.core import positions as positions_mod


def _btns(markup):
    """Flatten an InlineKeyboardMarkup to a list of (text, callback_data)."""
    return [(b.text, b.callback_data) for row in markup.inline_keyboard for b in row]


def _open(trade_id, cid, question, side="YES", amount=10.0, yes=0.5):
    """Insert an open position into the tmp_db and return its id."""
    market = types.SimpleNamespace(
        condition_id=cid, question=question, slug="", yes_price=yes,
        event_id="", category="crypto", end_date="",
    )
    return positions_mod.open_position(trade_id, market, side, amount)


def _run_button(data):
    """Drive _button_cmd with a fake callback query; return (text, markup) of the
    single edit_message_text call it makes."""
    edits = []

    async def edit(text, reply_markup=None):
        edits.append((text, reply_markup))

    async def answer():
        pass

    query = types.SimpleNamespace(data=data, answer=answer, edit_message_text=edit)
    update = types.SimpleNamespace(callback_query=query)
    asyncio.run(telegram_bot._button_cmd(update, None))
    return edits[0] if edits else (None, None)


@pytest.fixture
def enabled(monkeypatch):
    """Configure a token+chat and capture sent text instead of hitting the network."""
    monkeypatch.setattr(config, "TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setattr(config, "TELEGRAM_CHAT_ID", "123")
    sent: list[str] = []
    monkeypatch.setattr(telegram_bot, "_send", lambda text: (sent.append(text), True)[1])
    telegram_bot._drawdown_alerted.clear()
    return sent


# --- Graceful disable --------------------------------------------------------

@pytest.mark.parametrize("token,chat", [("", "123"), ("tok", ""), ("", "")])
def test_disabled_is_noop(monkeypatch, token, chat):
    monkeypatch.setattr(config, "TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setattr(config, "TELEGRAM_CHAT_ID", chat)

    class BoomHTTP:
        def post(self, *a, **k):
            raise AssertionError("must not hit the network when disabled")

    monkeypatch.setattr(telegram_bot, "httpx", BoomHTTP())
    telegram_bot._drawdown_alerted.clear()

    assert telegram_bot._enabled() is False
    pos = {"market_question": "Q", "side": "YES", "entry_yes_price": 0.5, "amount_usd": 10}
    assert telegram_bot.notify_position_opened(pos) is False
    assert telegram_bot.notify_position_closed(pos, -5.0, -1.0, "sl") is False
    assert telegram_bot.notify_half_closed(pos, 5.0, 1.0) is False
    assert telegram_bot.notify_daily_summary({"opened": 1, "closed": 0}) is False
    assert telegram_bot.notify_drawdown_alert({"id": 1, **pos}, -30.0) is False


def test_start_bot_polling_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(config, "TELEGRAM_CHAT_ID", "")
    assert telegram_bot.start_bot_polling() is None


def test_start_bot_polling_skipped_when_channel_monitor_owns_token(monkeypatch):
    # Channel monitor already long-polls this token -> skip interactive polling.
    monkeypatch.setattr(config, "TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setattr(config, "TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(config, "TELEGRAM_CHANNEL_IDS", ["-100123"])
    assert telegram_bot.start_bot_polling() is None


# --- Message formats ---------------------------------------------------------

def test_notify_position_opened_format(enabled):
    pos = {
        "market_question": "Will BTC hit 100k?",
        "side": "YES",
        "entry_yes_price": 0.42,
        "amount_usd": 12.5,
        "edge": 0.10,
        "confidence": 0.7,
    }
    assert telegram_bot.notify_position_opened(pos) is True
    assert enabled[-1] == (
        "🟢 NEW POSITION\n"
        "Market: Will BTC hit 100k?\n"
        "Side: YES | Entry: 0.420 | Amount: $12.50\n"
        "Edge: 10.0% | Conf: 70%"
    )


def test_notify_position_closed_format(enabled):
    pos = {"market_question": "Will X?", "side": "YES", "entry_yes_price": 0.5}
    assert telegram_bot.notify_position_closed(pos, -12.3, -4.5, "sl") is True
    assert enabled[-1] == (
        "🔴 CLOSED\n"
        "Market: Will X?\n"
        "PnL: -4.50 (-12.3%) | Reason: sl"
    )


def test_notify_half_closed_format(enabled):
    pos = {"market_question": "Will X?"}
    assert telegram_bot.notify_half_closed(pos, 20.0, 3.0) is True
    assert enabled[-1] == (
        "🟡 HALF CLOSED\n"
        "Market: Will X?\n"
        "Locked: $3.00 (+20.0%)"
    )


def test_notify_daily_summary_format(enabled):
    stats = {"opened": 3, "closed": 2, "realized": 5.25, "unrealized": -1.5, "win_rate": 50.0}
    assert telegram_bot.notify_daily_summary(stats) is True
    assert enabled[-1] == (
        "📊 DAILY SUMMARY\n"
        "Trades: 3 opened, 2 closed\n"
        "Realized PnL: +5.25\n"
        "Unrealized: -1.50\n"
        "Win Rate: 50.0%"
    )


def test_notify_daily_summary_handles_none_winrate(enabled):
    assert telegram_bot.notify_daily_summary({"opened": 0, "closed": 0, "win_rate": None}) is True
    assert "Win Rate: 0%" in enabled[-1]


def test_notify_drawdown_alert_format_and_dedup(enabled):
    pos = {"id": 7, "market_question": "Will X?"}
    assert telegram_bot.notify_drawdown_alert(pos, -30.0) is True
    assert enabled[-1] == (
        "⚠️ DRAWDOWN ALERT\n"
        "Market: Will X?\n"
        "PnL: -30.0% — consider closing"
    )
    # Same position again is suppressed until it closes.
    assert telegram_bot.notify_drawdown_alert(pos, -35.0) is False
    assert len(enabled) == 1
    # Closing it clears the dedup, so a future drawdown can alert again.
    telegram_bot.notify_position_closed(pos, -35.0, -5.0, "sl")
    assert telegram_bot.notify_drawdown_alert(pos, -40.0) is True


# --- Interactive views -------------------------------------------------------

def test_balance_view_has_no_close_buttons(tmp_db):
    _open(1, "c1", "Will A happen?")
    _open(2, "c2", "Will B happen?")
    text, markup = telegram_bot._build_balance()
    assert text.startswith("💰 BALANCE")
    btns = _btns(markup)
    # Summary only — no per-position Close buttons.
    assert all(not cb.startswith("close:") for _, cb in btns)
    assert btns == [("⬅️ Back to Portfolio", "portfolio"), ("🔄 Refresh", "balance")]


def test_portfolio_view_has_refresh_and_balance(tmp_db):
    pid = _open(1, "c1", "Will A happen?")
    text, markup = telegram_bot._build_portfolio()
    assert text.startswith("💼 PORTFOLIO")
    btns = _btns(markup)
    assert (f"🔴 Close #{pid}", f"close:{pid}") in btns
    assert ("📈 Refresh", "refresh") in btns
    assert ("💰 Balance", "balance") in btns


def test_close_confirmation_has_back_to_portfolio_button(tmp_db):
    text, markup = telegram_bot._build_close_confirmation("✅ Closed #5 — Foo")
    assert text.startswith("✅ Closed #5 — Foo")
    assert "💼 PORTFOLIO" in text  # embeds the refreshed portfolio list
    btns = _btns(markup)
    assert btns[0] == ("📊 Portfolio", "portfolio")  # back button on top


def test_button_navigation_between_portfolio_and_balance(tmp_db):
    _open(1, "c1", "Will A happen?")
    btext, bmarkup = _run_button("balance")
    assert btext.startswith("💰 BALANCE")
    assert all(not cb.startswith("close:") for _, cb in _btns(bmarkup))
    ptext, pmarkup = _run_button("portfolio")
    assert ptext.startswith("💼 PORTFOLIO")
    assert ("💰 Balance", "balance") in _btns(pmarkup)


def test_button_close_auto_refreshes_portfolio(tmp_db):
    pid = _open(1, "c1", "Will A happen?")
    text, markup = _run_button(f"close:{pid}")
    # Confirmation header + the refreshed (now empty) portfolio list.
    assert "✅ Closed" in text
    assert "No open positions" in text
    assert _btns(markup)[0] == ("📊 Portfolio", "portfolio")
    # The position is actually closed in the DB.
    assert positions_mod.get_open_positions() == []


def test_button_close_missing_position(tmp_db):
    text, _ = _run_button("close:999")
    assert "not found or already closed" in text
