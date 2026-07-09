"""Telegram bot: graceful no-op when unconfigured, exact message formats, and
the interactive portfolio / balance / close-and-refresh UX."""
import asyncio
import types

import pytest

from locus import config
from locus.core import telegram_bot
from locus.core import positions as positions_mod


@pytest.fixture(autouse=True)
def _default_dry_run(monkeypatch):
    """Pin DRY_RUN on for the suite so mode-dependent views/notifications are
    deterministic regardless of the developer's .env (which may set live mode).
    Live-mode tests flip it back off explicitly."""
    monkeypatch.setattr(config, "DRY_RUN", True)


@pytest.fixture(autouse=True)
def _authorized_chat(monkeypatch):
    """Every interactive handler is auth-gated on the configured chat id; pin a
    known id so the helpers below can build authorized updates. Unauthorized-
    path tests pass a different chat id explicitly."""
    monkeypatch.setattr(config, "TELEGRAM_CHAT_ID", "123")


def _chat(chat_id="123"):
    return types.SimpleNamespace(id=chat_id)


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


def _run_button(data, chat_id="123"):
    """Drive _button_cmd with a fake callback query; return (text, markup) of the
    single edit_message_text call it makes."""
    edits = []

    async def edit(text, reply_markup=None):
        edits.append((text, reply_markup))

    async def answer():
        pass

    query = types.SimpleNamespace(data=data, answer=answer, edit_message_text=edit)
    update = types.SimpleNamespace(callback_query=query, effective_chat=_chat(chat_id))
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
    assert telegram_bot.notify_passive_filled(pos) is False
    assert telegram_bot.notify_passive_expired("Q", 0.0) is False


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


def test_notify_position_opened_shows_filled_cost(enabled):
    # A live fill below the nominal bet shows both the filled cost and nominal.
    pos = {
        "market_question": "Will BTC hit 100k?",
        "side": "YES",
        "entry_yes_price": 0.42,
        "amount_usd": 25.0,
        "actual_cost_usd": 21.0,
        "edge": 0.10,
        "confidence": 0.7,
    }
    assert telegram_bot.notify_position_opened(pos) is True
    assert "Amount: $21.00 (filled) | Nominal: $25.00" in enabled[-1]


def test_notify_position_opened_no_filled_line_when_cost_equals_nominal(enabled):
    pos = {
        "market_question": "Will BTC hit 100k?", "side": "YES",
        "entry_yes_price": 0.42, "amount_usd": 25.0, "actual_cost_usd": 25.0,
        "edge": 0.1, "confidence": 0.7,
    }
    assert telegram_bot.notify_position_opened(pos) is True
    assert "Amount: $25.00\n" in enabled[-1]
    assert "filled" not in enabled[-1]


def test_notify_position_closed_format(enabled):
    pos = {"market_question": "Will X?", "side": "YES", "entry_yes_price": 0.5}
    assert telegram_bot.notify_position_closed(pos, -12.3, -4.5, "sl") is True
    # No actual_cost_usd -> % unchanged (price-based), shown to 2 decimals.
    assert enabled[-1] == (
        "🔴 CLOSED\n"
        "Market: Will X?\n"
        "PnL: -4.50 (-12.30%) | Reason: sl"
    )


def test_notify_position_closed_pct_shown_as_is(enabled):
    # positions._close already computes the % as realized return on the
    # chunk's ACTUAL cost (the fill basis), so the notification shows it
    # unchanged — a second rebase would double-adjust.
    pos = {"market_question": "Will X?", "side": "YES", "entry_yes_price": 0.5,
           "amount_usd": 25.0, "actual_cost_usd": 21.0}
    assert telegram_bot.notify_position_closed(pos, 75.71, 15.90, "tp_decision") is True
    assert enabled[-1] == (
        "🔴 CLOSED\n"
        "Market: Will X?\n"
        "PnL: +15.90 (+75.71%) | Reason: tp_decision"
    )


def test_notify_half_closed_format(enabled):
    # Three-line picture: this chunk's realized, the position's cumulative
    # realized so far, and what's still riding.
    pos = {"market_question": "Will X?", "realized_pnl_usd": 3.0,
           "token_count": 20.0, "amount_usd": 10.0}
    assert telegram_bot.notify_half_closed(pos, 20.0, 3.0) is True
    assert enabled[-1] == (
        "🟡 HALF CLOSED\n"
        "Market: Will X?\n"
        "This sale: +3.00 (+20.0%)\n"
        "Position total realized so far: +3.00\n"
        "Still open: 20 tokens (~$10) riding"
    )


def test_notify_half_closed_near_zero_chunk_reads_as_spread(enabled):
    # A ~$0 chunk is correct (the spread ate the profit — Polymarket has no
    # fees); the message notes the spread instead of looking broken.
    pos = {"market_question": "Will X?", "realized_pnl_usd": 1.2,
           "token_count": 15.0, "amount_usd": 8.0}
    assert telegram_bot.notify_half_closed(pos, -0.04, -0.004) is True
    assert enabled[-1] == (
        "🟡 HALF CLOSED\n"
        "Market: Will X?\n"
        "This sale: -0.00 (-0.0%) (spread)\n"
        "Position total realized so far: +1.20\n"
        "Still open: 15 tokens (~$8) riding"
    )


def test_notify_half_closed_without_token_count(enabled):
    # Dry-run/legacy rows have no token_count — the riding line degrades to
    # dollars only, and the cumulative falls back to this chunk.
    pos = {"market_question": "Will X?", "amount_usd": 12.5}
    assert telegram_bot.notify_half_closed(pos, 20.0, 2.5) is True
    assert "Still open: ~$12 riding" in enabled[-1]
    assert "Position total realized so far: +2.50" in enabled[-1]


def test_notify_position_closed_shows_total_after_partials(enabled):
    # A full close after earlier half-closes: the final chunk and the
    # position's cumulative realized differ — show both.
    pos = {"market_question": "Will X?", "realized_pnl_usd": 5.0}
    assert telegram_bot.notify_position_closed(pos, 40.0, 2.0, "manual") is True
    assert enabled[-1] == (
        "🔴 CLOSED\n"
        "Market: Will X?\n"
        "This close: +2.00 (+40.00%) | Reason: manual\n"
        "Position total: +5.00 (incl. earlier partial sales)"
    )


def test_notify_position_closed_simple_when_total_matches_chunk(enabled):
    # No prior partial realizations: cumulative == chunk -> keep it simple.
    pos = {"market_question": "Will X?", "realized_pnl_usd": -4.5}
    assert telegram_bot.notify_position_closed(pos, -12.3, -4.5, "sl") is True
    assert enabled[-1] == (
        "🔴 CLOSED\n"
        "Market: Will X?\n"
        "PnL: -4.50 (-12.30%) | Reason: sl"
    )


def test_notify_passive_filled_format(enabled):
    pos = {"market_question": "Will X?", "side": "YES", "price": 0.41,
           "token_count": 24.0, "actual_cost_usd": 9.84}
    assert telegram_bot.notify_passive_filled(pos) is True
    assert enabled[-1] == (
        "🔵 PASSIVE FILL\n"
        "Market: Will X?\n"
        "Side: YES | Entry: 0.410\n"
        "Filled: 24 tokens for $9.84"
    )


def test_notify_passive_expired_no_fill(enabled):
    assert telegram_bot.notify_passive_expired("Will X?", 0.0) is True
    assert enabled[-1] == (
        "⚪ PASSIVE EXPIRED\n"
        "Market: Will X?\n"
        "No fill — nothing opened"
    )


def test_notify_passive_expired_partial_fill_wording(enabled):
    assert telegram_bot.notify_passive_expired("Will X?", 12.0) is True
    assert enabled[-1] == (
        "⚪ PASSIVE EXPIRED\n"
        "Market: Will X?\n"
        "Partial fill: 12 tokens opened as a position"
    )


def test_notify_passive_chased_away(enabled):
    assert telegram_bot.notify_passive_expired("Will X?", 0.0,
                                               reason="chased_away") is True
    assert enabled[-1] == (
        "⚪ PASSIVE CHASED AWAY\n"
        "Market: Will X?\n"
        "No fill — nothing opened"
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


def test_close_notifications_cumulative_through_real_path(tmp_db, enabled, monkeypatch):
    # End-to-end through positions._close: a half close then the full close.
    # The half-close message must show this chunk AND the cumulative realized;
    # the final close must show both the last chunk and the position total —
    # all on the same realized_pnl_usd basis calibration-report reads.
    import json
    market = types.SimpleNamespace(
        condition_id="c1", question="Will X happen?", slug="", yes_price=0.5,
        event_id="", category="crypto", end_date="", volume=5000,
    )
    positions_mod.open_position(1, market, "YES", 25.0, token_count=50.0)
    pos = positions_mod.get_open_positions()[0]

    def create(**kwargs):
        text = json.dumps({"decision": "close_half", "reasoning": "r"})
        return types.SimpleNamespace(content=[types.SimpleNamespace(text=text)])
    fake = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    monkeypatch.setattr(positions_mod.anthropic, "Anthropic", lambda api_key=None: fake)

    # Half close at 0.60: 25 of 50 tokens sold, +$2.50 on a $12.50 chunk.
    positions_mod.reevaluate(pos, trigger="news_reeval", yes_price=0.60)
    assert enabled[-1] == (
        "🟡 HALF CLOSED\n"
        "Market: Will X happen?\n"
        "This sale: +2.50 (+20.0%)\n"
        "Position total realized so far: +2.50\n"
        "Still open: 25 tokens (~$12) riding"
    )

    # Full close of the remainder at the marked 0.60: another +$2.50 chunk,
    # position total +$5.00 — both shown.
    positions_mod.close_manual(pos["id"])
    assert "This close: +2.50 (+20.00%)" in enabled[-1]
    assert "Position total: +5.00 (incl. earlier partial sales)" in enabled[-1]


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


def test_balance_deployed_counts_only_open_positions(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "PERFORMANCE_START_DATE", "")
    # Two open positions = $100 currently deployed.
    _open(1, "c1", "A", amount=50.0)
    _open(2, "c2", "B", amount=50.0)
    # A closed position whose $161 must NOT inflate deployed (capital returned).
    pid = _open(3, "c3", "C", amount=161.0)
    # Mark it off its 0.50 entry so the manual close realizes a real (non-zero)
    # PnL and counts as a closed trade — a break-even ($0.00) close is a non-event.
    conn = tmp_db._conn()
    conn.execute("UPDATE positions SET current_yes_price=0.60 WHERE id=?", (pid,))
    conn.commit(); conn.close()
    positions_mod.close_manual(pid)

    text, _ = telegram_bot._build_balance()
    assert "Deployed: $100.00" in text
    assert "Open: 2 |" in text
    assert "Closed: 1" in text


def test_balance_respects_performance_start_date(tmp_db, monkeypatch):
    # An open position from before the window must be excluded.
    old = _open(1, "c1", "OLD", amount=80.0)
    conn = tmp_db._conn()
    conn.execute("UPDATE positions SET opened_at='2020-01-01 00:00:00' WHERE id=?", (old,))
    conn.commit()
    conn.close()
    # An open position within the window.
    _open(2, "c2", "NEW", amount=30.0)
    monkeypatch.setattr(config, "PERFORMANCE_START_DATE", "2026-01-01")

    text, _ = telegram_bot._build_balance()
    assert "Deployed: $30.00" in text
    assert "Open: 1 |" in text


def test_portfolio_view_has_refresh_and_balance(tmp_db):
    pid = _open(1, "c1", "Will A happen?")
    text, markup = telegram_bot._build_portfolio()
    assert text.startswith("💼 PORTFOLIO")
    btns = _btns(markup)
    assert (f"🔴 Close #{pid}", f"close:{pid}") in btns
    assert ("📈 Refresh", "refresh") in btns
    assert ("💰 Balance", "balance") in btns


def test_portfolio_pct_shown_as_stored(tmp_db):
    # Stored unrealized_pnl_pct is already marked on the actual fill basis
    # (positions.pnl_pct_basis) — the portfolio shows it unchanged; a second
    # rebase would double-adjust.
    pid = _open(1, "c1", "Will A happen?", amount=25.0)
    conn = tmp_db._conn()
    conn.execute(
        "UPDATE positions SET unrealized_pnl_pct=23.8, actual_cost_usd=21.0 WHERE id=?",
        (pid,),
    )
    conn.commit()
    conn.close()
    text, _ = telegram_bot._build_portfolio()
    assert "(+23.8%)" in text  # stored basis-aware pct, displayed as-is


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


def _run_button_with_edit(data, edit):
    async def answer():
        pass

    query = types.SimpleNamespace(data=data, answer=answer, edit_message_text=edit)
    update = types.SimpleNamespace(callback_query=query, effective_chat=_chat())
    asyncio.run(telegram_bot._button_cmd(update, None))


def test_refresh_swallows_not_modified_error(tmp_db):
    _open(1, "c1", "Will A happen?")

    async def edit(text, reply_markup=None):
        raise RuntimeError("Message is not modified: same content and markup")

    # Tapping Refresh on an unchanged view must be a no-op, not an error.
    _run_button_with_edit("refresh", edit)


def test_other_edit_errors_propagate(tmp_db):
    _open(1, "c1", "Will A happen?")

    async def edit(text, reply_markup=None):
        raise RuntimeError("network down")

    with pytest.raises(RuntimeError, match="network down"):
        _run_button_with_edit("refresh", edit)


# --- Live-mode views & notifications -----------------------------------------

def test_build_balance_live_shows_real_balance(tmp_db, monkeypatch):
    from locus.core import executor
    monkeypatch.setattr(config, "DRY_RUN", False)
    monkeypatch.setattr(config, "PERFORMANCE_START_DATE", "")
    monkeypatch.setattr(executor, "get_live_balance", lambda: 245.0)
    _open(1, "c1", "A", amount=50.0)

    text, _ = telegram_bot._build_balance()
    assert "💰 BALANCE 🟢 LIVE MODE" in text
    assert "Real Balance: $245.00" in text
    # Live mode shows the real balance, not the computed deployed-capital figure.
    assert "Deployed:" not in text


def test_build_balance_live_unavailable_on_fetch_failure(tmp_db, monkeypatch):
    from locus.core import executor
    monkeypatch.setattr(config, "DRY_RUN", False)
    monkeypatch.setattr(executor, "get_live_balance", lambda: None)

    text, _ = telegram_bot._build_balance()
    assert "🟢 LIVE MODE" in text
    assert "Real Balance: unavailable" in text


def test_build_balance_dry_run_shows_deployed(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "PERFORMANCE_START_DATE", "")
    _open(1, "c1", "A", amount=50.0)

    text, _ = telegram_bot._build_balance()
    assert "💰 BALANCE 🔵 DRY RUN" in text
    assert "Deployed: $50.00" in text


def test_build_portfolio_badge_reflects_mode(tmp_db, monkeypatch):
    _open(1, "c1", "Will A happen?")
    monkeypatch.setattr(config, "DRY_RUN", False)
    assert telegram_bot._build_portfolio()[0].startswith("💼 PORTFOLIO 🟢 LIVE")
    monkeypatch.setattr(config, "DRY_RUN", True)
    assert telegram_bot._build_portfolio()[0].startswith("💼 PORTFOLIO 🔵 DRY RUN")


def test_build_portfolio_badge_when_empty(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", False)
    text, _ = telegram_bot._build_portfolio()
    assert text.startswith("💼 PORTFOLIO 🟢 LIVE")
    assert "No open positions" in text


def test_notify_position_opened_live_header(enabled, monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", False)
    pos = {
        "market_question": "Will BTC hit 100k?", "side": "YES",
        "entry_yes_price": 0.42, "amount_usd": 12.5, "edge": 0.10, "confidence": 0.7,
    }
    assert telegram_bot.notify_position_opened(pos) is True
    assert enabled[-1] == (
        "🟢 LIVE POSITION\n"
        "Market: Will BTC hit 100k?\n"
        "Side: YES | Entry: 0.420 | Amount: $12.50\n"
        "Edge: 10.0% | Conf: 70%"
    )


# --- /positions operator view (Close / Half / Force close) --------------------

def _open_sized(trade_id, cid, question, amount, tokens, yes=0.5, side="YES"):
    """Open a position with a real token_count so the view shows true holdings."""
    market = types.SimpleNamespace(
        condition_id=cid, question=question, slug="", yes_price=yes,
        event_id="", category="crypto", end_date="",
    )
    return positions_mod.open_position(trade_id, market, side, amount,
                                       token_count=tokens)


def test_positions_view_renders(tmp_db):
    pid = _open_sized(1, "c1", "Will the Maine candidate drop out?", 2.03, 3.5,
                      yes=0.05)
    text, markup = telegram_bot._build_positions()
    assert text.startswith("📋 POSITIONS")
    assert f"#{pid} YES Will the Maine candidate drop out?" in text
    assert "3.5 tok | cost $2.03 | value $0.18" in text  # 3.5 x 0.05
    btns = _btns(markup)
    assert (f"🔴 Close #{pid}", f"pclose:{pid}") in btns
    assert (f"🟡 Half #{pid}", f"phalf:{pid}") in btns
    assert (f"⚠️ Force #{pid}", f"pforce:{pid}") in btns
    assert ("🔄 Refresh", "positions") in btns


def test_positions_command_renders(tmp_db):
    _open_sized(1, "c1", "Will A happen?", 10.0, 20.0)
    replies = []

    async def reply_text(text, reply_markup=None):
        replies.append(text)

    update = types.SimpleNamespace(
        message=types.SimpleNamespace(reply_text=reply_text),
        effective_chat=_chat(),
    )
    asyncio.run(telegram_bot._positions_cmd(update, None))
    assert replies and replies[0].startswith("📋 POSITIONS")


def test_positions_close_success(tmp_db):
    pid = _open_sized(1, "c1", "Will A happen?", 10.0, 20.0)
    text, _ = _run_button(f"pclose:{pid}")
    assert f"✅ Closed #{pid}" in text
    assert "No open positions" in text  # embedded refreshed list
    assert positions_mod.get_open_positions() == []


@pytest.mark.parametrize("status,phrase", [
    ("skipped_thin_book", "below exchange minimums"),
    ("skipped_empty_book", "no bids at all"),
    ("skipped_wide_spread", "spread too wide"),
])
def test_positions_close_failure_reason_surfaces(tmp_db, monkeypatch, status, phrase):
    from locus.core import executor
    monkeypatch.setattr(config, "DRY_RUN", False)

    def fake_close(condition_id, side, shares, max_spread=None, allow_topup=False):
        return {"status": status, "order_id": None, "price": None, "shares": None}

    monkeypatch.setattr(executor, "close_position_live", fake_close)
    pid = _open_sized(1, "c1", "Will A happen?", 10.0, 20.0)

    text, _ = _run_button(f"pclose:{pid}")
    assert f"❌ Closed #{pid} failed" in text
    assert phrase in text
    assert positions_mod.get_open_positions()  # still open


def test_close_failure_reports_topup_skip(tmp_db, monkeypatch):
    # A sub-minimum holding whose top-up was skipped: the operator must learn
    # the position is genuinely stuck, and why.
    from locus.core import executor
    monkeypatch.setattr(config, "DRY_RUN", False)

    def fake_close(condition_id, side, shares, max_spread=None, allow_topup=False):
        return {"status": "skipped_thin_book", "order_id": None, "price": None,
                "shares": None, "topup_skipped": "no_bid_liquidity"}

    monkeypatch.setattr(executor, "close_position_live", fake_close)
    pid = _open_sized(1, "c1", "Will A happen?", 0.72, 3.5, yes=0.05)

    text, _ = _run_button(f"pclose:{pid}")
    assert "top-up skipped: no real bid liquidity" in text
    assert "stuck until resolution" in text


def test_half_button_realizes_half(tmp_db, enabled):
    pid = _open_sized(1, "c1", "Will A happen?", 40.0, 80.0)
    text, _ = _run_button(f"phalf:{pid}")
    assert f"✅ Half closed #{pid}" in text
    p = positions_mod.get_open_positions()[0]
    assert p["token_count"] == pytest.approx(40.0)
    assert p["amount_usd"] == pytest.approx(20.0)
    # The manual action produced the standard half-close notification.
    assert enabled[-1].startswith("🟡 HALF CLOSED")


def test_half_button_escalates_dust_to_full_close(tmp_db, enabled):
    # 3.5 tokens at 0.05 is worth $0.18 — far below the remainder floor, so
    # the manual Half escalates to a FULL close with the reason shown.
    pid = _open_sized(1, "c1", "Will A happen?", 2.03, 3.5, yes=0.05)
    text, _ = _run_button(f"phalf:{pid}")
    assert f"✅ Closed #{pid}" in text
    assert "half escalated to FULL close" in text
    assert positions_mod.get_open_positions() == []
    assert enabled[-1].startswith("🔴 CLOSED")


# --- Force close: two-step confirmation ----------------------------------------

def test_force_close_first_tap_previews_only(tmp_db, monkeypatch):
    monkeypatch.setattr(telegram_bot, "_fetch_best_bid", lambda p: 0.007)
    pid = _open_sized(1, "c1", "Will A happen?", 0.72, 18.25, yes=0.05)

    text, markup = _run_button(f"pforce:{pid}")
    # Preview shows the realized outcome at the live best bid...
    assert "Best bid 0.007 vs mark 0.050" in text
    assert "Selling 18.25 tokens returns ~$0.13 (you paid $0.72)" in text
    assert "This realizes -82%" in text
    assert "Confirm?" in text
    btns = _btns(markup)
    assert ("⚠️ Yes, force", f"pforceyes:{pid}") in btns
    assert ("Cancel", "positions") in btns
    # ...and nothing was sold.
    assert positions_mod.get_open_positions()


def test_force_close_second_tap_executes(tmp_db, monkeypatch):
    monkeypatch.setattr(telegram_bot, "_fetch_best_bid", lambda p: 0.007)
    pid = _open_sized(1, "c1", "Will A happen?", 0.72, 18.25, yes=0.05)

    _run_button(f"pforce:{pid}")  # preview
    assert positions_mod.get_open_positions()  # still open after tap 1
    text, _ = _run_button(f"pforceyes:{pid}")  # confirm
    assert f"✅ Force closed #{pid}" in text
    assert positions_mod.get_open_positions() == []


def test_force_preview_falls_back_to_mark_without_live_bid(tmp_db, monkeypatch):
    monkeypatch.setattr(telegram_bot, "_fetch_best_bid", lambda p: None)
    pid = _open_sized(1, "c1", "Will A happen?", 0.72, 18.25, yes=0.05)
    text, _ = _run_button(f"pforce:{pid}")
    assert "live bid unavailable" in text
    assert "Confirm?" in text


def test_force_close_subminimum_holding_tries_topup(tmp_db, monkeypatch):
    # A confirmed force close on a sub-minimum holding must go through the
    # executor with top-up armed and the spread gate disabled.
    from locus.core import executor
    monkeypatch.setattr(config, "DRY_RUN", False)
    calls = []

    def fake_close(condition_id, side, shares, max_spread=None, allow_topup=False):
        calls.append({"shares": shares, "max_spread": max_spread,
                      "allow_topup": allow_topup})
        return {"status": "executed", "order_id": "F-1", "price": 0.04,
                "shares": shares, "sold_shares": shares, "remaining_shares": 0.0}

    monkeypatch.setattr(executor, "close_position_live", fake_close)
    pid = _open_sized(1, "c1", "Will A happen?", 0.72, 3.5, yes=0.05)

    text, _ = _run_button(f"pforceyes:{pid}")
    assert calls and calls[0]["allow_topup"] is True
    assert calls[0]["max_spread"] == float("inf")
    assert f"✅ Force closed #{pid}" in text


# --- Auth: only the configured chat id may act ---------------------------------

def test_unauthorized_chat_button_rejected(tmp_db):
    pid = _open_sized(1, "c1", "Will A happen?", 10.0, 20.0)
    text, markup = _run_button(f"pclose:{pid}", chat_id="999")
    assert text is None  # no edit performed at all
    assert positions_mod.get_open_positions()  # position untouched


def test_unauthorized_positions_command_ignored(tmp_db):
    replies = []

    async def reply_text(text, reply_markup=None):
        replies.append(text)

    update = types.SimpleNamespace(
        message=types.SimpleNamespace(reply_text=reply_text),
        effective_chat=_chat("999"),
    )
    asyncio.run(telegram_bot._positions_cmd(update, None))
    asyncio.run(telegram_bot._portfolio_cmd(update, None))
    assert replies == []


def test_no_configured_chat_id_rejects_everything(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "TELEGRAM_CHAT_ID", "")
    pid = _open_sized(1, "c1", "Will A happen?", 10.0, 20.0)
    text, _ = _run_button(f"pclose:{pid}", chat_id="")
    assert text is None
    assert positions_mod.get_open_positions()
