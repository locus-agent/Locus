"""
Telegram bot for real-time trading notifications and an interactive portfolio.

Two independent halves:

1. Notifications (sendMessage over the Telegram HTTP API via httpx). These are
   fire-and-forget, synchronous, and thread-safe, so they can be called straight
   from the pipeline's executor threads (executor, positions, journal). They are
   no-ops — never raising — when TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is unset.

2. Interactive command bot (python-telegram-bot Application long-polling).
   /portfolio and /positions render the portfolio card: one row per open
   position (PnL% or "stuck" for sub-minimum holdings) with [📊 #id] detail
   buttons — NO sell buttons in the list. A detail card shows the full
   DB-authoritative numbers, enriched (never overwritten) by Polymarket's
   public Data API: live curPrice, event link/end date, and an independent
   PnL cross-check (✓ / ⚠️ расходится past tolerance / unavailable). Selling
   happens only from detail cards: [Close] [Half] [Force] on a normal card;
   a provably-unsellable (stuck) card offers only Force, whose two-step
   preview shows the realized outcome at the current best bid first. Every
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
import time
from datetime import datetime

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

# --- Polymarket public Data API (read-only enrichment) ------------------------
#
# GET {POLYMARKET_DATA_HOST}/positions?user={proxy wallet} — no auth. Used ONLY
# for presentation extras: the live curPrice, event slug / end date for links,
# and an INDEPENDENT cross-check of our PnL. Our DB (token_count,
# actual_cost_usd, realized_pnl_usd) stays the source of truth for OUR numbers
# — API values are never written back. Any API failure degrades to DB-only
# cards; it must never break a card or its buttons.

_API_POSITIONS_TTL = 45.0  # seconds; a /positions render + Refresh taps share one fetch

# {"ts": monotonic timestamp of last attempt, "rows": list | None}. A failed
# fetch caches None for the TTL too, so a Refresh burst against a down API
# doesn't hammer the endpoint.
_api_positions_cache: dict = {"ts": 0.0, "rows": None}


def _fetch_api_positions() -> list[dict] | None:
    """Rows from the Data API for our wallet (config.POLYMARKET_FUNDER_ADDRESS),
    cached for _API_POSITIONS_TTL seconds. None when the wallet is unset, the
    call fails, or the response isn't a list — callers degrade to DB-only."""
    now = time.monotonic()
    if now - _api_positions_cache["ts"] < _API_POSITIONS_TTL:
        return _api_positions_cache["rows"]
    _api_positions_cache["ts"] = now
    rows = None
    wallet = config.POLYMARKET_FUNDER_ADDRESS
    if wallet:
        try:
            resp = httpx.get(
                f"{config.POLYMARKET_DATA_HOST}/positions",
                params={"user": wallet}, timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                rows = data
            else:
                log.warning("[telegram] Data API positions: unexpected shape %s",
                            type(data).__name__)
        except Exception as e:
            log.warning(f"[telegram] Data API positions fetch failed: {e}")
    _api_positions_cache["rows"] = rows
    return rows


def _api_row_for(position: dict, api_rows: list[dict] | None) -> dict | None:
    """The Data API row for one of our positions: matched by conditionId +
    outcome (our side). None when the API is down or the position is absent."""
    for row in api_rows or []:
        if (row.get("conditionId") == position["condition_id"]
                and str(row.get("outcome", "")).upper() == position["side"].upper()):
            return row
    return None


def _usd_signed(v: float) -> str:
    """-0.59 -> '-$0.59', 3.6 -> '+$3.60' (sign before the currency symbol)."""
    return f"{'+' if v >= 0 else '-'}${abs(v):.2f}"


def _short_name(question: str, limit: int = 26) -> str:
    """One-line market name for portfolio rows: drop the 'Will …?' scaffolding
    and truncate."""
    name = (question or "").strip()
    if name.lower().startswith("will "):
        name = name[5:]
    name = name.rstrip("?")
    return name[:limit].rstrip()


def _end_date_dt(end_date) -> datetime | None:
    """Parse an ISO/Gamma end date into a datetime, or None."""
    if not end_date:
        return None
    try:
        return datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _position_card_data(p: dict, api_rows: list[dict] | None) -> dict:
    """Every display number for one open position.

    DB-authoritative: tokens, cost basis, realized come from our row. The
    matched API row contributes the live curPrice (the held outcome token's
    price — no side adjustment needed), the event link / end date, and the
    independent PnL cross-check; without it everything falls back to the DB
    mark and the verification line reads 'unavailable'."""
    from locus.core import positions
    from locus.core.performance import position_shares

    tokens = position_shares(
        p["side"], p["entry_yes_price"], p.get("amount_usd") or 0.0,
        token_count=p.get("token_count"),
    )
    cost = positions.position_cost_basis(p)
    entry_price = cost / tokens if tokens > 0 else 0.0

    api = _api_row_for(p, api_rows)
    cur_price = None
    if api is not None:
        try:
            cur_price = float(api["curPrice"])
        except (KeyError, TypeError, ValueError):
            cur_price = None
    if cur_price is None:
        yes = p.get("current_yes_price")
        yes = p["entry_yes_price"] if yes is None else yes
        price = yes if p["side"] == "YES" else 1.0 - yes
    else:
        price = cur_price

    value = tokens * price
    pnl_usd = value - cost
    pnl_pct = pnl_usd / cost * 100.0 if cost > 0 else 0.0
    # Stuck = below the exchange sell minimums: can NOT be sold as-is.
    stuck = (tokens < config.MIN_ORDER_SHARES
             or value < config.MIN_ORDER_USD)

    # Independent cross-check of PnL against Polymarket's own accounting.
    # Within tolerance when the dollar OR the percent difference is small;
    # beyond both -> accounting drift (we've shipped a bad entry price
    # before), flagged on the card and logged.
    verify = None
    if api is not None:
        try:
            api_cash = float(api["cashPnl"])
            api_pct = float(api["percentPnl"])
        except (KeyError, TypeError, ValueError):
            api_cash = api_pct = None
        if api_cash is not None:
            ok = (abs(api_cash - pnl_usd) <= 0.10
                  or abs(api_pct - pnl_pct) <= 2.0)
            if not ok:
                log.warning(
                    "[telegram] PnL cross-check DISAGREES on position %s: ours "
                    "%+.2f (%+.1f%%) vs Polymarket %+.2f (%+.1f%%) — check the "
                    "entry price / cost basis", p["id"], pnl_usd, pnl_pct,
                    api_cash, api_pct,
                )
            verify = {"cash": api_cash, "pct": api_pct, "ok": ok}

    # End date: DB first (kept fresh from Gamma by positions.refresh_end_dates),
    # API as fallback. The Data API returns epoch-zero placeholders for some
    # markets ('1970-01-01' on the live Maine row) — treat anything pre-2000
    # as unknown rather than showing "ends Jan 1970".
    end_dt = None
    for candidate in (p.get("end_date"), (api or {}).get("endDate")):
        dt = _end_date_dt(candidate)
        if dt is not None and dt.year >= 2000:
            end_dt = dt
            break
    event_slug = (api or {}).get("eventSlug")
    if event_slug:
        link = f"https://polymarket.com/event/{event_slug}"
    elif p.get("slug"):
        link = f"https://polymarket.com/market/{p['slug']}"
    else:
        link = None

    return {
        "tokens": tokens, "cost": cost, "entry_price": entry_price,
        "price": price, "value": value, "pnl_usd": pnl_usd, "pnl_pct": pnl_pct,
        "stuck": stuck, "api": api, "verify": verify,
        "end_dt": end_dt, "link": link,
    }


def _position_history(p: dict) -> dict:
    """DB context for the detail card: how many partial sales this position
    has had (close_half / partial_close rows in exit_decisions), the trade's
    original stake (for the 'из $X' display after partial sales), and whether
    the entry came from a passive limit order (a pending_orders row exists
    for its trade)."""
    from locus.memory import logger as db

    conn = db._conn()
    try:
        sales = conn.execute(
            "SELECT COUNT(*) AS c FROM exit_decisions "
            "WHERE position_id=? AND decision IN ('close_half', 'partial_close')",
            (p["id"],),
        ).fetchone()["c"]
        original = None
        passive = False
        if p.get("trade_id"):
            row = conn.execute(
                "SELECT amount_usd FROM trades WHERE id=?", (p["trade_id"],)
            ).fetchone()
            original = row["amount_usd"] if row else None
            passive = conn.execute(
                "SELECT 1 FROM pending_orders WHERE trade_id=? LIMIT 1",
                (p["trade_id"],),
            ).fetchone() is not None
    finally:
        conn.close()
    return {"partial_sales": sales, "original_stake": original, "passive": passive}


_PORTFOLIO_PAGE_ROWS = 20  # positions per portfolio page (Telegram 4096-char cap)


def _build_portfolio(page: int = 0):
    """(text, InlineKeyboardMarkup) for the portfolio card (/portfolio and
    /positions): one line per open position (colored dot, id/side/name, PnL%
    or 'stuck', current value), a USDC balance line when readable, and a
    footer with open/realized totals. NO sell buttons here — selling happens
    only from a detail card where the numbers are visible; each position gets
    a [📊 #id] detail button (3 per row). Paginates past _PORTFOLIO_PAGE_ROWS."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    from locus.core import positions
    from locus.core import executor
    from locus.core.performance import compute_performance

    open_pos = positions.get_open_positions()
    api_rows = _fetch_api_positions()

    lines = [f"💼 PORTFOLIO · {_mode_badge()}"]
    if not config.DRY_RUN:
        balance = executor.get_live_balance()
        if balance is not None:
            lines.append(f"USDC: ${balance:.2f}")
    lines.append("")

    pages = max(1, -(-len(open_pos) // _PORTFOLIO_PAGE_ROWS))
    page = max(0, min(page, pages - 1))
    chunk = open_pos[page * _PORTFOLIO_PAGE_ROWS:(page + 1) * _PORTFOLIO_PAGE_ROWS]

    open_total = 0.0
    for p in open_pos:
        open_total += positions.position_cost_basis(p)

    if open_pos:
        for p in chunk:
            d = _position_card_data(p, api_rows)
            dot = "🟡" if d["stuck"] else ("🟩" if d["pnl_usd"] >= 0 else "🟥")
            # A PnL% on an unsellable holding is misleading — say "stuck".
            pct_part = "stuck" if d["stuck"] else f"{d['pnl_pct']:+.1f}%"
            lines.append(
                f"{dot} #{p['id']} {p['side']} · {_short_name(p['market_question'])}"
                f"  {pct_part}  ${d['value']:.2f}"
            )
    else:
        lines.append("No open positions.")

    perf = compute_performance()
    lines.append("")
    lines.append(
        f"Открыто: ${open_total:.2f} · "
        f"Realized: {perf['realized_pnl_usd']:+.2f} · "
        f"{perf['closed_count']} closed"
    )
    if pages > 1:
        lines.append(f"Стр. {page + 1}/{pages}")

    rows = []
    btn_row = []
    for p in chunk:
        btn_row.append(InlineKeyboardButton(f"📊 #{p['id']}",
                                            callback_data=f"pos:{p['id']}"))
        if len(btn_row) == 3:
            rows.append(btn_row)
            btn_row = []
    if btn_row:
        rows.append(btn_row)
    if pages > 1:
        nav = []
        if page > 0:
            nav.append(InlineKeyboardButton("⬅️", callback_data=f"portfolio:{page - 1}"))
        if page < pages - 1:
            nav.append(InlineKeyboardButton("➡️", callback_data=f"portfolio:{page + 1}"))
        rows.append(nav)
    rows.append([
        InlineKeyboardButton("🔄 Refresh", callback_data=f"portfolio:{page}"),
        InlineKeyboardButton("💰 Balance", callback_data="balance"),
    ])
    return "\n".join(lines), InlineKeyboardMarkup(rows)


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


def _build_position_detail(pid: int):
    """(text, InlineKeyboardMarkup) for one position's detail card — the ONLY
    place with sell buttons, so the operator always sees the numbers before
    acting.

    Normal card: our cost basis / shares / live price / value / PnL, the
    Data-API cross-check line (✓ when Polymarket's cashPnl agrees with ours,
    ⚠️ расходится past tolerance, 'unavailable' when the API is down), the end
    date and a Polymarket link, and [Close] [Half] [Force] buttons.

    Stuck card (holding below the exchange sell minimums): additionally shows
    the original stake and realized-so-far after partial sales, an explicit
    "нельзя продать" block with the real top-up blocker when one was recorded,
    and NO plain Close/Half buttons — Force (with its preview) is the only
    exit for a provably unsellable holding."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    from locus.core import positions

    p = next((x for x in positions.get_open_positions() if x["id"] == pid), None)
    if p is None:
        return _build_action_result(
            f"⚠️ Position #{pid} not found or already closed.")

    d = _position_card_data(p, _fetch_api_positions())
    hist = _position_history(p)

    header = f"📊 Position #{pid} · {_mode_badge()}"
    if hist["passive"]:
        header += " · passive fill"
    dot = "🟡" if d["stuck"] else ("🟩" if d["pnl_usd"] >= 0 else "🟥")
    lines = [header, f"{dot} {p['side']} · {p['market_question']}", ""]

    invested = f"Вложено:     ${d['cost']:.2f}"
    if hist["partial_sales"] and hist["original_stake"]:
        invested += f" ← из ${hist['original_stake']:.2f}"
    lines.append(invested)
    lines.append(f"Shares:      {_shares(round(d['tokens'], 3))} @ ${d['entry_price']:.3f}")
    lines.append(f"Тек. цена:   ${d['price']:.3f}")
    lines.append(f"Стоимость:   ${d['value']:.2f}")
    pnl_dot = "🟢" if d["pnl_usd"] >= 0 else "🔴"
    lines.append(f"PnL:         {pnl_dot} {_usd_signed(d['pnl_usd'])} ({d['pnl_pct']:+.1f}%)")
    if hist["partial_sales"]:
        realized = p.get("realized_pnl_usd") or 0.0
        n = hist["partial_sales"]
        sales_word = "продажа" if n == 1 else ("продажи" if n < 5 else "продаж")
        lines.append(f"Realized:    {_usd_signed(realized)} · {n} {sales_word}")

    lines.append("")
    if d["verify"] is not None:
        v = d["verify"]
        mark = "✓" if v["ok"] else "⚠️ расходится"
        lines.append(
            f"🔍 Polymarket API: {_usd_signed(v['cash'])} ({v['pct']:+.1f}%) {mark}")
    else:
        lines.append("🔍 Polymarket API: unavailable")

    if d["stuck"]:
        lines.append("")
        lines.append(
            f"⚠️ Нельзя продать. Ниже минимумов биржи "
            f"({config.MIN_ORDER_SHARES:g} shares / ${config.MIN_ORDER_USD:g})."
        )
        from locus.core.positions import _last_close_failure
        skip = (_last_close_failure.get(pid) or {}).get("topup_skipped")
        if skip:
            lines.append(f"   Top-up отклонён: {_TOPUP_SKIP_TEXT.get(skip, skip)}.")
        when = d["end_dt"].strftime("%Y-%m-%d") if d["end_dt"] else "дата неизвестна"
        lines.append(f"   → застряла до резолюции ({when}).")
        lines.append(f"   При выигрыше вернёт ${d['tokens']:.2f}")

    end_dt = d["end_dt"]
    footer = f"⏱ ends {end_dt.strftime('%b %Y')}" if end_dt else "⏱ ends: unknown"
    lines.append("")
    lines.append(footer)
    lines.append(f"🕒 {datetime.now().strftime('%H:%M')}")

    if d["stuck"]:
        rows = [[InlineKeyboardButton("⚠️ Force (превью)",
                                      callback_data=f"pforce:{pid}")]]
    else:
        rows = [[
            InlineKeyboardButton("🔴 Close", callback_data=f"pclose:{pid}"),
            InlineKeyboardButton("½ Half", callback_data=f"phalf:{pid}"),
            InlineKeyboardButton("⚠️ Force", callback_data=f"pforce:{pid}"),
        ]]
    rows.append([
        InlineKeyboardButton("🔄 Refresh", callback_data=f"pos:{pid}"),
        InlineKeyboardButton("📁 Portfolio", callback_data="portfolio"),
    ])
    if d["link"]:
        rows.append([InlineKeyboardButton("🔗 Polymarket", url=d["link"])])
    return "\n".join(lines), InlineKeyboardMarkup(rows)


def _build_action_result(header: str):
    """(text, markup) for a manual-action outcome: the result header above the
    freshly-refreshed portfolio card."""
    text, markup = _build_portfolio()
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
    """(text, markup) for the force-close confirmation step: the FULL realized
    outcome (held / paid / mark / best bid / proceeds / realized $ and %) at
    the CURRENT best bid, plus a dead-book warning when the bid sits far below
    the mark. Nothing is sold until the [✅ Да, force] tap."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    from locus.core import positions

    p = next((x for x in positions.get_open_positions() if x["id"] == pid), None)
    if p is None:
        return _build_action_result(
            f"⚠️ Position #{pid} not found or already closed.")

    d = _position_card_data(p, _fetch_api_positions())
    tokens, cost, mark = d["tokens"], d["cost"], d["price"]
    bid = _fetch_best_bid(p)
    bid_known = bid is not None
    if not bid_known:
        bid = mark
    proceeds = tokens * bid
    realized = proceeds - cost
    realized_pct = realized / cost * 100.0 if cost > 0 else 0.0

    lines = [
        f"⚠️ FORCE CLOSE #{pid} — {p['market_question'][:50]}",
        f"Держим: {_shares(round(tokens, 2))} tok · Заплачено: ${cost:.2f}",
        f"Марк-цена: {mark:.3f} · Лучший бид: {bid:.3f}"
        + ("" if bid_known else " (бид недоступен — берём марк)"),
        f"Вернётся ≈ ${proceeds:.2f}",
        f"Реализует {_usd_signed(realized)} ({realized_pct:+.0f}%)",
    ]
    if bid_known and mark > 0 and bid < mark * 0.5:
        lines.append("⚠️ Мёртвый стакан: лучший бид далеко от марк-цены.")
    lines.append("Это необратимо. Продолжить?")
    markup = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Да, force", callback_data=f"pforceyes:{pid}"),
        InlineKeyboardButton("✖️ Отмена", callback_data=f"pos:{pid}"),
    ]])
    return "\n".join(lines), markup


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


async def _portfolio_cmd(update, context):
    """/portfolio, /positions (and /start): show the portfolio card."""
    if not _authorized(update):
        return
    text, markup = _build_portfolio()
    await update.message.reply_text(text, reply_markup=markup)


async def _button_cmd(update, context):
    """Inline-button taps: navigate (portfolio/balance/detail cards),
    close/half/force-close a position from its detail card."""
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

    if data in ("portfolio", "refresh", "positions"):
        text, markup = _build_portfolio()
    elif data.startswith("portfolio:"):
        text, markup = _build_portfolio(_pid() or 0)
    elif data == "balance":
        text, markup = _build_balance()
    elif data.startswith("pos:"):
        pid = _pid()
        if pid is None:
            return
        text, markup = _build_position_detail(pid)
    elif data.startswith("pclose:"):
        from locus.core import positions
        pid = _pid()
        if pid is None:
            return
        result = positions.close_manual(pid)
        text, markup = _build_action_result(
            _manual_action_header(pid, result, "Closed"))
    elif data.startswith("phalf:"):
        from locus.core import positions
        pid = _pid()
        if pid is None:
            return
        result = positions.close_manual_half(pid)
        label = "Closed" if (result or {}).get("escalated") else "Half closed"
        text, markup = _build_action_result(
            _manual_action_header(pid, result, label))
    elif data.startswith("pforce:"):
        # Step 1 of 2: preview only — show the realized outcome at the current
        # best bid. Nothing is sold until the [✅ Да, force] tap.
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
        text, markup = _build_action_result(
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
        app.add_handler(CommandHandler(["portfolio", "positions", "start"],
                                       _portfolio_cmd))
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
