"""
Position tracking and exits — rules trigger, the agent decides.

Open positions are tracked in the positions table and marked to the live
watcher prices each pipeline cycle. Exits (all simulated in dry-run, like
entries):

- HARD stop-loss at STOP_LOSS_PCT: unconditional close, no Claude call.
  Safety never depends on a model call.
- Rule triggers (take_profit at TAKE_PROFIT_TRIGGER_PCT, drawdown at
  REEVAL_LOSS_PCT, contradicting material news) cause ONE Claude
  re-evaluation with full context; Claude answers hold / close /
  close_half. Decisions + reasoning land in exit_decisions — the public
  "agent changed its mind" log.
- Cooldown: max one re-evaluation per position per REEVAL_COOLDOWN_HOURS,
  unless a different trigger type fires.
- Market resolution remains the natural exit otherwise.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone

import anthropic

from locus import config
from locus.markets import gamma
from locus.memory import logger
from locus.core.performance import position_pnl, position_pnl_basis, position_shares
from locus.core import telegram_bot
from locus.core import executor

# Map a position's granular exit_reason to a canonical re-entry close reason.
# 'resolution' is excluded from watching (no point watching a resolved market).
_CLOSE_REASON_MAP = {
    "sl": "sl",
    "tp_decision": "tp",
    "news_decision": "news",
    "drawdown_decision": "sl",  # loss-driven exit -> treat as strict as a stop
    "resolution": "resolution",
}


def _canonical_close_reason(exit_reason: str) -> str:
    """Bucket an exit_reason into sl/tp/news/resolution for re-entry rules."""
    return _CLOSE_REASON_MAP.get(exit_reason, "sl")

log = logging.getLogger(__name__)


# --- Re-entry 2.0: exit_reason calibration gate -------------------------------
#
# A granular re-entry gate keyed on *why* the position closed. Unlike the
# bucketed reentry.check_reentry_opportunity (sl/tp/news), this inspects the raw
# exit_reason against the calibrated REENTRY_ALLOWED_REASONS / _BLOCKED_REASONS
# lists: exits that left a clean thesis (tp_decision, manual, near_certain_*,
# resolution) may re-enter; exits where the market beat us (drawdown_decision,
# time_pressure, hard_sl, news_decision, already_priced_in) never do.

def _normalize_exit_reason(exit_reason: str) -> str:
    """Collapse a stored exit_reason to its re-entry list key. near_certain
    exits carry a price suffix ("near_certain_yes_0.96"); the hard stop loss is
    stored as "sl" but the calibration lists call it "hard_sl"."""
    r = (exit_reason or "").lower()
    if r.startswith("near_certain_yes"):
        return "near_certain_yes"
    if r.startswith("near_certain_no"):
        return "near_certain_no"
    if r == "sl":
        return "hard_sl"
    return r


def event_reentry_count(conn, event_id: str | None) -> int:
    """How many re-entry trades (edge_type='reentry') already exist for this
    event_id. 0 when the event is unknown."""
    if not event_id:
        return 0
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM trades WHERE event_id=? AND edge_type='reentry'",
        (event_id,),
    ).fetchone()
    return row["c"] if row else 0


def check_reentry_opportunity(
    exit_reason: str,
    materiality: float,
    base_size_usd: float,
    hours_since_close: float,
    hours_to_resolution: float | None = None,
    event_id: str | None = None,
    conn=None,
) -> dict | None:
    """Re-entry 2.0 gate: decide whether a watched market may be re-entered,
    driven by the granular exit_reason and the new classification's materiality.

    Returns None when re-entry is blocked (each rejection logs its reason), or
    {"size_usd": float, "reason": str} when allowed — size already scaled by
    REENTRY_SIZE_FACTOR.
    """
    if not config.REENTRY_ENABLED:
        log.info("[positions] Re-entry blocked: feature disabled (REENTRY_ENABLED=false)")
        return None

    reason_key = _normalize_exit_reason(exit_reason)

    # Explicit block list takes precedence over everything else.
    if reason_key in config.REENTRY_BLOCKED_REASONS:
        log.info(f"[positions] Re-entry blocked: exit_reason '{reason_key}' is on the block list")
        return None

    # Only re-enter on an explicitly allowed exit reason (unknown reasons block).
    if reason_key not in config.REENTRY_ALLOWED_REASONS:
        log.info(f"[positions] Re-entry blocked: exit_reason '{reason_key}' not on the allow list")
        return None

    # Cooldown: don't re-enter the same market within REENTRY_MIN_HOURS of exit.
    if hours_since_close < config.REENTRY_MIN_HOURS:
        log.info(
            f"[positions] Re-entry blocked: only {hours_since_close:.1f}h since close "
            f"(< {config.REENTRY_MIN_HOURS}h cooldown)"
        )
        return None

    # Don't re-enter a market that resolves too soon for the thesis to play out
    # (skip the check when the close time is unknown — a None hours_to_resolution).
    if (hours_to_resolution is not None
            and hours_to_resolution < config.REENTRY_MIN_HOURS_TO_RESOLUTION):
        log.info(
            f"[positions] Re-entry blocked: {hours_to_resolution:.1f}h to resolution "
            f"(< {config.REENTRY_MIN_HOURS_TO_RESOLUTION}h)"
        )
        return None

    # Per-event cap: at most REENTRY_MAX_PER_EVENT re-entries across one event.
    if conn is not None and event_id:
        existing = event_reentry_count(conn, event_id)
        if existing >= config.REENTRY_MAX_PER_EVENT:
            log.info(
                f"[positions] Re-entry blocked: event {event_id} already has "
                f"{existing} re-entry(s) (max {config.REENTRY_MAX_PER_EVENT})"
            )
            return None

    # Materiality floor.
    if materiality < config.REENTRY_MIN_MATERIALITY:
        log.info(
            f"[positions] Re-entry blocked: materiality {materiality:.2f} "
            f"< {config.REENTRY_MIN_MATERIALITY}"
        )
        return None

    size_usd = round(base_size_usd * config.REENTRY_SIZE_FACTOR, 2)
    reason = (
        f"exit_reason '{reason_key}', materiality {materiality:.2f}, "
        f"size ${size_usd:.2f} ({config.REENTRY_SIZE_FACTOR:g}x of ${base_size_usd:.2f})"
    )
    log.info(f"[positions] Re-entry allowed: {reason}")
    return {"size_usd": size_usd, "reason": reason}

EXIT_PROMPT = """You are Locus, an autonomous agent trading niche Polymarket prediction \
markets. One of your open positions hit a re-evaluation trigger. Decide what to do with it.

## Position
Market: {question}
Side: {side} | Entry YES price: {entry_yes_price:.3f} | Current YES price: {current_yes_price:.3f}
Position size: ${amount_usd:.2f} | Unrealized PnL: {pnl_pct:+.1f}%
Opened: {opened_at} | Time to market close: {time_remaining}

## Why you opened it
Headline then: {entry_headline}
Your reasoning then: {entry_reasoning}

## Trigger
{trigger_description}

## Task
Decide: "hold" (keep the position), "close" (exit fully at the current price), or
"close_half" (realize half, let the rest ride). Consider whether your original thesis
still holds, whether the move already captured the edge, and the time remaining.
Locking in profit on thin markets is often better than waiting for resolution;
holding a loser needs a reason beyond hope.

Respond with ONLY valid JSON:
{{"decision": "hold" | "close" | "close_half", "reasoning": "<1-2 sentences>"}}"""

TRIGGER_DESCRIPTIONS = {
    "take_profit": "Unrealized PnL crossed +{tp:.0f}% (take-profit review).",
    "drawdown": "Unrealized PnL fell below {dd:.0f}% (drawdown review).",
    "news": "Fresh material headline contradicts your side: {detail}",
}

# trigger -> (status, exit_reason) when the decision is to close
CLOSE_MAP = {
    "take_profit": ("closed_tp", "tp_decision"),
    "drawdown": ("closed_drawdown", "drawdown_decision"),
    "news": ("closed_news", "news_decision"),
}


def pnl_pct(side: str, entry_yes: float, now_yes: float) -> float:
    entry = entry_yes if side == "YES" else 1.0 - entry_yes
    now = now_yes if side == "YES" else 1.0 - now_yes
    entry = min(max(entry, 1e-6), 1.0)
    return (now / entry - 1.0) * 100.0


def pnl_pct_basis(position: dict, yes_price_now: float) -> float:
    """PnL% of a position marked at yes_price_now, on the actual fill basis
    when known — the % counterpart of performance.position_pnl_basis
    (token_count real shares x side price vs amount_usd real cost), so the
    stop-loss/take-profit/drawdown triggers, the stored unrealized mark, and
    the displays all judge the same dollars a close would realize. Falls back
    to the legacy price-ratio pnl_pct for dry-run/legacy rows (no
    token_count)."""
    amount = position.get("amount_usd") or 0.0
    shares = position.get("token_count")
    if shares is not None and shares > 0 and amount > 0:
        now = yes_price_now if position["side"] == "YES" else 1.0 - yes_price_now
        return (shares * now / amount - 1.0) * 100.0
    return pnl_pct(position["side"], position["entry_yes_price"], yes_price_now)


def position_cost_basis(position: dict) -> float:
    """USD actually paid for the position — the real filled notional
    (actual_cost_usd) when known, else the nominal stake (amount_usd).

    After share-size rounding and fees a live fill typically costs less than the
    nominal bet (a $25 order fills for ~$21), and Polymarket reports returns
    against what you actually paid. PnL% displays use this so our figures match
    the Polymarket UI; dry-run/legacy rows with no actual_cost_usd fall back to
    amount_usd, where cost == nominal and nothing changes."""
    actual = position.get("actual_cost_usd")
    if actual is not None and actual > 0:
        return actual
    return position.get("amount_usd") or 0.0


def pnl_pct_on_cost(position: dict, price_pnl_pct: float) -> float:
    """Re-express a price-based PnL% as a percentage of the actual cost basis,
    matching Polymarket's return-on-cost.

    The price-based pnl_pct equals pnl_usd / amount_usd * 100, so multiplying by
    amount_usd / cost_basis rebases it onto the real dollars paid:
    price_pnl_pct * amount_usd / cost_basis == pnl_usd / cost_basis * 100. With
    no actual_cost_usd the basis is amount_usd and the value is unchanged."""
    amount = position.get("amount_usd") or 0.0
    cost = position_cost_basis(position)
    if cost <= 0 or amount <= 0:
        return price_pnl_pct
    return price_pnl_pct * amount / cost


def open_position(trade_id: int, market, side: str, amount_usd: float,
                  headline: str = "", reasoning: str = "",
                  actual_cost_usd: float | None = None,
                  token_count: float | None = None) -> int | None:
    """Record a new open position for an executed/dry-run trade.

    `actual_cost_usd` is the real USD that filled on a live BUY (filled_shares *
    price from the exchange reconcile); None for dry-run/simulated fills, where the
    nominal amount_usd is the cost basis. When it's known, that filled cost — not
    the nominal bet — becomes the position's notional: a GTC order can fill only
    partially (e.g. $4.41 of a $32.81 bet), and the held share count, exposure, and
    PnL must reflect what we actually own, not what we asked for. It's also stored
    in actual_cost_usd so PnL% displays match Polymarket's return-on-cost.

    `token_count` is the real filled share count (filled_cost / fill_price). Stored
    so a live SELL flattens exactly the tokens we own; None for dry-run/simulated
    fills, where position_shares falls back to amount_usd / entry_yes_price."""
    notional = (actual_cost_usd if (actual_cost_usd is not None and actual_cost_usd > 0)
                else amount_usd)
    conn = logger._conn()
    cur = conn.execute(
        """INSERT OR IGNORE INTO positions
           (trade_id, condition_id, market_question, slug, side,
            entry_yes_price, amount_usd, headline, reasoning,
            current_yes_price, unrealized_pnl_pct, event_id, category, end_date,
            actual_cost_usd, token_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)""",
        (trade_id, market.condition_id, market.question,
         getattr(market, "slug", "") or None, side,
         market.yes_price, notional, headline, reasoning, market.yes_price,
         getattr(market, "event_id", "") or None,
         getattr(market, "category", "") or None,
         getattr(market, "end_date", "") or None,
         actual_cost_usd, token_count),
    )
    conn.commit()
    position_id = cur.lastrowid if cur.rowcount else None
    conn.close()
    return position_id


def backfill_positions() -> int:
    """Create positions for pre-existing trades that lack one (idempotent)."""
    conn = logger._conn()
    rows = conn.execute(
        """SELECT t.id, t.market_id, t.market_question, t.side, t.market_price,
                  t.amount_usd, t.headlines, t.reasoning
           FROM trades t LEFT JOIN positions p ON p.trade_id = t.id
           WHERE p.id IS NULL AND t.status IN ('dry_run', 'executed', 'filled')"""
    ).fetchall()
    for t in rows:
        conn.execute(
            """INSERT OR IGNORE INTO positions
               (trade_id, condition_id, market_question, side, entry_yes_price,
                amount_usd, headline, reasoning, current_yes_price, unrealized_pnl_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
            (t["id"], t["market_id"], t["market_question"], t["side"],
             t["market_price"], t["amount_usd"],
             (t["headlines"] or "").splitlines()[0] if t["headlines"] else "",
             t["reasoning"] or "", t["market_price"]),
        )
    conn.commit()
    conn.close()
    if rows:
        log.info(f"[positions] Backfilled {len(rows)} positions from existing trades")
    return len(rows)


def get_open_positions(since: str | None = None) -> list[dict]:
    """Open positions, newest first. `since` (an ISO date/datetime) is a
    display-only filter — when set, only positions opened on or after it are
    returned. Default None returns every open position (used by the pipeline's
    risk gates and internal exit management, which must see the full book)."""
    conn = logger._conn()
    sql = ("SELECT p.*, t.edge_type FROM positions p "
           "LEFT JOIN trades t ON p.trade_id = t.id WHERE p.status = 'open'")
    params: list = []
    if since:
        sql += " AND p.opened_at >= ?"
        params.append(since)
    sql += " ORDER BY p.id DESC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_closed_positions(limit: int = 10, since: str | None = None) -> list[dict]:
    """Most recently closed positions. `since` (an ISO date/datetime) is a
    display-only filter on the open date; default None returns all."""
    conn = logger._conn()
    sql = "SELECT * FROM positions WHERE status != 'open'"
    params: list = []
    if since:
        sql += " AND opened_at >= ?"
        params.append(since)
    sql += " ORDER BY closed_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_exit_decisions(limit: int = 5) -> list[dict]:
    conn = logger._conn()
    rows = conn.execute(
        """SELECT d.*, p.market_question, p.side
           FROM exit_decisions d JOIN positions p ON d.position_id = p.id
           ORDER BY d.id DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_partial_closes(since: str | None = None) -> list[dict]:
    """close_half exit decisions joined with their position, newest first.

    A close_half realizes half the stake but leaves the position open, so it
    never lands in get_closed_positions. The dashboard's closed-positions
    display merges these in (see export_status). `since` filters on the
    position's open date, matching get_closed_positions."""
    conn = logger._conn()
    sql = """SELECT d.id AS decision_id, d.created_at, d.pnl_pct, d.yes_price,
                    p.id AS position_id, p.market_question, p.slug, p.side,
                    p.entry_yes_price, p.exit_yes_price, p.status,
                    p.realized_pnl_usd, p.event_id, p.opened_at, p.closed_at
             FROM exit_decisions d JOIN positions p ON d.position_id = p.id
             WHERE d.decision = 'close_half'"""
    params: list = []
    if since:
        sql += " AND p.opened_at >= ?"
        params.append(since)
    sql += " ORDER BY d.id DESC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --- Correlation risk ---------------------------------------------------------
#
# One headline can only open one position (the pipeline's headline cap), but
# distinct headlines about the same subject — three Trump markets, two SpaceX
# markets — can quietly stack into one concentrated bet. The correlation
# tracker estimates topic overlap between a candidate market and the open book
# and flags over-concentration before another correlated position is opened.

# Market-question scaffolding we don't treat as a topic ("Will Trump win?").
_TOPIC_STOPWORDS = {
    "will", "who", "what", "when", "where", "which", "the", "a", "an", "is",
    "are", "be", "to", "of", "in", "on", "at", "by", "for", "and", "or", "win",
    "wins", "won", "first", "before", "after", "between", "during", "this",
    "that", "next", "new", "no", "yes", "it", "its", "his", "her", "than",
    "with", "from", "into", "over", "under", "out", "up", "down", "end",
}

# Topics/entities worth tracking even when they appear lowercase in a question.
_TOPIC_KEYWORDS = {
    "trump", "biden", "harris", "vance", "desantis", "newsom", "obama",
    "putin", "zelensky", "musk", "spacex", "tesla", "openai", "anthropic",
    "nvidia", "apple", "google", "meta", "microsoft",
    "ai", "agi", "election", "primary", "senate", "congress", "president",
    "governor", "fed", "inflation", "rates", "recession", "gdp", "jobs",
    "bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "doge", "xrp",
    "ukraine", "russia", "israel", "gaza", "iran", "china", "taiwan", "korea",
    "nfl", "nba", "mlb", "nhl", "ufc", "olympics", "superbowl", "worldcup",
    "oil", "gold", "stock", "ipo", "nyse", "nasdaq",
}


def extract_topics(question: str) -> set[str]:
    """Pull topic keywords (people, entities, themes) out of a market question.

    Two complementary passes: capitalized proper nouns / acronyms as they
    appear in the text (Trump, SpaceX, AI, NYSE), plus a curated keyword list
    matched case-insensitively so lowercase themes ("election", "crypto") are
    caught regardless of capitalization. Everything is lowercased so the two
    passes share a namespace and set intersection means "shared topic".
    """
    text = question or ""
    topics: set[str] = set()
    for token in re.findall(r"\b[A-Z][A-Za-z]+\b", text):
        word = token.lower()
        if word not in _TOPIC_STOPWORDS:
            topics.add(word)
    lowered = text.lower()
    for kw in _TOPIC_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", lowered):
            topics.add(kw)
    return topics


def check_correlation_risk(new_market_question: str, new_side: str,
                           open_positions: list[dict]) -> dict:
    """Estimate concentration risk of adding a position correlated with the book.

    `new_side` is accepted for caller context/logging; risk is driven purely by
    topic overlap and combined dollar exposure of the correlated open positions:

      high   — 3+ correlated positions OR combined exposure > $75
      medium — 2 correlated positions OR combined exposure > $50
      low    — otherwise
    """
    new_topics = extract_topics(new_market_question)
    correlated: list[dict] = []
    if new_topics:
        for p in open_positions:
            shared = new_topics & extract_topics(p.get("market_question", ""))
            if shared:
                correlated.append({
                    "market_question": p.get("market_question", ""),
                    "side": p.get("side"),
                    "amount_usd": p.get("amount_usd") or 0.0,
                    "shared_topics": sorted(shared),
                })

    total_exposure = sum(c["amount_usd"] for c in correlated)
    count = len(correlated)
    if count >= 3 or total_exposure > 75:
        risk_level = "high"
    elif count >= 2 or total_exposure > 50:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "correlated_positions": correlated,
        "total_exposure_usd": round(total_exposure, 2),
        "risk_level": risk_level,
    }


def check_category_exposure(category: str, open_positions: list[dict]) -> dict:
    """Combined open exposure in a market category vs its configured hard cap.

    Sums amount_usd over open positions in `category` (falling back to inferring
    the category from each position's question when the stored category is
    missing — e.g. legacy rows). The new trade's own size is not added: the gate
    blocks once a category's *existing* exposure is over its hard limit, and
    warns inside the soft band (CATEGORY_SOFT_LIMIT_PCT..100% of the limit).

    Returns {allowed, warning, current_usd, limit_usd, pct}:
      allowed — existing exposure is at/under the hard limit
      warning — allowed, but at/above CATEGORY_SOFT_LIMIT_PCT of the limit
    """
    category = category or "other"
    limits = config.MAX_EXPOSURE_PER_CATEGORY
    limit_usd = float(limits.get(category, limits.get("other", 0)))

    current = 0.0
    for p in open_positions:
        pcat = p.get("category") or gamma._infer_category(p.get("market_question", ""), [])
        if pcat == category:
            current += p.get("amount_usd") or 0.0

    pct = (current / limit_usd) if limit_usd > 0 else 0.0
    allowed = current <= limit_usd
    warning = allowed and pct >= config.CATEGORY_SOFT_LIMIT_PCT

    # Log exposure on every check — the per-category book is otherwise invisible.
    log.info(
        f"[positions] Category exposure {category}: "
        f"${current:.0f}/${limit_usd:.0f} ({pct:.0%})"
    )

    return {
        "allowed": allowed,
        "warning": warning,
        "current_usd": round(current, 2),
        "limit_usd": limit_usd,
        "pct": round(pct, 4),
    }


def check_trigger(position: dict, current_pnl_pct: float) -> str | None:
    """Price-based trigger detection (news triggers come from the pipeline).
    stop_loss is checked first and handled WITHOUT a Claude call."""
    if current_pnl_pct <= config.STOP_LOSS_PCT:
        return "stop_loss"
    if current_pnl_pct >= config.TAKE_PROFIT_TRIGGER_PCT:
        return "take_profit"
    if current_pnl_pct <= config.REEVAL_LOSS_PCT:
        return "drawdown"
    return None


def hours_to_close(end_date: str | None, now: datetime | None = None) -> float | None:
    """Hours until the market closes, or None when the close time is unknown or
    unparseable (legacy positions without a stored end_date)."""
    if not end_date:
        return None
    try:
        dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return (dt - now).total_seconds() / 3600.0


# Near-certain exits are skipped inside this many hours of close: near expiry a
# market naturally drifts to 0/1, so a 0.95+ price there isn't a signal to act on.
NEAR_CERTAIN_MIN_HOURS_TO_CLOSE = 1.0


def check_hard_exit(position: dict, current_pnl_pct: float,
                    now: datetime | None = None) -> tuple[str, str] | None:
    """Hard, model-free exits on top of the Claude re-eval path. Returns an
    (action, reason) pair to execute without a Claude call, or None.

    - ("time_pressure", "time_pressure"): a deep loser running into market close
      (time-to-close < TIME_PRESSURE_HOURS AND PnL < TIME_PRESSURE_LOSS_PCT).
      Needs a known close time; skipped when end_date is unknown.
    - ("force_close", "near_certain_yes_X.XX" / "near_certain_no_X.XX"): the held
      side is all but resolved (YES price >= NEAR_CERTAIN_THRESHOLD, or NO price
      <= 1 - threshold), so lock it in rather than wait for resolution. Skipped
      within NEAR_CERTAIN_MIN_HOURS_TO_CLOSE of close, where prices drift to
      certainty on their own. Uses the position's current_yes_price mark.
    """
    ttc = hours_to_close(position.get("end_date"), now)

    # Time-pressure: deep loser with little time left to recover (needs a known
    # close time).
    if (ttc is not None and ttc < config.TIME_PRESSURE_HOURS
            and current_pnl_pct < config.TIME_PRESSURE_LOSS_PCT):
        return "time_pressure", "time_pressure"

    # Near-certain: the held side is essentially resolved. Skip in the final hour
    # before close, where any market converges to 0/1 regardless.
    price = position.get("current_yes_price")
    side = position.get("side")
    near_expiry = ttc is not None and ttc < NEAR_CERTAIN_MIN_HOURS_TO_CLOSE
    if price is not None and side and not near_expiry:
        if side == "YES" and price >= config.NEAR_CERTAIN_THRESHOLD:
            log.info(f"[positions] NEAR CERTAIN: closing YES position at {price:.2f}")
            return "force_close", f"near_certain_yes_{price:.2f}"
        if side == "NO" and price <= (1.0 - config.NEAR_CERTAIN_THRESHOLD):
            log.info(f"[positions] NEAR CERTAIN: closing NO position at {price:.2f}")
            return "force_close", f"near_certain_no_{price:.2f}"

    return None


def cooldown_allows(position: dict, trigger: str, now: datetime | None = None) -> bool:
    """One re-evaluation per REEVAL_COOLDOWN_HOURS, unless the trigger TYPE
    differs from the last one (a new kind of event deserves a fresh look)."""
    last = position.get("last_reeval_at")
    if not last:
        return True
    if position.get("last_trigger") != trigger:
        return True
    now = now or datetime.now(timezone.utc)
    last_dt = datetime.fromisoformat(last.replace(" ", "T"))
    if last_dt.tzinfo is None:
        last_dt = last_dt.replace(tzinfo=timezone.utc)
    return (now - last_dt).total_seconds() >= config.REEVAL_COOLDOWN_HOURS * 3600


def _live_close(
    position: dict, exit_reason: str, fraction: float
) -> tuple[str, str | None, float | None, float | None, float | None]:
    """Flatten `fraction` of the position on the CLOB (live trading only).

    Returns (outcome, order_id, remaining_shares, sold_shares, fill_price).
    sold_shares is the share count that actually matched, and fill_price is
    the SELL fill price of the HELD side's token (i.e. the NO-token price for
    a NO position) — together the real proceeds, so _close can realize the
    sold chunk at the price it actually sold for. Both are None when no CLOB
    sell happened.

      - ("closed", None, None, None, None): nothing to sell on the exchange —
        every close while DRY_RUN and every resolution close (settles
        on-chain) skip the CLOB, so the local close is always safe to record
        as a simulated/settled fill at the marked price.
      - ("closed", order_id, remaining, sold, price): the live SELL flattened
        everything it set out to (any remainder is dust); record the close
        with the real Polymarket order_id.
      - ("partial", order_id, remaining, sold, price): the SELL filled but
        `remaining` tokens are STILL HELD on-chain (thin book / partial fill).
        The caller must KEEP the position open, shrink token_count to
        `remaining`, and realize only the sold chunk — recording a close here
        would hide live exposure.
      - ("failed", None, None, None, None): the live SELL could NOT be
        confirmed (close_failed, a thin/empty/wide book, or a missing client).
        We still hold the full position on-chain, so the caller MUST keep it
        open and NOT record the close.

    CRITICAL: a recorded close on an unconfirmed/partial sell silently hides live
    exposure. Only a confirmed full 'closed' (or a no-sell-needed close) may
    record a flat position."""
    log.info(
        "[positions] _live_close ENTER: condition_id=%s side=%s exit_reason=%s "
        "fraction=%.2f config.DRY_RUN=%s on \"%s\"",
        position["condition_id"], position["side"], exit_reason, fraction,
        config.DRY_RUN, position["market_question"][:40],
    )
    if config.DRY_RUN or exit_reason == "resolution":
        log.info(
            "[positions] _live_close: NO CLOB sell placed (dry_run=%s, exit_reason=%s) "
            "— close recorded locally only for \"%s\"",
            config.DRY_RUN, exit_reason, position["market_question"][:40],
        )
        return "closed", None, None, None, None
    # Prefer the real filled token_count (the BUY filled at the ask, so deriving
    # shares from amount_usd / entry_yes_price over-counts); scale by the close
    # fraction. Falls back to the derivation for dry-run/legacy positions.
    shares = position_shares(
        position["side"], position["entry_yes_price"], position["amount_usd"],
        token_count=position.get("token_count"),
    ) * fraction
    log.info(
        "[positions] _live_close: placing LIVE SELL — condition_id=%s side=%s "
        "fraction=%.2f shares=%.4f reason=%s on \"%s\"",
        position["condition_id"], position["side"], fraction, shares, exit_reason,
        position["market_question"][:40],
    )
    result = executor.close_position_live(
        position["condition_id"], position["side"], shares
    )
    log.info(
        "[positions] _live_close: close_position_live returned status=%s order_id=%s "
        "price=%s shares=%s", result["status"], result.get("order_id"),
        result.get("price"), result.get("shares"),
    )
    if result["status"] == "executed":
        sold = result.get("sold_shares", result.get("shares"))
        log.info(
            f"[positions] Live close order {result['order_id']} placed: SELL "
            f"{sold} {position['side']} @ "
            f"{result['price']} on \"{position['market_question'][:40]}\""
        )
        return ("closed", result["order_id"], result.get("remaining_shares"),
                sold, result.get("price"))
    if result["status"] == "partial_close":
        # Sold part of the holding; tokens still remain on-chain. Keep the position
        # open and let the caller shrink token_count to the real remainder and
        # realize the sold chunk at its fill price.
        remaining = result.get("remaining_shares")
        log.warning(
            f"[positions] PARTIAL close on \"{position['market_question'][:40]}\": "
            f"sold {result.get('sold_shares')} {position['side']}, "
            f"{remaining} tokens remain — position kept OPEN, close NOT recorded"
        )
        return ("partial", result["order_id"], remaining,
                result.get("sold_shares"), result.get("price"))
    # SELL not confirmed. Do NOT record the close — keep the position open.
    detail = result.get("error")
    log.error(
        f"[positions] LIVE CLOSE FAILED (status {result['status']}"
        + (f": {detail}" if detail else "")
        + f") on \"{position['market_question'][:40]}\" — position kept OPEN, "
        "close NOT recorded (we still hold this on-chain)"
    )
    return "failed", None, None, None, None


def _close(conn, position: dict, yes_price: float, status: str, exit_reason: str,
           fraction: float = 1.0) -> float:
    """Realize `fraction` of the position at yes_price. Live trading places a
    real CLOB sell to flatten the shares; dry-run simulates the fill.

    Concurrency-safe: the realizing UPDATE only matches while the row is still
    status='open', so when two closers race (management cycle vs. calibrator
    vs. news re-eval), exactly one realizes PnL — the other returns 0.0 with a
    warning."""
    log.info(
        "[positions] _close ENTER: position_id=%s side=%s status=%s exit_reason=%s "
        "fraction=%.2f yes_price=%s config.DRY_RUN=%s on \"%s\"",
        position["id"], position["side"], status, exit_reason, fraction, yes_price,
        config.DRY_RUN, position["market_question"][:40],
    )
    log.info(
        "[positions] _close: calling _live_close (position_id=%s, exit_reason=%s, "
        "fraction=%.2f)", position["id"], exit_reason, fraction,
    )
    outcome, exit_order_id, remaining_shares, sold_shares, fill_price = _live_close(
        position, exit_reason, fraction
    )
    log.info(
        "[positions] _close: _live_close returned outcome=%s exit_order_id=%s "
        "remaining_shares=%s sold_shares=%s fill_price=%s (position_id=%s)",
        outcome, exit_order_id, remaining_shares, sold_shares, fill_price,
        position["id"],
    )
    amount = position["amount_usd"] or 0.0
    token_count = position.get("token_count")
    # The share count the SELL was sized from (the real holding when known) —
    # the denominator for "what fraction actually sold".
    shares_total = position_shares(
        position["side"], position["entry_yes_price"], amount,
        token_count=token_count,
    )
    live_fill = bool(sold_shares and fill_price)

    if outcome == "partial":
        # Live SELL only partially flattened — tokens still held on-chain. Do
        # NOT mark closed; shrink token_count to the real remainder so the
        # next cycle sells exactly what's left, and keep status='open'. The
        # sold chunk DID produce proceeds, so realize its PnL now, at ITS fill
        # price against its proportional share of the cost basis — otherwise
        # it would silently be re-realized later at whatever price the
        # remainder exits.
        current_pct = pnl_pct_basis(position, yes_price)
        if live_fill and shares_total > 0:
            sold_fraction = min(sold_shares / shares_total, 1.0)
            realized = sold_shares * fill_price - amount * sold_fraction
        else:
            # No usable fill data — keep the position's basis untouched.
            sold_fraction = 0.0
            realized = 0.0
        log.warning(
            "[positions] Partial close on position %s: sold %s @ %s "
            "(realized $%+.2f), %s tokens remain — position kept open",
            position["id"], sold_shares, fill_price, realized, remaining_shares,
        )
        cur = conn.execute(
            """UPDATE positions SET token_count=?, amount_usd=amount_usd*?,
               realized_pnl_usd=COALESCE(realized_pnl_usd,0)+?,
               current_yes_price=? WHERE id=? AND status='open'""",
            (remaining_shares, 1.0 - sold_fraction, realized, yes_price,
             position["id"]),
        )
        if cur.rowcount == 0:
            log.warning(
                "[positions] Partial close SKIPPED: position %s no longer open "
                "(concurrent close won the race) — realizing nothing on \"%s\"",
                position["id"], position["market_question"][:40],
            )
            return 0.0
        conn.execute(
            """INSERT INTO exit_decisions
               (position_id, trigger, decision, reasoning, pnl_pct, yes_price)
               VALUES (?, ?, 'partial_close', ?, ?, ?)""",
            (position["id"], exit_reason or status or "close",
             f"Partial live SELL ({exit_reason or status}): sold {sold_shares} "
             f"@ {fill_price} (realized ${realized:+.2f}); {remaining_shares} "
             "tokens remain, position kept open",
             round(current_pct, 2), yes_price),
        )
        return realized
    if outcome == "failed":
        # Live SELL not confirmed — we still hold this position. Record the failed
        # attempt for tracking (status 'close_failed'), keep the row open, and
        # realize nothing. The next management cycle will retry the close.
        current_pct = pnl_pct_basis(position, yes_price)
        conn.execute(
            """INSERT INTO exit_decisions
               (position_id, trigger, decision, reasoning, pnl_pct, yes_price)
               VALUES (?, ?, 'close_failed', ?, ?, ?)""",
            (position["id"], exit_reason or status or "close",
             f"Live SELL not confirmed ({exit_reason or status}); position kept open",
             round(current_pct, 2), yes_price),
        )
        return 0.0
    now_iso = datetime.now(timezone.utc).isoformat()
    # Realized PnL is proceeds minus cost, on the actual basis. A live fill
    # realizes exactly what the SELL produced (sold_shares x the side-token
    # fill price); dry-run and resolution closes simulate/settle at the given
    # yes_price on the stored basis (position_pnl_basis: token_count real
    # shares when known, else the legacy amount/entry derivation).
    #
    # The `AND status='open'` guard makes concurrent closers safe: several
    # paths (the 30s management cycle, the calibrator's resolution close, a
    # news re-eval) close from stale position snapshots on separate threads,
    # and an unguarded UPDATE would let two of them each add `realized` to
    # realized_pnl_usd. The loser of the race matches zero rows and realizes
    # nothing.
    if fraction >= 1.0:
        if live_fill:
            # The whole position exits: proceeds minus the full remaining cost
            # (a sub-dust unsold remainder is written off with the close).
            realized = sold_shares * fill_price - amount
        else:
            realized = position_pnl_basis(position, yes_price)
        chunk_cost = amount
        cur = conn.execute(
            """UPDATE positions SET status=?, exit_yes_price=?, exit_reason=?,
               realized_pnl_usd=COALESCE(realized_pnl_usd,0)+?, closed_at=?,
               current_yes_price=?, unrealized_pnl_pct=0, exit_order_id=?
               WHERE id=? AND status='open'""",
            (status, yes_price, exit_reason, realized, now_iso, yes_price,
             exit_order_id, position["id"]),
        )
        if cur.rowcount == 0:
            log.warning(
                "[positions] Close SKIPPED: position %s no longer open "
                "(concurrent close won the race, reason=%s) — realizing nothing "
                "on \"%s\"", position["id"], exit_reason,
                position["market_question"][:40],
            )
            return 0.0
        # Re-entry: keep watching the market for a thesis reversal (every close
        # type except resolution — a resolved market has nothing left to trade).
        reason = _canonical_close_reason(exit_reason)
        if reason != "resolution":
            logger.watch_closed_position(
                conn,
                condition_id=position["condition_id"],
                market_question=position["market_question"],
                original_side=position["side"],
                original_entry_price=position["entry_yes_price"],
                close_reason=reason,
                watch_hours=config.REENTRY_WATCH_HOURS,
                exit_reason=exit_reason,
            )
    else:
        if live_fill and shares_total > 0:
            # Realize what actually sold, at its fill price; shrink the basis
            # by the fraction that actually sold (not the requested fraction).
            sold_fraction = min(sold_shares / shares_total, 1.0)
            realized = sold_shares * fill_price - amount * sold_fraction
            new_token_count = (remaining_shares if remaining_shares is not None
                               else (max(token_count - sold_shares, 0.0)
                                     if token_count else None))
        else:
            sold_fraction = fraction
            realized = position_pnl_basis(position, yes_price, fraction)
            new_token_count = (token_count * (1.0 - fraction)
                               if token_count else token_count)
        chunk_cost = amount * sold_fraction
        cur = conn.execute(
            """UPDATE positions SET amount_usd=amount_usd*?,
               realized_pnl_usd=COALESCE(realized_pnl_usd,0)+?,
               current_yes_price=?, exit_order_id=?, token_count=?
               WHERE id=? AND status='open'""",
            (1.0 - sold_fraction, realized, yes_price, exit_order_id,
             new_token_count, position["id"]),
        )
        if cur.rowcount == 0:
            log.warning(
                "[positions] Half-close SKIPPED: position %s no longer open "
                "(concurrent close won the race) — realizing nothing on \"%s\"",
                position["id"], position["market_question"][:40],
            )
            return 0.0

    # Real-time notification — single choke point for every close path (manual,
    # resolution, stop, hard exit, re-eval, half close). The % is the realized
    # return on the closed chunk's actual cost.
    realized_pct = realized / chunk_cost * 100.0 if chunk_cost > 0 else 0.0
    if fraction >= 1.0:
        telegram_bot.notify_position_closed(position, realized_pct, realized, exit_reason)
    else:
        telegram_bot.notify_half_closed(position, realized_pct, realized)
    return realized


def close_on_resolution(trade_id: int, exit_yes_price: float) -> None:
    """Natural exit: the market resolved (called by the calibrator)."""
    conn = logger._conn()
    row = conn.execute(
        "SELECT * FROM positions WHERE trade_id=? AND status='open'", (trade_id,)
    ).fetchone()
    if row:
        realized = _close(conn, dict(row), exit_yes_price, "resolved", "resolution")
        conn.commit()
        log.info(f"[positions] Resolution closed position {row['id']} pnl ${realized:+.2f}")
    conn.close()


def close_manual(position_id: int) -> dict | None:
    """User-requested exit: close an open position at its last-marked price.

    Returns a result dict (id, market_question, side, price, pnl_pct, realized)
    on success, or None when the position doesn't exist or is already closed.
    The close is logged to exit_decisions as an explicit user decision."""
    conn = logger._conn()
    row = conn.execute("SELECT * FROM positions WHERE id=?", (position_id,)).fetchone()
    if row is None or row["status"] != "open":
        conn.close()
        return None

    position = dict(row)
    # Close at the last price marked into the positions table (falling back to
    # entry if the position was never marked — leaves PnL at 0).
    yes_price = position["current_yes_price"]
    if yes_price is None:
        yes_price = position["entry_yes_price"]
    current_pnl = pnl_pct_basis(position, yes_price)

    # _close places the real CLOB sell when live (DRY_RUN=false); dry-run
    # simulates the fill at the marked price. When live, a SELL that can't be
    # confirmed leaves the position open and records nothing (_close logs the
    # failed attempt itself), so re-read the row to see if it actually closed.
    realized = _close(conn, position, yes_price, "closed_manual", "manual")
    still_open = conn.execute(
        "SELECT status FROM positions WHERE id=?", (position_id,)
    ).fetchone()["status"] == "open"
    if still_open:
        conn.commit()
        conn.close()
        log.error(
            f"[positions] Manual close of position {position_id} FAILED — live "
            f"SELL not confirmed; position still open"
        )
        return {
            "id": position_id,
            "market_question": position["market_question"],
            "side": position["side"],
            "price": yes_price,
            "pnl_pct": current_pnl,
            # A partial sell realizes its sold chunk even though the position
            # stays open; report those real dollars, not a flat 0.
            "realized": realized,
            "close_failed": True,
        }
    conn.execute(
        """INSERT INTO exit_decisions (position_id, trigger, decision, reasoning, pnl_pct, yes_price)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (position_id, "manual", "close", "Manual close requested by user",
         round(current_pnl, 2), yes_price),
    )
    conn.commit()
    conn.close()
    log.info(
        f"[positions] Manual close of position {position_id} "
        f"pnl {current_pnl:+.1f}% realized ${realized:+.2f}"
    )
    return {
        "id": position_id,
        "market_question": position["market_question"],
        "side": position["side"],
        "price": yes_price,
        "pnl_pct": current_pnl,
        "realized": realized,
        "close_failed": False,
    }


def _mark_reconciled_closed(conn, position_id: int) -> None:
    """Flip an open position to closed_reconciled (a no-fill bookkeeping close):
    the on-chain token balance says nothing is held anymore, so stamp
    status/exit_reason/closed_at. The reconcile itself realizes $0 — but any
    realized_pnl_usd already on the row (e.g. close_half chunks realized before
    the remainder went missing) is REAL money that was actually realized, so it
    is preserved, never overwritten. Distinct status/exit_reason keep the close
    auditable."""
    now_iso = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """UPDATE positions
           SET status='closed_reconciled', exit_reason='reconcile_mismatch',
               unrealized_pnl_pct=0, closed_at=?
           WHERE id=?""",
        (now_iso, position_id),
    )


def _verify_position_holding(client, position: dict) -> tuple[str, float | None]:
    """Verify one open position against its real on-chain token balance.

    Resolves the position's outcome token_id from its condition_id/side, then
    asks the exchange for the held share count. Returns (state, held):
      - ("ok", shares)        held > 0 — genuinely open
      - ("mismatch", 0.0)     held == 0 — DB says open, exchange holds nothing
      - ("unknown", None)     balance couldn't be verified (no client, token
                              unresolved, or held_token_shares returned None —
                              e.g. the client lacks the call); never auto-closed
    """
    if client is None:
        return "unknown", None
    try:
        token_id = executor._resolve_token_id(
            client, position["condition_id"], position["side"]
        )
    except Exception as e:
        log.warning("[positions] reconcile: token resolve failed for position %s (%s)",
                    position["id"], e)
        return "unknown", None
    if not token_id:
        return "unknown", None

    held = executor.held_token_shares(client, token_id)
    if held is None:
        return "unknown", None
    if held <= 0:
        return "mismatch", 0.0
    return "ok", held


def reconcile_positions(fix: bool = False, client=None) -> dict:
    """Sync DB open positions against actual Polymarket (on-chain) state.

    For each status='open' position, check the real held token balance via
    executor.held_token_shares. A position the exchange holds nothing for is a
    MISMATCH (a phantom open — e.g. an order that never actually filled, or a
    close that updated the chain but not the DB). A position with a positive
    balance is OK. A position whose balance can't be verified (no CLOB client,
    token unresolved, or the client lacks the balance call) is UNKNOWN and is
    NEVER auto-closed — reconciliation only ever closes positions it has
    positively confirmed are empty.

    `fix=False` (default) reports only. `fix=True` closes each MISMATCH as
    closed_reconciled / reconcile_mismatch, realizing nothing further (prior
    partial realizations on the row are preserved). `client` is the
    CLOB client; built via executor.create_clob_client() when omitted (injectable
    for tests). Returns a report dict:
        {"entries": [{"id", "market_question", "state", "held", "line"}, ...],
         "ok": [ids], "mismatches": [ids], "unknown": [ids], "fixed": [ids]}
    """
    open_positions = get_open_positions()
    if client is None and open_positions:
        try:
            client = executor.create_clob_client()
        except Exception as e:
            # No usable client — every position becomes UNKNOWN (never closed).
            log.warning("[positions] reconcile: CLOB client unavailable (%s); "
                        "all positions will report UNKNOWN", e)
            client = None

    entries: list[dict] = []
    ok_ids: list[int] = []
    mismatch_ids: list[int] = []
    unknown_ids: list[int] = []

    for position in open_positions:
        pid = position["id"]
        state, held = _verify_position_holding(client, position)
        if state == "ok":
            ok_ids.append(pid)
            line = f"OK: ID={pid} held={held:g} tokens — matches DB"
        elif state == "mismatch":
            mismatch_ids.append(pid)
            line = f"MISMATCH: ID={pid} DB says open but held=0 on Polymarket"
        else:
            unknown_ids.append(pid)
            line = f"UNKNOWN: could not verify ID={pid}"
        entries.append({
            "id": pid,
            "market_question": position["market_question"],
            "state": state,
            "held": held,
            "line": line,
        })

    fixed_ids: list[int] = []
    if fix and mismatch_ids:
        conn = logger._conn()
        try:
            for pid in mismatch_ids:
                _mark_reconciled_closed(conn, pid)
                fixed_ids.append(pid)
                log.info("[positions] reconcile FIX: position %s closed_reconciled "
                         "(reconcile_mismatch, realized $0)", pid)
            conn.commit()
        finally:
            conn.close()

    return {
        "entries": entries,
        "ok": ok_ids,
        "mismatches": mismatch_ids,
        "unknown": unknown_ids,
        "fixed": fixed_ids,
    }


def _format_time_remaining_for(condition_id: str) -> str:
    return "unknown"  # end date isn't stored on positions; kept simple


def reevaluate(position: dict, trigger: str, trigger_detail: str = "",
               yes_price: float | None = None) -> dict | None:
    """ONE Claude call with full context; execute the decision (simulated)."""
    yes_price = yes_price if yes_price is not None else position.get("current_yes_price")
    if yes_price is None:
        return None
    current_pnl = pnl_pct_basis(position, yes_price)

    desc = TRIGGER_DESCRIPTIONS.get(trigger, "{detail}").format(
        tp=config.TAKE_PROFIT_TRIGGER_PCT, dd=config.REEVAL_LOSS_PCT, detail=trigger_detail
    )
    prompt = EXIT_PROMPT.format(
        question=position["market_question"],
        side=position["side"],
        entry_yes_price=position["entry_yes_price"],
        current_yes_price=yes_price,
        amount_usd=position["amount_usd"],
        pnl_pct=current_pnl,
        opened_at=position.get("opened_at", "unknown"),
        time_remaining=_format_time_remaining_for(position["condition_id"]),
        entry_headline=position.get("headline") or "(not recorded)",
        entry_reasoning=position.get("reasoning") or "(not recorded)",
        trigger_description=desc,
    )

    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=config.SCORING_MODEL, max_tokens=250, temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1].removeprefix("json").strip()
        result = json.loads(text)
        decision = result.get("decision", "hold")
        if decision not in ("hold", "close", "close_half"):
            decision = "hold"
        reasoning = result.get("reasoning", "")
    except Exception as e:
        log.warning(f"[positions] Re-evaluation failed for {position['id']}: {e}")
        return None

    now_iso = datetime.now(timezone.utc).isoformat()
    conn = logger._conn()
    if decision == "close":
        status, reason = CLOSE_MAP.get(trigger, ("closed_news", "news_decision"))
        realized = _close(conn, position, yes_price, status, reason)
    elif decision == "close_half":
        realized = _close(conn, position, yes_price, "", "", fraction=0.5)
    else:
        realized = 0.0
    conn.execute(
        "UPDATE positions SET last_reeval_at=?, last_trigger=? WHERE id=?",
        (now_iso, trigger, position["id"]),
    )
    conn.execute(
        """INSERT INTO exit_decisions (position_id, trigger, decision, reasoning, pnl_pct, yes_price)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (position["id"], trigger, decision, reasoning, round(current_pnl, 2), yes_price),
    )
    conn.commit()
    conn.close()
    log.info(
        f"[positions] Re-eval ({trigger}) -> {decision} on \"{position['market_question'][:40]}\" "
        f"pnl {current_pnl:+.1f}%" + (f" realized ${realized:+.2f}" if realized else "")
    )
    return {"decision": decision, "reasoning": reasoning, "realized": realized}


# Open positions can carry a NULL end_date: some Gamma markets (e.g. election
# "Winner" outcomes) have no market-level endDate, and rows opened before the
# event-endDate parser fallback existed stored nothing. Without an end_date the
# time_pressure hard exit can never fire, so the management cycle periodically
# re-fetches those markets and fills the date in when Gamma reports one.
_END_DATE_BACKFILL_INTERVAL_SECONDS = 3600.0
_last_end_date_backfill = 0.0


def backfill_missing_end_dates() -> int:
    """Fill end_date on open positions where it is NULL/empty, from a fresh
    Gamma fetch of their markets (market endDate, falling back to the parent
    event's endDate — gamma._end_date_from_raw). Returns how many positions
    were updated; positions whose market still reports no end date are left
    NULL and retried on a later cycle."""
    conn = logger._conn()
    try:
        rows = conn.execute(
            "SELECT id, condition_id FROM positions "
            "WHERE status='open' AND (end_date IS NULL OR end_date='')"
        ).fetchall()
        if not rows:
            return 0
        markets = gamma.fetch_markets_by_condition_ids(
            [r["condition_id"] for r in rows]
        )
        updated = 0
        for r in rows:
            end_date = (markets.get(r["condition_id"]) or {}).get("end_date")
            if not end_date:
                continue
            conn.execute(
                "UPDATE positions SET end_date=? "
                "WHERE id=? AND (end_date IS NULL OR end_date='')",
                (end_date, r["id"]),
            )
            updated += 1
            log.info(
                "[positions] end_date backfilled on position %s: %s "
                "(time-based exit protection restored)", r["id"], end_date,
            )
        conn.commit()
        return updated
    finally:
        conn.close()


def update_and_manage(prices: dict[str, float]) -> dict:
    """Mark open positions to `prices`, apply the hard stop-loss, and run
    Claude re-evaluations for rule triggers (respecting the cooldown).
    Called from the pipeline's periodic cycle, off the event loop."""
    # Backfill missing end_dates first (throttled to once an hour — one cheap
    # batch fetch, and only while NULL-end_date positions exist), so the
    # time_pressure/near-certain checks below see the date this same cycle.
    global _last_end_date_backfill
    now_mono = time.monotonic()
    if now_mono - _last_end_date_backfill >= _END_DATE_BACKFILL_INTERVAL_SECONDS:
        _last_end_date_backfill = now_mono
        try:
            backfill_missing_end_dates()
        except Exception as e:
            log.warning(f"[positions] end_date backfill error: {e}")

    stats = {"updated": 0, "stop_losses": 0, "time_pressure_exits": 0,
             "near_certain_exits": 0, "reevals": 0}
    conn = logger._conn()
    for position in get_open_positions():
        yes_price = prices.get(position["condition_id"])
        if yes_price is None:
            continue
        current_pnl = pnl_pct_basis(position, yes_price)
        conn.execute(
            "UPDATE positions SET current_yes_price=?, unrealized_pnl_pct=? WHERE id=?",
            (yes_price, round(current_pnl, 2), position["id"]),
        )
        # Keep the in-memory mark fresh so the near-certain hard exit sees the
        # live price, not the stale stored one.
        position["current_yes_price"] = yes_price
        stats["updated"] += 1

        # Drawdown alert (config is a fraction; current_pnl is in percent), sent
        # at most once per position by the bot's own dedup.
        if current_pnl <= -config.TELEGRAM_DRAWDOWN_ALERT_PCT * 100:
            telegram_bot.notify_drawdown_alert(position, current_pnl)

        trigger = check_trigger(position, current_pnl)
        hard = check_hard_exit(position, current_pnl)
        if trigger == "stop_loss":
            # Hard stop: unconditional, never waits on a model call.
            realized = _close(conn, position, yes_price, "closed_sl", "sl")
            stats["stop_losses"] += 1
            log.warning(
                f"[positions] STOP LOSS on \"{position['market_question'][:40]}\" "
                f"pnl {current_pnl:+.1f}% realized ${realized:+.2f}"
            )
        elif hard is not None:
            # Hard, model-free exit (time-pressure or near-certain). Runs before
            # the re-eval path, like the stop loss.
            action, reason = hard
            if action == "time_pressure":
                realized = _close(conn, position, yes_price, "closed_time", "time_pressure")
                stats["time_pressure_exits"] += 1
                log.warning(
                    f"[positions] TIME PRESSURE exit on \"{position['market_question'][:40]}\" "
                    f"pnl {current_pnl:+.1f}% (<{config.TIME_PRESSURE_HOURS:.0f}h to close) "
                    f"realized ${realized:+.2f}"
                )
            else:  # "force_close" — near-certain
                realized = _close(conn, position, yes_price, "closed_near_certain", reason)
                stats["near_certain_exits"] += 1
                log.warning(
                    f"[positions] NEAR CERTAIN exit on \"{position['market_question'][:40]}\" "
                    f"at {yes_price:.2f} ({reason}) realized ${realized:+.2f}"
                )
        elif trigger:
            # A big unrealized winner forces a re-eval even inside the cooldown:
            # we don't want a runaway gain locked out of review by a recent look.
            force = current_pnl >= config.REEVAL_FORCE_PCT
            if trigger and (force or cooldown_allows(position, trigger)):
                if force and not cooldown_allows(position, trigger):
                    log.info(
                        f"[positions] Force re-eval: PnL {current_pnl:.0f}% exceeds "
                        f"force threshold, bypassing cooldown"
                    )
                conn.commit()  # persist the mark before the slow Claude call
                stats["reevals"] += 1
                reevaluate(position, trigger, yes_price=yes_price)
    conn.commit()
    conn.close()
    return stats


def trigger_news_reeval(condition_id: str, direction: str, materiality: float,
                        headline: str, news_source: str = "") -> bool:
    """Pipeline hook: a fresh material headline contradicts a held side.

    A high-materiality Truth Social post (direct from Trump, e.g. a first-person
    denial or cancellation) that contradicts the held side FORCES an immediate
    re-evaluation, bypassing the per-position cooldown — we don't wait for the
    drawdown trigger when the principal himself just contradicted the thesis."""
    if materiality < config.NEWS_REEVAL_MATERIALITY:
        return False
    force = (
        (news_source or "").lower() == "truthsocial"
        and materiality >= config.TRUTHSOCIAL_REEVAL_MATERIALITY
    )
    for position in get_open_positions():
        if position["condition_id"] != condition_id:
            continue
        contradicts = (position["side"] == "YES" and direction == "bearish") or (
            position["side"] == "NO" and direction == "bullish"
        )
        if contradicts and (force or cooldown_allows(position, "news")):
            detail = f'"{headline[:120]}" ({direction}, materiality {materiality:.2f})'
            if force:
                detail = "Truth Social contra-post (forced) — " + detail
            reevaluate(position, "news", trigger_detail=detail)
            return True
    return False
