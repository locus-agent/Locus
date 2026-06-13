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
from datetime import datetime, timezone

import anthropic

from locus import config
from locus.memory import logger
from locus.core.performance import position_pnl

log = logging.getLogger(__name__)

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


def open_position(trade_id: int, market, side: str, amount_usd: float,
                  headline: str = "", reasoning: str = "") -> int | None:
    """Record a new open position for an executed/dry-run trade."""
    conn = logger._conn()
    cur = conn.execute(
        """INSERT OR IGNORE INTO positions
           (trade_id, condition_id, market_question, slug, side,
            entry_yes_price, amount_usd, headline, reasoning,
            current_yes_price, unrealized_pnl_pct)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
        (trade_id, market.condition_id, market.question,
         getattr(market, "slug", "") or None, side,
         market.yes_price, amount_usd, headline, reasoning, market.yes_price),
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


def get_open_positions() -> list[dict]:
    conn = logger._conn()
    rows = conn.execute(
        """SELECT p.*, t.edge_type FROM positions p
           LEFT JOIN trades t ON p.trade_id = t.id
           WHERE p.status = 'open' ORDER BY p.id DESC"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_closed_positions(limit: int = 10) -> list[dict]:
    conn = logger._conn()
    rows = conn.execute(
        "SELECT * FROM positions WHERE status != 'open' ORDER BY closed_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
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


def _close(conn, position: dict, yes_price: float, status: str, exit_reason: str,
           fraction: float = 1.0) -> float:
    """Realize `fraction` of the position at yes_price (simulated fill)."""
    realized = position_pnl(position["side"], position["entry_yes_price"],
                            yes_price, position["amount_usd"] * fraction)
    now_iso = datetime.now(timezone.utc).isoformat()
    if fraction >= 1.0:
        conn.execute(
            """UPDATE positions SET status=?, exit_yes_price=?, exit_reason=?,
               realized_pnl_usd=COALESCE(realized_pnl_usd,0)+?, closed_at=?,
               current_yes_price=?, unrealized_pnl_pct=0 WHERE id=?""",
            (status, yes_price, exit_reason, realized, now_iso, yes_price, position["id"]),
        )
    else:
        conn.execute(
            """UPDATE positions SET amount_usd=amount_usd*?,
               realized_pnl_usd=COALESCE(realized_pnl_usd,0)+?,
               current_yes_price=? WHERE id=?""",
            (1.0 - fraction, realized, yes_price, position["id"]),
        )
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


def _format_time_remaining_for(condition_id: str) -> str:
    return "unknown"  # end date isn't stored on positions; kept simple


def reevaluate(position: dict, trigger: str, trigger_detail: str = "",
               yes_price: float | None = None) -> dict | None:
    """ONE Claude call with full context; execute the decision (simulated)."""
    yes_price = yes_price if yes_price is not None else position.get("current_yes_price")
    if yes_price is None:
        return None
    current_pnl = pnl_pct(position["side"], position["entry_yes_price"], yes_price)

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


def update_and_manage(prices: dict[str, float]) -> dict:
    """Mark open positions to `prices`, apply the hard stop-loss, and run
    Claude re-evaluations for rule triggers (respecting the cooldown).
    Called from the pipeline's periodic cycle, off the event loop."""
    stats = {"updated": 0, "stop_losses": 0, "reevals": 0}
    conn = logger._conn()
    for position in get_open_positions():
        yes_price = prices.get(position["condition_id"])
        if yes_price is None:
            continue
        current_pnl = pnl_pct(position["side"], position["entry_yes_price"], yes_price)
        conn.execute(
            "UPDATE positions SET current_yes_price=?, unrealized_pnl_pct=? WHERE id=?",
            (yes_price, round(current_pnl, 2), position["id"]),
        )
        stats["updated"] += 1

        trigger = check_trigger(position, current_pnl)
        if trigger == "stop_loss":
            # Hard stop: unconditional, never waits on a model call.
            realized = _close(conn, position, yes_price, "closed_sl", "sl")
            stats["stop_losses"] += 1
            log.warning(
                f"[positions] STOP LOSS on \"{position['market_question'][:40]}\" "
                f"pnl {current_pnl:+.1f}% realized ${realized:+.2f}"
            )
        elif trigger and cooldown_allows(position, trigger):
            conn.commit()  # persist the mark before the slow Claude call
            stats["reevals"] += 1
            reevaluate(position, trigger, yes_price=yes_price)
    conn.commit()
    conn.close()
    return stats


def trigger_news_reeval(condition_id: str, direction: str, materiality: float,
                        headline: str) -> bool:
    """Pipeline hook: a fresh material headline contradicts a held side."""
    if materiality < config.NEWS_REEVAL_MATERIALITY:
        return False
    for position in get_open_positions():
        if position["condition_id"] != condition_id:
            continue
        contradicts = (position["side"] == "YES" and direction == "bearish") or (
            position["side"] == "NO" and direction == "bullish"
        )
        if contradicts and cooldown_allows(position, "news"):
            detail = f'"{headline[:120]}" ({direction}, materiality {materiality:.2f})'
            reevaluate(position, "news", trigger_detail=detail)
            return True
    return False
