"""
The Agent's Journal — once a day, Locus looks at its own last 24 hours and
writes a short first-person entry: what it noticed, what surprised it, what
it suspects it got wrong, what it wants to watch tomorrow. Entries are stored
in the journal table and published on the dashboard.

One Claude (Sonnet) call per day. Triggered from the pipeline's periodic
cycle: the first cycle after JOURNAL_HOUR_UTC writes that day's entry.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone

import anthropic

from locus import config
from locus.memory import logger

log = logging.getLogger(__name__)

JOURNAL_HOUR_UTC = 21
# The daily missed-opportunity sweep runs an hour after the journal, once a day.
MISSED_OPPORTUNITY_HOUR_UTC = 22

JOURNAL_PROMPT = """You are Locus, an autonomous agent that reads breaking news and trades \
niche Polymarket prediction markets (dry-run mode unless stated otherwise). Once a day you \
write a short public journal entry reflecting on your own performance.

Here is your last 24 hours, as data:

{stats_json}

Field notes: "actions" counts your decisions per classification — "signal" means you traded, \
"skip" means no edge, "stale" means you wanted to trade but the news was too old (your stale \
gate stopped you), "capped" means a second trade on the same headline was blocked. \
"match_sources" shows how each news-market pair was found: kw = keyword overlap, \
sem = embedding similarity, both = both. "top_materiality" are the news-market pairs you \
scored as most material. "closed_today" are positions you exited today, with their \
exit_reason and realized_pnl_usd (profit/loss in dollars). "resolutions" are markets that \
closed, grading your past calls.

Write your journal entry for {date}.

Rules:
- 100 to 200 words, first person, plain prose (no headers, no bullet points, no emojis).
- Honest, dry, a little self-deprecating. No hype. No exclamation marks.
- Reference concrete numbers from the data — at least three of them.
- Cover, in whatever order feels natural: what you noticed, what surprised you, what you got
  wrong or suspect you got wrong, and one thing you want to watch tomorrow.
- If the day was quiet or your numbers are unimpressive, say so plainly. Do not invent drama
  and do not anthropomorphize markets.
- Do not address the reader. Do not explain what you are. Just write the entry.

Output only the entry text."""


# A line that is only a date — "June 12, 2026", "2026-06-12", optionally
# bolded. The model likes to open with one; the dashboard already renders
# the date as a label, so it would show twice.
_DATE_LINE_RE = re.compile(
    r"^\s*[*_#]*\s*(?:"
    r"\d{4}-\d{2}-\d{2}"
    r"|(?:january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r")\s*[*_#]*\s*$",
    re.IGNORECASE,
)


def _strip_leading_date_lines(text: str) -> str:
    lines = text.splitlines()
    while lines and (not lines[0].strip() or _DATE_LINE_RE.match(lines[0])):
        lines.pop(0)
    return "\n".join(lines).strip()


def gather_daily_stats(extra: dict | None = None) -> dict:
    """Collect the last 24h of activity from the database."""
    now = datetime.now(timezone.utc)
    since = (now - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
    conn = logger._conn()

    def counts(query, *args):
        return {r[0] or "unknown": r[1] for r in conn.execute(query, args).fetchall()}

    actions = counts(
        "SELECT action, COUNT(*) FROM classifications WHERE created_at >= ? GROUP BY action",
        since,
    )
    directions = counts(
        "SELECT direction, COUNT(*) FROM classifications WHERE created_at >= ? GROUP BY direction",
        since,
    )
    match_sources = counts(
        "SELECT match_source, COUNT(*) FROM classifications WHERE created_at >= ? GROUP BY match_source",
        since,
    )

    top_materiality = [
        dict(r)
        for r in conn.execute(
            """SELECT market_question, headline, direction, materiality, action
               FROM classifications WHERE created_at >= ? AND materiality IS NOT NULL
               ORDER BY materiality DESC LIMIT 3""",
            (since,),
        ).fetchall()
    ]

    trades = [
        dict(r)
        for r in conn.execute(
            """SELECT market_question, side, amount_usd, market_price, status, created_at
               FROM trades WHERE created_at >= ? ORDER BY id DESC LIMIT 10""",
            (since,),
        ).fetchall()
    ]
    error_trades = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE created_at >= ? AND status LIKE 'error%'", (since,)
    ).fetchone()[0]

    resolutions = conn.execute(
        """SELECT COUNT(*) AS total, COALESCE(SUM(correct), 0) AS correct
           FROM calibration WHERE resolved_at >= ?""",
        (since,),
    ).fetchone()
    lessons = [
        r["lesson"]
        for r in conn.execute(
            "SELECT lesson FROM lessons WHERE created_at >= ? ORDER BY id DESC LIMIT 3", (since,)
        ).fetchall()
    ]

    # Positions closed today (since 00:00 UTC) — so the journal can reflect on
    # profitable exits, not just open positions. closed_at is stored via
    # datetime.isoformat(), so a midnight-UTC ISO boundary compares as a string.
    # A market may close more than once in a day (a close_half/manual partial
    # close, or a re-entry that closes again), each its own row sharing one
    # condition_id, so group by condition_id and SUM realized_pnl_usd into one
    # total per market. MAX(closed_at) makes SQLite's bare-column rule take
    # market_question / exit_reason from the final close.
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    closed_today = [
        {
            "market_question": r["market_question"],
            "exit_reason": r["exit_reason"],
            "realized_pnl_usd": r["realized_pnl_usd"],
        }
        for r in conn.execute(
            """SELECT market_question, exit_reason,
                      SUM(realized_pnl_usd) AS realized_pnl_usd,
                      MAX(closed_at) AS closed_at
               FROM positions WHERE status != 'open' AND closed_at >= ?
               GROUP BY condition_id
               ORDER BY closed_at DESC""",
            (today_start,),
        ).fetchall()
    ]

    news_count = conn.execute(
        "SELECT COUNT(*) FROM news_events WHERE created_at >= ?", (since,)
    ).fetchone()[0]
    # Largest silence between consecutive news events — a data-gap indicator.
    gaps = conn.execute(
        """SELECT created_at FROM news_events WHERE created_at >= ?
           ORDER BY created_at""",
        (since,),
    ).fetchall()
    largest_gap_min = 0.0
    times = [datetime.fromisoformat(r[0].replace(" ", "T")) for r in gaps]
    for a, b in zip(times, times[1:]):
        largest_gap_min = max(largest_gap_min, (b - a).total_seconds() / 60)

    conn.close()

    stats = {
        "date": now.strftime("%Y-%m-%d"),
        "mode": "dry-run" if config.DRY_RUN else "LIVE",
        "classifications_24h": sum(actions.values()),
        "actions": actions,
        "directions": directions,
        "match_sources": match_sources,
        "top_materiality": top_materiality,
        "trades_24h": trades,
        "closed_today": closed_today,
        "error_trades_24h": error_trades,
        "resolutions_24h": {"total": resolutions["total"], "correct": resolutions["correct"]},
        "new_lessons_24h": lessons,
        "claude_calls_saved_24h": actions.get("prefiltered", 0) + actions.get("cached", 0),
        "news_events_24h": news_count,
        "largest_news_gap_minutes": round(largest_gap_min, 1),
    }
    if extra:
        stats["task_restarts"] = extra.get("task_restarts", 0)
    return stats


def write_journal_entry(extra: dict | None = None, date: str | None = None) -> str | None:
    """Generate and store today's entry. Returns the text, or None on failure."""
    stats = gather_daily_stats(extra)
    date = date or stats["date"]

    prompt = JOURNAL_PROMPT.format(
        stats_json=json.dumps(stats, indent=1, default=str), date=date
    )
    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=config.SCORING_MODEL,
            max_tokens=400,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        entry = _strip_leading_date_lines(response.content[0].text.strip())
    except Exception as e:
        log.warning(f"[journal] Entry generation failed: {e}")
        return None

    logger.log_journal_entry(date, entry, json.dumps(stats, default=str))
    log.info(f"[journal] Wrote entry for {date} ({len(entry.split())} words)")
    return entry


def _daily_summary_stats(now: datetime | None = None) -> dict:
    """Numbers for the Telegram daily summary: positions opened/closed and
    realized PnL since 00:00 UTC, plus the portfolio's current unrealized PnL
    and win rate."""
    now = now or datetime.now(timezone.utc)
    today_start = now.strftime("%Y-%m-%d 00:00:00")
    conn = logger._conn()
    opened = conn.execute(
        "SELECT COUNT(*) FROM positions WHERE opened_at >= ?", (today_start,)
    ).fetchone()[0]
    closed = conn.execute(
        "SELECT COUNT(*) FROM positions WHERE status != 'open' AND closed_at >= ?",
        (today_start,),
    ).fetchone()[0]
    realized = conn.execute(
        "SELECT COALESCE(SUM(realized_pnl_usd), 0) FROM positions WHERE closed_at >= ?",
        (today_start,),
    ).fetchone()[0]
    conn.close()

    from locus.core.performance import compute_performance
    perf = compute_performance()
    return {
        "opened": opened,
        "closed": closed,
        "realized": realized,
        "unrealized": perf["unrealized_pnl_usd"],
        "win_rate": perf["win_rate_pct"],
    }


def maybe_write_journal(extra: dict | None = None, now: datetime | None = None) -> str | None:
    """Daily trigger: write today's entry on the first call after
    JOURNAL_HOUR_UTC. Idempotent — the journal table's UNIQUE date plus this
    check make double-writes impossible."""
    if not config.JOURNAL_ENABLED:
        return None
    now = now or datetime.now(timezone.utc)
    if now.hour < JOURNAL_HOUR_UTC:
        return None
    today = now.strftime("%Y-%m-%d")
    if logger.has_journal_for(today):
        return None
    entry = write_journal_entry(extra, date=today)

    # Push a daily summary to Telegram once the entry is written (best-effort).
    if entry is not None:
        try:
            from locus.core import telegram_bot
            telegram_bot.notify_daily_summary(_daily_summary_stats(now))
        except Exception as e:
            log.warning(f"[journal] Daily summary notification failed: {e}")

    # Weekly meta-prompt evolution: after the journal is written, evolve the
    # classification prompt if a week has passed since the last evolution (or if
    # there are seven days of lessons, for the first one). Never fatal.
    if entry is not None:
        try:
            from locus.memory import meta_evolver
            if meta_evolver.should_evolve(now):
                result = meta_evolver.evolve_prompt_sync()
                if result:
                    log.info(
                        f"[journal] Meta-prompt evolved to v{result['version']} "
                        f"({result['lessons_count']} lessons)"
                    )
        except Exception as e:
            log.warning(f"[journal] Prompt evolution failed: {e}")

    return entry


# The meta-table key holding the UTC date the missed-opportunity sweep last
# ran. Persisted in the DB (not a module global) so the once-per-day guard
# survives the frequent agent restarts — an in-memory flag let a whole day
# slip when no single process lived through the post-22:00-UTC window.
_MISSED_OPPORTUNITY_META_KEY = "last_missed_opportunity_date"


def maybe_check_missed_opportunities(now: datetime | None = None) -> int:
    """Daily trigger: run the calibrator's missed-opportunity sweep on the first
    call at/after MISSED_OPPORTUNITY_HOUR_UTC each day. Scheduled after the daily
    journal entry and before the weekly meta_evolver check. Returns the number
    of lessons logged (0 when it didn't run or found nothing).

    The last-run date lives in the DB meta table, so the guard is restart-safe.
    If one or more whole days were missed (last run older than yesterday), the
    sweep just runs for today — the missed days are accepted as lost, with a
    warning so the gap is visible."""
    if not config.MISSED_OPPORTUNITY_ENABLED:
        return 0
    now = now or datetime.now(timezone.utc)
    if now.hour < MISSED_OPPORTUNITY_HOUR_UTC:
        return 0
    today = now.strftime("%Y-%m-%d")
    last_run = logger.get_meta(_MISSED_OPPORTUNITY_META_KEY)
    if last_run == today:
        return 0
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    if last_run is not None and last_run < yesterday:
        log.warning(
            f"[journal] Missed-opportunity sweep gap: last ran {last_run}, "
            f"today is {today} — skipped day(s) are lost, resuming with today's sweep"
        )
    logger.set_meta(_MISSED_OPPORTUNITY_META_KEY, today)
    from locus.memory import calibrator
    return calibrator.check_missed_opportunities()


if __name__ == "__main__":
    import sys

    text = write_journal_entry()
    if text is None:
        sys.exit(1)
    print(text)
