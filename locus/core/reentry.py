"""
Re-entry logic.

After a position closes (for any reason other than the market resolving) the
market is kept on a watch list for `REENTRY_WATCH_HOURS`. If a fresh
classification fires on a watched market, we may re-enter — but the bar depends
on *why* we exited:

- closed on contradicting news ('news'): re-enter only if the new news supports
  the ORIGINAL side with materiality >= REENTRY_NEWS_MATERIALITY (the news that
  pushed us out has itself reversed).
- stopped out ('sl'): stricter — materiality >= REENTRY_SL_MATERIALITY AND the
  reversal is confirmed by >= REENTRY_SL_MIN_SOURCES distinct sources.
- took profit ('tp'): never re-enter — we already captured the edge, don't chase.

Re-entry happens at most MAX_REENTRY_PER_MARKET times per market. The pipeline
calls this after a classification clears the direction/materiality gates.
"""
from __future__ import annotations

from datetime import datetime, timezone

from locus import config


def _now_str(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M:%S")


def get_watched_markets(db_conn, now: datetime | None = None) -> list[dict]:
    """Active watch rows: still inside the watch window and under the re-entry
    cap. Most recently closed first."""
    rows = db_conn.execute(
        """SELECT * FROM watched_closed_positions
           WHERE watch_until > ? AND reentry_count < ?
           ORDER BY id DESC""",
        (_now_str(now), config.MAX_REENTRY_PER_MARKET),
    ).fetchall()
    return [dict(r) for r in rows]


def find_watched_market(db_conn, condition_id: str, now: datetime | None = None) -> dict | None:
    """The active watch row for one market (latest), or None if not watched."""
    row = db_conn.execute(
        """SELECT * FROM watched_closed_positions
           WHERE condition_id = ? AND watch_until > ? AND reentry_count < ?
           ORDER BY id DESC LIMIT 1""",
        (condition_id, _now_str(now), config.MAX_REENTRY_PER_MARKET),
    ).fetchone()
    return dict(row) if row else None


def record_reentry(db_conn, condition_id: str, now: datetime | None = None) -> None:
    """Consume one re-entry from the market's active watch window (caller
    commits)."""
    db_conn.execute(
        """UPDATE watched_closed_positions SET reentry_count = reentry_count + 1
           WHERE condition_id = ? AND watch_until > ?""",
        (condition_id, _now_str(now)),
    )


def _supports_original_side(direction: str, original_side: str) -> bool:
    """A bullish read supports a YES position; bearish supports NO."""
    return (
        (direction == "bullish" and original_side == "YES")
        or (direction == "bearish" and original_side == "NO")
    )


def check_reentry_opportunity(
    watched_market: dict,
    new_classification,
    confirming_source_count: int = 0,
) -> dict:
    """Decide whether a new classification justifies re-entering a watched
    market. `new_classification` is anything with `.direction` and
    `.materiality` (the Classification dataclass, or a stub in tests).

    Returns {should_reenter: bool, reason: str}.
    """
    close_reason = watched_market.get("close_reason")
    original_side = watched_market.get("original_side")
    direction = getattr(new_classification, "direction", "neutral")
    materiality = getattr(new_classification, "materiality", 0.0)

    # Took profit: never chase a market we already made money on.
    if close_reason == "tp":
        return {"should_reenter": False, "reason": "took profit — don't chase"}

    # Re-entry is always back into the original thesis.
    if not _supports_original_side(direction, original_side):
        return {
            "should_reenter": False,
            "reason": f"new {direction} read doesn't support original {original_side}",
        }

    if close_reason == "news":
        if materiality >= config.REENTRY_NEWS_MATERIALITY:
            return {
                "should_reenter": True,
                "reason": (
                    f"news-close reversed: {direction} supports {original_side} "
                    f"(materiality {materiality:.2f})"
                ),
            }
        return {
            "should_reenter": False,
            "reason": (
                f"materiality {materiality:.2f} < {config.REENTRY_NEWS_MATERIALITY} "
                f"for news re-entry"
            ),
        }

    if close_reason == "sl":
        if (
            materiality >= config.REENTRY_SL_MATERIALITY
            and confirming_source_count >= config.REENTRY_SL_MIN_SOURCES
        ):
            return {
                "should_reenter": True,
                "reason": (
                    f"stop-loss reversal confirmed: materiality {materiality:.2f}, "
                    f"{confirming_source_count} sources"
                ),
            }
        return {
            "should_reenter": False,
            "reason": (
                f"stop-loss re-entry needs materiality >= {config.REENTRY_SL_MATERIALITY} "
                f"and >= {config.REENTRY_SL_MIN_SOURCES} sources "
                f"(have {materiality:.2f}, {confirming_source_count})"
            ),
        }

    return {"should_reenter": False, "reason": f"no re-entry rule for close reason {close_reason!r}"}
