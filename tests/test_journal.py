"""Journal daily trigger: fires once, after 21:00 UTC, never twice."""
from datetime import datetime, timedelta, timezone

from locus import config
from locus.core import journal


def test_gather_daily_stats_on_empty_db(tmp_db):
    stats = journal.gather_daily_stats()
    assert stats["classifications_24h"] == 0
    assert stats["news_events_24h"] == 0
    assert stats["resolutions_24h"] == {"total": 0, "correct": 0}
    assert stats["closed_today"] == []


def _insert_position(tmp_db, *, question, status, exit_reason, pnl, closed_at):
    conn = tmp_db._conn()
    conn.execute(
        """INSERT INTO positions
           (condition_id, market_question, side, entry_yes_price, amount_usd,
            status, realized_pnl_usd, exit_reason, closed_at)
           VALUES (?, ?, 'YES', 0.5, 10, ?, ?, ?, ?)""",
        ("cond", question, status, pnl, exit_reason, closed_at),
    )
    conn.commit()
    conn.close()


def test_gather_daily_stats_includes_closed_today(tmp_db):
    now = datetime.now(timezone.utc)
    today_iso = now.replace(hour=1).isoformat()
    yesterday_iso = (now - timedelta(days=1)).isoformat()

    # Closed today (a profitable take-profit) — should appear.
    _insert_position(
        tmp_db, question="Will X win?", status="closed_tp",
        exit_reason="take_profit", pnl=4.2, closed_at=today_iso,
    )
    # Closed yesterday — should be excluded by the 00:00 UTC boundary.
    _insert_position(
        tmp_db, question="Old market?", status="closed_sl",
        exit_reason="stop_loss", pnl=-3.0, closed_at=yesterday_iso,
    )
    # Still open (no closed_at) — should be excluded.
    _insert_position(
        tmp_db, question="Open market?", status="open",
        exit_reason=None, pnl=0, closed_at=None,
    )

    stats = journal.gather_daily_stats()
    closed = stats["closed_today"]
    assert len(closed) == 1
    assert closed[0]["market_question"] == "Will X win?"
    assert closed[0]["exit_reason"] == "take_profit"
    assert closed[0]["realized_pnl_usd"] == 4.2


def test_trigger_respects_hour_and_writes_once(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "JOURNAL_ENABLED", True)
    written = []
    monkeypatch.setattr(
        journal, "write_journal_entry",
        lambda extra=None, date=None: (written.append(date), "entry text")[1],
    )

    before = datetime(2026, 6, 12, 20, 59, tzinfo=timezone.utc)
    after = datetime(2026, 6, 12, 21, 1, tzinfo=timezone.utc)

    assert journal.maybe_write_journal(now=before) is None
    assert written == []

    assert journal.maybe_write_journal(now=after) == "entry text"
    assert written == ["2026-06-12"]

    # Once the entry exists in the DB, later cycles are no-ops.
    tmp_db.log_journal_entry("2026-06-12", "entry text", "{}")
    assert journal.maybe_write_journal(now=after) is None
    assert written == ["2026-06-12"]


def test_disabled_flag_short_circuits(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "JOURNAL_ENABLED", False)
    called = []
    monkeypatch.setattr(journal, "write_journal_entry", lambda *a, **k: called.append(1))
    late = datetime(2026, 6, 12, 23, 0, tzinfo=timezone.utc)
    assert journal.maybe_write_journal(now=late) is None
    assert called == []


def test_strip_leading_date_lines():
    strip = journal._strip_leading_date_lines
    assert strip("June 12, 2026\n\nThe entry text.") == "The entry text."
    assert strip("2026-06-12\nThe entry text.") == "The entry text."
    assert strip("**June 12, 2026**\nThe entry text.") == "The entry text."
    assert strip("June 3rd, 2026\nText.") == "Text."
    # date mentioned mid-text is untouched; no leading date is a no-op
    assert strip("On June 12, 2026 I traded.") == "On June 12, 2026 I traded."
    assert strip("Plain entry.") == "Plain entry."
