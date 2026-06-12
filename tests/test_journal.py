"""Journal daily trigger: fires once, after 21:00 UTC, never twice."""
from datetime import datetime, timezone

from locus import config
from locus.core import journal


def test_gather_daily_stats_on_empty_db(tmp_db):
    stats = journal.gather_daily_stats()
    assert stats["classifications_24h"] == 0
    assert stats["news_events_24h"] == 0
    assert stats["resolutions_24h"] == {"total": 0, "correct": 0}


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
