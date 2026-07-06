"""The missed-opportunity sweep's once-per-day guard is persisted in the DB
meta table (not a module global), so it survives the agent's frequent restarts
— the bug where no single process lived through the post-22:00-UTC window and
a whole day's sweep silently skipped."""
import importlib
import logging
from datetime import datetime, timezone

import pytest


@pytest.fixture
def journal(tmp_db, monkeypatch):
    from locus import config
    from locus.core import journal
    from locus.memory import calibrator

    monkeypatch.setattr(config, "MISSED_OPPORTUNITY_ENABLED", True)
    calls = []
    monkeypatch.setattr(
        calibrator, "check_missed_opportunities", lambda: calls.append(1) or 3
    )
    journal._sweep_calls = calls  # test-only handle
    return journal


def _at(hour, day=5):
    return datetime(2026, 7, day, hour, 30, tzinfo=timezone.utc)


def test_hour_gate_respected(journal):
    assert journal.maybe_check_missed_opportunities(now=_at(21)) == 0
    assert journal._sweep_calls == []
    assert journal.maybe_check_missed_opportunities(now=_at(22)) == 3
    assert journal._sweep_calls == [1]


def test_disabled_flag(journal, monkeypatch):
    from locus import config

    monkeypatch.setattr(config, "MISSED_OPPORTUNITY_ENABLED", False)
    assert journal.maybe_check_missed_opportunities(now=_at(23)) == 0
    assert journal._sweep_calls == []


def test_no_double_run_same_day(journal):
    assert journal.maybe_check_missed_opportunities(now=_at(22)) == 3
    assert journal.maybe_check_missed_opportunities(now=_at(23)) == 0
    assert journal._sweep_calls == [1]


def test_guard_survives_restart(journal, tmp_db, monkeypatch):
    """A process restart (fresh module state, same DB) must not rerun the sweep."""
    assert journal.maybe_check_missed_opportunities(now=_at(22)) == 3

    from locus.core import journal as journal_mod
    from locus.memory import calibrator

    reloaded = importlib.reload(journal_mod)
    calls = []
    monkeypatch.setattr(
        calibrator, "check_missed_opportunities", lambda: calls.append(1) or 3
    )
    assert reloaded.maybe_check_missed_opportunities(now=_at(23)) == 0
    assert calls == []

    # Next day it runs again.
    assert reloaded.maybe_check_missed_opportunities(now=_at(22, day=6)) == 3
    assert calls == [1]


def test_next_day_runs_without_gap_warning(journal, caplog):
    assert journal.maybe_check_missed_opportunities(now=_at(22, day=5)) == 3
    with caplog.at_level(logging.WARNING):
        assert journal.maybe_check_missed_opportunities(now=_at(22, day=6)) == 3
    assert "gap" not in caplog.text.lower()


def test_gap_detected_logs_warning_and_resumes(journal, tmp_db, caplog):
    """A fully missed day (last run older than yesterday) resumes with today's
    sweep and logs a warning so the gap is visible."""
    tmp_db.set_meta(journal._MISSED_OPPORTUNITY_META_KEY, "2026-07-03")
    with caplog.at_level(logging.WARNING):
        assert journal.maybe_check_missed_opportunities(now=_at(22, day=5)) == 3
    assert "gap" in caplog.text.lower()
    assert "2026-07-03" in caplog.text
    assert tmp_db.get_meta(journal._MISSED_OPPORTUNITY_META_KEY) == "2026-07-05"


def test_first_run_ever_no_warning(journal, caplog):
    """No persisted date at all (fresh install) is not a gap."""
    with caplog.at_level(logging.WARNING):
        assert journal.maybe_check_missed_opportunities(now=_at(22)) == 3
    assert "gap" not in caplog.text.lower()


def test_meta_kv_roundtrip(tmp_db):
    assert tmp_db.get_meta("nope") is None
    tmp_db.set_meta("k", "v1")
    assert tmp_db.get_meta("k") == "v1"
    tmp_db.set_meta("k", "v2")
    assert tmp_db.get_meta("k") == "v2"
