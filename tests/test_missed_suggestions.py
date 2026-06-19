"""Missed-opportunity-driven threshold suggestions (conservative mode): the
recurring-pattern analysis, the decay window, and the pending-suggestion queue."""
from datetime import datetime, timedelta, timezone

import pytest

from locus import config
from locus.memory import calibrator
from locus.core import export_status


@pytest.fixture(autouse=True)
def _pin_thresholds(monkeypatch):
    monkeypatch.setattr(config, "MISSED_STALE_GEOPOLITICAL_THRESHOLD", 3)
    monkeypatch.setattr(config, "MISSED_MATERIALITY_THRESHOLD", 2)
    monkeypatch.setattr(config, "STALE_WINDOW_INCREASE_PCT", 0.25)
    monkeypatch.setattr(config, "MISSED_ADJUSTMENT_DECAY_DAYS", 30)


def _seed_lesson(db, question, action, pct_move, *, ago_days=0.0, direction="bullish"):
    """Insert a missed-opportunity lesson (reflection set) at NOW - ago_days."""
    created = (datetime.now(timezone.utc) - timedelta(days=ago_days)).strftime("%Y-%m-%d %H:%M:%S")
    conn = db._conn()
    conn.execute(
        """INSERT INTO lessons (market_question, classification, actual_direction,
                                lesson, reflection, action, pct_move, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (question, direction, direction, "missed", "reflection", action, pct_move, created),
    )
    conn.commit()
    conn.close()


def _missed(question, action, pct, materiality=0.4, direction="bullish", category="other"):
    return {"market_question": question, "action": action, "pct": pct,
            "materiality": materiality, "direction": direction, "category": category}


GEO_Q = "Will Iran sign a nuclear deal with the US?"
GEO_Q2 = "Will Russia withdraw troops from Ukraine?"
GEO_Q3 = "Will the Iran sanctions be lifted?"
CRYPTO_Q = "Will Bitcoin close above $100k in July?"


# --- stale geopolitical: 3-in-14-days ----------------------------------------

def test_stale_geopolitical_below_threshold_no_suggestion(tmp_db):
    # Only 2 qualifying geo stale misses -> no suggestion yet.
    _seed_lesson(tmp_db, GEO_Q, "stale", 30.0)
    _seed_lesson(tmp_db, GEO_Q2, "stale", 25.0)
    calibrator._analyze_missed_pattern(_missed(GEO_Q2, "stale", 0.25))
    assert tmp_db.get_pending_suggestions() == []


def test_stale_geopolitical_at_threshold_suggests(tmp_db):
    _seed_lesson(tmp_db, GEO_Q, "stale", 30.0)
    _seed_lesson(tmp_db, GEO_Q2, "stale", 20.0)
    _seed_lesson(tmp_db, GEO_Q3, "stale", 40.0)
    calibrator._analyze_missed_pattern(_missed(GEO_Q3, "stale", 0.40))
    pending = tmp_db.get_pending_suggestions()
    assert len(pending) == 1
    s = pending[0]
    assert s["suggestion_type"] == "stale_geopolitical"
    assert s["miss_count"] == 3
    assert s["avg_pct_move"] == pytest.approx(30.0)         # (30+20+40)/3
    assert "MAX_NEWS_AGE_SECONDS_GEOPOLITICAL by 25%" in s["suggestion_text"]


def test_stale_geopolitical_old_misses_outside_window_excluded(tmp_db):
    # Two recent + one 20 days old -> only 2 within the 14-day window -> no suggestion.
    _seed_lesson(tmp_db, GEO_Q, "stale", 30.0)
    _seed_lesson(tmp_db, GEO_Q2, "stale", 25.0)
    _seed_lesson(tmp_db, GEO_Q3, "stale", 40.0, ago_days=20.0)
    calibrator._analyze_missed_pattern(_missed(GEO_Q2, "stale", 0.25))
    assert tmp_db.get_pending_suggestions() == []


def test_non_geopolitical_stale_does_not_count(tmp_db):
    # Three stale misses but on a crypto market -> not geopolitical -> no suggestion.
    for _ in range(3):
        _seed_lesson(tmp_db, CRYPTO_Q, "stale", 30.0)
    calibrator._analyze_missed_pattern(_missed(CRYPTO_Q, "stale", 0.30, category="crypto"))
    assert tmp_db.get_pending_suggestions() == []


def test_stale_geopolitical_not_re_raised_when_pending(tmp_db):
    _seed_lesson(tmp_db, GEO_Q, "stale", 30.0)
    _seed_lesson(tmp_db, GEO_Q2, "stale", 20.0)
    _seed_lesson(tmp_db, GEO_Q3, "stale", 40.0)
    calibrator._analyze_missed_pattern(_missed(GEO_Q3, "stale", 0.40))
    calibrator._analyze_missed_pattern(_missed(GEO_Q3, "stale", 0.40))  # again
    assert len(tmp_db.get_pending_suggestions()) == 1  # no duplicate


# --- low_materiality category grouping ---------------------------------------

def test_low_materiality_same_category_suggests(tmp_db):
    _seed_lesson(tmp_db, CRYPTO_Q, "low_materiality", 25.0)
    _seed_lesson(tmp_db, "Will Ethereum flip Bitcoin?", "low_materiality", 30.0)
    calibrator._analyze_missed_pattern(
        _missed(CRYPTO_Q, "low_materiality", 0.25, category="crypto"))
    pending = tmp_db.get_pending_suggestions()
    assert len(pending) == 1
    assert pending[0]["suggestion_type"] == "low_materiality"
    assert pending[0]["category"] == "crypto"
    assert "crypto low_materiality misses" in pending[0]["suggestion_text"]


def test_low_materiality_mixed_categories_do_not_aggregate(tmp_db):
    # One crypto + one politics low_materiality miss -> neither category reaches 2.
    _seed_lesson(tmp_db, CRYPTO_Q, "low_materiality", 25.0)
    _seed_lesson(tmp_db, "Will Trump win the 2028 election?", "low_materiality", 30.0)
    calibrator._analyze_missed_pattern(
        _missed(CRYPTO_Q, "low_materiality", 0.25, category="crypto"))
    assert tmp_db.get_pending_suggestions() == []


# --- high-materiality skip -> lesson -----------------------------------------

def test_high_materiality_skip_logs_lesson(tmp_db):
    calibrator._analyze_missed_pattern(
        _missed(CRYPTO_Q, "skip", 0.30, materiality=0.6, category="crypto"))
    lessons = tmp_db.get_all_lessons()
    assert any("Direct Evidence Rule may be too strict" in l["lesson"] for l in lessons)
    # No adjustment suggestion for the skip path.
    assert tmp_db.get_pending_suggestions() == []


def test_high_materiality_skip_below_floor_does_nothing(tmp_db):
    calibrator._analyze_missed_pattern(
        _missed(CRYPTO_Q, "skip", 0.30, materiality=0.50, category="crypto"))  # < 0.55
    assert tmp_db.get_all_lessons() == []


# --- decay window + pending queue + export -----------------------------------

def test_decay_removes_old_suggestions(tmp_db):
    sid = tmp_db.log_adjustment_suggestion("low_materiality", "old one", category="crypto")
    # Backdate it past the 30-day decay window.
    conn = tmp_db._conn()
    old = (datetime.now(timezone.utc) - timedelta(days=31)).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("UPDATE adjustment_suggestions SET created_at = ? WHERE id = ?", (old, sid))
    conn.commit()
    conn.close()
    assert tmp_db.get_pending_suggestions() == []
    # A fresh one is still returned.
    tmp_db.log_adjustment_suggestion("stale_geopolitical", "fresh", category="geopolitical")
    assert len(tmp_db.get_pending_suggestions()) == 1


def test_mark_reviewed_drops_from_pending(tmp_db):
    sid = tmp_db.log_adjustment_suggestion("low_materiality", "review me", category="crypto")
    assert len(tmp_db.get_pending_suggestions()) == 1
    tmp_db.mark_suggestion_reviewed(sid)
    assert tmp_db.get_pending_suggestions() == []


def test_suggestion_stored_and_exported(tmp_db):
    tmp_db.log_adjustment_suggestion(
        "stale_geopolitical", "3+ geopolitical stale misses (avg +30%) — ...",
        category="geopolitical", avg_pct_move=30.0, miss_count=3,
    )
    rows = [export_status._suggestion_row(s) for s in tmp_db.get_pending_suggestions()]
    assert len(rows) == 1
    r = rows[0]
    assert r["type"] == "stale_geopolitical"
    assert r["category"] == "geopolitical"
    assert r["avg_pct_move"] == 30.0
    assert r["miss_count"] == 3
    assert "id" in r and "time" in r
