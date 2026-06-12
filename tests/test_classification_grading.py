"""Non-traded classification grading: move math, batching, grade-once."""
import sqlite3
from datetime import datetime, timedelta, timezone

from locus import config
from locus.memory import calibrator


def test_grade_move_thresholds():
    t = 0.02
    assert calibrator.grade_move("bullish", 0.50, 0.55, t) is True
    assert calibrator.grade_move("bullish", 0.50, 0.51, t) is False  # within noise
    assert calibrator.grade_move("bullish", 0.50, 0.40, t) is False
    assert calibrator.grade_move("bearish", 0.50, 0.40, t) is True
    assert calibrator.grade_move("bearish", 0.50, 0.49, t) is False
    assert calibrator.grade_move("neutral", 0.50, 0.90, t) is False


def test_price_at_or_after():
    hist = [{"t": 100, "p": 0.5}, {"t": 200, "p": 0.6}, {"t": 300, "p": 0.7}]
    assert calibrator._price_at_or_after(hist, 150) == 0.6
    assert calibrator._price_at_or_after(hist, 300) == 0.7
    assert calibrator._price_at_or_after(hist, 999) == 0.7  # falls back to last
    assert calibrator._price_at_or_after([], 100) is None


def _insert_classification(db, direction, yes_price, token, hours_ago):
    created = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.execute(
        """INSERT INTO classifications
           (market_question, headline, direction, materiality, action,
            condition_id, yes_price, yes_token_id, created_at)
           VALUES (?, ?, ?, 0.7, 'skip', 'cid', ?, ?, ?)""",
        ("Will X happen?", "headline", direction, yes_price, token, created),
    )
    conn.commit()
    cid = cur.lastrowid
    conn.close()
    return cid


def test_grades_directional_rows_once(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "CALIBRATION_HORIZON_HOURS", 24.0)
    monkeypatch.setattr(config, "CALIBRATION_MOVE_THRESHOLD", 0.02)
    # bullish call 30h ago at 0.50; price later 0.60 -> correct
    _insert_classification(tmp_db, "bullish", 0.50, "tokA", hours_ago=30)
    # bearish call 30h ago at 0.50; price later 0.60 -> incorrect
    _insert_classification(tmp_db, "bearish", 0.50, "tokA", hours_ago=30)
    # too recent: must not be graded yet
    _insert_classification(tmp_db, "bullish", 0.50, "tokA", hours_ago=1)

    fetches = []

    def fake_history(token_id, start_ts, end_ts):
        fetches.append(token_id)
        return [{"t": start_ts, "p": 0.50}, {"t": end_ts, "p": 0.60}]

    monkeypatch.setattr(calibrator, "_fetch_price_history", fake_history)
    monkeypatch.setattr(calibrator.time, "sleep", lambda s: None)

    assert calibrator.grade_classifications() == 2
    assert fetches == ["tokA"]  # one history fetch covers both rows

    grades = tmp_db.get_classification_grades_with_meta()
    assert len(grades) == 2
    by_dir = {g["classification"]: g["correct"] for g in grades}
    assert by_dir == {"bullish": 1, "bearish": 0}

    # second run: nothing left old enough and ungraded
    assert calibrator.grade_classifications() == 0
    assert fetches == ["tokA"]


def test_missing_history_marks_ungradeable_not_retried(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "CALIBRATION_HORIZON_HOURS", 24.0)
    _insert_classification(tmp_db, "bullish", 0.50, "tokB", hours_ago=30)
    monkeypatch.setattr(calibrator, "_fetch_price_history", lambda *a: [])
    monkeypatch.setattr(calibrator.time, "sleep", lambda s: None)

    assert calibrator.grade_classifications() == 0
    # row got a NULL-correct grade: excluded from stats, never retried
    assert tmp_db.get_classification_grades_with_meta() == []
    assert tmp_db.get_ungraded_directional_classifications(min_age_hours=24.0) == []


def test_track_record_includes_classification_grades(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "CALIBRATION_HORIZON_HOURS", 24.0)
    monkeypatch.setattr(config, "CALIBRATION_MOVE_THRESHOLD", 0.02)
    _insert_classification(tmp_db, "bullish", 0.50, "tokC", hours_ago=30)
    monkeypatch.setattr(
        calibrator, "_fetch_price_history",
        lambda t, s, e: [{"t": s, "p": 0.50}, {"t": e, "p": 0.60}],
    )
    monkeypatch.setattr(calibrator.time, "sleep", lambda s: None)
    calibrator.grade_classifications()

    from locus import memory
    record = memory.get_track_record()
    assert record["total"] == 1
    assert record["accuracy"] == 100.0
