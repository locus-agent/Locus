"""Track-record and lessons caches: TTL, invalidation, correctness."""
from locus import memory
from locus.memory import logger as logger_mod


def test_track_record_cached_within_ttl(tmp_db, monkeypatch):
    memory.invalidate_track_record_cache()
    calls = {"n": 0}
    real = logger_mod.get_calibration_with_trades

    def counting():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(memory.logger, "get_calibration_with_trades", counting)

    clock = {"t": 1000.0}
    monkeypatch.setattr(memory.time, "monotonic", lambda: clock["t"])

    memory.get_track_record()
    memory.get_track_record()
    memory.get_track_record()
    assert calls["n"] == 1  # served from cache

    clock["t"] += memory.TRACK_RECORD_TTL_SECONDS + 1
    memory.get_track_record()
    assert calls["n"] == 2  # TTL expired -> recomputed

    memory.invalidate_track_record_cache()
    memory.get_track_record()
    assert calls["n"] == 3  # explicit invalidation -> recomputed

    memory.invalidate_track_record_cache()


def test_lessons_cache_invalidated_by_new_lesson(tmp_db, monkeypatch):
    logger_mod._lessons_cache = None
    clock = {"t": 1000.0}
    monkeypatch.setattr(logger_mod, "_monotonic", lambda: clock["t"])

    assert logger_mod.get_recent_lessons(limit=5) == []
    # cached: a row inserted behind the cache's back is not yet visible...
    assert logger_mod.get_recent_lessons(limit=5) == []

    # ...but log_lesson invalidates, so the new lesson appears immediately
    logger_mod.log_lesson(1, "Q?", "bullish", "bearish", "lesson text")
    lessons = logger_mod.get_recent_lessons(limit=5)
    assert len(lessons) == 1 and lessons[0]["lesson"] == "lesson text"

    logger_mod._lessons_cache = None


def test_lessons_cache_respects_limit_change(tmp_db, monkeypatch):
    logger_mod._lessons_cache = None
    monkeypatch.setattr(logger_mod, "_monotonic", lambda: 1000.0)
    logger_mod.log_lesson(1, "Q?", "bullish", "bearish", "a")
    logger_mod.log_lesson(2, "Q?", "bullish", "bearish", "b")
    assert len(logger_mod.get_recent_lessons(limit=1)) == 1
    assert len(logger_mod.get_recent_lessons(limit=5)) == 2  # different limit -> fresh query
    logger_mod._lessons_cache = None
