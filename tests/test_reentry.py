"""Re-entry logic: close-reason rules, watch-window expiry, the per-market cap,
and the close -> watch hook."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from locus import config
from locus.core import reentry

NOW = datetime(2026, 6, 14, 12, 0, 0, tzinfo=timezone.utc)


def cls(direction, materiality):
    """Minimal stand-in for a Classification (only the fields reentry reads)."""
    return SimpleNamespace(direction=direction, materiality=materiality)


def watched(close_reason, original_side="YES"):
    return {
        "condition_id": "m1", "market_question": "Will m1?",
        "original_side": original_side, "original_entry_price": 0.4,
        "close_reason": close_reason, "reentry_count": 0,
    }


# --- check_reentry_opportunity rules -------------------------------------

def test_tp_close_never_reenters():
    d = reentry.check_reentry_opportunity(watched("tp"), cls("bullish", 0.99))
    assert d["should_reenter"] is False
    assert "profit" in d["reason"].lower()


def test_news_close_reenters_on_original_direction():
    # YES position closed on news; fresh bullish news (supports YES) above 0.45.
    d = reentry.check_reentry_opportunity(watched("news"), cls("bullish", 0.5))
    assert d["should_reenter"] is True


def test_news_close_below_materiality_blocks():
    d = reentry.check_reentry_opportunity(watched("news"), cls("bullish", 0.4))
    assert d["should_reenter"] is False


def test_news_close_wrong_direction_blocks():
    # Bearish news does not support re-entering a YES position.
    d = reentry.check_reentry_opportunity(watched("news"), cls("bearish", 0.9))
    assert d["should_reenter"] is False
    assert "support" in d["reason"].lower()


def test_news_close_no_position_direction_match():
    # NO position closed on news; bearish news supports NO.
    d = reentry.check_reentry_opportunity(
        watched("news", original_side="NO"), cls("bearish", 0.5)
    )
    assert d["should_reenter"] is True


def test_sl_close_requires_high_materiality_and_two_sources():
    w = watched("sl")
    # materiality below 0.55 -> blocked even with sources
    assert reentry.check_reentry_opportunity(w, cls("bullish", 0.5), confirming_source_count=3)["should_reenter"] is False
    # enough materiality but only one source -> blocked
    assert reentry.check_reentry_opportunity(w, cls("bullish", 0.6), confirming_source_count=1)["should_reenter"] is False
    # both conditions met -> re-enter
    assert reentry.check_reentry_opportunity(w, cls("bullish", 0.6), confirming_source_count=2)["should_reenter"] is True


def test_unknown_close_reason_does_not_reenter():
    d = reentry.check_reentry_opportunity(watched("drawdown"), cls("bullish", 0.99), confirming_source_count=5)
    assert d["should_reenter"] is False


# --- watch table: window, cap, hook --------------------------------------

def _watch(db, condition_id="m1", reason="news", side="YES", watch_hours=72, now=NOW):
    conn = db._conn()
    try:
        logger_inserted = db.watch_closed_position(
            conn, condition_id, f"Will {condition_id}?", side, 0.4, reason, watch_hours, now=now
        )
        conn.commit()
    finally:
        conn.close()
    return logger_inserted


def test_watched_market_is_found_within_window(tmp_db):
    _watch(tmp_db)
    conn = tmp_db._conn()
    try:
        assert reentry.find_watched_market(conn, "m1", now=NOW) is not None
        assert len(reentry.get_watched_markets(conn, now=NOW)) == 1
    finally:
        conn.close()


def test_expired_watch_window_is_excluded(tmp_db):
    _watch(tmp_db, watch_hours=72)
    later = NOW + timedelta(hours=73)  # past watch_until
    conn = tmp_db._conn()
    try:
        assert reentry.find_watched_market(conn, "m1", now=later) is None
        assert reentry.get_watched_markets(conn, now=later) == []
    finally:
        conn.close()


def test_max_reentry_count_excludes_market(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "MAX_REENTRY_PER_MARKET", 1)
    _watch(tmp_db)
    conn = tmp_db._conn()
    try:
        reentry.record_reentry(conn, "m1", now=NOW)
        conn.commit()
        # reentry_count is now 1 == MAX -> no longer active
        assert reentry.find_watched_market(conn, "m1", now=NOW) is None
        assert reentry.get_watched_markets(conn, now=NOW) == []
    finally:
        conn.close()


def test_watch_dedupes_active_window(tmp_db):
    assert _watch(tmp_db) is True   # first insert
    assert _watch(tmp_db) is False  # already watched, skipped
    conn = tmp_db._conn()
    try:
        rows = conn.execute("SELECT COUNT(*) c FROM watched_closed_positions").fetchone()
        assert rows["c"] == 1
    finally:
        conn.close()


def test_count_active_watched_markets(tmp_db):
    _watch(tmp_db, condition_id="m1")
    _watch(tmp_db, condition_id="m2")
    # count_active_watched_markets uses real "now"; rows watch 72h from NOW
    # (2026), so without a now arg they may be expired. Query relative to NOW.
    conn = tmp_db._conn()
    try:
        assert len(reentry.get_watched_markets(conn, now=NOW)) == 2
    finally:
        conn.close()


# --- close -> watch hook (positions._close) ------------------------------

def test_close_populates_watch_table_except_resolution(tmp_db):
    from locus.core import positions

    market = SimpleNamespace(condition_id="c1", market_question="Will c1?")
    pos = {
        "id": 1, "condition_id": "c1", "market_question": "Will c1?",
        "side": "YES", "entry_yes_price": 0.4, "amount_usd": 10.0,
    }
    conn = tmp_db._conn()
    try:
        # seed a position row so the UPDATE in _close has a target
        conn.execute(
            """INSERT INTO positions (id, trade_id, condition_id, market_question,
               side, entry_yes_price, amount_usd, status, current_yes_price)
               VALUES (1, 1, 'c1', 'Will c1?', 'YES', 0.4, 10.0, 'open', 0.4)""",
        )
        conn.commit()
        positions._close(conn, pos, 0.2, "closed_sl", "sl")
        conn.commit()
        watched_row = reentry.find_watched_market(conn, "c1")
        assert watched_row is not None
        assert watched_row["close_reason"] == "sl"
        assert watched_row["original_side"] == "YES"
    finally:
        conn.close()


def test_resolution_close_is_not_watched(tmp_db):
    from locus.core import positions

    pos = {
        "id": 2, "condition_id": "c2", "market_question": "Will c2?",
        "side": "NO", "entry_yes_price": 0.6, "amount_usd": 10.0,
    }
    conn = tmp_db._conn()
    try:
        conn.execute(
            """INSERT INTO positions (id, trade_id, condition_id, market_question,
               side, entry_yes_price, amount_usd, status, current_yes_price)
               VALUES (2, 2, 'c2', 'Will c2?', 'NO', 0.6, 10.0, 'open', 0.6)""",
        )
        conn.commit()
        positions._close(conn, pos, 1.0, "resolved", "resolution")
        conn.commit()
        assert reentry.find_watched_market(conn, "c2") is None
    finally:
        conn.close()
