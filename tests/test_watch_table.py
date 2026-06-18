"""Re-entry watch table: the watch-window query helpers (logger.find_watched_market
/ record_reentry / count_active_watched_markets / watch_closed_position) and the
position close -> watch hook (positions._close). The re-entry *decision* lives in
positions.check_reentry_opportunity (see test_reentry_v2.py)."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from locus import config

NOW = datetime(2026, 6, 14, 12, 0, 0, tzinfo=timezone.utc)


def _watch(db, condition_id="m1", reason="news", side="YES", watch_hours=72, now=NOW):
    conn = db._conn()
    try:
        inserted = db.watch_closed_position(
            conn, condition_id, f"Will {condition_id}?", side, 0.4, reason, watch_hours,
            now=now, exit_reason="news_decision",
        )
        conn.commit()
    finally:
        conn.close()
    return inserted


def test_watched_market_is_found_within_window(tmp_db):
    _watch(tmp_db)
    conn = tmp_db._conn()
    try:
        assert tmp_db.find_watched_market(conn, "m1", now=NOW) is not None
    finally:
        conn.close()
    assert tmp_db.count_active_watched_markets(now=NOW) == 1


def test_expired_watch_window_is_excluded(tmp_db):
    _watch(tmp_db, watch_hours=72)
    later = NOW + timedelta(hours=73)  # past watch_until
    conn = tmp_db._conn()
    try:
        assert tmp_db.find_watched_market(conn, "m1", now=later) is None
    finally:
        conn.close()
    assert tmp_db.count_active_watched_markets(now=later) == 0


def test_max_reentry_count_excludes_market(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "MAX_REENTRY_PER_MARKET", 1)
    _watch(tmp_db)
    conn = tmp_db._conn()
    try:
        tmp_db.record_reentry(conn, "m1", now=NOW)
        conn.commit()
        # reentry_count is now 1 == MAX -> no longer active
        assert tmp_db.find_watched_market(conn, "m1", now=NOW) is None
    finally:
        conn.close()
    assert tmp_db.count_active_watched_markets(now=NOW) == 0


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
    assert tmp_db.count_active_watched_markets(now=NOW) == 2


# --- close -> watch hook (positions._close) ------------------------------

def test_close_populates_watch_table_with_granular_exit_reason(tmp_db):
    from locus.core import positions

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
        watched_row = tmp_db.find_watched_market(conn, "c1")
        assert watched_row is not None
        # the bucketed close_reason and the granular exit_reason are both stored
        assert watched_row["close_reason"] == "sl"
        assert watched_row["exit_reason"] == "sl"
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
        assert tmp_db.find_watched_market(conn, "c2") is None
    finally:
        conn.close()
