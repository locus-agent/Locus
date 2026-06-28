"""The classifications.published_at column: round-trips through log_classification
and its migration is idempotent on a DB that predates the column."""


def _columns(tmp_db):
    conn = tmp_db._conn()
    try:
        cur = conn.execute("PRAGMA table_info(classifications)")
        return {row[1] for row in cur.fetchall()}
    finally:
        conn.close()


def test_published_at_column_exists(tmp_db):
    assert "published_at" in _columns(tmp_db)


def test_published_at_round_trips(tmp_db):
    cid = tmp_db.log_classification(
        market_question="Will X happen?", headline="h", news_source="rss",
        direction="bullish", materiality=0.4, edge=0.2, action="signal",
        condition_id="c1", yes_price=0.5,
        published_at="2026-06-23T14:30:00+00:00",
    )
    conn = tmp_db._conn()
    try:
        row = conn.execute(
            "SELECT published_at FROM classifications WHERE id = ?", (cid,)
        ).fetchone()
    finally:
        conn.close()
    assert row["published_at"] == "2026-06-23T14:30:00+00:00"


def test_published_at_defaults_to_null(tmp_db):
    # No publication time given (the received_at fallback case) -> NULL column.
    cid = tmp_db.log_classification(
        market_question="Will X happen?", headline="h", news_source="rss",
        direction="bullish", materiality=0.4, edge=0.2, action="signal",
        condition_id="c1", yes_price=0.5,
    )
    conn = tmp_db._conn()
    try:
        row = conn.execute(
            "SELECT published_at FROM classifications WHERE id = ?", (cid,)
        ).fetchone()
    finally:
        conn.close()
    assert row["published_at"] is None


def test_migration_is_idempotent(tmp_db):
    # Running the column migration again on an already-migrated DB is a no-op
    # (no duplicate-column error) and the column survives.
    conn = tmp_db._conn()
    try:
        tmp_db._migrate_classification_columns(conn)
        tmp_db._migrate_classification_columns(conn)
    finally:
        conn.close()
    assert "published_at" in _columns(tmp_db)


def test_migration_adds_column_to_legacy_table(tmp_db):
    # Simulate a pre-column DB: drop and recreate classifications without
    # published_at, then migrate.
    conn = tmp_db._conn()
    try:
        conn.execute("DROP TABLE classifications")
        conn.execute(
            """CREATE TABLE classifications (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   market_question TEXT NOT NULL,
                   action TEXT NOT NULL,
                   created_at TEXT NOT NULL DEFAULT (datetime('now'))
               )"""
        )
        conn.commit()
        assert "published_at" not in {
            r[1] for r in conn.execute("PRAGMA table_info(classifications)").fetchall()
        }
        tmp_db._migrate_classification_columns(conn)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(classifications)").fetchall()}
    finally:
        conn.close()
    assert "published_at" in cols
