from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from time import monotonic as _monotonic
from pathlib import Path

from locus import config

DB_PATH = config.PROJECT_ROOT / "trades.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = _conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT NOT NULL,
            market_question TEXT NOT NULL,
            claude_score REAL NOT NULL,
            market_price REAL NOT NULL,
            edge REAL NOT NULL,
            side TEXT NOT NULL,
            amount_usd REAL NOT NULL,
            order_id TEXT,
            status TEXT NOT NULL DEFAULT 'dry_run',
            reasoning TEXT,
            headlines TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            -- V2 columns
            news_source TEXT,
            classification TEXT,
            materiality REAL,
            news_latency_ms INTEGER,
            classification_latency_ms INTEGER,
            total_latency_ms INTEGER
        );

        CREATE TABLE IF NOT EXISTS outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER NOT NULL REFERENCES trades(id),
            resolved_at TEXT,
            result TEXT,
            pnl REAL,
            UNIQUE(trade_id)
        );

        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            markets_scanned INTEGER DEFAULT 0,
            signals_found INTEGER DEFAULT 0,
            trades_placed INTEGER DEFAULT 0,
            status TEXT DEFAULT 'running'
        );

        CREATE TABLE IF NOT EXISTS news_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            headline TEXT NOT NULL,
            source TEXT NOT NULL,
            received_at TEXT NOT NULL,
            latency_ms INTEGER,
            matched_markets INTEGER DEFAULT 0,
            triggered_trades INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS calibration (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER REFERENCES trades(id),
            classification TEXT,
            materiality REAL,
            entry_price REAL,
            exit_price REAL,
            actual_direction TEXT,
            correct INTEGER,
            resolved_at TEXT,
            UNIQUE(trade_id)
        );

        CREATE TABLE IF NOT EXISTS lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER REFERENCES trades(id),
            market_question TEXT,
            classification TEXT,
            actual_direction TEXT,
            lesson TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_question TEXT NOT NULL,
            headline TEXT,
            news_source TEXT,
            direction TEXT,
            materiality REAL,
            edge REAL,
            expected_edge REAL,
            vol_adj REAL,
            action TEXT NOT NULL,
            match_source TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS classification_grades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            classification_id INTEGER NOT NULL UNIQUE REFERENCES classifications(id),
            direction TEXT,
            materiality REAL,
            entry_price REAL,
            price_after REAL,
            horizon_hours REAL,
            correct INTEGER,
            resolved_at TEXT
        );

        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER UNIQUE REFERENCES trades(id),
            condition_id TEXT NOT NULL,
            market_question TEXT NOT NULL,
            slug TEXT,
            side TEXT NOT NULL,
            entry_yes_price REAL NOT NULL,
            amount_usd REAL NOT NULL,
            headline TEXT,
            reasoning TEXT,
            status TEXT NOT NULL DEFAULT 'open',
            current_yes_price REAL,
            unrealized_pnl_pct REAL,
            realized_pnl_usd REAL DEFAULT 0,
            exit_yes_price REAL,
            exit_reason TEXT,
            last_reeval_at TEXT,
            last_trigger TEXT,
            end_date TEXT,
            opened_at TEXT NOT NULL DEFAULT (datetime('now')),
            closed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS exit_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            position_id INTEGER NOT NULL REFERENCES positions(id),
            trigger TEXT NOT NULL,
            decision TEXT NOT NULL,
            reasoning TEXT,
            pnl_pct REAL,
            yes_price REAL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL UNIQUE,
            entry TEXT NOT NULL,
            stats_snapshot TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        -- Meta-prompt evolution: each weekly self-improvement of the
        -- classification prompt is stored here (and as a versioned file under
        -- docs/prompts/). The classifier loads the latest version at runtime.
        CREATE TABLE IF NOT EXISTS prompt_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version INTEGER NOT NULL UNIQUE,
            prompt_text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            lessons_count INTEGER,
            accuracy_at_creation REAL
        );

        -- Markets we recently exited and keep watching for a thesis reversal
        -- (re-entry logic). One unexpired row per market; reentry_count caps
        -- how many times we re-enter (see core/reentry.py).
        CREATE TABLE IF NOT EXISTS watched_closed_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            condition_id TEXT NOT NULL,
            market_question TEXT,
            original_side TEXT,
            original_entry_price REAL,
            close_reason TEXT,
            closed_at TEXT NOT NULL DEFAULT (datetime('now')),
            watch_until TEXT,
            reentry_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    # Add V2 columns to existing trades table if missing
    _migrate_v2_columns(conn)
    _migrate_classification_columns(conn)
    _migrate_event_columns(conn)
    _migrate_position_category(conn)
    conn.close()


def _migrate_v2_columns(conn):
    """Add V2 columns to trades table if they don't exist."""
    cursor = conn.execute("PRAGMA table_info(trades)")
    columns = {row[1] for row in cursor.fetchall()}
    new_cols = [
        ("news_source", "TEXT"),
        ("classification", "TEXT"),
        ("materiality", "REAL"),
        ("news_latency_ms", "INTEGER"),
        ("classification_latency_ms", "INTEGER"),
        ("total_latency_ms", "INTEGER"),
        # Edge type behind the signal: 'news' (default), 'momentum', 'arbitrage'.
        ("edge_type", "TEXT"),
        # Claude's win-probability estimate (0.5-1.0) used for Kelly sizing.
        ("confidence", "REAL"),
    ]
    for col_name, col_type in new_cols:
        if col_name not in columns:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
    conn.commit()


def _migrate_classification_columns(conn):
    """Add newer columns to the classifications table if they don't exist."""
    cursor = conn.execute("PRAGMA table_info(classifications)")
    columns = {row[1] for row in cursor.fetchall()}
    new_cols = [
        ("match_source", "TEXT"),
        # Market context captured at classification time, so non-traded
        # directional calls can be graded against later price moves.
        ("condition_id", "TEXT"),
        ("yes_price", "REAL"),
        ("yes_token_id", "TEXT"),
        # Edge type behind a signal classification, for calibration by type.
        ("edge_type", "TEXT"),
        # Claude's win-probability estimate (0.5-1.0).
        ("confidence", "REAL"),
        # Multi-LLM ensemble: agreement score (0-1) and whether two models
        # were actually blended for this classification.
        ("consensus_score", "REAL"),
        ("ensemble_used", "INTEGER"),
        # Enhanced-edge sizing inputs: edge discounted by confidence, and the
        # volatility adjustment that penalizes near-certain markets.
        ("expected_edge", "REAL"),
        ("vol_adj", "REAL"),
    ]
    for col_name, col_type in new_cols:
        if col_name not in columns:
            conn.execute(f"ALTER TABLE classifications ADD COLUMN {col_name} {col_type}")
    conn.commit()


def _migrate_event_columns(conn):
    """Add the Gamma event_id column to trades, classifications, and positions
    (event context awareness — sibling outcomes share an event_id)."""
    for table in ("trades", "classifications", "positions"):
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = {row[1] for row in cursor.fetchall()}
        if "event_id" not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN event_id TEXT")


def _migrate_position_category(conn):
    """Add the inferred market category to positions (per-category exposure
    limits). Legacy rows stay NULL and fall back to question-based inference."""
    cursor = conn.execute("PRAGMA table_info(positions)")
    columns = {row[1] for row in cursor.fetchall()}
    if "category" not in columns:
        conn.execute("ALTER TABLE positions ADD COLUMN category TEXT")
    # Market close time, captured at open, for the time-pressure hard exit.
    # Legacy rows stay NULL and are skipped by the exit (unknown time-to-close).
    if "end_date" not in columns:
        conn.execute("ALTER TABLE positions ADD COLUMN end_date TEXT")
    conn.commit()


def log_trade(
    market_id: str,
    market_question: str,
    claude_score: float,
    market_price: float,
    edge: float,
    side: str,
    amount_usd: float,
    order_id: str | None = None,
    status: str = "dry_run",
    reasoning: str = "",
    headlines: str = "",
    news_source: str | None = None,
    classification: str | None = None,
    materiality: float | None = None,
    news_latency_ms: int | None = None,
    classification_latency_ms: int | None = None,
    total_latency_ms: int | None = None,
    edge_type: str | None = None,
    confidence: float | None = None,
    event_id: str | None = None,
) -> int:
    conn = _conn()
    cur = conn.execute(
        """INSERT INTO trades
           (market_id, market_question, claude_score, market_price, edge,
            side, amount_usd, order_id, status, reasoning, headlines,
            news_source, classification, materiality,
            news_latency_ms, classification_latency_ms, total_latency_ms, edge_type,
            confidence, event_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (market_id, market_question, claude_score, market_price, edge,
         side, amount_usd, order_id, status, reasoning, headlines,
         news_source, classification, materiality,
         news_latency_ms, classification_latency_ms, total_latency_ms, edge_type,
         confidence, event_id),
    )
    trade_id = cur.lastrowid
    conn.commit()
    conn.close()
    return trade_id


def log_news_event(
    headline: str,
    source: str,
    received_at: str,
    latency_ms: int = 0,
    matched_markets: int = 0,
    triggered_trades: int = 0,
) -> int:
    conn = _conn()
    cur = conn.execute(
        """INSERT INTO news_events
           (headline, source, received_at, latency_ms, matched_markets, triggered_trades)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (headline, source, received_at, latency_ms, matched_markets, triggered_trades),
    )
    event_id = cur.lastrowid
    conn.commit()
    conn.close()
    return event_id


def log_calibration(
    trade_id: int,
    classification: str,
    materiality: float,
    entry_price: float,
    exit_price: float | None = None,
    actual_direction: str | None = None,
    correct: bool | None = None,
    resolved_at: str | None = None,
):
    conn = _conn()
    conn.execute(
        """INSERT OR REPLACE INTO calibration
           (trade_id, classification, materiality, entry_price, exit_price,
            actual_direction, correct, resolved_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (trade_id, classification, materiality, entry_price, exit_price,
         actual_direction, 1 if correct else (0 if correct is not None else None),
         resolved_at),
    )
    conn.commit()
    conn.close()


def log_journal_entry(date: str, entry: str, stats_snapshot: str) -> int:
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO journal (date, entry, stats_snapshot) VALUES (?, ?, ?)",
        (date, entry, stats_snapshot),
    )
    journal_id = cur.lastrowid
    conn.commit()
    conn.close()
    return journal_id


def get_journal_entries(limit: int = 3) -> list[dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT date, entry, created_at FROM journal ORDER BY date DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def has_journal_for(date: str) -> bool:
    conn = _conn()
    row = conn.execute("SELECT 1 FROM journal WHERE date = ?", (date,)).fetchone()
    conn.close()
    return row is not None


def get_trades_for_performance() -> list[dict]:
    """Real (dry-run or live) trades with the fields PnL math needs."""
    conn = _conn()
    rows = conn.execute(
        """SELECT id, market_id, side, amount_usd, market_price
           FROM trades WHERE status IN ('dry_run', 'executed', 'filled')"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_calibration_exits() -> dict[int, float]:
    """{trade_id: exit YES price} for resolved trades."""
    conn = _conn()
    rows = conn.execute(
        "SELECT trade_id, exit_price FROM calibration WHERE exit_price IS NOT NULL"
    ).fetchall()
    conn.close()
    return {r["trade_id"]: r["exit_price"] for r in rows}


def get_calibrated_trade_ids() -> set[int]:
    """Trade ids already graded — used to resolve each trade exactly once."""
    conn = _conn()
    rows = conn.execute("SELECT trade_id FROM calibration").fetchall()
    conn.close()
    return {r["trade_id"] for r in rows}


def find_recent_classification(
    headline: str,
    condition_id: str,
    yes_price: float,
    price_tolerance: float,
    max_age_hours: float,
) -> dict | None:
    """A reusable prior result for the same (headline, market) pair: real
    classification (not prefiltered/cached/error), recent, and made when the
    market price was within price_tolerance of the current one."""
    conn = _conn()
    row = conn.execute(
        """SELECT direction, materiality, confidence, yes_price, created_at
           FROM classifications
           WHERE headline = ? AND condition_id = ?
             AND action NOT IN ('prefiltered', 'cached', 'error')
             AND direction IS NOT NULL
             AND yes_price IS NOT NULL
             AND ABS(yes_price - ?) <= ?
             AND created_at >= datetime('now', ?)
           ORDER BY id DESC LIMIT 1""",
        (headline, condition_id, yes_price, price_tolerance, f"-{max_age_hours} hours"),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_ungraded_directional_classifications(min_age_hours: float, limit: int = 500) -> list[dict]:
    """Directional (non-neutral) classifications with market context, old
    enough to grade, that have no grade row yet."""
    conn = _conn()
    rows = conn.execute(
        """SELECT c.id, c.direction, c.materiality, c.yes_price, c.yes_token_id, c.created_at
           FROM classifications c
           LEFT JOIN classification_grades g ON g.classification_id = c.id
           WHERE g.id IS NULL
             AND c.direction IN ('bullish', 'bearish')
             AND c.condition_id IS NOT NULL
             AND c.yes_token_id IS NOT NULL
             AND c.yes_price IS NOT NULL
             AND c.created_at <= datetime('now', ?)
           ORDER BY c.yes_token_id, c.id
           LIMIT ?""",
        (f"-{min_age_hours} hours", limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_classification_grade(
    classification_id: int,
    direction: str,
    materiality: float | None,
    entry_price: float,
    price_after: float | None,
    horizon_hours: float,
    correct: bool | None,
    resolved_at: str,
) -> None:
    conn = _conn()
    conn.execute(
        """INSERT OR IGNORE INTO classification_grades
           (classification_id, direction, materiality, entry_price, price_after,
            horizon_hours, correct, resolved_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (classification_id, direction, materiality, entry_price, price_after,
         horizon_hours, 1 if correct else (0 if correct is not None else None), resolved_at),
    )
    conn.commit()
    conn.close()


def get_classification_grades_with_meta() -> list[dict]:
    """Graded classification rows joined with question/source, for track record."""
    conn = _conn()
    rows = conn.execute("""
        SELECT g.correct, g.direction AS classification,
               c.market_question, c.news_source
        FROM classification_grades g
        JOIN classifications c ON g.classification_id = c.id
        WHERE g.correct IS NOT NULL
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_calibration_with_trades() -> list[dict]:
    """Resolved calibration records joined with their trade's question and news source."""
    conn = _conn()
    rows = conn.execute("""
        SELECT c.trade_id, c.classification, c.correct,
               t.market_question, t.news_source
        FROM calibration c
        JOIN trades t ON c.trade_id = t.id
        WHERE c.correct IS NOT NULL
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_graded_rows_with_prices() -> list[dict]:
    """Every graded row (correct IS NOT NULL) from both grading tables, with the
    entry price, the predicted direction, and the realized/marked exit price —
    the raw input for price-bucket accuracy analysis.

    `calibration` rows are resolved trades (exit = resolution price);
    `classification_grades` rows are non-traded directional calls marked at the
    CLOB price CALIBRATION_HORIZON_HOURS later. Both share the same shape:
    (direction, entry_price, exit_price, correct)."""
    conn = _conn()
    rows = conn.execute("""
        SELECT classification AS direction, entry_price, exit_price, correct
        FROM calibration
        WHERE correct IS NOT NULL AND entry_price IS NOT NULL AND exit_price IS NOT NULL
        UNION ALL
        SELECT direction, entry_price, price_after AS exit_price, correct
        FROM classification_grades
        WHERE correct IS NOT NULL AND entry_price IS NOT NULL AND price_after IS NOT NULL
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_lesson(trade_id: int, market_question: str, classification: str, actual_direction: str, lesson: str) -> int:
    global _lessons_cache
    _lessons_cache = None
    conn = _conn()
    cur = conn.execute(
        """INSERT INTO lessons (trade_id, market_question, classification, actual_direction, lesson)
           VALUES (?, ?, ?, ?, ?)""",
        (trade_id, market_question, classification, actual_direction, lesson),
    )
    lesson_id = cur.lastrowid
    conn.commit()
    conn.close()
    return lesson_id


LESSONS_TTL_SECONDS = 300.0
_lessons_cache: tuple[float, int, list[dict]] | None = None


def get_recent_lessons(limit: int = 5) -> list[dict]:
    """Cached for LESSONS_TTL_SECONDS (queried on every classify call);
    log_lesson() invalidates."""
    global _lessons_cache
    if (
        _lessons_cache is not None
        and _lessons_cache[1] == limit
        and _monotonic() - _lessons_cache[0] < LESSONS_TTL_SECONDS
    ):
        return _lessons_cache[2]
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM lessons ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    result = [dict(r) for r in rows]
    _lessons_cache = (_monotonic(), limit, result)
    return result


def get_all_lessons() -> list[dict]:
    """Every stored lesson, oldest first — for meta-prompt evolution."""
    conn = _conn()
    rows = conn.execute("SELECT * FROM lessons ORDER BY created_at").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_earliest_lesson_date() -> str | None:
    """created_at of the oldest lesson, or None when there are no lessons.
    Anchors the 'first evolution after 7 days of data' check."""
    conn = _conn()
    row = conn.execute("SELECT MIN(created_at) AS earliest FROM lessons").fetchone()
    conn.close()
    return row["earliest"] if row else None


# --- Meta-prompt evolution ---------------------------------------------------

def get_latest_prompt_version() -> dict | None:
    """The most recent evolved classification prompt, or None when none exist."""
    conn = _conn()
    row = conn.execute(
        "SELECT * FROM prompt_versions ORDER BY version DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_prompt_version_count() -> int:
    conn = _conn()
    n = conn.execute("SELECT COUNT(*) AS c FROM prompt_versions").fetchone()["c"]
    conn.close()
    return n


def save_prompt_version(
    version: int, prompt_text: str, lessons_count: int, accuracy_at_creation: float | None
) -> int:
    """Record a newly evolved prompt version. Versions are unique and monotonic."""
    conn = _conn()
    cur = conn.execute(
        """INSERT INTO prompt_versions
           (version, prompt_text, lessons_count, accuracy_at_creation)
           VALUES (?, ?, ?, ?)""",
        (version, prompt_text, lessons_count, accuracy_at_creation),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def log_classification(
    market_question: str,
    headline: str,
    news_source: str,
    direction: str,
    materiality: float,
    edge: float | None,
    action: str,
    match_source: str | None = None,
    condition_id: str | None = None,
    yes_price: float | None = None,
    yes_token_id: str | None = None,
    edge_type: str | None = None,
    confidence: float | None = None,
    event_id: str | None = None,
    consensus_score: float | None = None,
    ensemble_used: bool | None = None,
    expected_edge: float | None = None,
    vol_adj: float | None = None,
) -> int:
    conn = _conn()
    cur = conn.execute(
        """INSERT INTO classifications
           (market_question, headline, news_source, direction, materiality, edge, action,
            match_source, condition_id, yes_price, yes_token_id, edge_type, confidence,
            event_id, consensus_score, ensemble_used, expected_edge, vol_adj)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (market_question, headline, news_source, direction, materiality, edge, action,
         match_source, condition_id, yes_price, yes_token_id, edge_type, confidence,
         event_id,
         consensus_score,
         None if ensemble_used is None else int(ensemble_used),
         expected_edge, vol_adj),
    )
    classification_id = cur.lastrowid
    conn.commit()
    conn.close()
    return classification_id


def get_recent_closed_position_pnls(limit: int, since: str | None = None) -> list[float]:
    """Realized PnL of the most recently closed positions (newest first).

    Only fully-closed positions (status LIKE 'closed_%'); a NULL realized PnL
    is treated as 0.0 (a break-even, non-winning close). Backs dynamic Kelly
    win-rate sizing. `since` (an ISO date/datetime) restricts to positions
    opened on or after it — the PERFORMANCE_START_DATE window — before taking
    the most recent `limit`; default None counts all history."""
    conn = _conn()
    sql = "SELECT realized_pnl_usd FROM positions WHERE status LIKE 'closed_%'"
    params: list = []
    if since:
        sql += " AND opened_at >= ?"
        params.append(since)
    sql += " ORDER BY closed_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [r["realized_pnl_usd"] or 0.0 for r in rows]


def get_recent_classifications(limit: int = 20) -> list[dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM classifications ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_classification_count_since(since: str, action: str | None = None) -> int:
    conn = _conn()
    if action:
        row = conn.execute(
            "SELECT COUNT(*) as c FROM classifications WHERE created_at >= ? AND action = ?",
            (since, action),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) as c FROM classifications WHERE created_at >= ?", (since,)
        ).fetchone()
    conn.close()
    return row["c"]


def get_confirming_sources(condition_id: str, direction: str, since: str) -> set[str]:
    """Distinct news_source values among directional classifications for this
    market in this direction since `since`. Backs the high-materiality
    multi-source confirmation gate in pipeline.gate_trade."""
    conn = _conn()
    rows = conn.execute(
        """SELECT DISTINCT news_source FROM classifications
           WHERE condition_id = ? AND direction = ?
             AND news_source IS NOT NULL AND news_source != ''
             AND created_at >= ?""",
        (condition_id, direction, since),
    ).fetchall()
    conn.close()
    return {r["news_source"] for r in rows}


def watch_closed_position(
    conn,
    condition_id: str,
    market_question: str,
    original_side: str,
    original_entry_price: float | None,
    close_reason: str,
    watch_hours: float,
    now: datetime | None = None,
) -> bool:
    """Record a just-closed position as watched for re-entry (caller commits).

    Skips if an unexpired watch row already exists for this market, so one close
    -> at most one re-entry window. Returns True if a row was inserted.
    """
    now = now or datetime.now(timezone.utc)
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    existing = conn.execute(
        "SELECT 1 FROM watched_closed_positions WHERE condition_id=? AND watch_until > ? LIMIT 1",
        (condition_id, now_str),
    ).fetchone()
    if existing:
        return False
    watch_until = (now + timedelta(hours=watch_hours)).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """INSERT INTO watched_closed_positions
           (condition_id, market_question, original_side, original_entry_price,
            close_reason, closed_at, watch_until, reentry_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
        (condition_id, market_question, original_side, original_entry_price,
         close_reason, now_str, watch_until),
    )
    return True


def count_active_watched_markets(now: datetime | None = None) -> int:
    """Distinct markets still in their re-entry watch window and under the cap."""
    now = now or datetime.now(timezone.utc)
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    conn = _conn()
    row = conn.execute(
        """SELECT COUNT(DISTINCT condition_id) AS c FROM watched_closed_positions
           WHERE watch_until > ? AND reentry_count < ?""",
        (now_str, config.MAX_REENTRY_PER_MARKET),
    ).fetchone()
    conn.close()
    return row["c"]


def get_earliest_classification_price(condition_id: str, lookback_hours: float) -> float | None:
    """Oldest stored YES price for a market within the lookback window — the
    baseline the current price is compared against to detect a momentum move."""
    conn = _conn()
    row = conn.execute(
        """SELECT yes_price FROM classifications
           WHERE condition_id = ? AND yes_price IS NOT NULL
             AND created_at >= datetime('now', ?)
           ORDER BY created_at ASC, id ASC LIMIT 1""",
        (condition_id, f"-{lookback_hours} hours"),
    ).fetchone()
    conn.close()
    return row["yes_price"] if row else None


def get_edge_type_breakdown_since(since: str, action: str = "signal") -> dict[str, int]:
    """Count of `action` classifications by edge_type within the window."""
    conn = _conn()
    rows = conn.execute(
        """SELECT edge_type, COUNT(*) AS c FROM classifications
           WHERE created_at >= ? AND action = ? AND edge_type IS NOT NULL
           GROUP BY edge_type""",
        (since, action),
    ).fetchall()
    conn.close()
    return {r["edge_type"]: r["c"] for r in rows}


def get_matched_headline_count_since(since: str) -> int:
    """Headlines that produced at least one classification (i.e. matched)."""
    conn = _conn()
    row = conn.execute(
        "SELECT COUNT(DISTINCT headline) as c FROM classifications WHERE created_at >= ?",
        (since,),
    ).fetchone()
    conn.close()
    return row["c"]


def get_trade_count_since(since: str) -> int:
    conn = _conn()
    row = conn.execute(
        "SELECT COUNT(*) as c FROM trades WHERE created_at >= ?", (since,)
    ).fetchone()
    conn.close()
    return row["c"]


def get_news_event_count_since(since: str) -> int:
    conn = _conn()
    row = conn.execute(
        "SELECT COUNT(*) as c FROM news_events WHERE created_at >= ?", (since,)
    ).fetchone()
    conn.close()
    return row["c"]


def log_run_start() -> int:
    conn = _conn()
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO pipeline_runs (started_at) VALUES (?)", (now,)
    )
    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id


def log_run_end(run_id: int, markets_scanned: int, signals_found: int, trades_placed: int, status: str = "completed"):
    conn = _conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """UPDATE pipeline_runs
           SET finished_at=?, markets_scanned=?, signals_found=?, trades_placed=?, status=?
           WHERE id=?""",
        (now, markets_scanned, signals_found, trades_placed, status, run_id),
    )
    conn.commit()
    conn.close()


def get_daily_pnl() -> float:
    conn = _conn()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row = conn.execute(
        """SELECT COALESCE(SUM(
               CASE WHEN status IN ('filled','executed') THEN -amount_usd ELSE 0 END
           ), 0) as spent
           FROM trades WHERE created_at LIKE ?""",
        (f"{today}%",),
    ).fetchone()
    conn.close()
    return row["spent"]


def get_recent_trades(limit: int = 20, unresolved_only: bool = False) -> list[dict]:
    """Most recent trades, newest first. With unresolved_only=True, return only
    trades not yet graded in the calibration table (LEFT JOIN ... IS NULL) — so
    the calibrator scans only what's still open, not a fixed slice of all-time
    trades that fills up with already-resolved rows."""
    conn = _conn()
    if unresolved_only:
        rows = conn.execute(
            """SELECT t.* FROM trades t
               LEFT JOIN calibration c ON c.trade_id = t.id
               WHERE c.trade_id IS NULL
               ORDER BY t.created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_news_events(limit: int = 20) -> list[dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM news_events ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_trade_stats() -> dict:
    conn = _conn()
    total = conn.execute("SELECT COUNT(*) as c FROM trades").fetchone()["c"]
    by_status = conn.execute(
        "SELECT status, COUNT(*) as c FROM trades GROUP BY status"
    ).fetchall()
    conn.close()
    return {
        "total_trades": total,
        "by_status": {r["status"]: r["c"] for r in by_status},
    }


def get_calibration_stats() -> dict:
    conn = _conn()
    total = conn.execute("SELECT COUNT(*) as c FROM calibration WHERE correct IS NOT NULL").fetchone()["c"]
    if total == 0:
        conn.close()
        return {"total": 0, "accuracy": 0.0, "by_source": {}, "by_classification": {}}

    correct = conn.execute("SELECT COUNT(*) as c FROM calibration WHERE correct = 1").fetchone()["c"]

    by_source = {}
    rows = conn.execute("""
        SELECT t.news_source as source, COUNT(*) as total,
               SUM(CASE WHEN c.correct = 1 THEN 1 ELSE 0 END) as wins
        FROM calibration c JOIN trades t ON c.trade_id = t.id
        WHERE c.correct IS NOT NULL AND t.news_source IS NOT NULL
        GROUP BY t.news_source
    """).fetchall()
    for r in rows:
        by_source[r["source"]] = round(r["wins"] / r["total"] * 100, 1) if r["total"] > 0 else 0

    by_cls = {}
    rows = conn.execute("""
        SELECT classification, COUNT(*) as total,
               SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as wins
        FROM calibration WHERE correct IS NOT NULL
        GROUP BY classification
    """).fetchall()
    for r in rows:
        by_cls[r["classification"]] = round(r["wins"] / r["total"] * 100, 1) if r["total"] > 0 else 0

    conn.close()
    return {
        "total": total,
        "accuracy": round(correct / total * 100, 1),
        "by_source": by_source,
        "by_classification": by_cls,
    }


def get_latency_stats() -> dict:
    conn = _conn()
    row = conn.execute("""
        SELECT
            AVG(total_latency_ms) as avg_total,
            MIN(total_latency_ms) as min_total,
            MAX(total_latency_ms) as max_total,
            AVG(news_latency_ms) as avg_news,
            AVG(classification_latency_ms) as avg_class,
            COUNT(*) as count
        FROM trades
        WHERE total_latency_ms IS NOT NULL
    """).fetchone()
    conn.close()
    if not row or row["count"] == 0:
        return {"avg_total_ms": 0, "min_total_ms": 0, "max_total_ms": 0,
                "avg_news_ms": 0, "avg_class_ms": 0, "count": 0}
    return {
        "avg_total_ms": round(row["avg_total"] or 0),
        "min_total_ms": round(row["min_total"] or 0),
        "max_total_ms": round(row["max_total"] or 0),
        "avg_news_ms": round(row["avg_news"] or 0),
        "avg_class_ms": round(row["avg_class"] or 0),
        "count": row["count"],
    }


init_db()
