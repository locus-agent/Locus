"""entry_volume_usd: the market's USD volume captured on positions at open
time (docs/UNIVERSE_ANALYSIS.md instrumentation recommendation). Covers the
idempotent schema migration, capture in open_position (with the NULL fallback
for unknown/zero volume), and the calibration report's entry-volume bucket
section — including all-NULL legacy data."""
from types import SimpleNamespace

import pytest

from locus.core import performance, positions
from locus.markets.gamma import Market


def _mkt(volume=64_000.0, cid="cond-vol"):
    return Market(cid, "Will X happen?", "politics", 0.5, 0.5, volume,
                  "2030-01-01T00:00:00Z", True, [])


def _trade(tmp_db, cid="cond-vol"):
    return tmp_db.log_trade(cid, "Will X happen?", 0.7, 0.5, 0.2, "YES", 25.0)


def _position_row(tmp_db, cid):
    conn = tmp_db._conn()
    row = dict(conn.execute(
        "SELECT * FROM positions WHERE condition_id = ?", (cid,)).fetchone())
    conn.close()
    return row


# --- migration ----------------------------------------------------------------

def test_migration_is_idempotent(tmp_db):
    # tmp_db already ran init_db once; running it again must not raise or
    # duplicate the column.
    tmp_db.init_db()
    tmp_db.init_db()
    conn = tmp_db._conn()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(positions)").fetchall()]
    conn.close()
    assert cols.count("entry_volume_usd") == 1


def test_migration_adds_column_to_legacy_table(tmp_db):
    # Simulate a pre-column DB: drop and recreate positions without the column,
    # then re-run init_db — the migration must add it.
    conn = tmp_db._conn()
    conn.execute("DROP TABLE positions")
    conn.execute(
        """CREATE TABLE positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER UNIQUE REFERENCES trades(id),
            condition_id TEXT NOT NULL,
            market_question TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_yes_price REAL NOT NULL,
            amount_usd REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            opened_at TEXT NOT NULL DEFAULT (datetime('now'))
        )"""
    )
    conn.commit()
    conn.close()
    tmp_db.init_db()
    conn = tmp_db._conn()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(positions)").fetchall()]
    conn.close()
    assert cols.count("entry_volume_usd") == 1


# --- capture on open ----------------------------------------------------------

def test_open_position_stores_market_volume(tmp_db):
    trade_id = _trade(tmp_db)
    positions.open_position(trade_id, _mkt(volume=64_000.0), "YES", 25.0)
    assert _position_row(tmp_db, "cond-vol")["entry_volume_usd"] == pytest.approx(64_000.0)


def test_open_position_zero_volume_stored_as_null(tmp_db):
    # A zero volume means the caller had no real market data (e.g. a passive
    # fill's Market rebuilt from the pending_orders row) — stored NULL, not $0.
    trade_id = _trade(tmp_db)
    positions.open_position(trade_id, _mkt(volume=0.0), "YES", 25.0)
    assert _position_row(tmp_db, "cond-vol")["entry_volume_usd"] is None


def test_open_position_missing_volume_attr_stored_as_null(tmp_db):
    trade_id = _trade(tmp_db, cid="cond-bare")
    bare = SimpleNamespace(condition_id="cond-bare", question="Bare market?",
                           yes_price=0.5)
    positions.open_position(trade_id, bare, "YES", 25.0)
    assert _position_row(tmp_db, "cond-bare")["entry_volume_usd"] is None


# --- calibration report section -----------------------------------------------

def _close(tmp_db, cid, pnl):
    conn = tmp_db._conn()
    conn.execute(
        """UPDATE positions SET status = 'closed_manual', realized_pnl_usd = ?,
                  closed_at = datetime('now') WHERE condition_id = ?""",
        (pnl, cid),
    )
    conn.commit()
    conn.close()


def test_report_buckets_by_entry_volume(tmp_db):
    for cid, vol, pnl in [
        ("c-tiny", 10_000.0, 2.0),      # [<$50k], win
        ("c-mid", 120_000.0, -3.0),     # [$50k-$200k], loss
        ("c-upper", 350_000.0, 1.0),    # [$200k-$500k], win
        ("c-big", 900_000.0, 4.0),      # [>$500k], win
    ]:
        trade_id = _trade(tmp_db, cid=cid)
        positions.open_position(trade_id, _mkt(volume=vol, cid=cid), "YES", 25.0)
        _close(tmp_db, cid, pnl)

    rep = performance.calibration_report()
    by = {e["label"]: e for e in rep["by_entry_volume"]}
    assert by["[<$50k]"]["n"] == 1
    assert by["[<$50k]"]["win_rate"] == pytest.approx(100.0)
    assert by["[$50k-$200k]"]["n"] == 1
    assert by["[$50k-$200k]"]["total_pnl"] == pytest.approx(-3.0)
    assert by["[$200k-$500k]"]["n"] == 1
    assert by["[>$500k]"]["n"] == 1
    assert "unknown (pre-column)" not in by
    # buckets are emitted in ascending volume order
    labels = [e["label"] for e in rep["by_entry_volume"]]
    assert labels.index("[<$50k]") < labels.index("[>$500k]")

    text = performance.format_calibration_report(rep)
    assert "9. BREAKDOWN BY ENTRY VOLUME BUCKET" in text
    assert "[>$500k]" in text


def test_report_renders_with_all_null_legacy_volumes(tmp_db):
    # Legacy rows (pre-column) have NULL entry_volume_usd: the section must
    # group them as 'unknown (pre-column)' and still render.
    conn = tmp_db._conn()
    for i, pnl in enumerate([5.0, -2.0], start=1):
        conn.execute(
            """INSERT INTO positions (trade_id, condition_id, market_question,
                                      side, entry_yes_price, amount_usd, status,
                                      realized_pnl_usd, closed_at)
               VALUES (?, ?, 'Legacy?', 'YES', 0.5, 20.0, 'closed_manual', ?,
                       '2026-06-16 12:00:00')""",
            (i, f"legacy{i}", pnl),
        )
    conn.commit()
    conn.close()

    rep = performance.calibration_report()
    by = {e["label"]: e for e in rep["by_entry_volume"]}
    assert list(by) == ["unknown (pre-column)"]
    assert by["unknown (pre-column)"]["n"] == 2
    assert by["unknown (pre-column)"]["total_pnl"] == pytest.approx(3.0)

    text = performance.format_calibration_report(rep)
    assert "9. BREAKDOWN BY ENTRY VOLUME BUCKET" in text
    assert "unknown (pre-column)" in text
