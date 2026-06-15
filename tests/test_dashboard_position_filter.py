"""Dashboard display filter: DASHBOARD_POSITIONS_START_DATE hides old positions
from the exported open/closed lists, without touching the DB the pipeline,
calibration, circuit breaker, and performance read from."""
import pytest

from locus import config
from locus.core import positions, export_status
from locus.core.performance import compute_performance
from locus.markets.gamma import Market


OLD = "2026-06-10 09:00:00"
NEW = "2026-06-14 09:00:00"
CUTOFF = "2026-06-14"


def _make_position(tmp_db, cond, opened_at, side="YES", entry=0.5, amount=25.0,
                   closed=False, closed_at=None, realized=0.0):
    """Create a position with a controlled opened_at (and optional closed state)."""
    trade_id = tmp_db.log_trade(
        market_id=cond, market_question=f"Q {cond}?", claude_score=0.7,
        market_price=entry, edge=0.2, side=side, amount_usd=amount,
        status="dry_run", classification="bullish", materiality=0.7,
    )
    mkt = Market(cond, f"Q {cond}?", "ai", entry, round(1 - entry, 4),
                 5000, "", True, [])
    positions.open_position(trade_id, mkt, side, amount)
    conn = tmp_db._conn()
    if closed:
        conn.execute(
            "UPDATE positions SET opened_at=?, status='closed_manual', closed_at=?, "
            "exit_yes_price=?, exit_reason='manual', realized_pnl_usd=? WHERE trade_id=?",
            (opened_at, closed_at or opened_at, entry, realized, trade_id),
        )
    else:
        conn.execute("UPDATE positions SET opened_at=? WHERE trade_id=?",
                     (opened_at, trade_id))
    conn.commit()
    conn.close()
    return trade_id


# --- the SQL-level filter on the position helpers --------------------------

def test_get_open_positions_filtered_by_since(tmp_db):
    _make_position(tmp_db, "old1", OLD)
    _make_position(tmp_db, "new1", NEW)
    # No filter -> the full book (what the risk gates / exit management see).
    assert len(positions.get_open_positions()) == 2
    # With a cutoff -> only positions opened on or after it.
    filtered = positions.get_open_positions(since=CUTOFF)
    assert [p["condition_id"] for p in filtered] == ["new1"]


def test_get_closed_positions_filtered_by_since(tmp_db):
    _make_position(tmp_db, "oldC", OLD, closed=True, closed_at="2026-06-11 10:00:00")
    _make_position(tmp_db, "newC", NEW, closed=True, closed_at="2026-06-15 10:00:00")
    assert len(positions.get_closed_positions()) == 2
    filtered = positions.get_closed_positions(since=CUTOFF)
    assert [p["condition_id"] for p in filtered] == ["newC"]


def test_position_opened_on_cutoff_is_included(tmp_db):
    # opened_at exactly on the cutoff date is kept (filter is inclusive).
    _make_position(tmp_db, "edge", "2026-06-14 00:00:01")
    assert len(positions.get_open_positions(since=CUTOFF)) == 1


# --- the export integration (no real-file / git side effects) --------------

@pytest.fixture
def isolated_export(tmp_path, monkeypatch):
    """Point the export at a throwaway status.json and disable archives/push."""
    monkeypatch.setattr(config, "AUTO_PUSH_STATUS", False)
    monkeypatch.setattr(export_status, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(export_status, "_export_archives", lambda: None)


def _questions(rows):
    return {r["market_question"] for r in rows}


def test_export_excludes_old_positions_when_date_set(tmp_db, isolated_export, monkeypatch):
    monkeypatch.setattr(config, "DASHBOARD_POSITIONS_START_DATE", CUTOFF)
    _make_position(tmp_db, "old1", OLD)
    _make_position(tmp_db, "new1", NEW)
    _make_position(tmp_db, "oldC", OLD, closed=True, closed_at="2026-06-11 10:00:00")
    _make_position(tmp_db, "newC", NEW, closed=True, closed_at="2026-06-15 10:00:00")

    status = export_status.export_status()
    assert _questions(status["open_positions"]) == {"Q new1?"}
    assert _questions(status["closed_positions"]) == {"Q newC?"}


def test_export_includes_all_positions_when_date_unset(tmp_db, isolated_export, monkeypatch):
    monkeypatch.setattr(config, "DASHBOARD_POSITIONS_START_DATE", "")
    _make_position(tmp_db, "old1", OLD)
    _make_position(tmp_db, "new1", NEW)
    _make_position(tmp_db, "oldC", OLD, closed=True, closed_at="2026-06-11 10:00:00")
    _make_position(tmp_db, "newC", NEW, closed=True, closed_at="2026-06-15 10:00:00")

    status = export_status.export_status()
    assert _questions(status["open_positions"]) == {"Q old1?", "Q new1?"}
    assert _questions(status["closed_positions"]) == {"Q oldC?", "Q newC?"}


def test_filter_does_not_affect_performance(tmp_db, isolated_export, monkeypatch):
    # The dashboard hides the old position, but performance/calibration still
    # count it — they read the positions table directly, not the filtered lists.
    monkeypatch.setattr(config, "DASHBOARD_POSITIONS_START_DATE", CUTOFF)
    _make_position(tmp_db, "oldC", OLD, closed=True, closed_at="2026-06-11 10:00:00",
                   realized=10.0)
    _make_position(tmp_db, "newC", NEW, closed=True, closed_at="2026-06-15 10:00:00",
                   realized=5.0)

    status = export_status.export_status()
    # Only the new one is shown...
    assert _questions(status["closed_positions"]) == {"Q newC?"}
    # ...but performance still sees both closes and both realized PnLs.
    perf = compute_performance()
    assert perf["closed_count"] == 2
    assert perf["realized_pnl_usd"] == pytest.approx(15.0)
    # And the unfiltered helper still returns the whole book.
    assert len(positions.get_closed_positions()) == 2
