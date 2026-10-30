"""Paper (simulated dry-run) vs live (real on-chain) split: the mode is read
off the ORIGINATING trade's status (trades.status via positions.trade_id /
calibration.trade_id) — no separate paper/live column anywhere. Covers
compute_performance, memory.get_track_record, and export_status.export_status
(the dashboard's live-only `performance`/`track_record` plus the separate
`performance_paper`/`track_record_paper` fields)."""
import pytest

from locus import config, memory
from locus.core import performance
from locus.core.performance import compute_performance
from locus.markets.gamma import Market


@pytest.fixture(autouse=True)
def _neutralize_start_dates(monkeypatch):
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_START_DATE", "")
    monkeypatch.setattr(config, "PERFORMANCE_START_DATE", "")


def _position(tmp_db, mid, side, price, status, amount=25.0):
    """Open (and the caller may later close) a position whose ORIGINATING
    trade has the given status ('dry_run' = paper, 'executed' = live)."""
    from locus.core import positions

    trade_id = tmp_db.log_trade(
        market_id=mid, market_question=f"Will {mid}?", claude_score=0.7,
        market_price=price, edge=0.2, side=side, amount_usd=amount,
        status=status, classification="bullish", materiality=0.7,
    )
    mkt = Market(mid, f"Will {mid}?", "ai", price, round(1 - price, 4),
                5000, "", True, [])
    positions.open_position(trade_id, mkt, side, amount)
    return trade_id


def _live(tmp_db, mid, side, price, amount=25.0):
    return _position(tmp_db, mid, side, price, "executed", amount)


def _paper(tmp_db, mid, side, price, amount=25.0):
    return _position(tmp_db, mid, side, price, "dry_run", amount)


# --- compute_performance mode split ------------------------------------------

def test_compute_performance_default_pools_everything(tmp_db, monkeypatch):
    from locus.core import positions

    t_live = _live(tmp_db, "m1", "YES", 0.50)
    positions.close_on_resolution(t_live, 1.0)  # +25
    t_paper = _paper(tmp_db, "m2", "YES", 0.50)
    positions.close_on_resolution(t_paper, 0.0)  # -25

    perf = compute_performance(current_prices={})
    assert perf["mode"] == "all"
    assert perf["trades_total"] == 2
    assert perf["closed_count"] == 2
    assert perf["realized_pnl_usd"] == pytest.approx(0.0)  # +25 - 25


def test_compute_performance_live_excludes_paper(tmp_db, monkeypatch):
    from locus.core import positions

    t_live = _live(tmp_db, "m1", "YES", 0.50)
    positions.close_on_resolution(t_live, 1.0)  # +25
    t_paper = _paper(tmp_db, "m2", "YES", 0.50)
    positions.close_on_resolution(t_paper, 0.0)  # -25, must not show up

    perf = compute_performance(current_prices={}, mode="live")
    assert perf["mode"] == "live"
    assert perf["trades_total"] == 1
    assert perf["closed_count"] == 1
    assert perf["wins"] == 1 and perf["losses"] == 0
    assert perf["realized_pnl_usd"] == pytest.approx(25.0)


def test_compute_performance_paper_excludes_live(tmp_db, monkeypatch):
    from locus.core import positions

    t_live = _live(tmp_db, "m1", "YES", 0.50)
    positions.close_on_resolution(t_live, 1.0)  # +25, must not show up
    t_paper = _paper(tmp_db, "m2", "YES", 0.50)
    positions.close_on_resolution(t_paper, 0.0)  # -25

    perf = compute_performance(current_prices={}, mode="paper")
    assert perf["mode"] == "paper"
    assert perf["trades_total"] == 1
    assert perf["closed_count"] == 1
    assert perf["wins"] == 0 and perf["losses"] == 1
    assert perf["realized_pnl_usd"] == pytest.approx(-25.0)


def test_compute_performance_counts_open_positions_per_mode(tmp_db):
    _live(tmp_db, "m1", "YES", 0.50)
    _paper(tmp_db, "m2", "YES", 0.50)

    live_perf = compute_performance(current_prices={}, mode="live")
    paper_perf = compute_performance(current_prices={}, mode="paper")
    assert live_perf["open_count"] == 1
    assert paper_perf["open_count"] == 1


def test_compute_performance_orphan_position_excluded_from_live_and_paper(tmp_db):
    # A position with no linked trade (trade_id NULL) can't be attributed to
    # either mode — it must never silently count as live OR paper.
    conn = tmp_db._conn()
    conn.execute(
        """INSERT INTO positions (trade_id, condition_id, market_question, side,
                                  entry_yes_price, amount_usd, status,
                                  realized_pnl_usd, closed_at)
           VALUES (NULL, 'condX', 'Orphan?', 'YES', 0.4, 10.0, 'closed_manual',
                   3.0, '2026-06-16 12:00:00')""",
    )
    conn.commit()
    conn.close()

    assert compute_performance(current_prices={}, mode="live")["closed_count"] == 0
    assert compute_performance(current_prices={}, mode="paper")["closed_count"] == 0
    assert compute_performance(current_prices={}, mode="all")["closed_count"] == 1


def test_compute_performance_invalid_mode_raises():
    with pytest.raises(ValueError):
        compute_performance(mode="bogus")


# --- memory.get_track_record mode split --------------------------------------

def _seed_calibration(tmp_db, trade_status, correct, question="Will X?", source="rss"):
    trade_id = tmp_db.log_trade(
        market_id="c1", market_question=question, claude_score=0.7,
        market_price=0.5, edge=0.2, side="YES", amount_usd=25.0,
        status=trade_status, classification="bullish", materiality=0.7,
        news_source=source,
    )
    tmp_db.log_calibration(
        trade_id=trade_id, classification="bullish", materiality=0.7,
        entry_price=0.5, exit_price=1.0 if correct else 0.0,
        actual_direction="bullish" if correct else "bearish", correct=correct,
    )
    return trade_id


@pytest.fixture(autouse=True)
def _clear_track_record_cache():
    memory.invalidate_track_record_cache()
    yield
    memory.invalidate_track_record_cache()


def test_get_track_record_default_pools_everything(tmp_db):
    _seed_calibration(tmp_db, "executed", correct=True)
    _seed_calibration(tmp_db, "dry_run", correct=False)

    record = memory.get_track_record()
    assert record["mode"] == "all"
    assert record["total"] == 2
    assert record["accuracy"] == pytest.approx(50.0)


def test_get_track_record_live_excludes_paper(tmp_db):
    _seed_calibration(tmp_db, "executed", correct=True)
    _seed_calibration(tmp_db, "dry_run", correct=False)

    record = memory.get_track_record(mode="live")
    assert record["mode"] == "live"
    assert record["total"] == 1
    assert record["accuracy"] == pytest.approx(100.0)


def test_get_track_record_paper_excludes_live(tmp_db):
    _seed_calibration(tmp_db, "executed", correct=True)
    _seed_calibration(tmp_db, "dry_run", correct=False)

    record = memory.get_track_record(mode="paper")
    assert record["mode"] == "paper"
    assert record["total"] == 1
    assert record["accuracy"] == pytest.approx(0.0)


def test_get_track_record_nontraded_classifications_only_in_all(tmp_db):
    # A non-traded classification_grades row has no trade link at all — it
    # must never count as "live" or "paper" (there's no trades.status to
    # read), only ever appearing under mode="all".
    conn = tmp_db._conn()
    conn.execute(
        """INSERT INTO classifications
           (headline, market_question, direction, materiality, news_source, action)
           VALUES ('H', 'Will X?', 'bullish', 0.5, 'rss', 'signal')""",
    )
    cid = conn.execute("SELECT id FROM classifications").fetchone()["id"]
    conn.commit()
    conn.close()
    tmp_db.log_classification_grade(
        classification_id=cid, direction="bullish", materiality=0.5,
        entry_price=0.5, price_after=0.6, horizon_hours=4.0,
        correct=True, resolved_at="2026-06-16 12:00:00",
    )

    assert memory.get_track_record(mode="all")["total"] == 1
    assert memory.get_track_record(mode="live")["total"] == 0
    assert memory.get_track_record(mode="paper")["total"] == 0


def test_get_track_record_cache_is_keyed_per_mode(tmp_db):
    _seed_calibration(tmp_db, "executed", correct=True)

    live1 = memory.get_track_record(mode="live")
    paper1 = memory.get_track_record(mode="paper")
    assert live1["total"] == 1
    assert paper1["total"] == 0  # not a stale copy of the live result

    # Add a paper row; the paper cache must still be fresh enough to see it
    # once TTL passes or cache is invalidated — invalidate explicitly here to
    # assert the two mode-caches are genuinely independent, not aliased.
    memory.invalidate_track_record_cache()
    _seed_calibration(tmp_db, "dry_run", correct=False, question="Will Y?")
    live2 = memory.get_track_record(mode="live")
    paper2 = memory.get_track_record(mode="paper")
    assert live2["total"] == 1   # unchanged
    assert paper2["total"] == 1  # now sees the new paper row


def test_get_track_record_invalid_mode_raises():
    with pytest.raises(ValueError):
        memory.get_track_record(mode="bogus")


# --- export_status: performance/track_record are live-only, *_paper separate -

def test_export_status_performance_is_live_only(tmp_db, monkeypatch):
    from locus.core import positions, export_status

    monkeypatch.setattr(config, "WATCH_START_TIME", None)
    t_live = _live(tmp_db, "m1", "YES", 0.50)
    positions.close_on_resolution(t_live, 1.0)  # +25
    t_paper = _paper(tmp_db, "m2", "YES", 0.50)
    positions.close_on_resolution(t_paper, 0.0)  # -25

    status = export_status.export_status()

    assert status["performance"]["mode"] == "live"
    assert status["performance"]["realized_pnl_usd"] == pytest.approx(25.0)
    assert status["performance_paper"]["mode"] == "paper"
    assert status["performance_paper"]["realized_pnl_usd"] == pytest.approx(-25.0)
    # Never silently merged.
    assert status["performance"]["realized_pnl_usd"] != \
        status["performance"]["realized_pnl_usd"] + status["performance_paper"]["realized_pnl_usd"]


def test_export_status_track_record_is_live_only(tmp_db, monkeypatch):
    from locus.core import export_status

    monkeypatch.setattr(config, "WATCH_START_TIME", None)
    _seed_calibration(tmp_db, "executed", correct=True)
    _seed_calibration(tmp_db, "dry_run", correct=False)

    status = export_status.export_status()

    assert status["track_record"]["mode"] == "live"
    assert status["track_record"]["total"] == 1
    assert status["track_record"]["accuracy"] == pytest.approx(100.0)
    assert status["track_record_paper"]["mode"] == "paper"
    assert status["track_record_paper"]["total"] == 1
    assert status["track_record_paper"]["accuracy"] == pytest.approx(0.0)
