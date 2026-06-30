"""Read-only calibration_report analytics: groups CLOSED positions (joined to
their trade's classification) by materiality, category, direction, fill quality,
exit reason, entry-price bucket, and time-to-resolution, plus best/worst
leaderboards. Pure reads — it must never modify the DB."""
import pytest

from locus.core import performance


def _seed(tmp_db, rows):
    """Insert (trade, position) pairs straight into the schema. Each row is a
    dict with optional materiality/direction/category/cost/exit_reason/status/
    entry_yes_price/opened_at/end_date/market_question so tests can exercise
    NULLs and odd values without the full open path."""
    conn = tmp_db._conn()
    for i, r in enumerate(rows, start=1):
        question = r.get("market_question", f"Market {i}?")
        conn.execute(
            """INSERT INTO trades (id, market_id, market_question, claude_score,
                                   market_price, edge, side, amount_usd, status,
                                   materiality, classification)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'executed', ?, ?)""",
            (i, f"cond{i}", question, 0.7, 0.5, 0.2, r.get("side", "YES"),
             r.get("amount_usd", 20.0), r.get("materiality"), r.get("direction")),
        )
        conn.execute(
            """INSERT INTO positions (trade_id, condition_id, market_question, side,
                                      entry_yes_price, amount_usd, status,
                                      realized_pnl_usd, exit_reason, category,
                                      actual_cost_usd, opened_at, end_date)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (i, f"cond{i}", question, r.get("side", "YES"),
             r.get("entry_yes_price", 0.5), r.get("amount_usd", 20.0),
             r.get("status", "closed_manual"), r.get("realized_pnl_usd"),
             r.get("exit_reason"), r.get("category"), r.get("actual_cost_usd"),
             r.get("opened_at", "2026-06-15 12:00:00"), r.get("end_date")),
        )
    conn.commit()
    conn.close()


def test_summary_counts_wins_and_flags_small_sample(tmp_db):
    _seed(tmp_db, [
        {"realized_pnl_usd": 5.0, "materiality": 0.45},
        {"realized_pnl_usd": -3.0, "materiality": 0.55},
        {"realized_pnl_usd": 2.0, "materiality": 0.72},
        # an OPEN position must be ignored entirely
        {"realized_pnl_usd": 99.0, "materiality": 0.5, "status": "open"},
    ])
    rep = performance.calibration_report()
    s = rep["summary"]
    assert s["n"] == 3                       # the open row excluded
    assert s["wins"] == 2
    assert s["win_rate"] == pytest.approx(66.7)
    assert s["total_pnl"] == pytest.approx(4.0)
    assert s["small_sample"] is True         # 3 < 25
    assert s["small_sample_threshold"] == 25


def test_materiality_buckets_and_below_floor(tmp_db):
    _seed(tmp_db, [
        {"realized_pnl_usd": 1.0, "materiality": 0.30},   # [0.27-0.40)
        {"realized_pnl_usd": 1.0, "materiality": 0.39},   # [0.27-0.40)
        {"realized_pnl_usd": -1.0, "materiality": 0.65},  # [0.60-0.70)
        {"realized_pnl_usd": 1.0, "materiality": 0.25},   # below floor -> [<0.27]
        {"realized_pnl_usd": 1.0, "materiality": None},   # NULL -> unknown
    ])
    rep = performance.calibration_report()
    by = {e["label"]: e for e in rep["by_materiality"]}
    assert by["[0.27-0.40)"]["n"] == 2
    assert by["[0.27-0.40)"]["win_rate"] == pytest.approx(100.0)
    assert by["[0.60-0.70)"]["n"] == 1
    assert by["[0.60-0.70)"]["win_rate"] == pytest.approx(0.0)
    # sub-floor and NULL materiality are surfaced, never dropped
    assert by["[<0.27]"]["n"] == 1
    assert by["unknown"]["n"] == 1
    # ordering: defined buckets come before the catch-alls
    labels = [e["label"] for e in rep["by_materiality"]]
    assert labels.index("[0.27-0.40)") < labels.index("[<0.27]")


def test_fill_quality_partial_vs_full_vs_unknown(tmp_db):
    _seed(tmp_db, [
        # partial: cost 8 of 20 nominal (< 50%)
        {"realized_pnl_usd": -2.0, "amount_usd": 20.0, "actual_cost_usd": 8.0},
        # full: cost 18 of 20 nominal (>= 50%)
        {"realized_pnl_usd": 4.0, "amount_usd": 20.0, "actual_cost_usd": 18.0},
        {"realized_pnl_usd": 1.0, "amount_usd": 20.0, "actual_cost_usd": 19.0},
        # unknown: no cost recorded (legacy/dry-run)
        {"realized_pnl_usd": 3.0, "amount_usd": 20.0, "actual_cost_usd": None},
    ])
    rep = performance.calibration_report()
    by = {e["label"]: e for e in rep["by_fill_quality"]}
    assert by["partial fill (<50% nominal)"]["n"] == 1
    assert by["partial fill (<50% nominal)"]["win_rate"] == pytest.approx(0.0)
    assert by["full fill (>=50% nominal)"]["n"] == 2
    assert by["full fill (>=50% nominal)"]["win_rate"] == pytest.approx(100.0)
    assert by["unknown (no cost data)"]["n"] == 1


def test_category_direction_exit_reason_and_nulls(tmp_db):
    _seed(tmp_db, [
        {"realized_pnl_usd": 5.0, "category": "crypto", "direction": "bullish",
         "exit_reason": "tp_decision"},
        {"realized_pnl_usd": -1.0, "category": "crypto", "direction": "bearish",
         "exit_reason": "drawdown_decision"},
        # NULL category / direction / exit_reason must bucket as 'unknown', not crash
        {"realized_pnl_usd": 2.0, "category": None, "direction": None,
         "exit_reason": None},
    ])
    rep = performance.calibration_report()

    cat = {e["label"]: e for e in rep["by_category"]}
    assert cat["crypto"]["n"] == 2
    assert cat["unknown"]["n"] == 1

    direction = {e["label"]: e for e in rep["by_direction"]}
    assert direction["bullish"]["n"] == 1
    assert direction["bearish"]["n"] == 1
    assert direction["unknown"]["n"] == 1

    exits = {e["label"]: e for e in rep["by_exit_reason"]}
    assert exits["tp_decision"]["total_pnl"] == pytest.approx(5.0)
    assert exits["unknown"]["n"] == 1

    # the formatter renders without error and shows the small-sample warning
    text = performance.format_calibration_report(rep)
    assert "CALIBRATION REPORT" in text
    assert "SMALL SAMPLE WARNING" in text


def test_entry_price_buckets_flag_longshots(tmp_db):
    _seed(tmp_db, [
        # two cheap longshots (< 0.15), both losers
        {"realized_pnl_usd": -5.0, "entry_yes_price": 0.07},
        {"realized_pnl_usd": -4.0, "entry_yes_price": 0.085},
        {"realized_pnl_usd": 3.0, "entry_yes_price": 0.25},   # [0.15-0.35]
        {"realized_pnl_usd": 1.0, "entry_yes_price": 0.50},   # [0.35-0.65]
        {"realized_pnl_usd": 2.0, "entry_yes_price": 0.90},   # [>0.85]
    ])
    rep = performance.calibration_report()
    by = {e["label"]: e for e in rep["by_entry_price"]}
    assert by["[<0.15 longshot]"]["n"] == 2
    assert by["[<0.15 longshot]"]["win_rate"] == pytest.approx(0.0)
    assert by["[<0.15 longshot]"]["total_pnl"] == pytest.approx(-9.0)
    assert by["[0.15-0.35]"]["n"] == 1
    assert by["[0.35-0.65]"]["n"] == 1
    assert by["[>0.85]"]["n"] == 1
    # buckets are emitted in ascending price order
    labels = [e["label"] for e in rep["by_entry_price"]]
    assert labels.index("[<0.15 longshot]") < labels.index("[>0.85]")


def test_time_to_resolution_with_null_end_date(tmp_db):
    _seed(tmp_db, [
        # opened 2026-06-15 12:00, resolves +12h -> [<24h]
        {"realized_pnl_usd": 1.0, "opened_at": "2026-06-15 12:00:00",
         "end_date": "2026-06-16T00:00:00Z"},
        # +3 days -> [1-7 days]
        {"realized_pnl_usd": 1.0, "opened_at": "2026-06-15 12:00:00",
         "end_date": "2026-06-18T12:00:00Z"},
        # +10 days -> [>7 days]
        {"realized_pnl_usd": -1.0, "opened_at": "2026-06-15 12:00:00",
         "end_date": "2026-06-25T12:00:00Z"},
        # NULL end_date -> unknown, must not crash
        {"realized_pnl_usd": 1.0, "opened_at": "2026-06-15 12:00:00",
         "end_date": None},
    ])
    rep = performance.calibration_report()
    by = {e["label"]: e for e in rep["by_time_to_resolution"]}
    assert by["[<24h]"]["n"] == 1
    assert by["[1-7 days]"]["n"] == 1
    assert by["[>7 days]"]["n"] == 1
    assert by["unknown"]["n"] == 1


def test_best_and_worst_trades_ordering(tmp_db):
    _seed(tmp_db, [
        {"realized_pnl_usd": 10.0, "market_question": "Big winner?",
         "direction": "bullish", "entry_yes_price": 0.4},
        {"realized_pnl_usd": -8.0, "market_question": "Big loser?",
         "direction": "bearish", "entry_yes_price": 0.6},
        {"realized_pnl_usd": 2.0, "market_question": "Small winner?"},
        {"realized_pnl_usd": -1.0, "market_question": "Small loser?"},
    ])
    rep = performance.calibration_report()
    # best is highest-PnL first; worst is lowest-PnL first
    assert rep["best_trades"][0]["market_question"] == "Big winner?"
    assert rep["best_trades"][0]["realized_pnl_usd"] == pytest.approx(10.0)
    assert rep["best_trades"][0]["direction"] == "bullish"
    assert rep["worst_trades"][0]["market_question"] == "Big loser?"
    assert rep["worst_trades"][0]["realized_pnl_usd"] == pytest.approx(-8.0)
    # capped at TOP_TRADES_N
    assert len(rep["best_trades"]) <= performance.TOP_TRADES_N
    # the new sections render in the text output
    text = performance.format_calibration_report(rep)
    assert "ENTRY PRICE BUCKET" in text
    assert "TIME TO RESOLUTION" in text
    assert "TOP 5 BEST TRADES" in text
    assert "TOP 5 WORST TRADES" in text


def test_empty_db_does_not_crash(tmp_db):
    rep = performance.calibration_report()
    assert rep["summary"]["n"] == 0
    assert rep["summary"]["win_rate"] is None
    assert rep["summary"]["small_sample"] is True
    assert rep["best_trades"] == []
    assert rep["worst_trades"] == []
    # every section is empty but present, and the formatter still renders
    text = performance.format_calibration_report(rep)
    assert "CALIBRATION REPORT" in text
    assert "(no data)" in text


def test_report_is_read_only(tmp_db):
    _seed(tmp_db, [{"realized_pnl_usd": 1.0, "materiality": 0.5}])
    conn = tmp_db._conn()
    before = conn.execute("SELECT count(*) FROM positions").fetchone()[0]
    conn.close()

    performance.calibration_report()

    conn = tmp_db._conn()
    after = conn.execute("SELECT count(*) FROM positions").fetchone()[0]
    conn.close()
    assert before == after == 1
