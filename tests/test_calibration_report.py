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
                                   materiality, classification, confidence,
                                   created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'executed', ?, ?, ?, ?)""",
            (i, f"cond{i}", question, 0.7, 0.5, 0.2, r.get("side", "YES"),
             r.get("amount_usd", 20.0), r.get("materiality"), r.get("direction"),
             r.get("confidence"),
             r.get("trade_created_at", "2026-06-15 12:00:00")),
        )
        status = r.get("status", "closed_manual")
        # Every real close path stamps closed_at; only open rows leave it NULL.
        closed_at = r.get(
            "closed_at", None if status == "open" else "2026-06-16 12:00:00"
        )
        conn.execute(
            """INSERT INTO positions (trade_id, condition_id, market_question, side,
                                      entry_yes_price, amount_usd, status,
                                      realized_pnl_usd, exit_reason, category,
                                      actual_cost_usd, opened_at, end_date, closed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (i, f"cond{i}", question, r.get("side", "YES"),
             r.get("entry_yes_price", 0.5), r.get("amount_usd", 20.0),
             status, r.get("realized_pnl_usd"),
             r.get("exit_reason"), r.get("category"), r.get("actual_cost_usd"),
             r.get("opened_at", "2026-06-15 12:00:00"), r.get("end_date"),
             closed_at),
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


# --- Brier score (probability calibration) ------------------------------------

def _seed_prompt_versions(tmp_db, versions):
    """Insert (version, created_at) rows into prompt_versions."""
    conn = tmp_db._conn()
    for v, created in versions:
        conn.execute(
            "INSERT INTO prompt_versions (version, prompt_text, created_at) "
            "VALUES (?, ?, ?)",
            (v, f"prompt v{v}", created),
        )
    conn.commit()
    conn.close()


def test_brier_hand_computed_from_confidence(tmp_db):
    # Stored confidence IS the prediction. Hand-computed:
    #   win  predicted 0.8 -> (0.8-1)^2 = 0.04
    #   loss predicted 0.6 -> (0.6-0)^2 = 0.36
    #   win  predicted 0.7 -> (0.7-1)^2 = 0.09
    # Brier = (0.04 + 0.36 + 0.09) / 3 = 0.16333...
    _seed(tmp_db, [
        {"realized_pnl_usd": 5.0, "confidence": 0.8},
        {"realized_pnl_usd": -3.0, "confidence": 0.6},
        {"realized_pnl_usd": 2.0, "confidence": 0.7},
        {"realized_pnl_usd": 0.0, "confidence": 0.9},   # break-even: unknowable
        {"realized_pnl_usd": None, "confidence": 0.9},  # never realized: unknowable
    ])
    rep = performance.calibration_report()
    b = rep["brier"]
    assert b["overall"]["n"] == 3
    assert b["overall"]["brier"] == pytest.approx(0.1633, abs=1e-4)
    assert b["proxy_counts"] == {"confidence": 3, "entry_implied": 0}
    assert b["excluded"] == 2


def test_brier_entry_implied_fallback_documents_proxy(tmp_db):
    # No confidence stored -> the entry-implied fallback: entry price for a
    # YES-side, 1 - price for a NO-side. Hand-computed:
    #   YES @ 0.30, win  -> (0.30-1)^2 = 0.49
    #   NO  @ 0.60, loss -> predicted 0.40 -> (0.40-0)^2 = 0.16
    # Brier = (0.49 + 0.16) / 2 = 0.325
    _seed(tmp_db, [
        {"realized_pnl_usd": 4.0, "side": "YES", "entry_yes_price": 0.30},
        {"realized_pnl_usd": -2.0, "side": "NO", "entry_yes_price": 0.60},
    ])
    rep = performance.calibration_report()
    b = rep["brier"]
    assert b["overall"]["brier"] == pytest.approx(0.325)
    assert b["proxy_counts"] == {"confidence": 0, "entry_implied": 2}
    # the rendered report says which proxy was used and prints the reference
    text = performance.format_calibration_report(rep)
    assert "BRIER SCORE" in text
    assert "0.25 = random guessing" in text
    assert "entry-implied" in text
    assert "win-probability" in text


def test_brier_by_category_with_small_n_flag(tmp_db):
    _seed(tmp_db, [
        {"realized_pnl_usd": 1.0, "confidence": 1.0, "category": "crypto"},   # 0.00
        {"realized_pnl_usd": -1.0, "confidence": 1.0, "category": "crypto"},  # 1.00
        {"realized_pnl_usd": 1.0, "confidence": 0.5, "category": "politics"}, # 0.25
    ])
    rep = performance.calibration_report()
    by = {e["label"]: e for e in rep["brier"]["by_category"]}
    assert by["crypto"]["n"] == 2
    assert by["crypto"]["brier"] == pytest.approx(0.5)
    assert by["politics"]["brier"] == pytest.approx(0.25)
    # small-n honesty: every bucket here is tiny and flagged in the rendering
    text = performance.format_calibration_report(rep)
    assert "(small n)" in text


def test_brier_by_prompt_version_time_linkage(tmp_db):
    # No positions->prompt_versions FK exists anywhere in the schema, so the
    # version is inferred by time: the newest prompt_versions row created
    # at/before each trade's created_at (what classifier.get_active_prompt had
    # loaded); earlier trades belong to 'v0 (baseline)'.
    _seed_prompt_versions(tmp_db, [
        (1, "2026-06-10 00:00:00"),
        (2, "2026-06-20 00:00:00"),
    ])
    _seed(tmp_db, [
        {"realized_pnl_usd": 1.0, "confidence": 0.9,
         "trade_created_at": "2026-06-01 00:00:00"},   # before v1 -> v0: 0.01
        {"realized_pnl_usd": 1.0, "confidence": 0.8,
         "trade_created_at": "2026-06-12 00:00:00"},   # v1: 0.04
        {"realized_pnl_usd": -1.0, "confidence": 0.8,
         "trade_created_at": "2026-06-15 00:00:00"},   # v1: 0.64
        {"realized_pnl_usd": 1.0, "confidence": 0.6,
         "trade_created_at": "2026-06-25 00:00:00"},   # v2: 0.16
    ])
    rep = performance.calibration_report()
    by = {e["label"]: e for e in rep["brier"]["by_prompt_version"]}
    assert by["v0 (baseline)"]["n"] == 1
    assert by["v0 (baseline)"]["brier"] == pytest.approx(0.01)
    assert by["v1"]["n"] == 2
    assert by["v1"]["brier"] == pytest.approx((0.04 + 0.64) / 2)
    assert by["v2"]["brier"] == pytest.approx(0.16)
    # baseline first, then versions ascending
    labels = [e["label"] for e in rep["brier"]["by_prompt_version"]]
    assert labels == ["v0 (baseline)", "v1", "v2"]
    # the rendering is honest that the linkage is inferred, not a real join
    text = performance.format_calibration_report(rep)
    assert "no direct position->prompt_version link" in text


def test_brier_prompt_version_unknown_without_trade_link(tmp_db):
    # A closed position with no linked trade can't be attributed to a version
    # (or a confidence): it lands in the honest 'unknown' bucket with the
    # entry-implied prediction, never guessed.
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
    rep = performance.calibration_report()
    by = {e["label"]: e for e in rep["brier"]["by_prompt_version"]}
    assert by["unknown (no linked trade)"]["n"] == 1
    assert by["unknown (no linked trade)"]["brier"] == pytest.approx((0.4 - 1.0) ** 2)


def test_brier_all_baseline_when_no_prompt_versions(tmp_db):
    # With no evolved prompts, every trade used the hardcoded baseline — the
    # report says so instead of inventing a version.
    _seed(tmp_db, [{"realized_pnl_usd": 1.0, "confidence": 0.7}])
    rep = performance.calibration_report()
    b = rep["brier"]
    assert b["prompt_versions_known"] is False
    assert [e["label"] for e in b["by_prompt_version"]] == ["v0 (baseline)"]
    text = performance.format_calibration_report(rep)
    assert "No evolved prompt versions exist yet" in text


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
