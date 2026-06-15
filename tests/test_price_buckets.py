"""Price-bucket accuracy analysis: bucket assignment + boundaries, the
directional-PnL helper, color thresholds relative to overall accuracy, and the
end-to-end aggregation (with its calibration-driven cache)."""
import pytest

from locus.memory import calibrator
from locus.memory.calibrator import (
    bucket_for_price,
    bucket_color,
    directional_pnl_pct,
    get_accuracy_by_price_bucket,
    invalidate_price_bucket_cache,
)


@pytest.fixture(autouse=True)
def _clean_bucket_cache():
    """The price-bucket cache is module-global; reset it around every test so
    neither these tests nor others see a stale cross-test result."""
    invalidate_price_bucket_cache()
    yield
    invalidate_price_bucket_cache()


# --- bucket assignment + boundaries ------------------------------------------

def test_bucket_for_price_assignment():
    assert bucket_for_price(0.07) == "very_low"
    assert bucket_for_price(0.20) == "low"
    assert bucket_for_price(0.40) == "mid"
    assert bucket_for_price(0.60) == "fair"
    assert bucket_for_price(0.78) == "high"
    assert bucket_for_price(0.92) == "extreme"


def test_bucket_for_price_boundaries_are_lower_inclusive():
    # Each boundary belongs to the higher bucket (lower bound inclusive).
    assert bucket_for_price(0.00) == "very_low"
    assert bucket_for_price(0.15) == "low"
    assert bucket_for_price(0.30) == "mid"
    assert bucket_for_price(0.50) == "fair"
    assert bucket_for_price(0.70) == "high"
    assert bucket_for_price(0.85) == "extreme"
    # The very top is inclusive; just under a boundary stays in the lower bucket.
    assert bucket_for_price(1.00) == "extreme"
    assert bucket_for_price(0.1499) == "very_low"
    assert bucket_for_price(0.8499) == "high"


def test_bucket_for_price_out_of_range():
    assert bucket_for_price(-0.01) is None
    assert bucket_for_price(1.01) is None
    assert bucket_for_price(None) is None


# --- directional PnL ---------------------------------------------------------

def test_directional_pnl_pct():
    # Bullish: bought YES at 0.50, marked at 0.60 -> +20%.
    assert directional_pnl_pct("bullish", 0.50, 0.60) == pytest.approx(20.0)
    # Bullish that went the wrong way is negative.
    assert directional_pnl_pct("bullish", 0.50, 0.40) == pytest.approx(-20.0)
    # Bearish: bought NO at 0.50, YES fell to 0.40 -> +20%.
    assert directional_pnl_pct("bearish", 0.50, 0.40) == pytest.approx(20.0)
    assert directional_pnl_pct("bearish", 0.80, 0.90) == pytest.approx(-50.0)
    # Degenerate entry prices yield 0, not a divide-by-zero.
    assert directional_pnl_pct("bullish", 0.0, 0.5) == 0.0
    assert directional_pnl_pct("bearish", 1.0, 0.5) == 0.0


# --- color thresholds relative to overall accuracy ---------------------------

def test_bucket_color_thresholds():
    overall = 50.0
    # >= overall + 5 -> green (boundary inclusive).
    assert bucket_color(55.0, overall) == "green"
    assert bucket_color(60.0, overall) == "green"
    # <= overall - 5 -> red (boundary inclusive).
    assert bucket_color(45.0, overall) == "red"
    assert bucket_color(30.0, overall) == "red"
    # Strictly within the band -> yellow.
    assert bucket_color(54.9, overall) == "yellow"
    assert bucket_color(45.1, overall) == "yellow"
    assert bucket_color(50.0, overall) == "yellow"


# --- end-to-end aggregation --------------------------------------------------

def _add_calibration(tmp_db, trade_id, direction, entry, exit_, correct):
    conn = tmp_db._conn()
    conn.execute(
        """INSERT INTO calibration
           (trade_id, classification, materiality, entry_price, exit_price, correct, resolved_at)
           VALUES (?, ?, 0.5, ?, ?, ?, '2026-06-16T00:00:00')""",
        (trade_id, direction, entry, exit_, correct),
    )
    conn.commit()
    conn.close()


def _add_grade(tmp_db, cls_id, direction, entry, after, correct):
    conn = tmp_db._conn()
    conn.execute(
        """INSERT INTO classification_grades
           (classification_id, direction, materiality, entry_price, price_after, horizon_hours, correct, resolved_at)
           VALUES (?, ?, 0.5, ?, ?, 6, ?, '2026-06-16T00:00:00')""",
        (cls_id, direction, entry, after, correct),
    )
    conn.commit()
    conn.close()


def test_get_accuracy_by_price_bucket_aggregates_both_tables(tmp_db):
    invalidate_price_bucket_cache()
    # very_low bucket: two calls, one correct -> 50% accuracy.
    _add_calibration(tmp_db, 1, "bullish", 0.05, 0.10, 1)   # +100% pnl, correct
    _add_grade(tmp_db, 1, "bullish", 0.10, 0.08, 0)         # -20% pnl, wrong
    # high bucket: one correct bearish call from the grades table.
    _add_grade(tmp_db, 2, "bearish", 0.80, 0.70, 1)         # NO bet, +50% pnl

    result = get_accuracy_by_price_bucket(use_cache=False)
    assert result["overall_total"] == 3
    # 2 of 3 correct overall.
    assert result["overall_accuracy"] == pytest.approx(66.7, abs=0.1)

    by_name = {b["name"]: b for b in result["buckets"]}
    vlow = by_name["very_low"]
    assert vlow["total"] == 2 and vlow["correct"] == 1
    assert vlow["accuracy_pct"] == pytest.approx(50.0)
    # avg pnl = mean(+100, -20) = +40%.
    assert vlow["avg_pnl_pct"] == pytest.approx(40.0)
    # 50% is well below overall 66.7% -> red.
    assert vlow["color"] == "red"

    high = by_name["high"]
    assert high["total"] == 1 and high["accuracy_pct"] == pytest.approx(100.0)
    assert high["color"] == "green"

    # Untouched buckets report zeros with the neutral "none" color.
    assert by_name["mid"]["total"] == 0
    assert by_name["mid"]["color"] == "none"
    assert by_name["mid"]["range"] == "0.30-0.50"


def test_price_bucket_cache_holds_until_invalidated(tmp_db):
    invalidate_price_bucket_cache()
    _add_calibration(tmp_db, 1, "bullish", 0.40, 0.50, 1)
    first = get_accuracy_by_price_bucket()  # populates the cache
    assert first["overall_total"] == 1

    # New graded row, but the cache is returned unchanged...
    _add_calibration(tmp_db, 2, "bullish", 0.40, 0.50, 1)
    assert get_accuracy_by_price_bucket()["overall_total"] == 1

    # ...until a calibration run invalidates it (as check_resolutions does).
    invalidate_price_bucket_cache()
    assert get_accuracy_by_price_bucket()["overall_total"] == 2
