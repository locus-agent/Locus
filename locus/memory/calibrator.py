"""
Calibration engine — tracks classification accuracy over time.
Determines if the system's classifications actually predict market movements.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from locus import config
from locus.memory import logger
from locus import memory
from locus.core import positions
from locus.markets import gamma

log = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"


@dataclass
class CalibrationReport:
    total: int
    accuracy: float
    by_source: dict[str, float]
    by_classification: dict[str, float]
    recommendation: str


# --- Price-bucket accuracy analysis ------------------------------------------
# Buckets graded classifications by their entry (YES) price to see where the
# directional calls actually pay off. Lower bound inclusive, upper bound
# exclusive, except the top bucket which includes a 1.00 price.
PRICE_BUCKETS: list[tuple[str, float, float]] = [
    ("very_low", 0.00, 0.15),
    ("low", 0.15, 0.30),
    ("mid", 0.30, 0.50),
    ("fair", 0.50, 0.70),
    ("high", 0.70, 0.85),
    ("extreme", 0.85, 1.00),
]

# Accuracy this many points above/below the overall accuracy flips a bucket's
# color (green = clearly better, red = clearly worse, yellow = in line).
BUCKET_COLOR_MARGIN = 5.0

# Cached price-bucket analysis. This scans every graded row, so it is computed
# once and reused across the (30s) export cycles; the calibrator invalidates it
# whenever it grades new rows. None until first computed.
_price_bucket_cache: dict | None = None


def invalidate_price_bucket_cache() -> None:
    """Drop the cached price-bucket analysis so the next call recomputes it.
    Called when the calibrator resolves/grades new rows."""
    global _price_bucket_cache
    _price_bucket_cache = None


def bucket_for_price(price: float) -> str | None:
    """Bucket name for a YES price, or None if the price is outside [0, 1].
    Lower bound inclusive, upper exclusive; the top bucket includes 1.00."""
    if price is None or price < 0.0 or price > 1.0:
        return None
    for name, lo, hi in PRICE_BUCKETS:
        if lo <= price < hi:
            return name
    return PRICE_BUCKETS[-1][0]  # price == 1.00 lands in the top bucket


def directional_pnl_pct(direction: str, entry_price: float, exit_price: float) -> float:
    """Percent return of a $1 bet placed in the predicted direction at entry,
    marked at exit_price. Bullish buys YES at entry_price; bearish buys NO at
    (1 - entry_price). Degenerate 0/1 entry prices yield 0."""
    if direction == "bullish":
        return (exit_price - entry_price) / entry_price * 100 if entry_price > 0 else 0.0
    if direction == "bearish":
        return (entry_price - exit_price) / (1 - entry_price) * 100 if entry_price < 1 else 0.0
    return 0.0


def bucket_color(accuracy_pct: float, overall_accuracy: float) -> str:
    """Color a bucket's accuracy relative to the overall accuracy:
    green >= overall + margin, red <= overall - margin, yellow in between."""
    if accuracy_pct >= overall_accuracy + BUCKET_COLOR_MARGIN:
        return "green"
    if accuracy_pct <= overall_accuracy - BUCKET_COLOR_MARGIN:
        return "red"
    return "yellow"


def get_accuracy_by_price_bucket(use_cache: bool = True) -> dict:
    """Accuracy and average directional PnL by entry-price bucket.

    Returns {overall_accuracy, overall_total, buckets: [...]} where each bucket
    carries name, range, total, correct, accuracy_pct, avg_pnl_pct, and a color
    (relative to overall_accuracy). Empty buckets report zeros with color
    "none". Cached across cycles; pass use_cache=False to force a recompute."""
    global _price_bucket_cache
    if use_cache and _price_bucket_cache is not None:
        return _price_bucket_cache

    rows = logger.get_graded_rows_with_prices()
    overall_total = len(rows)
    overall_correct = sum(r["correct"] for r in rows)
    overall_accuracy = round(overall_correct / overall_total * 100, 1) if overall_total else 0.0

    grouped: dict[str, list[dict]] = {name: [] for name, _, _ in PRICE_BUCKETS}
    for r in rows:
        name = bucket_for_price(r["entry_price"])
        if name is not None:
            grouped[name].append(r)

    buckets = []
    for name, lo, hi in PRICE_BUCKETS:
        group = grouped[name]
        total = len(group)
        correct = sum(g["correct"] for g in group)
        accuracy_pct = round(correct / total * 100, 1) if total else 0.0
        avg_pnl_pct = (
            round(
                sum(directional_pnl_pct(g["direction"], g["entry_price"], g["exit_price"])
                    for g in group) / total,
                2,
            )
            if total else 0.0
        )
        buckets.append({
            "name": name,
            "range": f"{lo:.2f}-{hi:.2f}",
            "total": total,
            "correct": correct,
            "accuracy_pct": accuracy_pct,
            "avg_pnl_pct": avg_pnl_pct,
            "color": bucket_color(accuracy_pct, overall_accuracy) if total else "none",
        })

    result = {
        "overall_accuracy": overall_accuracy,
        "overall_total": overall_total,
        "buckets": buckets,
    }
    _price_bucket_cache = result
    return result


def grade_direction(entry_price: float, exit_price: float) -> str:
    """The market's actual direction between trade entry and resolution."""
    if exit_price > entry_price:
        return "bullish"
    if exit_price < entry_price:
        return "bearish"
    return "neutral"


def check_resolutions():
    """
    Check if any open trades have resolved. Update calibration table.
    Queries Gamma API for market resolution status. Trades already graded
    (present in the calibration table) are skipped, so each trade is
    resolved — and its lesson generated — exactly once.
    """
    trades = logger.get_recent_trades(limit=2000, unresolved_only=True)
    already_calibrated = logger.get_calibrated_trade_ids()
    unresolved = [
        t for t in trades
        if t.get("classification")
        and t.get("status") in ("dry_run", "executed")
        and t["id"] not in already_calibrated
    ]

    if not unresolved:
        return 0

    resolved_count = 0
    for trade in unresolved:
        market_id = trade["market_id"]

        try:
            # NB: the param is condition_ids (plural) — Gamma silently ignores
            # unknown params and returns the default market list, so the
            # singular form made this loop compare trades to random markets.
            resp = httpx.get(
                f"{GAMMA_API}/markets",
                params={"condition_ids": market_id},
                timeout=10,
            )
            data = resp.json()
            items = data if isinstance(data, list) else data.get("data", [])

            if not items:
                continue

            market_data = items[0]
            if not market_data.get("closed", False):
                continue

            # Market resolved — determine direction
            outcome_prices = market_data.get("outcomePrices", "")
            if isinstance(outcome_prices, str):
                import json
                try:
                    prices = json.loads(outcome_prices)
                except Exception:
                    continue
            else:
                prices = outcome_prices

            if not prices or len(prices) < 2:
                continue

            exit_price = float(prices[0])
            entry_price = trade["market_price"]
            actual_direction = grade_direction(entry_price, exit_price)

            classification = trade.get("classification", "neutral")
            correct = classification == actual_direction

            logger.log_calibration(
                trade_id=trade["id"],
                classification=classification,
                materiality=trade.get("materiality", 0),
                entry_price=entry_price,
                exit_price=exit_price,
                actual_direction=actual_direction,
                correct=correct,
                resolved_at=datetime.now(timezone.utc).isoformat(),
            )

            if not correct:
                memory.record_lesson(trade, actual_direction, entry_price, exit_price)

            positions.close_on_resolution(trade["id"], exit_price)
            resolved_count += 1

        except Exception as e:
            log.debug(f"[calibrator] Error checking {market_id}: {e}")
            continue

    if resolved_count:
        memory.invalidate_track_record_cache()
        invalidate_price_bucket_cache()
        log.info(f"[calibrator] Resolved {resolved_count} trades")
    return resolved_count


CLOB_API = "https://clob.polymarket.com"


def grade_move(direction: str, entry_price: float, price_after: float, threshold: float) -> bool:
    """A directional call is correct if the price moved the predicted way by
    more than `threshold` over the horizon."""
    delta = price_after - entry_price
    if direction == "bullish":
        return delta > threshold
    if direction == "bearish":
        return delta < -threshold
    return False


def _fetch_price_history(token_id: str, start_ts: int, end_ts: int) -> list[dict]:
    resp = httpx.get(
        f"{CLOB_API}/prices-history",
        params={"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": 60},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("history", [])


def _price_at_or_after(history: list[dict], ts: int) -> float | None:
    """First recorded price at/after ts, else the last one before it."""
    for point in history:
        if point["t"] >= ts:
            return float(point["p"])
    return float(history[-1]["p"]) if history else None


def grade_classifications(max_tokens_per_run: int = 50) -> int:
    """Grade directional non-traded classifications against the market's
    price move CALIBRATION_HORIZON_HOURS later (CLOB price history).

    This is what gives the track record a real sample size: trades are a
    handful per day, directional classifications are hundreds. One history
    fetch per market token covers every row in that market. Rows whose
    history is unavailable are stored with correct=NULL so they are not
    retried forever (and excluded from accuracy stats).
    """
    horizon_h = config.CALIBRATION_HORIZON_HOURS
    threshold = config.CALIBRATION_MOVE_THRESHOLD
    rows = logger.get_ungraded_directional_classifications(min_age_hours=horizon_h)
    if not rows:
        return 0

    by_token: dict[str, list[dict]] = {}
    for r in rows:
        by_token.setdefault(r["yes_token_id"], []).append(r)

    graded = 0
    now_iso = datetime.now(timezone.utc).isoformat()
    for token_id, group in list(by_token.items())[:max_tokens_per_run]:
        created = [
            datetime.fromisoformat(r["created_at"].replace(" ", "T")).replace(tzinfo=timezone.utc)
            for r in group
        ]
        start_ts = int(min(created).timestamp())
        end_ts = int(max(created).timestamp() + (horizon_h + 2) * 3600)

        try:
            history = _fetch_price_history(token_id, start_ts, end_ts)
        except Exception as e:
            log.debug(f"[calibrator] History fetch failed for {token_id[:16]}: {e}")
            continue  # transient: retry next run

        for r, created_at in zip(group, created):
            target_ts = int(created_at.timestamp() + horizon_h * 3600)
            price_after = _price_at_or_after(history, target_ts)
            correct = (
                grade_move(r["direction"], r["yes_price"], price_after, threshold)
                if price_after is not None else None
            )
            logger.log_classification_grade(
                classification_id=r["id"],
                direction=r["direction"],
                materiality=r["materiality"],
                entry_price=r["yes_price"],
                price_after=price_after,
                horizon_hours=horizon_h,
                correct=correct,
                resolved_at=now_iso,
            )
            if correct is not None:
                graded += 1
        time.sleep(0.1)

    if graded:
        memory.invalidate_track_record_cache()
        invalidate_price_bucket_cache()
        log.info(f"[calibrator] Graded {graded} non-traded classifications")
    return graded


# Minimum materiality for a declined classification to count as a candidate
# missed opportunity (a directional read weak enough below this wasn't a real
# signal we "missed").
MISSED_MIN_MATERIALITY = 0.35


def check_missed_opportunities() -> int:
    """Look back at directional classifications we declined to trade and flag
    the ones where the market moved our way anyway, as missed opportunities.

    Pulls the top candidates (by materiality), batch-fetches their current
    prices in one Gamma call, drops resolved markets, and for any whose price
    moved at least MISSED_OPPORTUNITY_THRESHOLD in the predicted direction
    (relative to entry) logs a lesson nudging the materiality bar lower for that
    category. Returns the number of lessons logged."""
    if not config.MISSED_OPPORTUNITY_ENABLED:
        return 0

    rows = logger.get_missed_opportunity_candidates(
        min_materiality=MISSED_MIN_MATERIALITY,
        min_entry_price=config.MISSED_MIN_ENTRY_PRICE,
        limit=50,
    )
    if not rows:
        return 0

    # One batched price fetch covers every candidate (not N single-market calls).
    markets = gamma.fetch_markets_by_condition_ids([r["condition_id"] for r in rows])

    logged = 0
    for r in rows:
        market = markets.get(r["condition_id"])
        # Missing from Gamma or resolved -> can't (or needn't) grade the move.
        if not market or market.get("closed"):
            continue
        current = market.get("yes_price")
        entry = r["yes_price"]
        if current is None or entry is None or entry <= 0:
            continue

        direction = r["direction"]
        if direction == "bullish":
            pct = (current - entry) / entry
        else:  # bearish — a profitable NO move is the price falling
            pct = (entry - current) / entry

        if pct < config.MISSED_OPPORTUNITY_THRESHOLD:
            continue

        category = gamma._infer_category(r["market_question"], [])
        lesson = (
            f"Missed strong {direction} signal (mat={r['materiality']:.2f}) on "
            f"'{r['market_question'][:50]}' — price moved +{pct * 100:.0f}%. "
            f"Consider lowering min_materiality for {category} category."
        )
        logger.log_lesson(
            trade_id=None,
            market_question=r["market_question"],
            classification=direction,
            actual_direction=direction,
            lesson=lesson,
        )
        logged += 1

    if logged:
        log.info(f"[calibrator] Logged {logged} missed-opportunity lessons")
    return logged


def run_cycle() -> dict:
    """One full calibration pass: resolve trades, grade non-traded
    classifications, return a summary. Used by the pipeline's schedule;
    `cli.py calibrate` remains the manual equivalent."""
    resolved = check_resolutions()
    graded = grade_classifications()
    record = memory.get_track_record()
    return {
        "resolved": resolved,
        "graded": graded,
        "total": record["total"],
        "accuracy": record["accuracy"],
    }


def get_report() -> CalibrationReport:
    """Generate a calibration report from stored data."""
    stats = logger.get_calibration_stats()

    if stats["total"] == 0:
        return CalibrationReport(
            total=0,
            accuracy=0.0,
            by_source={},
            by_classification={},
            recommendation="Not enough data — need at least 20 resolved trades for meaningful calibration.",
        )

    accuracy = stats["accuracy"]

    if accuracy >= 65:
        rec = f"Strong signal. {accuracy:.1f}% accuracy suggests real edge. Consider increasing bet sizes cautiously."
    elif accuracy >= 55:
        rec = f"Moderate signal. {accuracy:.1f}% accuracy is above chance but thin. Keep current sizing."
    elif accuracy >= 45:
        rec = f"Weak signal. {accuracy:.1f}% accuracy is near random. Review classification prompt and news sources."
    else:
        rec = f"Negative signal. {accuracy:.1f}% accuracy is below chance. PAUSE trading and investigate."

    return CalibrationReport(
        total=stats["total"],
        accuracy=accuracy,
        by_source=stats["by_source"],
        by_classification=stats["by_classification"],
        recommendation=rec,
    )


if __name__ == "__main__":
    print("Checking resolutions...")
    count = check_resolutions()
    print(f"Resolved: {count}")

    report = get_report()
    print(f"\nCalibration Report:")
    print(f"  Total: {report.total}")
    print(f"  Accuracy: {report.accuracy:.1f}%")
    print(f"  By source: {report.by_source}")
    print(f"  By classification: {report.by_classification}")
    print(f"  Recommendation: {report.recommendation}")
