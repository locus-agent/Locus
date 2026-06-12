"""
Calibration engine — tracks classification accuracy over time.
Determines if the system's classifications actually predict market movements.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from locus import config
from locus.memory import logger
from locus import memory

log = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"


@dataclass
class CalibrationReport:
    total: int
    accuracy: float
    by_source: dict[str, float]
    by_classification: dict[str, float]
    recommendation: str


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
    trades = logger.get_recent_trades(limit=100)
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

            resolved_count += 1

        except Exception as e:
            log.debug(f"[calibrator] Error checking {market_id}: {e}")
            continue

    if resolved_count:
        log.info(f"[calibrator] Resolved {resolved_count} trades")
    return resolved_count


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
