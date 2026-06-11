"""
Exports a public snapshot of pipeline status to docs/status.json, used by the
GitHub Pages dashboard (docs/index.html). Contains no API keys or private data —
just classification history, accuracy stats, and lessons learned.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
import logger
import memory

log = logging.getLogger(__name__)

REPO_DIR = Path(__file__).parent
STATUS_PATH = REPO_DIR / "docs" / "status.json"

_last_push_at = float("-inf")


def export_status(headlines_last_cycle: int = 0, markets_tracked: int = 0) -> dict:
    """Write a snapshot of current pipeline status to docs/status.json."""
    now = datetime.now(timezone.utc)
    today_start = now.strftime("%Y-%m-%d 00:00:00")
    since_24h = (now - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")

    signals_24h = logger.get_classification_count_since(since_24h, action="signal")
    classifications_24h = logger.get_classification_count_since(since_24h)

    status = {
        "generated_at": now.isoformat(),
        "dry_run": config.DRY_RUN,
        "markets_tracked": markets_tracked,
        "track_record": memory.get_track_record(),
        "headlines_scanned_today": logger.get_news_event_count_since(today_start),
        "headlines_last_cycle": headlines_last_cycle,
        "signals_24h": signals_24h,
        "classifications_24h": classifications_24h,
        "selectivity_pct": (
            round(signals_24h / classifications_24h * 100, 1)
            if classifications_24h else None
        ),
        "pipeline_24h": {
            "news": logger.get_news_event_count_since(since_24h),
            "matched": logger.get_matched_headline_count_since(since_24h),
            "classified": classifications_24h,
            "signals": signals_24h,
            "trades": logger.get_trade_count_since(since_24h),
        },
        "open_positions": [
            {
                "time": t["created_at"],
                "market_question": t["market_question"],
                "side": t["side"],
                "entry_price": t["market_price"],
                "edge": t["edge"],
                "amount_usd": t["amount_usd"],
                "status": t["status"],
            }
            for t in logger.get_recent_trades(limit=10)
        ],
        "recent_classifications": [
            {
                "time": c["created_at"],
                "market_question": c["market_question"],
                "headline": c["headline"],
                "direction": c["direction"],
                "materiality": c["materiality"],
                "edge": c["edge"],
                "action": c["action"],
                "match_source": c["match_source"],
            }
            for c in logger.get_recent_classifications(limit=20)
        ],
        "lessons": [
            {
                "time": l["created_at"],
                "market_question": l["market_question"],
                "classification": l["classification"],
                "actual_direction": l["actual_direction"],
                "lesson": l["lesson"],
            }
            for l in logger.get_recent_lessons(limit=5)
        ],
    }

    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2))

    _auto_push_status()
    return status


def _auto_push_status():
    """Commit and push docs/status.json if it changed, at most once per
    AUTO_PUSH_MIN_INTERVAL_SECONDS. Never raises — logs a warning instead."""
    global _last_push_at

    if not config.AUTO_PUSH_STATUS:
        return

    now = time.monotonic()
    if now - _last_push_at < config.AUTO_PUSH_MIN_INTERVAL_SECONDS:
        return

    try:
        diff = subprocess.run(
            ["git", "diff", "--quiet", "--", "docs/status.json"],
            cwd=REPO_DIR, capture_output=True,
        )
        if diff.returncode == 0:
            return  # no changes to commit

        subprocess.run(
            ["git", "add", "docs/status.json"],
            cwd=REPO_DIR, check=True, capture_output=True, timeout=30,
        )
        subprocess.run(
            ["git", "commit", "-m", "update dashboard data"],
            cwd=REPO_DIR, check=True, capture_output=True, timeout=30,
        )
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=REPO_DIR, check=True, capture_output=True, timeout=30,
        )
        _last_push_at = now
    except Exception as e:
        log.warning(f"[export_status] Auto-push of dashboard data failed: {e}")


if __name__ == "__main__":
    export_status()
    print(f"Wrote {STATUS_PATH}")
