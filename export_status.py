"""
Exports a public snapshot of pipeline status to docs/status.json, used by the
GitHub Pages dashboard (docs/index.html). Contains no API keys or private data —
just classification history, accuracy stats, and lessons learned.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
import logger
import memory

STATUS_PATH = Path(__file__).parent / "docs" / "status.json"


def export_status(headlines_last_cycle: int = 0) -> dict:
    """Write a snapshot of current pipeline status to docs/status.json."""
    now = datetime.now(timezone.utc)
    today_start = now.strftime("%Y-%m-%d 00:00:00")
    since_24h = (now - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")

    status = {
        "generated_at": now.isoformat(),
        "dry_run": config.DRY_RUN,
        "track_record": memory.get_track_record(),
        "headlines_scanned_today": logger.get_news_event_count_since(today_start),
        "headlines_last_cycle": headlines_last_cycle,
        "signals_24h": logger.get_classification_count_since(since_24h, action="signal"),
        "recent_classifications": [
            {
                "time": c["created_at"],
                "market_question": c["market_question"],
                "headline": c["headline"],
                "direction": c["direction"],
                "materiality": c["materiality"],
                "edge": c["edge"],
                "action": c["action"],
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
    return status


if __name__ == "__main__":
    export_status()
    print(f"Wrote {STATUS_PATH}")
