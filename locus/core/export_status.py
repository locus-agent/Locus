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

from locus import config
from locus.memory import logger
from locus import memory
from locus.core.performance import compute_performance, compute_live_readiness
from locus.core import positions

log = logging.getLogger(__name__)

REPO_DIR = config.PROJECT_ROOT
STATUS_PATH = REPO_DIR / "docs" / "status.json"
JOURNAL_PATH = REPO_DIR / "docs" / "journal.json"
DECISIONS_PATH = REPO_DIR / "docs" / "exit_decisions.json"
# All files the auto-pusher is allowed to commit.
PUSH_PATHS = ["docs/status.json", "docs/journal.json", "docs/exit_decisions.json"]

_last_push_at = float("-inf")
# Max row ids already exported — archives rewrite only when new rows exist,
# so the 30s cycle stays cheap. No generated_at inside the archives: content
# is byte-identical unless rows changed, keeping auto-push commits honest.
_archive_state: dict[str, int | None] = {"journal": None, "decisions": None}


def _export_archives() -> None:
    """Write full-history journal.json / exit_decisions.json when new rows exist."""
    conn = logger._conn()
    journal_max = conn.execute("SELECT COALESCE(MAX(id), 0) FROM journal").fetchone()[0]
    decisions_max = conn.execute("SELECT COALESCE(MAX(id), 0) FROM exit_decisions").fetchone()[0]
    conn.close()

    if journal_max != _archive_state["journal"]:
        entries = logger.get_journal_entries(limit=100000)
        JOURNAL_PATH.write_text(json.dumps({"entries": entries}, indent=1))
        _archive_state["journal"] = journal_max

    if decisions_max != _archive_state["decisions"]:
        decisions = [
            {
                "time": d["created_at"],
                "market_question": d["market_question"],
                "side": d["side"],
                "trigger": d["trigger"],
                "decision": d["decision"],
                "pnl_pct": d["pnl_pct"],
                "yes_price": d["yes_price"],
                "reasoning": d["reasoning"],
            }
            for d in positions.get_recent_exit_decisions(limit=100000)
        ]
        DECISIONS_PATH.write_text(json.dumps({"decisions": decisions}, indent=1))
        _archive_state["decisions"] = decisions_max


def export_status(headlines_last_cycle: int = 0, markets_tracked: int = 0, classify_error_streak: int = 0, current_prices: dict | None = None) -> dict:
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
        "classify_error_streak": classify_error_streak,
        "track_record": memory.get_track_record(),
        "headlines_scanned_today": logger.get_news_event_count_since(today_start),
        "headlines_last_cycle": headlines_last_cycle,
        "signals_24h": signals_24h,
        "classifications_24h": classifications_24h,
        "selectivity_pct": (
            round(signals_24h / classifications_24h * 100, 1)
            if classifications_24h else None
        ),
        "gates_24h": {
            "stale": logger.get_classification_count_since(since_24h, action="stale"),
            "capped": logger.get_classification_count_since(since_24h, action="capped"),
        },
        "pipeline_24h": {
            "news": logger.get_news_event_count_since(since_24h),
            "matched": logger.get_matched_headline_count_since(since_24h),
            "classified": classifications_24h,
            "signals": signals_24h,
            "trades": logger.get_trade_count_since(since_24h),
        },
        "performance": compute_performance(current_prices),
        "live_readiness": compute_live_readiness(),
        "open_positions": [
            {
                "time": p["opened_at"],
                "market_question": p["market_question"],
                "slug": p["slug"],
                "side": p["side"],
                "entry_price": p["entry_yes_price"],
                "current_price": p["current_yes_price"],
                "pnl_pct": p["unrealized_pnl_pct"],
                "amount_usd": p["amount_usd"],
            }
            for p in positions.get_open_positions()[:10]
        ],
        "closed_positions": [
            {
                "time": p["closed_at"],
                "market_question": p["market_question"],
                "slug": p["slug"],
                "side": p["side"],
                "entry_price": p["entry_yes_price"],
                "exit_price": p["exit_yes_price"],
                "realized_pnl_usd": p["realized_pnl_usd"],
                "exit_reason": p["exit_reason"],
            }
            for p in positions.get_closed_positions(limit=10)
        ],
        "exit_decisions": [
            {
                "time": d["created_at"],
                "market_question": d["market_question"],
                "side": d["side"],
                "trigger": d["trigger"],
                "decision": d["decision"],
                "reasoning": d["reasoning"],
                "pnl_pct": d["pnl_pct"],
            }
            for d in positions.get_recent_exit_decisions(limit=5)
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
        "journal": [
            {"date": j["date"], "entry": j["entry"]}
            for j in logger.get_journal_entries(limit=3)
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

    try:
        _export_archives()
    except Exception as e:
        log.warning(f"[export_status] Archive export failed: {e}")

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
        paths = [p for p in PUSH_PATHS if (REPO_DIR / p).exists()]
        if not paths:
            return
        # status --porcelain (unlike diff) also sees brand-new untracked
        # archive files, so the first journal.json gets committed too.
        changed = subprocess.run(
            ["git", "status", "--porcelain", "--", *paths],
            cwd=REPO_DIR, capture_output=True, text=True,
        )
        if not changed.stdout.strip():
            return  # no changes to commit

        # Add only the dashboard data files, then commit pathspec-limited:
        # a bare `git commit` commits the whole index, sweeping in anything
        # a human (or agent) had staged when the 30s cycle fired. Unrelated
        # staged work stays staged.
        subprocess.run(
            ["git", "add", "--", *paths],
            cwd=REPO_DIR, check=True, capture_output=True, timeout=30,
        )
        subprocess.run(
            ["git", "commit", "-m", "update dashboard data", "--", *paths],
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
