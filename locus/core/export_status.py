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
CLASSIFICATIONS_PATH = REPO_DIR / "docs" / "classifications.json"
# All files the auto-pusher is allowed to commit.
PUSH_PATHS = [
    "docs/status.json", "docs/journal.json", "docs/exit_decisions.json",
    "docs/classifications.json",
]

# Rows shown inline on the main dashboard; the rest live in the archive page.
RECENT_CLASSIFICATIONS_LIMIT = 5

_last_push_at = float("-inf")
# Max row ids already exported — archives rewrite only when new rows exist,
# so the 30s cycle stays cheap. No generated_at inside the archives: content
# is byte-identical unless rows changed, keeping auto-push commits honest.
_archive_state: dict[str, int | None] = {"journal": None, "decisions": None, "classifications": None}


def _classification_row(c: dict) -> dict:
    """Public shape of a classification row (shared by inline + archive)."""
    return {
        "time": c["created_at"],
        "market_question": c["market_question"],
        "headline": c["headline"],
        "direction": c["direction"],
        "materiality": c["materiality"],
        "confidence": c["confidence"],
        "edge": c["edge"],
        "action": c["action"],
        "match_source": c["match_source"],
    }


def _export_archives() -> None:
    """Write full-history journal / exit_decisions / classifications archives
    when new rows exist (byte-identical otherwise, so auto-push stays quiet)."""
    conn = logger._conn()
    journal_max = conn.execute("SELECT COALESCE(MAX(id), 0) FROM journal").fetchone()[0]
    decisions_max = conn.execute("SELECT COALESCE(MAX(id), 0) FROM exit_decisions").fetchone()[0]
    classifications_max = conn.execute("SELECT COALESCE(MAX(id), 0) FROM classifications").fetchone()[0]
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

    if classifications_max != _archive_state["classifications"]:
        classifications = [
            _classification_row(c)
            for c in logger.get_recent_classifications(limit=100000)
        ]
        CLASSIFICATIONS_PATH.write_text(
            json.dumps({"classifications": classifications}, indent=1)
        )
        _archive_state["classifications"] = classifications_max


def export_status(headlines_last_cycle: int = 0, markets_tracked: int = 0, classify_error_streak: int = 0, current_prices: dict | None = None, whale_last_check: str | None = None) -> dict:
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
            "correlation_block": logger.get_classification_count_since(since_24h, action="correlation_block"),
            "orderbook_skip": logger.get_classification_count_since(since_24h, action="orderbook_skip"),
            "needs_confirmation": logger.get_classification_count_since(since_24h, action="needs_confirmation"),
            "event_exposure_block": logger.get_classification_count_since(since_24h, action="event_exposure_block"),
            # Opportunities, not blocks: whale-triggered investigations and
            # re-entries into recently closed markets.
            "whale_triggered": logger.get_classification_count_since(since_24h, action="whale_triggered"),
            "reentry_triggered": logger.get_classification_count_since(since_24h, action="reentry_triggered"),
        },
        "watched_markets_count": logger.count_active_watched_markets(),
        "whale": {
            "watched_wallets": len(config.WHALE_WALLETS),
            "triggered_24h": logger.get_classification_count_since(since_24h, action="whale_triggered"),
            "last_check": whale_last_check,
        },
        # Signals in the last 24h broken down by edge type (news/momentum/
        # arbitrage). Zero-filled so the dashboard has stable keys.
        "edge_types_24h": {
            "news": 0, "momentum": 0, "arbitrage": 0,
            **logger.get_edge_type_breakdown_since(since_24h, action="signal"),
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
                "edge_type": p.get("edge_type"),
                "event_id": p.get("event_id"),
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
                "event_id": p.get("event_id"),
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
            _classification_row(c)
            for c in logger.get_recent_classifications(limit=RECENT_CLASSIFICATIONS_LIMIT)
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
