"""Public-data loaders for the logged-classification replay backtest.

Everything here reads PUBLIC, unauthenticated Polymarket endpoints and caches
the responses on disk (``backtest_data/`` at the project root, gitignored), so
a second run is offline and reproducible:

- Gamma ``/public-search``      -> market metadata by exact question text
                                   (token ids, outcomes, closed flag,
                                   final outcomePrices = resolution)
- CLOB ``/prices-history``      -> hourly YES-token price path

What is NOT available from public APIs (documented in BACKTEST.md): historical
order-book snapshots (depth/spread at a past moment). The replay therefore uses
a parametric book model (see ``locus.backtest.costs``).

The classification log itself comes from the repo's own git history: every
``docs/status``-cycle commit rewrote ``docs/classifications.json`` with the last
5000 classifications, so walking the history recovers the live agent's actual
Claude outputs (direction, materiality, confidence) with their timestamps.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from locus import config

log = logging.getLogger(__name__)

GAMMA = "https://gamma-api.polymarket.com"
DATA_DIR = config.PROJECT_ROOT / "backtest_data"
CLASSIFICATIONS_PATH = DATA_DIR / "logged_classifications.json"
MARKETS_PATH = DATA_DIR / "markets.json"
HISTORY_DIR = DATA_DIR / "history"


def extract_logged_classifications(stride: int = 4, repo_dir: Path | None = None) -> list[dict]:
    """Walk the git history of docs/classifications.json and union every row.

    Requires a full (non-shallow) clone: ``git fetch --unshallow``. ``stride``
    samples every Nth commit — each snapshot holds the last 5000 rows, so a
    small stride loses nothing (stride 4 vs 40 changed the row count by <5%).
    """
    repo = repo_dir or config.PROJECT_ROOT
    hashes = subprocess.run(
        ["git", "log", "--format=%h", "--", "docs/classifications.json"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.split()
    rows: dict[tuple, dict] = {}
    for h in hashes[::max(1, stride)]:
        out = subprocess.run(
            ["git", "show", f"{h}:docs/classifications.json"],
            cwd=repo, capture_output=True, text=True,
        ).stdout
        try:
            data = json.loads(out)
        except ValueError:
            continue
        for r in data["classifications"] if isinstance(data, dict) else data:
            rows[(r.get("time"), r.get("market_question"), r.get("headline"))] = r
    return sorted(rows.values(), key=lambda r: r.get("time") or "")


def load_or_extract_classifications(stride: int = 4) -> list[dict]:
    if CLASSIFICATIONS_PATH.exists():
        return json.loads(CLASSIFICATIONS_PATH.read_text())
    rows = extract_logged_classifications(stride)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CLASSIFICATIONS_PATH.write_text(json.dumps(rows))
    return rows


def _search_market(question: str) -> dict | None:
    """Exact-question match via Gamma public search (includes closed markets)."""
    for attempt in range(3):
        try:
            r = httpx.get(
                f"{GAMMA}/public-search",
                params={"q": question, "limit_per_type": 10,
                        "keep_closed_markets": 1, "events_status": "all"},
                timeout=20,
            )
            if r.status_code == 429:
                time.sleep(2 + attempt * 3)
                continue
            r.raise_for_status()
            for ev in r.json().get("events", []) or []:
                for m in ev.get("markets", []) or []:
                    if (m.get("question") or "").strip() == question.strip():
                        return {
                            "question": m.get("question"),
                            "condition_id": m.get("conditionId"),
                            "outcomes": m.get("outcomes"),
                            "outcome_prices": m.get("outcomePrices"),
                            "clob_token_ids": m.get("clobTokenIds"),
                            "closed": bool(m.get("closed")),
                            "end_date": m.get("endDate"),
                            "closed_time": m.get("closedTime"),
                            "volume": m.get("volume"),
                            "category": ev.get("category") or m.get("category"),
                            "event_id": ev.get("id"),
                        }
            return {}
        except Exception as e:  # network hiccup -> retry
            log.debug("search failed for %r: %s", question[:50], e)
            time.sleep(1 + attempt)
    return None


def fetch_markets(questions: list[str], workers: int = 6) -> dict[str, dict]:
    """question -> market metadata ({} when not found). Cached in markets.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = json.loads(MARKETS_PATH.read_text()) if MARKETS_PATH.exists() else {}
    todo = [q for q in dict.fromkeys(questions) if q not in cache]
    if todo:
        with ThreadPoolExecutor(workers) as pool:
            for q, res in zip(todo, pool.map(_search_market, todo)):
                if res is not None:  # None = transient failure, retry next run
                    cache[q] = res
        MARKETS_PATH.write_text(json.dumps(cache))
    return cache


def _yes_token(meta: dict) -> str | None:
    try:
        tokens = json.loads(meta["clob_token_ids"]) if isinstance(meta.get("clob_token_ids"), str) else meta.get("clob_token_ids")
        outcomes = json.loads(meta["outcomes"]) if isinstance(meta.get("outcomes"), str) else meta.get("outcomes")
    except (TypeError, ValueError, KeyError):
        return None
    if not tokens:
        return None
    if outcomes and len(outcomes) == len(tokens):
        for o, t in zip(outcomes, tokens):
            if str(o).lower() == "yes":
                return t
    return tokens[0]  # YES=first convention (same as executor)


def resolution_yes(meta: dict) -> float | None:
    """1.0 / 0.0 when the market closed and resolved cleanly, else None."""
    if not meta.get("closed"):
        return None
    try:
        prices = json.loads(meta["outcome_prices"]) if isinstance(meta.get("outcome_prices"), str) else meta.get("outcome_prices")
        outcomes = json.loads(meta["outcomes"]) if isinstance(meta.get("outcomes"), str) else meta.get("outcomes")
    except (TypeError, ValueError, KeyError):
        return None
    if not prices:
        return None
    idx = 0
    if outcomes:
        for i, o in enumerate(outcomes):
            if str(o).lower() == "yes":
                idx = i
    p = float(prices[idx])
    if p >= 0.99:
        return 1.0
    if p <= 0.01:
        return 0.0
    return None  # 50/50 or unresolved payout


def _get_history(params: dict) -> list[dict] | None:
    for attempt in range(3):
        try:
            r = httpx.get(f"{config.POLYMARKET_HOST}/prices-history", params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 + attempt * 3)
                continue
            r.raise_for_status()
            return r.json().get("history", []) or []
        except Exception as e:
            log.debug("history failed: %s", e)
            time.sleep(1 + attempt)
    return None


def fetch_history(token_id: str, start_ts: int | None = None, end_ts: int | None = None,
                  chunk_days: int = 14) -> list[dict]:
    """Hourly price history for a token ({t, p} points), cached per token.

    ``interval=max`` returns nothing for many resolved markets, and explicit
    ranges longer than ~14-15 days are rejected ("interval is too long"), so
    the range is fetched in 14-day chunks.
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = HISTORY_DIR / f"{token_id[:40]}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached:
            return cached
    points: dict[int, float] = {}
    if start_ts is None:
        got = _get_history({"market": token_id, "interval": "max", "fidelity": 60})
        for x in got or []:
            points[int(x["t"])] = float(x["p"])
    else:
        end_ts = end_ts or int(time.time())
        cur = start_ts
        while cur < end_ts:
            nxt = min(cur + chunk_days * 86400, end_ts)
            got = _get_history({"market": token_id, "startTs": cur, "endTs": nxt, "fidelity": 60})
            if got is None:
                return []  # transient failure: don't cache a partial path
            for x in got:
                points[int(x["t"])] = float(x["p"])
            cur = nxt
    hist = [{"t": t, "p": points[t]} for t in sorted(points)]
    path.write_text(json.dumps(hist))
    return hist


def fetch_histories(ranges: dict[str, tuple[int, int]], workers: int = 6
                    ) -> dict[str, list[dict]]:
    """token_id -> (start_ts, end_ts) ranges -> token_id -> history."""
    ids = list(ranges)
    with ThreadPoolExecutor(workers) as pool:
        res = pool.map(lambda t: fetch_history(t, *ranges[t]), ids)
        return dict(zip(ids, res))


MINUTE_DIR = DATA_DIR / "minute"


def fetch_minute_day(token_id: str, day_start_ts: int) -> list[dict]:
    """Minute-level (fidelity=1) price points for one UTC day of a token,
    cached. Used for the realistic-latency entry price."""
    MINUTE_DIR.mkdir(parents=True, exist_ok=True)
    path = MINUTE_DIR / f"{token_id[:40]}_{day_start_ts}.json"
    if path.exists():
        return json.loads(path.read_text())
    got = _get_history({"market": token_id, "startTs": day_start_ts,
                        "endTs": day_start_ts + 86400 + 3600, "fidelity": 1})
    if got is None:
        return []
    hist = sorted(({"t": int(x["t"]), "p": float(x["p"])} for x in got), key=lambda x: x["t"])
    path.write_text(json.dumps(hist))
    return hist


def fetch_minute_days(pairs: list[tuple[str, int]], workers: int = 6) -> dict:
    """(token, day_start_ts) pairs -> {token: merged minute history}."""
    pairs = list(dict.fromkeys(pairs))
    out: dict[str, dict[int, float]] = {}
    with ThreadPoolExecutor(workers) as pool:
        for (tok, _), hist in zip(pairs, pool.map(lambda pr: fetch_minute_day(*pr), pairs)):
            d = out.setdefault(tok, {})
            for x in hist:
                d[x["t"]] = x["p"]
    return {t: [{"t": k, "p": d[k]} for k in sorted(d)] for t, d in out.items()}
