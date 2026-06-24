#!/usr/bin/env python3
"""
Real backtest pilot — replays resolved Polymarket crypto markets through the
actual classify() + detect_edge_v2() pipeline, using real headlines and
real CLOB price history (not synthetic data like backtest.py).

Pipeline:
  1. Gamma API: paginate resolved, niche-volume crypto markets.
  2. GDELT 2.0 DOC API: real headlines for each market's coin, 14-day window
     before close (cached per coin+date — see notes in the report below).
     If GDELT returns nothing (e.g. rate-limited), falls back to NewsAPI
     /v2/everything for the same query/window using the existing
     config.NEWSAPI_KEY (cheap: at most one call per coin+date group).
  3. CLOB /prices-history: real YES price right before each headline.
  4. Replay each (headline, market, price) through classify() + detect_edge_v2().
  5. Simulate PnL against the market's actual resolution.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from locus import config
from locus.core.classifier import classify
from locus.core.edge import detect_edge_v2
from locus.markets.gamma import Market, _infer_category
from locus.sources.news_stream import NewsEvent

console = Console()

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"
NEWSAPI_API = "https://newsapi.org/v2/everything"

# Gamma's offset-based pagination 422s past offset=10000 ("use /markets/keyset
# for deeper pagination"). A flat closedTime-desc scan therefore only reaches
# ~3 days back. To go deeper, the scan is split into end_date_min/max windows
# (each window gets its own 10k offset budget) and filtered server-side to
# Gamma's canonical Crypto tag, which keeps every window to a handful of pages.
MAX_OFFSET = 10000
LOOKBACK_DAYS = 30
WINDOW_DAYS = 3
CRYPTO_TAG_ID = 21  # Gamma's canonical "Crypto" tag
MIN_DURATION_DAYS = 2  # createdAt -> closedTime; excludes 5m/1h "Up or Down" markets
TARGET_MARKETS = 30
MAX_CLASSIFY_CALLS = 100
HEADLINES_PER_MARKET_CAP = 5
GDELT_MAXRECORDS = 10
GDELT_RATE_LIMIT_SECONDS = 5.5
GDELT_MAX_RETRIES = 3
NEWSAPI_PAGESIZE = 10  # fallback when GDELT returns nothing for a (coin, date) group

# Claude Haiku 4.5 pricing (per 1M tokens) — config.CLASSIFICATION_MODEL
HAIKU_INPUT_COST_PER_M = 1.00
HAIKU_OUTPUT_COST_PER_M = 5.00
EST_INPUT_TOKENS_PER_CALL = 450  # rough: prompt template + track record
EST_OUTPUT_TOKENS_PER_CALL = 100  # rough: max_tokens=200, JSON reply is short

# (substring in lowercased question, label, GDELT query clause)
CRYPTO_ASSETS = [
    ("bitcoin", "bitcoin", '"Bitcoin" OR "BTC"'),
    ("ethereum", "ethereum", '"Ethereum" OR "ETH"'),
    ("solana", "solana", '"Solana" OR "SOL"'),
    ("xrp", "xrp", '"XRP" OR "Ripple"'),
    ("dogecoin", "dogecoin", '"Dogecoin" OR "DOGE"'),
    ("cardano", "cardano", '"Cardano" OR "ADA"'),
    ("litecoin", "litecoin", '"Litecoin" OR "LTC"'),
    ("chainlink", "chainlink", '"Chainlink" OR "LINK"'),
    ("polygon", "polygon", '"Polygon" OR "MATIC"'),
    ("avalanche", "avalanche", '"Avalanche" OR "AVAX"'),
    ("polkadot", "polkadot", '"Polkadot" OR "DOT"'),
    ("shiba", "shiba inu", '"Shiba Inu" OR "SHIB"'),
    ("bnb", "bnb", '"BNB" OR "Binance Coin"'),
]


def _parse_dt(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def extract_keywords(question: str) -> tuple[str, str]:
    ql = question.lower()
    for substr, label, clause in CRYPTO_ASSETS:
        if substr in ql:
            return label, clause
    return "crypto", '"cryptocurrency" OR "crypto"'


@dataclass
class TradeRecord:
    market_question: str
    headline: str
    headline_date: str
    direction: str
    materiality: float
    side: str
    entry_price: float
    edge: float
    bet_amount: float
    resolved_yes: bool
    pnl: float
    won: bool


def fetch_crypto_markets() -> dict:
    """Scan Gamma for resolved, niche-volume crypto markets over LOOKBACK_DAYS.

    The scan walks end-date windows newest->oldest (WINDOW_DAYS each), with
    tag_id + volume bounds applied Gamma-side, so each window needs only a
    few pages and the whole lookback stays well under the 10k offset cap.
    Candidates are grouped by (coin label, closed date); each date has many
    distinct price-band questions, so we keep all of them and pick a diverse
    subset later. Boundary days appear in two windows; deduped by conditionId.
    """
    groups: dict[tuple[str, object], list[dict]] = {}
    oldest_closed = None
    newest_closed = None
    total_scanned = 0
    raw_candidates = 0
    pages_scanned = 0
    last_offset = None
    windows_scanned = 0
    seen_condition_ids: set[str] = set()

    now = datetime.now(timezone.utc)
    window_bounds = [
        (now - timedelta(days=w + WINDOW_DAYS), now - timedelta(days=w))
        for w in range(0, LOOKBACK_DAYS, WINDOW_DAYS)
    ]

    page_items: list[dict] = []
    for window_start, window_end in window_bounds:
        windows_scanned += 1
        for page in range((MAX_OFFSET // 100) + 1):
            offset = page * 100
            try:
                resp = httpx.get(
                    f"{GAMMA_API}/markets",
                    params={
                        "closed": "true",
                        "tag_id": CRYPTO_TAG_ID,
                        "volume_num_min": config.MIN_VOLUME_USD,
                        "volume_num_max": config.MAX_VOLUME_USD,
                        "end_date_min": window_start.strftime("%Y-%m-%d"),
                        "end_date_max": window_end.strftime("%Y-%m-%d"),
                        "order": "closedTime",
                        "ascending": "false",
                        "limit": 100,
                        "offset": offset,
                    },
                    timeout=20,
                )
                resp.raise_for_status()
                items = resp.json()
            except Exception as e:
                console.print(
                    f"  [yellow]Gamma error at offset {offset} "
                    f"(window {window_start.date()}..{window_end.date()}): {e}[/yellow]"
                )
                break

            if not isinstance(items, list) or not items:
                break

            pages_scanned += 1
            last_offset = offset
            total_scanned += len(items)
            page_items.extend(items)

            if len(items) < 100:
                break

    for m in page_items:
        cid = str(m.get("conditionId", "") or "")
        if cid and cid not in seen_condition_ids:
            seen_condition_ids.add(cid)
            closed_time = _parse_dt(m.get("closedTime", ""))
            created_at = _parse_dt(m.get("createdAt", ""))
            if closed_time is None:
                continue
            if oldest_closed is None or closed_time < oldest_closed:
                oldest_closed = closed_time
            if newest_closed is None or closed_time > newest_closed:
                newest_closed = closed_time

            if created_at is None:
                continue

            duration_days = (closed_time - created_at).total_seconds() / 86400.0
            if duration_days < MIN_DURATION_DAYS:
                continue

            question = m.get("question", "")
            ql = question.lower()
            if "up or down" in ql:
                continue

            tags = m.get("tags", None) or []
            if _infer_category(question, tags) != "crypto":
                continue

            outcome_prices_raw = m.get("outcomePrices", "")
            try:
                prices = (
                    json.loads(outcome_prices_raw)
                    if isinstance(outcome_prices_raw, str)
                    else outcome_prices_raw
                )
                yes_resolution = float(prices[0])
            except (json.JSONDecodeError, ValueError, TypeError, IndexError):
                continue
            if yes_resolution not in (0.0, 1.0):
                continue

            clob_ids_raw = m.get("clobTokenIds", "")
            try:
                token_ids = (
                    json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else clob_ids_raw
                )
            except (json.JSONDecodeError, TypeError):
                token_ids = []
            if not isinstance(token_ids, list) or len(token_ids) < 2:
                continue

            label, gdelt_clause = extract_keywords(question)
            group_key = (label, closed_time.date())

            raw_candidates += 1
            groups.setdefault(group_key, []).append(
                {
                    "condition_id": m.get("conditionId", ""),
                    "question": question,
                    "volume": float(m.get("volume", m.get("volumeNum", 0)) or 0),
                    "created_at": created_at,
                    "closed_time": closed_time,
                    "yes_resolution": yes_resolution,
                    "yes_token_id": token_ids[0],
                    "no_token_id": token_ids[1],
                    "label": label,
                    "gdelt_clause": gdelt_clause,
                    "group_key": group_key,
                }
            )

    return {
        "groups": groups,
        "oldest_closed": oldest_closed,
        "newest_closed": newest_closed,
        "pages_scanned": pages_scanned,
        "last_offset": last_offset,
        "total_scanned": total_scanned,
        "raw_candidates": raw_candidates,
        "windows_scanned": windows_scanned,
    }


def select_markets(groups: dict, target: int) -> list[dict]:
    """Stratified round-robin across (coin, date) groups for diversity."""
    sorted_keys = sorted(groups.keys())
    for key in sorted_keys:
        groups[key].sort(key=lambda c: c["volume"], reverse=True)

    selected = []
    round_idx = 0
    while len(selected) < target:
        added = False
        for key in sorted_keys:
            bucket = groups[key]
            if round_idx < len(bucket):
                selected.append(bucket[round_idx])
                added = True
                if len(selected) >= target:
                    break
        if not added:
            break
        round_idx += 1
    return selected


_last_gdelt_call_time = 0.0


def fetch_gdelt_headlines(gdelt_clause: str, window_start: datetime, window_end: datetime) -> tuple[list[dict], bool]:
    """Query GDELT for headlines in [window_start, window_end]. Returns (headlines, ok)."""
    global _last_gdelt_call_time

    params = {
        "query": f"({gdelt_clause}) sourcelang:eng",
        "mode": "ArtList",
        "format": "json",
        "maxrecords": GDELT_MAXRECORDS,
        "startdatetime": window_start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": window_end.strftime("%Y%m%d%H%M%S"),
        "sort": "datedesc",
    }

    for attempt in range(GDELT_MAX_RETRIES):
        elapsed = time.time() - _last_gdelt_call_time
        if elapsed < GDELT_RATE_LIMIT_SECONDS:
            time.sleep(GDELT_RATE_LIMIT_SECONDS - elapsed)

        try:
            resp = httpx.get(GDELT_API, params=params, timeout=30)
            _last_gdelt_call_time = time.time()
            if resp.status_code == 429:
                time.sleep(6 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            _last_gdelt_call_time = time.time()
            time.sleep(6 * (attempt + 1))
            continue

        articles = data.get("articles", [])
        out = []
        for a in articles:
            seendate = a.get("seendate", "")
            try:
                dt = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            out.append(
                {
                    "title": a.get("title", ""),
                    "date": dt,
                    "url": a.get("url", ""),
                    "domain": a.get("domain", ""),
                    "source_api": "gdelt",
                }
            )
        return out, True

    return [], False


def fetch_newsapi_headlines(query_clause: str, window_start: datetime, window_end: datetime) -> tuple[list[dict], bool]:
    """Query NewsAPI /v2/everything for headlines in [window_start, window_end].

    Fallback for when GDELT returns nothing (e.g. rate-limited). Returns
    (headlines, ok) where ok=False means the request failed outright (no key,
    HTTP error, etc.) — an empty-but-successful response returns ([], True).
    """
    if not config.NEWSAPI_KEY:
        return [], False

    params = {
        "q": query_clause,
        "from": window_start.strftime("%Y-%m-%dT%H:%M:%S"),
        "to": window_end.strftime("%Y-%m-%dT%H:%M:%S"),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": NEWSAPI_PAGESIZE,
        "apiKey": config.NEWSAPI_KEY,
    }

    try:
        resp = httpx.get(NEWSAPI_API, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return [], False

    if data.get("status") != "ok":
        return [], False

    out = []
    for a in data.get("articles", []):
        pub_str = a.get("publishedAt", "")
        dt = _parse_dt(pub_str)
        if dt is None:
            continue
        out.append(
            {
                "title": a.get("title", ""),
                "date": dt,
                "url": a.get("url", ""),
                "domain": (a.get("source") or {}).get("name", ""),
                "source_api": "newsapi",
            }
        )
    return out, True


def fetch_price_history(token_id: str, start_ts: int, end_ts: int, fidelity: int = 60) -> list[dict]:
    try:
        resp = httpx.get(
            f"{CLOB_API}/prices-history",
            params={"market": token_id, "startTs": int(start_ts), "endTs": int(end_ts), "fidelity": fidelity},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    return data.get("history", [])


def price_at_or_before(history: list[dict], ts: int) -> float | None:
    candidates = [h for h in history if h.get("t", 0) <= ts]
    if not candidates:
        return None
    best = max(candidates, key=lambda h: h["t"])
    try:
        return float(best["p"])
    except (TypeError, ValueError):
        return None


def _pick_spread(items: list, n: int) -> list:
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def run_pilot():
    console.print(Panel("Real Backtest Pilot — crypto markets, GDELT headlines, CLOB prices", style="bright_green"))

    console.print("\n[bold][1/5][/bold] Scanning resolved Polymarket markets (Gamma API)...")
    scan = fetch_crypto_markets()
    groups = scan["groups"]
    oldest, newest = scan["oldest_closed"], scan["newest_closed"]
    span_days = (newest - oldest).total_seconds() / 86400 if oldest and newest else 0.0

    console.print(
        f"  Scanned {scan['pages_scanned']} pages across {scan['windows_scanned']} "
        f"end-date windows of {WINDOW_DAYS}d ({scan['total_scanned']} markets total, "
        f"crypto tag + volume band filtered Gamma-side)"
    )
    console.print(f"  closedTime range covered: {oldest} -> {newest}  (~{span_days:.1f} days)")
    console.print(
        f"  Crypto candidates (duration>={MIN_DURATION_DAYS}d, excl. 'Up or Down', binary-resolved): "
        f"{scan['raw_candidates']} across {len(groups)} (coin, date) groups"
    )
    for key in sorted(groups.keys()):
        console.print(f"    {key[0]:10s} {key[1]} -> {len(groups[key])} markets")

    selected = select_markets(groups, TARGET_MARKETS)
    console.print(f"  Selected {len(selected)} markets via stratified round-robin across groups")

    if not selected:
        console.print("[red]No candidate markets found. Aborting.[/red]")
        return

    console.print("\n[bold][2/5][/bold] Fetching headlines (GDELT, NewsAPI fallback; cached per coin+date) and CLOB price history...")

    per_market_cap = max(1, min(HEADLINES_PER_MARKET_CAP, MAX_CLASSIFY_CALLS // len(selected)))
    console.print(
        f"  Per-market headline cap: {per_market_cap} "
        f"(target <= {MAX_CLASSIFY_CALLS} classify() calls across {len(selected)} markets)"
    )

    headline_cache: dict[tuple, tuple[list[dict], str]] = {}
    gdelt_attempted = 0
    gdelt_succeeded = 0
    newsapi_attempted = 0
    newsapi_succeeded = 0
    groups_used_gdelt = 0
    groups_used_newsapi = 0

    markets_zero_headlines = 0
    markets_no_price_history = 0
    markets_with_pairs = 0

    pairs: list[tuple[dict, dict, float]] = []

    for i, c in enumerate(selected):
        group_key = c["group_key"]
        if group_key not in headline_cache:
            date = group_key[1]
            window_end = datetime(date.year, date.month, date.day, 23, 59, 59, tzinfo=timezone.utc)
            window_start = window_end - timedelta(days=14)

            gdelt_attempted += 1
            headlines, ok = fetch_gdelt_headlines(c["gdelt_clause"], window_start, window_end)
            if ok:
                gdelt_succeeded += 1
            source_used = "gdelt" if headlines else "none"

            if not headlines:
                newsapi_attempted += 1
                headlines, ok2 = fetch_newsapi_headlines(c["gdelt_clause"], window_start, window_end)
                if ok2:
                    newsapi_succeeded += 1
                if headlines:
                    source_used = "newsapi"

            if source_used == "gdelt":
                groups_used_gdelt += 1
            elif source_used == "newsapi":
                groups_used_newsapi += 1

            headline_cache[group_key] = (headlines, source_used)

        headlines, source_used = headline_cache[group_key]
        label = f"[{i + 1}/{len(selected)}] {c['question'][:55]:55s}"

        if not headlines:
            markets_zero_headlines += 1
            console.print(f"  {label} -> 0 headlines (gdelt+newsapi)")
            continue

        history = fetch_price_history(
            c["yes_token_id"], int(c["created_at"].timestamp()), int(c["closed_time"].timestamp()), fidelity=60
        )
        time.sleep(0.15)

        if not history:
            markets_no_price_history += 1
            console.print(f"  {label} -> {len(headlines)} headlines ({source_used}), no price history")
            continue

        earliest_ts = history[0]["t"]
        usable_headlines = [h for h in headlines if h["date"].timestamp() >= earliest_ts]

        if not usable_headlines:
            markets_no_price_history += 1
            console.print(f"  {label} -> {len(headlines)} headlines ({source_used}), none within price-history window")
            continue

        n_pairs = 0
        for h in _pick_spread(usable_headlines, per_market_cap):
            price = price_at_or_before(history, int(h["date"].timestamp()))
            if price is None:
                continue
            pairs.append((c, h, price))
            n_pairs += 1

        if n_pairs:
            markets_with_pairs += 1

        console.print(f"  {label} -> {len(headlines)} headlines ({source_used}), {n_pairs} pairs")

    news_stats = {
        "gdelt_attempted": gdelt_attempted,
        "gdelt_succeeded": gdelt_succeeded,
        "newsapi_attempted": newsapi_attempted,
        "newsapi_succeeded": newsapi_succeeded,
        "groups_used_gdelt": groups_used_gdelt,
        "groups_used_newsapi": groups_used_newsapi,
        "total_groups": len(headline_cache),
    }

    console.print(f"\n  GDELT: {gdelt_succeeded}/{gdelt_attempted} queries succeeded, "
                   f"used for {groups_used_gdelt}/{len(headline_cache)} groups")
    console.print(f"  NewsAPI fallback: {newsapi_succeeded}/{newsapi_attempted} queries succeeded, "
                   f"used for {groups_used_newsapi}/{len(headline_cache)} groups")
    console.print(f"  Markets with zero matching headlines: {markets_zero_headlines}/{len(selected)}")
    console.print(f"  Markets with headlines but no usable price history: {markets_no_price_history}/{len(selected)}")
    console.print(f"  Markets with >=1 usable pair: {markets_with_pairs}/{len(selected)}")
    console.print(f"  Total (headline, market, price) pairs: {len(pairs)}")

    if not pairs:
        console.print("\n[red]No usable pairs -- cannot run classify(). See data-quality notes above.[/red]")
        _save_results(scan, selected, [], [], news_stats, markets_zero_headlines, markets_no_price_history, [])
        return

    if len(pairs) > MAX_CLASSIFY_CALLS:
        console.print(f"  Capping at {MAX_CLASSIFY_CALLS} pairs (had {len(pairs)})")
        pairs = pairs[:MAX_CLASSIFY_CALLS]

    console.print(f"\n[bold][3/5][/bold] Replaying {len(pairs)} pairs through classify() + detect_edge_v2()...")

    trades: list[TradeRecord] = []
    replay_log: list[dict] = []

    for i, (c, h, price_before) in enumerate(pairs):
        market = Market(
            condition_id=c["condition_id"],
            question=c["question"],
            category="crypto",
            yes_price=price_before,
            no_price=round(1.0 - price_before, 4),
            volume=c["volume"],
            end_date=c["closed_time"].isoformat(),
            active=False,
            tokens=[
                {"token_id": c["yes_token_id"], "outcome": "Yes", "price": price_before},
                {"token_id": c["no_token_id"], "outcome": "No", "price": round(1.0 - price_before, 4)},
            ],
        )

        news_event = NewsEvent(
            headline=h["title"],
            source=h["domain"] or h["source_api"],
            url=h["url"],
            received_at=h["date"],
            published_at=h["date"],
        )

        classification = classify(h["title"], market, source=news_event.source, as_of=h["date"])
        edge_metrics = detect_edge_v2(market, classification, news_event)
        signal = edge_metrics.signal if edge_metrics else None

        tag = "SIGNAL" if signal else "--"
        console.print(
            f"  [{i + 1}/{len(pairs)}] {classification.direction:8s} mat={classification.materiality:.2f} "
            f"price={price_before:.2f} -> {tag:6s} | {c['question'][:42]}"
        )

        replay_log.append(
            {
                "market_question": c["question"],
                "headline": h["title"],
                "headline_source": h["domain"] or h["source_api"],
                "headline_date": h["date"].isoformat(),
                "price_before": round(price_before, 4),
                "direction": classification.direction,
                "materiality": round(classification.materiality, 3),
                "signal": signal is not None,
            }
        )

        if not signal:
            continue

        resolved_yes = c["yes_resolution"] == 1.0
        if signal.side == "YES":
            entry = price_before
            won = resolved_yes
        else:
            entry = 1.0 - price_before
            won = not resolved_yes

        bet = signal.bet_amount
        if entry <= 0:
            pnl = 0.0
        elif won:
            pnl = bet * (1.0 / entry - 1.0)
        else:
            pnl = -bet

        trades.append(
            TradeRecord(
                market_question=c["question"],
                headline=h["title"],
                headline_date=h["date"].isoformat(),
                direction=classification.direction,
                materiality=round(classification.materiality, 3),
                side=signal.side,
                entry_price=round(entry, 4),
                edge=round(signal.edge, 4),
                bet_amount=bet,
                resolved_yes=resolved_yes,
                pnl=round(pnl, 2),
                won=won,
            )
        )

    console.print("\n[bold][4/5][/bold] Building report...")
    _print_report(scan, selected, pairs, trades, news_stats, markets_zero_headlines, markets_no_price_history, span_days)

    console.print("\n[bold][5/5][/bold] Saving backtest_results.json...")
    _save_results(scan, selected, pairs, trades, news_stats, markets_zero_headlines, markets_no_price_history, replay_log)


def _print_report(scan, selected, pairs, trades, news_stats, markets_zero_headlines, markets_no_price_history, span_days):
    console.print()

    total_pnl = sum(t.pnl for t in trades)
    wins = [t for t in trades if t.won]
    losses = [t for t in trades if not t.won]
    win_rate = (len(wins) / len(trades) * 100) if trades else 0.0
    avg_edge_winners = (sum(t.edge for t in wins) / len(wins)) if wins else 0.0
    avg_edge_losers = (sum(t.edge for t in losses) / len(losses)) if losses else 0.0

    table = Table(title="Real Backtest Pilot — Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Markets selected", str(len(selected)))
    table.add_row("Pairs tested (classify calls)", str(len(pairs)))
    table.add_row("Signals generated", str(len(trades)))
    table.add_row("Trades simulated", str(len(trades)))

    pnl_style = "bright_green" if total_pnl >= 0 else "red"
    pnl_str = f"+${total_pnl:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):.2f}"
    table.add_row("Total PnL", f"[{pnl_style}]{pnl_str}[/{pnl_style}]")

    if trades:
        wr_style = "bright_green" if win_rate >= 55 else ("yellow" if win_rate >= 45 else "red")
        table.add_row("Win rate", f"[{wr_style}]{win_rate:.1f}%[/{wr_style}] ({len(wins)}/{len(trades)})")
        table.add_row("Avg edge (winners)", f"{avg_edge_winners:.1%}" if wins else "n/a")
        table.add_row("Avg edge (losers)", f"{avg_edge_losers:.1%}" if losses else "n/a")
    else:
        table.add_row("Win rate", "n/a")
        table.add_row("Avg edge (winners)", "n/a")
        table.add_row("Avg edge (losers)", "n/a")

    console.print(table)

    dq = Table(title="Data Quality", show_header=True, header_style="bold yellow")
    dq.add_column("Check", style="bold")
    dq.add_column("Result", justify="right")
    dq.add_row("Gamma closedTime coverage", f"~{span_days:.1f} days ({LOOKBACK_DAYS}d end-date window scan)")
    dq.add_row("Crypto candidates found", f"{scan['raw_candidates']} across {len(scan['groups'])} (coin,date) groups")
    dq.add_row("GDELT queries OK", f"{news_stats['gdelt_succeeded']}/{news_stats['gdelt_attempted']}")
    dq.add_row(
        "NewsAPI fallback queries OK",
        f"{news_stats['newsapi_succeeded']}/{news_stats['newsapi_attempted']}",
    )
    dq.add_row(
        "Groups: GDELT / NewsAPI / none",
        f"{news_stats['groups_used_gdelt']} / {news_stats['groups_used_newsapi']} / "
        f"{news_stats['total_groups'] - news_stats['groups_used_gdelt'] - news_stats['groups_used_newsapi']}"
        f" (of {news_stats['total_groups']})",
    )
    dq.add_row("Markets w/ zero headlines", f"{markets_zero_headlines}/{len(selected)}")
    dq.add_row("Markets w/ no usable price history", f"{markets_no_price_history}/{len(selected)}")
    dq.add_row(
        "Markets w/ >=1 usable pair",
        f"{len(selected) - markets_zero_headlines - markets_no_price_history}/{len(selected)}",
    )
    console.print(dq)

    if trades:
        console.print()
        tt = Table(title="Simulated Trades", show_header=True, header_style="bold green")
        tt.add_column("Market", max_width=35)
        tt.add_column("Headline", max_width=35)
        tt.add_column("Dir.", width=8)
        tt.add_column("Side", width=4)
        tt.add_column("Entry", justify="right", width=6)
        tt.add_column("Resolved", width=8)
        tt.add_column("PnL", justify="right", width=9)

        for t in trades:
            pnl_str = f"+${t.pnl:.2f}" if t.pnl >= 0 else f"-${abs(t.pnl):.2f}"
            pnl_style = "bright_green" if t.pnl >= 0 else "red"
            tt.add_row(
                t.market_question[:35],
                t.headline[:35],
                t.direction[:8],
                t.side,
                f"{t.entry_price:.2f}",
                "YES" if t.resolved_yes else "NO",
                f"[{pnl_style}]{pnl_str}[/{pnl_style}]",
            )
        console.print(tt)
    else:
        console.print("\n[yellow]No signals generated -- no trades to show.[/yellow]")

    n_calls = len(pairs)
    est_cost = n_calls * (
        EST_INPUT_TOKENS_PER_CALL / 1e6 * HAIKU_INPUT_COST_PER_M
        + EST_OUTPUT_TOKENS_PER_CALL / 1e6 * HAIKU_OUTPUT_COST_PER_M
    )
    console.print(
        f"\n[dim]Estimated API cost: {n_calls} classify() calls x "
        f"(~{EST_INPUT_TOKENS_PER_CALL} in / ~{EST_OUTPUT_TOKENS_PER_CALL} out tokens) "
        f"@ Haiku 4.5 (${HAIKU_INPUT_COST_PER_M:.2f}/${HAIKU_OUTPUT_COST_PER_M:.2f} per 1M) "
        f"~= ${est_cost:.4f}[/dim]"
    )


def _save_results(scan, selected, pairs, trades, news_stats, markets_zero_headlines, markets_no_price_history, replay_log):
    total_pnl = sum(t.pnl for t in trades)
    wins = [t for t in trades if t.won]
    win_rate = (len(wins) / len(trades) * 100) if trades else 0.0

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status_note": {
            "status": "parked",
            "blocker": (
                "Deep-history news coverage. The windowed market scan now reaches "
                f"{LOOKBACK_DAYS} days of resolved markets, but the news sources don't: "
                f"GDELT proved unreliable (succeeded {news_stats['gdelt_succeeded']}/"
                f"{news_stats['gdelt_attempted']} queries, used for "
                f"{news_stats['groups_used_gdelt']}/{news_stats['total_groups']} groups this run) "
                "and NewsAPI's free tier only returns articles from the last ~30 days, so "
                f"{markets_zero_headlines}/{len(selected)} selected markets had zero usable "
                "headlines. Until a reliable historical headline source is added, the market "
                "scan outruns the news data. Parking the backtest line for now — live "
                "calibration on the expanded market universe (3,098 tracked niche markets) "
                "is the faster path to real numbers."
            ),
        },
        "config": {
            "max_offset": MAX_OFFSET,
            "lookback_days": LOOKBACK_DAYS,
            "window_days": WINDOW_DAYS,
            "crypto_tag_id": CRYPTO_TAG_ID,
            "min_duration_days": MIN_DURATION_DAYS,
            "target_markets": TARGET_MARKETS,
            "max_classify_calls": MAX_CLASSIFY_CALLS,
            "headlines_per_market_cap": HEADLINES_PER_MARKET_CAP,
            "min_volume_usd": config.MIN_VOLUME_USD,
            "max_volume_usd": config.MAX_VOLUME_USD,
            "materiality_threshold_bullish": config.MIN_MATERIALITY_BULLISH,
            "materiality_threshold_bearish": config.MIN_MATERIALITY_BEARISH,
            "edge_threshold": config.EDGE_THRESHOLD,
        },
        "coverage": {
            "pages_scanned": scan["pages_scanned"],
            "windows_scanned": scan["windows_scanned"],
            "last_offset": scan["last_offset"],
            "oldest_closed": scan["oldest_closed"].isoformat() if scan["oldest_closed"] else None,
            "newest_closed": scan["newest_closed"].isoformat() if scan["newest_closed"] else None,
            "raw_candidates": scan["raw_candidates"],
            "groups": {f"{k[0]}_{k[1]}": len(v) for k, v in scan["groups"].items()},
        },
        "selected_markets": [
            {
                "question": c["question"],
                "condition_id": c["condition_id"],
                "volume": c["volume"],
                "closed_time": c["closed_time"].isoformat(),
                "yes_resolution": c["yes_resolution"],
                "label": c["label"],
            }
            for c in selected
        ],
        "news_sources": news_stats,
        "data_quality": {
            "markets_zero_headlines": markets_zero_headlines,
            "markets_no_price_history": markets_no_price_history,
            "total_pairs": len(pairs),
        },
        "summary": {
            "pairs_tested": len(pairs),
            "signals_generated": len(trades),
            "total_pnl": round(total_pnl, 2),
            "win_rate_pct": round(win_rate, 1),
            "wins": len(wins),
            "losses": len(trades) - len(wins),
        },
        "trades": [asdict(t) for t in trades],
        "replay_log": replay_log,
    }

    with open("backtest_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    console.print("  Saved to backtest_results.json")


if __name__ == "__main__":
    run_pilot()
