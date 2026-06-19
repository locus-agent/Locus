from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

import httpx

from locus import config

GAMMA_API = "https://gamma-api.polymarket.com"


@dataclass
class Market:
    condition_id: str
    question: str
    category: str
    yes_price: float
    no_price: float
    volume: float
    end_date: str
    active: bool
    tokens: list[dict]
    description: str = ""
    spread: float = 0.0
    liquidity: float = 0.0
    slug: str = ""  # polymarket.com/event/<slug>
    event_id: str = ""  # Gamma event id; markets sharing it are sibling outcomes
    fee_rate: float = 0.0  # per-share fee rate by category (see _fee_rate_for_category)

    @property
    def implied_probability(self) -> float:
        return self.yes_price


# Gamma caps page size at 100 regardless of the requested limit.
_GAMMA_PAGE_SIZE = 100
# Per-stream pagination ceiling. A single volume-ordered stream can't reach the
# niche tail before Gamma's offset wall, so we fan out across tag_id-filtered
# streams instead, each paginated only this deep (offset 0..2000 = 20 pages).
_GAMMA_TAG_OFFSET_CAP = 2000

# Category tag slugs fetched in parallel (one tag_id stream each), spanning the
# niche markets the pipeline cares about plus the broad breaking-news topics.
_TAG_CATEGORIES = [
    "politics", "crypto", "sports", "science",
    "entertainment", "world", "business", "pop-culture",
]
# The general (untagged) stream only pulls high-volume markets, to catch large
# markets in categories outside the tag list without re-scanning the whole tail.
_GENERAL_VOLUME_MIN = 50_000
# Fallback when the /tags endpoint is unavailable: slice the volume axis into
# independent ranges and paginate each (no tag_id needed).
_FALLBACK_VOLUME_RANGES = [(1_000, 50_000), (50_000, 500_000)]

# Shared query params for every /markets call: active, open, orderbook-enabled,
# highest volume first.
_BASE_MARKET_PARAMS = {
    "active": True,
    "closed": False,
    "enableOrderBook": True,
    "order": "volume",
    "ascending": False,
}

# /tags caps page size at 100 like /markets, and the catalog is large (~6k
# tags) with the common category slugs scattered well past offset 2000, so
# paginate generously — but stop as soon as every tracked category slug is found.
_GAMMA_TAGS_MAX_PAGES = 120

# Process cache for the slug->tag_id map (tags change rarely). None until the
# first successful get_tags().
_tags_cache: dict[str, int] | None = None


def get_tags() -> dict[str, int]:
    """Fetch Gamma's tag catalog and return a {slug: tag_id} map, cached for the
    process. Paginates the /tags endpoint until every tracked category slug is
    found (or the catalog is exhausted). Raises on network/parse failure (and
    when the catalog comes back empty) so fetch_active_markets can fall back to
    volume-range slicing."""
    global _tags_cache
    if _tags_cache is not None:
        return _tags_cache

    mapping: dict[str, int] = {}
    needed = set(_TAG_CATEGORIES)
    with httpx.Client(timeout=15) as client:
        offset = 0
        for _ in range(_GAMMA_TAGS_MAX_PAGES):
            resp = client.get(
                f"{GAMMA_API}/tags",
                params={"limit": _GAMMA_PAGE_SIZE, "offset": offset},
            )
            resp.raise_for_status()
            data = resp.json()
            batch = data if isinstance(data, list) else data.get("data", [])
            if not batch:
                break
            for t in batch:
                slug = (t.get("slug") or "").lower()
                tid = t.get("id")
                if slug and tid is not None:
                    try:
                        mapping[slug] = int(tid)
                    except (ValueError, TypeError):
                        continue
            offset += len(batch)
            if len(batch) < _GAMMA_PAGE_SIZE:
                break
            if needed <= mapping.keys():  # every tracked category slug found
                break

    if not mapping:
        raise ValueError("Gamma /tags returned no usable tags")
    _tags_cache = mapping
    return mapping


def _parse_market(m: dict) -> Market | None:
    """Convert one raw Gamma market dict into a Market, or None when it can't be
    parsed or is a resolved/zero-info market. Pure (carries no dedup state)."""
    try:
        # Gamma API encodes outcomePrices as a JSON string of [yes, no].
        outcome_prices = m.get("outcomePrices", "")
        yes_price = 0.5
        no_price = 0.5
        if outcome_prices:
            try:
                prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
                if len(prices) >= 2:
                    yes_price = float(prices[0])
                    no_price = float(prices[1])
            except (json.JSONDecodeError, ValueError):
                pass

        clob_token_ids = m.get("clobTokenIds", "")
        if isinstance(clob_token_ids, str):
            try:
                clob_token_ids = json.loads(clob_token_ids)
            except json.JSONDecodeError:
                clob_token_ids = []

        token_list = []
        outcomes = ["Yes", "No"]
        for i, tid in enumerate(clob_token_ids if isinstance(clob_token_ids, list) else []):
            token_list.append({
                "token_id": tid,
                "outcome": outcomes[i] if i < len(outcomes) else f"Outcome_{i}",
                "price": yes_price if i == 0 else no_price,
            })

        vol = float(m.get("volume", m.get("volumeNum", 0)) or 0)
        question = m.get("question", "")

        # Skip resolved or low-info markets.
        if yes_price in (0.0, 1.0) and vol == 0:
            return None

        condition_id = m.get("conditionId", m.get("condition_id", m.get("id", "")))
        if not condition_id:
            return None

        event = (m.get("events") or [{}])[0]
        event_id = str(event.get("id", "") or "")

        category = _infer_category(question, m.get("tags", None) or [])
        return Market(
            condition_id=condition_id,
            question=question,
            category=category,
            yes_price=yes_price,
            no_price=no_price,
            volume=vol,
            end_date=m.get("endDate", m.get("end_date_iso", "")),
            active=m.get("active", True),
            tokens=token_list,
            description=m.get("description", "") or "",
            spread=float(m.get("spread", 0) or 0),
            liquidity=float(m.get("liquidityNum", m.get("liquidity", 0)) or 0),
            slug=event.get("slug", "") or m.get("slug", ""),
            event_id=event_id,
            fee_rate=_fee_rate_for_category(category),
        )
    except (KeyError, ValueError, TypeError):
        return None


def _parse_markets(raw: list[dict]) -> list[Market]:
    """Parse a batch of raw market dicts, dropping any that don't convert."""
    return [m for m in (_parse_market(r) for r in raw) if m is not None]


async def _paginate_async(
    client: httpx.AsyncClient, extra_params: dict, max_items: int,
) -> list[dict]:
    """Page /markets with the given extra params (a tag_id and/or volume bounds)
    until max_items are collected, the per-stream offset cap is hit, or a short
    page signals the end. Returns raw market dicts."""
    items: list[dict] = []
    offset = 0
    while offset < _GAMMA_TAG_OFFSET_CAP and len(items) < max_items:
        params = {
            **_BASE_MARKET_PARAMS, **extra_params,
            "limit": _GAMMA_PAGE_SIZE, "offset": offset,
        }
        resp = await client.get(f"{GAMMA_API}/markets", params=params)
        resp.raise_for_status()
        data = resp.json()
        batch = data if isinstance(data, list) else data.get("data", [])
        if not batch:
            break
        items.extend(batch)
        offset += len(batch)
        if len(batch) < _GAMMA_PAGE_SIZE:
            break
    return items


async def _fetch_with_tag(
    client: httpx.AsyncClient, tag_id: int, max_items: int = _GAMMA_TAG_OFFSET_CAP,
) -> list[Market]:
    """All active markets carrying a given tag_id, paginated up to the offset cap."""
    return _parse_markets(await _paginate_async(client, {"tag_id": tag_id}, max_items))


async def _fetch_general(
    client: httpx.AsyncClient, max_items: int = _GAMMA_TAG_OFFSET_CAP,
) -> list[Market]:
    """High-volume markets across all categories (no tag filter), to catch big
    markets in categories outside the tracked tag list."""
    return _parse_markets(
        await _paginate_async(client, {"volume_num_min": _GENERAL_VOLUME_MIN}, max_items)
    )


async def _fetch_volume_range(
    client: httpx.AsyncClient, vmin: float, vmax: float,
    max_items: int = _GAMMA_TAG_OFFSET_CAP,
) -> list[Market]:
    """Markets within a [vmin, vmax] volume slice — the tags-unavailable fallback."""
    return _parse_markets(await _paginate_async(
        client, {"volume_num_min": vmin, "volume_num_max": vmax}, max_items
    ))


def _merge_dedupe_sort(
    results: list, min_volume: float | None, max_volume: float | None,
    limit: int | None,
) -> list[Market]:
    """Merge the per-stream results (some may be Exceptions, since the gather
    uses return_exceptions=True), dedupe by condition_id, apply optional volume
    bounds, sort by volume DESC, and truncate to limit."""
    markets: list[Market] = []
    seen: set[str] = set()
    for res in results:
        if isinstance(res, Exception):
            print(f"[markets] fetch stream failed: {res}")
            continue
        for m in res or []:
            if m.condition_id in seen:
                continue
            if min_volume is not None and m.volume < min_volume:
                continue
            if max_volume is not None and m.volume > max_volume:
                continue
            seen.add(m.condition_id)
            markets.append(m)
    markets.sort(key=lambda x: x.volume, reverse=True)
    if limit is not None:
        markets = markets[:limit]
    return markets


async def _fetch_active_markets_async(
    limit: int | None, min_volume: float | None, max_volume: float | None,
) -> list[Market]:
    """Async core of fetch_active_markets: fan out one stream per category
    tag_id plus a general high-volume stream, in parallel. Falls back to
    parallel volume-range slicing when the /tags endpoint is unavailable."""
    max_items = limit if limit is not None else _GAMMA_TAG_OFFSET_CAP

    try:
        tags = get_tags()
    except Exception as e:
        print(f"[markets] tags endpoint failed ({e}); falling back to volume ranges")
        tags = None

    async with httpx.AsyncClient(timeout=20) as client:
        if tags:
            tasks = [
                _fetch_with_tag(client, tags[slug], max_items)
                for slug in _TAG_CATEGORIES if slug in tags
            ]
            tasks.append(_fetch_general(client, max_items))
        else:
            tasks = [
                _fetch_volume_range(client, vmin, vmax, max_items)
                for vmin, vmax in _FALLBACK_VOLUME_RANGES
            ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    return _merge_dedupe_sort(results, min_volume, max_volume, limit)


def fetch_active_markets(
    limit: int | None = 50,
    min_volume: float | None = None,
    max_volume: float | None = None,
) -> list[Market]:
    """Fetch active, orderbook-enabled markets from Polymarket's Gamma API.

    A single volume-ordered stream can't page deep enough to reach niche markets
    before Gamma's offset wall, so we fan out one tag_id-filtered stream per
    category (plus a general high-volume stream) in parallel, dedupe by
    condition_id, and sort by volume DESC. When the /tags endpoint is down we
    fall back to parallel volume-range slices; if Gamma is unreachable entirely
    we fall back to the CLOB API.

    `limit` truncates the merged result (None = full niche scan). `min_volume`/
    `max_volume`, when given, filter the merged result client-side. Synchronous:
    it drives its own event loop, so it must be called from a non-async context
    (as all callers do — the CLI directly, market_watcher via run_in_executor).
    """
    try:
        markets = asyncio.run(_fetch_active_markets_async(limit, min_volume, max_volume))
    except Exception as e:
        print(f"[markets] Gamma parallel fetch error: {e}, falling back to CLOB...")
        return _fetch_from_clob(limit or _GAMMA_PAGE_SIZE)
    if not markets:
        print("[markets] Gamma returned no markets, falling back to CLOB...")
        return _fetch_from_clob(limit or _GAMMA_PAGE_SIZE)
    return markets


def fetch_slug_by_condition_id(condition_id: str) -> str:
    """Look up a market's event slug by condition_id (used to backfill positions).

    Returns "" if the market is unknown to Gamma or the request fails.
    """
    if not condition_id:
        return ""
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(f"{GAMMA_API}/markets", params={"condition_ids": condition_id})
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError):
        return ""
    batch = data if isinstance(data, list) else data.get("data", [])
    if not batch:
        return ""
    m = batch[0]
    return (m.get("events") or [{}])[0].get("slug", "") or m.get("slug", "")


def fetch_markets_by_condition_ids(condition_ids: list[str]) -> dict[str, dict]:
    """Batch-fetch current state for a list of condition_ids in as few Gamma
    requests as possible (one per 100-id chunk), instead of N single-market
    calls. Returns a mapping condition_id -> {condition_id, yes_price, closed,
    question}. condition_ids Gamma doesn't return are simply absent from the
    map; callers should treat a missing id as resolved/unknown.
    """
    ids = [c for c in dict.fromkeys(condition_ids) if c]  # dedupe, drop blanks
    if not ids:
        return {}

    out: dict[str, dict] = {}
    try:
        with httpx.Client(timeout=15) as client:
            for start in range(0, len(ids), _GAMMA_PAGE_SIZE):
                chunk = ids[start:start + _GAMMA_PAGE_SIZE]
                resp = client.get(
                    f"{GAMMA_API}/markets",
                    params={"condition_ids": chunk, "limit": len(chunk)},
                )
                resp.raise_for_status()
                data = resp.json()
                batch = data if isinstance(data, list) else data.get("data", [])
                for m in batch:
                    cid = m.get("conditionId", m.get("condition_id", "")) or ""
                    if not cid:
                        continue
                    out[cid] = {
                        "condition_id": cid,
                        "yes_price": _parse_yes_price(m),
                        "closed": bool(m.get("closed", False)),
                        "question": m.get("question", ""),
                        "slug": (m.get("events") or [{}])[0].get("slug", "") or m.get("slug", ""),
                    }
    except (httpx.HTTPError, ValueError):
        # Network/parse failure: return whatever we collected. Missing ids are
        # treated as resolved/unknown by the caller, so a partial result is safe.
        pass
    return out


def _parse_yes_price(m: dict) -> float | None:
    """Extract the YES price from a Gamma market dict (outcomePrices is a JSON
    string of [yes, no]). Returns None when it can't be parsed."""
    outcome_prices = m.get("outcomePrices", "")
    if not outcome_prices:
        return None
    import json
    try:
        prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
        if prices and len(prices) >= 1:
            return float(prices[0])
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    return None


def _fetch_from_clob(limit: int) -> list[Market]:
    """Fallback: fetch from CLOB API directly."""
    markets = []
    try:
        resp = httpx.get(
            f"{config.POLYMARKET_HOST}/markets",
            params={"limit": limit, "active": True},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[markets] CLOB API error: {e}")
        return markets

    items = data if isinstance(data, list) else data.get("data", data.get("markets", []))

    for m in items:
        try:
            tokens = m.get("tokens", [])
            yes_price = 0.5
            no_price = 0.5
            for t in tokens:
                outcome = t.get("outcome", "").lower()
                price = float(t.get("price", 0.5))
                if outcome == "yes":
                    yes_price = price
                elif outcome == "no":
                    no_price = price

            category = _infer_category(m.get("question", ""), m.get("tags") or [])
            markets.append(Market(
                condition_id=m.get("condition_id", m.get("id", "")),
                question=m.get("question", ""),
                category=category,
                yes_price=yes_price,
                no_price=no_price,
                volume=float(m.get("volume", 0)),
                end_date=m.get("end_date_iso", m.get("end_date", "")),
                active=m.get("active", True),
                tokens=tokens,
                fee_rate=_fee_rate_for_category(category),
            ))
        except (KeyError, ValueError):
            continue

    return markets


# Order matters: first matching category wins. Keywords are matched on word
# boundaries (with optional plural "s") — substring matching mis-tagged e.g.
# "jail"/"chair"/"raise" as "ai". Longer forms ("technology", "cryptocurrency")
# are listed explicitly since the short forms no longer match inside them.
_CATEGORY_KEYWORDS = [
    ("ai", ["ai", "artificial intelligence", "openai", "chatgpt", "llm", "google ai", "anthropic"]),
    ("crypto", ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "crypto", "cryptocurrency",
                "blockchain", "defi", "nft", "token", "coinbase", "binance", "stablecoin", "usdc",
                "usdt", "web3", "altcoin", "memecoin", "polymarket"]),
    ("politics", ["election", "president", "congress", "senate", "trump", "biden", "political"]),
    ("technology", ["tech", "technology", "apple", "google", "microsoft", "software", "startup"]),
    ("sports", [
        # NFL/NBA/NHL/MLB
        "nfl", "nba", "nhl", "mlb", "quarterback", "touchdown", "super bowl",
        "basketball", "hockey", "baseball", "stanley cup", "world series",
        # Football (soccer)
        "premier league", "champions league", "la liga", "bundesliga", "serie a",
        "fifa", "soccer", "arsenal", "chelsea", "barcelona", "real madrid",
        "manchester", "liverpool", "goal", "transfer",
        # Tennis
        "tennis", "wimbledon", "us open", "french open", "australian open",
        "atp", "wta", "grand slam",
    ]),
]
_CATEGORY_PATTERNS = [
    (cat, re.compile(r"\b(?:" + "|".join(re.escape(kw) + r"s?" for kw in kws) + r")\b"))
    for cat, kws in _CATEGORY_KEYWORDS
]


def _infer_category(question: str, tags: list) -> str:
    """Infer category from question text and tags."""
    tag_str = " ".join(str(t).lower() for t in tags)
    combined = f"{question.lower()} {tag_str}"

    for cat, pattern in _CATEGORY_PATTERNS:
        if pattern.search(combined):
            return cat
    return "other"


def _fee_rate_for_category(category: str) -> float:
    """Per-share Polymarket fee rate for an inferred category. Falls back to the
    'other' rate for any category without an explicit entry. See config.FEE_RATES
    and edge.detect_edge_v2 for how the fee is applied (feeRate * p * (1 - p))."""
    rates = config.FEE_RATES
    return rates.get(category, rates.get("other", 0.05))


def filter_by_categories(markets: list[Market], categories: list[str] | None = None) -> list[Market]:
    """Filter markets to only target categories."""
    cats = categories or config.MARKET_CATEGORIES
    return [m for m in markets if m.category in cats]


def get_token_id(market: Market, side: str) -> str | None:
    """Get the token ID for a given side (YES/NO)."""
    for t in market.tokens:
        if t.get("outcome", "").upper() == side.upper():
            return t.get("token_id")
    return None


if __name__ == "__main__":
    all_markets = fetch_active_markets(limit=20)
    filtered = filter_by_categories(all_markets)
    print(f"\n--- {len(filtered)} markets in target categories (of {len(all_markets)} total) ---\n")
    for m in filtered[:15]:
        print(f"  [{m.category}] {m.question}")
        print(f"    YES: {m.yes_price:.2f} | NO: {m.no_price:.2f} | Vol: ${m.volume:,.0f}")
        print()
