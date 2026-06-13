from __future__ import annotations

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

    @property
    def implied_probability(self) -> float:
        return self.yes_price


# Gamma caps page size at 100 regardless of the requested limit.
_GAMMA_PAGE_SIZE = 100
# Gamma rejects offsets past 10,000 with a 422; beyond that we roll the
# volume cursor (see fetch_active_markets) instead of paging deeper.
_GAMMA_OFFSET_CAP = 10_000
# Safety cap on total requests per fetch.
_GAMMA_MAX_PAGES = 200


def fetch_active_markets(
    limit: int | None = 50,
    min_volume: float | None = None,
    max_volume: float | None = None,
) -> list[Market]:
    """
    Fetch active, orderbook-enabled markets from Polymarket's Gamma API.

    Paginates until `limit` markets are collected, or until the API is
    exhausted when limit is None. Volume bounds are applied Gamma-side
    (volume_num_min/max) so a full scan of the niche band stays cheap.

    Gamma rejects offsets past 10k, so on hitting that wall we restart at
    offset 0 with volume_num_max lowered to the smallest volume seen so far
    (results are volume-ordered; the one-row overlap is deduped below).
    """
    params = {
        "active": True,
        "closed": False,
        "enableOrderBook": True,
        "order": "volume",
        "ascending": False,
    }
    if min_volume is not None:
        params["volume_num_min"] = min_volume

    items = []
    offset = 0
    volume_ceiling = max_volume
    lowest_volume_seen = None
    try:
        with httpx.Client(timeout=15) as client:
            for _ in range(_GAMMA_MAX_PAGES):
                page_size = _GAMMA_PAGE_SIZE
                if limit is not None:
                    page_size = min(page_size, limit - len(items))
                    if page_size <= 0:
                        break

                if offset + page_size > _GAMMA_OFFSET_CAP:
                    # Roll the volume cursor instead of paging past the cap.
                    if lowest_volume_seen is None or lowest_volume_seen == volume_ceiling:
                        break
                    volume_ceiling = lowest_volume_seen
                    offset = 0

                page_params = {**params, "limit": page_size, "offset": offset}
                if volume_ceiling is not None:
                    page_params["volume_num_max"] = volume_ceiling

                resp = client.get(f"{GAMMA_API}/markets", params=page_params)
                resp.raise_for_status()
                data = resp.json()
                batch = data if isinstance(data, list) else data.get("data", [])
                if not batch:
                    break
                items.extend(batch)
                offset += len(batch)

                page_volumes = [
                    float(m.get("volumeNum", m.get("volume", 0)) or 0) for m in batch
                ]
                if page_volumes:
                    page_min = min(page_volumes)
                    if lowest_volume_seen is None or page_min < lowest_volume_seen:
                        lowest_volume_seen = page_min
    except Exception as e:
        if not items:
            print(f"[markets] Gamma API error: {e}, falling back to CLOB...")
            return _fetch_from_clob(limit or _GAMMA_PAGE_SIZE)
        # Mid-pagination failure: keep what we have rather than losing the refresh.
        print(f"[markets] Gamma pagination stopped early at offset {offset}: {e}")

    markets = []
    seen_ids = set()

    for m in items:
        try:
            # Gamma API uses outcomePrices as a JSON string
            outcome_prices = m.get("outcomePrices", "")
            yes_price = 0.5
            no_price = 0.5

            if outcome_prices:
                import json
                try:
                    prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
                    if len(prices) >= 2:
                        yes_price = float(prices[0])
                        no_price = float(prices[1])
                except (json.JSONDecodeError, ValueError):
                    pass

            # Also check tokens array
            tokens = m.get("tokens", m.get("clobTokenIds", []))
            if isinstance(tokens, str):
                import json
                try:
                    tokens = json.loads(tokens)
                except json.JSONDecodeError:
                    tokens = []

            # Build token list for order execution
            clob_token_ids = m.get("clobTokenIds", "")
            if isinstance(clob_token_ids, str):
                import json
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

            # Skip resolved or low-info markets
            if yes_price in (0.0, 1.0) and vol == 0:
                continue

            condition_id = m.get("conditionId", m.get("condition_id", m.get("id", "")))
            # Pagination ordered by live volume can shift between pages; dedupe.
            if condition_id in seen_ids:
                continue
            seen_ids.add(condition_id)

            markets.append(Market(
                condition_id=condition_id,
                question=question,
                category=_infer_category(question, m.get("tags", None) or []),
                yes_price=yes_price,
                no_price=no_price,
                volume=vol,
                end_date=m.get("endDate", m.get("end_date_iso", "")),
                active=m.get("active", True),
                tokens=token_list,
                description=m.get("description", "") or "",
                spread=float(m.get("spread", 0) or 0),
                liquidity=float(m.get("liquidityNum", m.get("liquidity", 0)) or 0),
                slug=(m.get("events") or [{}])[0].get("slug", "") or m.get("slug", ""),
            ))
        except (KeyError, ValueError, TypeError):
            continue

    # Sort by volume descending
    markets.sort(key=lambda x: x.volume, reverse=True)
    return markets


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

            markets.append(Market(
                condition_id=m.get("condition_id", m.get("id", "")),
                question=m.get("question", ""),
                category=_infer_category(m.get("question", ""), m.get("tags") or []),
                yes_price=yes_price,
                no_price=no_price,
                volume=float(m.get("volume", 0)),
                end_date=m.get("end_date_iso", m.get("end_date", "")),
                active=m.get("active", True),
                tokens=tokens,
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
