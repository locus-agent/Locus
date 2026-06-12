"""
News-to-market matching — routes breaking news to relevant active markets.
Two strategies: fast keyword overlap (no model, no API) and semantic search
over the embedded market index (market_index.py); the V2 pipeline unions both.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from locus import config
from locus.markets.gamma import Market

if TYPE_CHECKING:
    from locus.core.market_index import MarketIndex

log = logging.getLogger(__name__)


def extract_keywords(question: str) -> list[str]:
    """Extract meaningful keywords from a market question."""
    stopwords = {
        "will", "the", "a", "an", "be", "by", "in", "on", "at", "to",
        "of", "for", "is", "it", "this", "that", "and", "or", "not",
        "before", "after", "end", "yes", "no", "any", "has", "have",
        "does", "do", "than", "more", "less", "over", "under", "above",
        "below", "through", "during", "between", "reach", "exceed",
    }
    words = question.lower().split()
    keywords = [
        w.strip("?.,!\"'()[]")
        for w in words
        if w.strip("?.,!\"'()[]") not in stopwords and len(w.strip("?.,!\"'()[]")) > 2
    ]
    return keywords


_TOKEN_STRIP = "?.,!\"'()[]"


def tokenize(text: str) -> set[str]:
    """Lowercased word tokens, stripped of edge punctuation."""
    return {
        stripped
        for w in text.lower().split()
        if (stripped := w.strip(_TOKEN_STRIP))
    }


def match_news_to_markets(
    headline: str,
    markets: list[Market],
    max_matches: int = 5,
) -> list[Market]:
    """
    Find markets that a news headline is relevant to.
    Uses whole-token keyword overlap scoring — fast, no API call.
    (Substring matching let "cap" hit "capital" and "ai" hit "raise",
    burning classify calls on junk matches.)
    """
    headline_tokens = tokenize(headline)
    scored = []

    for market in markets:
        keywords = extract_keywords(market.question)
        if not keywords:
            continue

        # Count whole-token keyword hits
        hits = sum(1 for kw in keywords if kw in headline_tokens)
        if hits == 0:
            continue

        # Score = hits / total keywords (relevance ratio)
        score = hits / len(keywords)
        scored.append((score, market))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:max_matches]]


def match_news_to_markets_hybrid(
    headline: str,
    markets: list[Market],
    index: "MarketIndex | None" = None,
    max_matches: int = 5,
) -> list[tuple[Market, str]]:
    """
    Union of keyword and embedding matches, deduped by condition_id.
    Returns (market, match_source) pairs where match_source is
    "keyword", "embedding", or "both". Falls back to keyword-only while
    the embedding index is cold (or absent). Blocking (~50ms when warm) —
    the async pipeline calls this in an executor.
    """
    keyword_matches = match_news_to_markets(headline, markets, max_matches)
    sources: dict[str, str] = {m.condition_id: "keyword" for m in keyword_matches}
    matched: dict[str, Market] = {m.condition_id: m for m in keyword_matches}

    if index is not None and not index.ready:
        # First call after process start: load the persisted index now (we're
        # already off the event loop) rather than miss the startup news burst.
        index.warm()

    if index is not None and index.ready:
        by_id = {m.condition_id: m for m in markets}
        extras = 0
        hits = index.search(headline)
        for cid, _dist in sorted(hits.items(), key=lambda kv: kv[1]):
            market = by_id.get(cid)
            if market is None:
                continue  # index entry no longer tracked; next sync removes it
            if cid in sources:
                sources[cid] = "both"
            elif extras < config.EMBED_MAX_EXTRA_MATCHES:
                sources[cid] = "embedding"
                matched[cid] = market
                extras += 1

    return [(market, sources[cid]) for cid, market in matched.items()]


def match_news_to_markets_broad(
    headline: str,
    summary: str,
    markets: list[Market],
    max_matches: int = 5,
) -> list[Market]:
    """
    Broader matching using headline + summary text.
    Falls back to category matching if keyword matching returns nothing.
    """
    # Try keyword matching first
    matches = match_news_to_markets(headline, markets, max_matches)
    if matches:
        return matches

    # Fallback: match on category keywords in the headline
    combined = f"{headline} {summary}".lower()
    category_keywords = {
        "ai": ["ai", "openai", "gpt", "anthropic", "claude", "llm", "chatgpt", "gemini", "artificial intelligence"],
        "crypto": ["bitcoin", "ethereum", "solana", "crypto", "blockchain", "defi", "token", "btc", "eth"],
        "politics": ["trump", "biden", "congress", "senate", "election", "tariff", "fed", "white house"],
        "technology": ["apple", "google", "microsoft", "nvidia", "tech", "software", "startup"],
        "science": ["spacex", "nasa", "climate", "research", "discovery"],
    }

    matched_categories = set()
    for cat, kws in category_keywords.items():
        if any(kw in combined for kw in kws):
            matched_categories.add(cat)

    if not matched_categories:
        return []

    # Return markets in matching categories
    category_matches = [m for m in markets if m.category in matched_categories]
    return category_matches[:max_matches]


if __name__ == "__main__":
    from locus.markets.gamma import fetch_active_markets, filter_by_categories
    from locus import config

    print("Fetching markets...")
    all_m = fetch_active_markets(limit=100)
    filtered = filter_by_categories(all_m)
    niche = [m for m in filtered if config.MIN_VOLUME_USD <= m.volume <= config.MAX_VOLUME_USD]
    print(f"Niche markets: {len(niche)}")

    test_headlines = [
        "OpenAI reportedly testing GPT-5 internally with select partners",
        "Bitcoin ETF inflows hit $2.1B in single week",
        "Fed minutes signal growing consensus for summer rate cut",
    ]

    for h in test_headlines:
        matches = match_news_to_markets(h, niche)
        print(f"\n\"{h[:60]}...\"")
        print(f"  Matched {len(matches)} markets:")
        for m in matches:
            print(f"    [{m.category}] {m.question[:50]}")
