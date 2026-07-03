"""
Real-time news monitor — event-driven architecture.
Sources: Twitter API v2 filtered stream, Telegram channels, RSS fallback.
Emits NewsEvent objects into an asyncio queue as breaking news arrives.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
import logging
from collections import deque
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

import httpx
import numpy as np

from locus import config
from locus.supervisor import supervise
from locus.sources.scraper import scrape_all, scrape_newsapi, scrape_newsapi_top_headlines, NewsItem

log = logging.getLogger(__name__)

# Same local embedding model the matcher/market index use.
DEDUP_MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticDeduper:
    """Near-duplicate headline filter using MiniLM cosine similarity.

    Keeps the last `maxlen` headline embeddings with timestamps (a deque). A new
    headline is a duplicate when its cosine similarity to any embedding seen
    within the last `ttl_hours` exceeds `threshold` — catching reworded versions
    of the same story that a hash check misses. An exact-hash fast path skips the
    embedding for byte-identical repeats.

    Fails OPEN: if the model can't load/encode, the headline is forwarded rather
    than dropped (better a possible dup than a lost scoop). The model loads
    lazily on first use so importing this module stays light.
    """

    def __init__(self, threshold: float, maxlen: int = 500, ttl_hours: float = 2.0):
        self.threshold = threshold
        self.ttl = timedelta(hours=ttl_hours)
        self._recent: deque = deque(maxlen=maxlen)  # {embedding, timestamp, short_hash, headline}
        self._model = None

    def _encode(self, text: str):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(DEDUP_MODEL_NAME)
        return self._model.encode([text], normalize_embeddings=True)[0]

    @staticmethod
    def _short_hash(headline: str) -> str:
        return hashlib.sha1(headline.lower().strip().encode()).hexdigest()[:16]

    def check(self, headline: str, now: datetime | None = None):
        """Return (matched_entry, cosine) when `headline` duplicates a recent one,
        else None (and record it as freshly seen)."""
        now = now or datetime.now(timezone.utc)
        cutoff = now - self.ttl
        # Drop expired entries from the front (the deque is time-ordered).
        while self._recent and self._recent[0]["timestamp"] < cutoff:
            self._recent.popleft()

        short_hash = self._short_hash(headline)
        for entry in self._recent:
            if entry["short_hash"] == short_hash:
                return entry, 1.0  # exact repeat — no embedding needed

        try:
            emb = self._encode(headline)
        except Exception as e:
            log.debug(f"[dedup] encode failed, forwarding headline: {e}")
            return None  # fail open

        for entry in self._recent:
            sim = float(np.dot(emb, entry["embedding"]))
            if sim > self.threshold:
                return entry, sim

        self._recent.append({
            "embedding": emb,
            "timestamp": now,
            "short_hash": short_hash,
            "headline": headline,
        })
        return None


# Lazy module-level model for headline_similarity (shared across calls; the
# same MiniLM the deduper uses, loaded on first use so imports stay light).
_similarity_model = None


def headline_similarity(a: str, b: str) -> float:
    """Cosine similarity of two headlines under the dedup embedding model.

    Backs the pipeline's per-market classification cooldown: a pair scoring
    above config.DEDUP_COSINE_THRESHOLD is the same story reworded. Exact
    (case/whitespace-insensitive) matches short-circuit to 1.0 without touching
    the model. Fails OPEN like SemanticDeduper — an encode failure returns 0.0,
    so callers treat the pair as different and classify rather than skip."""
    if a.lower().strip() == b.lower().strip():
        return 1.0
    global _similarity_model
    try:
        if _similarity_model is None:
            from sentence_transformers import SentenceTransformer
            _similarity_model = SentenceTransformer(DEDUP_MODEL_NAME)
        emb = _similarity_model.encode([a, b], normalize_embeddings=True)
        return float(np.dot(emb[0], emb[1]))
    except Exception as e:
        log.debug(f"[dedup] similarity failed, treating as different: {e}")
        return 0.0


@dataclass
class NewsEvent:
    headline: str
    source: str  # "twitter", "telegram", "rss"
    url: str
    received_at: datetime
    published_at: datetime
    summary: str = ""
    raw_data: dict = field(default_factory=dict)
    latency_ms: int = 0  # publication -> receipt; -1 = publication time unknown

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.received_at).total_seconds()


class RecentKeys:
    """Bounded recently-seen set with insertion-order (FIFO) eviction.

    The old pattern — set(list(seen)[-N:]) — relied on set order, which is
    arbitrary: trimming could evict the *newest* keys and let recently seen
    headlines through dedup again. dict preserves insertion order.
    """

    def __init__(self, max_size: int = 10000, keep: int = 5000):
        self._keys: dict[str, None] = {}
        self._max_size = max_size
        self._keep = keep

    def seen(self, key: str) -> bool:
        """True if key was already recorded; records it otherwise."""
        if key in self._keys:
            return True
        self._keys[key] = None
        if len(self._keys) > self._max_size:
            self._keys = dict.fromkeys(list(self._keys)[-self._keep:])
        return False

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def __len__(self) -> int:
        return len(self._keys)


class TwitterStream:
    """Twitter API v2 filtered stream for real-time keyword monitoring."""

    def __init__(self, bearer_token: str, keywords: list[str]):
        self.bearer_token = bearer_token
        self.keywords = keywords
        self.base_url = "https://api.twitter.com/2"
        self.enabled = bool(bearer_token)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.bearer_token}"}

    async def setup_rules(self):
        """Set up filtered stream rules based on keywords."""
        if not self.enabled:
            return

        async with httpx.AsyncClient() as client:
            # Get existing rules
            resp = await client.get(
                f"{self.base_url}/tweets/search/stream/rules",
                headers=self._headers(),
                timeout=10,
            )
            existing = resp.json().get("data", [])

            # Delete existing rules
            if existing:
                ids = [r["id"] for r in existing]
                await client.post(
                    f"{self.base_url}/tweets/search/stream/rules",
                    headers=self._headers(),
                    json={"delete": {"ids": ids}},
                    timeout=10,
                )

            # Create new rules from keywords. Basic tier allows 5 rules of up
            # to 512 chars each, so batch generously to fit all keywords —
            # batch_size=5 overflowed past the rules[:5] cap and silently
            # dropped keywords once the list grew beyond 25.
            rules = []
            # Batch keywords into OR groups
            batch_size = 8
            for i in range(0, len(self.keywords), batch_size):
                batch = self.keywords[i:i + batch_size]
                value = " OR ".join(f'"{kw}"' for kw in batch)
                rules.append({"value": value, "tag": f"batch_{i // batch_size}"})

            if rules:
                await client.post(
                    f"{self.base_url}/tweets/search/stream/rules",
                    headers=self._headers(),
                    json={"add": rules[:5]},  # Basic tier: 5 rules max
                    timeout=10,
                )

    async def stream(self, queue: asyncio.Queue):
        """Connect to filtered stream and emit NewsEvents."""
        if not self.enabled:
            log.info("[twitter] No bearer token — stream disabled")
            return

        try:
            await self.setup_rules()
        except Exception as e:
            log.warning(f"[twitter] Failed to setup rules: {e}")
            return

        backoff = 1
        while True:
            try:
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "GET",
                        f"{self.base_url}/tweets/search/stream",
                        headers=self._headers(),
                        params={"tweet.fields": "created_at,author_id,text"},
                        timeout=None,
                    ) as resp:
                        backoff = 1
                        async for line in resp.aiter_lines():
                            if not line.strip():
                                continue
                            try:
                                import json
                                data = json.loads(line)
                                tweet = data.get("data", {})
                                text = tweet.get("text", "")
                                created = tweet.get("created_at", "")

                                now = datetime.now(timezone.utc)
                                try:
                                    pub = datetime.fromisoformat(created.replace("Z", "+00:00"))
                                    latency = int((now - pub).total_seconds() * 1000)
                                except (ValueError, AttributeError):
                                    pub = now
                                    latency = -1  # unknown publication time

                                event = NewsEvent(
                                    headline=text[:280],
                                    source="twitter",
                                    url=f"https://twitter.com/i/status/{tweet.get('id', '')}",
                                    received_at=now,
                                    published_at=pub,
                                    latency_ms=latency,
                                    raw_data=data,
                                )
                                await queue.put(event)
                            except Exception as e:
                                log.debug(f"[twitter] Parse error: {e}")

            except (httpx.HTTPError, Exception) as e:
                log.warning(f"[twitter] Stream error: {e}, reconnecting in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)


class TelegramMonitor:
    """Monitor Telegram channels via Bot API long polling."""

    def __init__(self, bot_token: str, channel_ids: list[str]):
        self.bot_token = bot_token
        self.channel_ids = channel_ids
        self.enabled = bool(bot_token) and bool(channel_ids)
        self.last_update_id = 0

    async def stream(self, queue: asyncio.Queue):
        """Poll for new messages and emit NewsEvents."""
        if not self.enabled:
            log.info("[telegram] No bot token or channels — monitor disabled")
            return

        base_url = f"https://api.telegram.org/bot{self.bot_token}"

        while True:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"{base_url}/getUpdates",
                        params={"offset": self.last_update_id + 1, "timeout": 30},
                        timeout=35,
                    )
                    data = resp.json()

                for update in data.get("result", []):
                    self.last_update_id = update["update_id"]
                    msg = update.get("channel_post") or update.get("message", {})
                    text = msg.get("text", "")
                    chat_id = str(msg.get("chat", {}).get("id", ""))

                    if not text or (self.channel_ids and chat_id not in self.channel_ids):
                        continue

                    now = datetime.now(timezone.utc)
                    msg_date = msg.get("date", 0)
                    pub = datetime.fromtimestamp(msg_date, tz=timezone.utc) if msg_date else now
                    latency = int((now - pub).total_seconds() * 1000) if msg_date else -1

                    event = NewsEvent(
                        headline=text[:500],
                        source="telegram",
                        url="",
                        received_at=now,
                        published_at=pub,
                        latency_ms=latency,
                        raw_data=update,
                    )
                    await queue.put(event)

            except Exception as e:
                # httpx errors embed the request URL, which contains the bot token
                msg = str(e).replace(self.bot_token, "***") if self.bot_token else str(e)
                log.warning(f"[telegram] Error: {msg}")
                await asyncio.sleep(5)


class RSSFallback:
    """Periodic RSS scraping as a fallback news source."""

    def __init__(self, interval_seconds: float = 120):
        self.interval = interval_seconds
        self._seen_headlines = RecentKeys(max_size=5000, keep=2000)

    async def stream(self, queue: asyncio.Queue):
        """Poll RSS feeds periodically and emit new headlines."""
        while True:
            try:
                items = await asyncio.get_event_loop().run_in_executor(
                    None, scrape_all
                )
                now = datetime.now(timezone.utc)
                new_count = 0

                for item in items:
                    key = item.headline.lower()[:80]
                    if self._seen_headlines.seen(key):
                        continue
                    new_count += 1

                    latency = (
                        int((now - item.published_at).total_seconds() * 1000)
                        if item.date_known else -1
                    )

                    # Truth Social is tagged as its own fast source upstream
                    # (scraper.scrape_rss); every other feed flattens to "rss".
                    event = NewsEvent(
                        headline=item.headline,
                        source="truthsocial" if item.source == "truthsocial" else "rss",
                        url=item.url,
                        received_at=now,
                        published_at=item.published_at,
                        summary=item.summary,
                        latency_ms=latency,
                    )
                    await queue.put(event)

                if new_count:
                    log.info(f"[rss] {new_count} new headlines")



            except Exception as e:
                log.warning(f"[rss] Error: {e}")

            await asyncio.sleep(self.interval)


class NewsAPISource:
    """Periodic NewsAPI.org polling — top headlines plus per-category searches.

    The free "Developer" plan allows config.NEWSAPI_DAILY_LIMIT requests/day
    across all endpoints, so requests are spread evenly across the day
    (interval = 24h / daily_limit) and round-robin through one spec per poll.
    scraper._newsapi_budget_ok() is the hard backstop against exceeding the
    daily cap.
    """

    def __init__(self, api_key: str, daily_limit: int = 100):
        self.enabled = bool(api_key)
        self.interval = max(60.0, 86400.0 / max(daily_limit, 1))
        self._seen_headlines = RecentKeys(max_size=5000, keep=2000)
        self._specs = (
            [("everything", q) for q in config.NEWSAPI_CATEGORY_QUERIES.values()]
            + [("top-headlines", c) for c in config.NEWSAPI_TOP_HEADLINE_CATEGORIES]
        )
        self._spec_idx = 0

    async def stream(self, queue: asyncio.Queue):
        """Poll NewsAPI periodically and emit new headlines."""
        if not self.enabled:
            log.info("[newsapi] No API key — source disabled")
            return

        while True:
            kind, arg = self._specs[self._spec_idx % len(self._specs)]
            self._spec_idx += 1

            try:
                if kind == "everything":
                    items = await asyncio.get_event_loop().run_in_executor(
                        None, scrape_newsapi, arg, config.NEWS_LOOKBACK_HOURS
                    )
                else:
                    items = await asyncio.get_event_loop().run_in_executor(
                        None, scrape_newsapi_top_headlines, arg
                    )

                now = datetime.now(timezone.utc)
                new_count = 0

                for item in items:
                    key = item.headline.lower()[:80]
                    if self._seen_headlines.seen(key):
                        continue
                    new_count += 1

                    latency = (
                        int((now - item.published_at).total_seconds() * 1000)
                        if item.date_known else -1
                    )

                    event = NewsEvent(
                        headline=item.headline,
                        source="newsapi",
                        url=item.url,
                        received_at=now,
                        published_at=item.published_at,
                        summary=item.summary,
                        latency_ms=latency,
                    )
                    await queue.put(event)

                if new_count:
                    log.info(f"[newsapi] {kind}:{arg} -> {new_count} new headlines")



            except Exception as e:
                log.warning(f"[newsapi] Error fetching {kind}:{arg}: {e}")

            await asyncio.sleep(self.interval)


class NewsAggregator:
    """Runs all news sources concurrently, deduplicates, emits to output queue."""

    def __init__(self, output_queue: asyncio.Queue):
        self.output_queue = output_queue
        self._internal_queue: asyncio.Queue = asyncio.Queue()
        # Cross-source dedup: semantic (cosine) rather than hash-only, so a story
        # reworded by a second outlet is still caught.
        self._deduper = SemanticDeduper(config.DEDUP_COSINE_THRESHOLD)

        self.twitter = TwitterStream(config.TWITTER_BEARER_TOKEN, config.TWITTER_KEYWORDS)
        self.telegram = TelegramMonitor(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHANNEL_IDS)
        self.rss = RSSFallback(interval_seconds=120)
        self.newsapi = NewsAPISource(config.NEWSAPI_KEY, config.NEWSAPI_DAILY_LIMIT)

        self.stats = {"twitter": 0, "telegram": 0, "rss": 0, "newsapi": 0, "total": 0, "deduped": 0}

    async def run(self):
        """Start all sources and the dedup router."""
        await asyncio.gather(
            supervise("twitter_stream", lambda: self.twitter.stream(self._internal_queue), self.stats),
            supervise("telegram_monitor", lambda: self.telegram.stream(self._internal_queue), self.stats),
            supervise("rss_fallback", lambda: self.rss.stream(self._internal_queue), self.stats),
            supervise("newsapi_source", lambda: self.newsapi.stream(self._internal_queue), self.stats),
            supervise("dedup_router", self._dedup_router, self.stats),
        )

    async def _dedup_router(self):
        """Deduplicate (semantic cosine) and forward events to the output queue.
        Embedding blocks briefly, so it runs off the event loop."""
        while True:
            event = await self._internal_queue.get()
            dup = await asyncio.get_event_loop().run_in_executor(
                None, self._deduper.check, event.headline
            )
            if dup is not None:
                entry, sim = dup
                self.stats["deduped"] += 1
                log.info(
                    f"Dedup blocked: similar to '{entry['headline'][:60]}' (cosine {sim:.2f})"
                )
                continue

            self.stats[event.source] = self.stats.get(event.source, 0) + 1
            self.stats["total"] += 1

            await self.output_queue.put(event)


if __name__ == "__main__":
    async def _test():
        q: asyncio.Queue = asyncio.Queue()
        agg = NewsAggregator(q)

        async def printer():
            while True:
                event = await q.get()
                print(f"[{event.source}] ({event.latency_ms}ms) {event.headline[:80]}")

        await asyncio.gather(agg.run(), printer())

    asyncio.run(_test())
