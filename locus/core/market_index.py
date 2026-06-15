"""
Semantic market index — a persistent Chroma collection of tracked markets,
embedded locally with sentence-transformers (all-MiniLM-L6-v2). No API calls,
no keys. market_watcher keeps it in sync; matcher queries it per headline.

Heavy imports (chromadb, torch) are deferred until first use so that modules
importing this one stay fast, and sync()/search() are blocking by design —
callers run them off the event loop.
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict

from locus import config
from locus.markets.gamma import Market

log = logging.getLogger(__name__)

CHROMA_PATH = str(config.PROJECT_ROOT / "chroma_db")
COLLECTION = "markets"
MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_BATCH = 128


class LRUCache:
    """Minimal LRU cache (OrderedDict-backed) — fronts the Chroma disk lookup
    for per-market embeddings. No third-party dependency; tracks hits/misses so
    the cache's effectiveness is observable. get() returns None on a miss."""

    def __init__(self, maxsize: int = 1000):
        self.maxsize = max(1, maxsize)
        self._data: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self._data:
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]
        self.misses += 1
        return None

    def put(self, key, value) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)  # evict least-recently-used

    def __contains__(self, key) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)


def _doc_text(market: Market) -> str:
    """Text embedded per market: question + truncated description."""
    return f"{market.question}\n{(market.description or '')[:500]}"


def _doc_hash(doc: str) -> str:
    return hashlib.sha1(doc.encode()).hexdigest()


class MarketIndex:
    def __init__(self, path: str = CHROMA_PATH):
        self._path = path
        self._model = None
        self._client = None
        self._collection = None
        # In-memory LRU of {condition_id: embedding}, fronting Chroma's disk
        # lookup for per-market embeddings (pre-warmed at startup).
        self._embed_cache = LRUCache(config.EMBEDDING_CACHE_SIZE)
        # True once the collection is queryable (persisted from a previous run,
        # or after the first sync). Until then, matching falls back to keywords.
        self.ready = False

    def _ensure_loaded(self):
        if self._collection is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=self._path)
            self._collection = self._client.get_or_create_collection(
                COLLECTION, metadata={"hnsw:space": "cosine"}
            )
            if self._collection.count() > 0:
                self.ready = True
                log.info(
                    f"[index] Loaded persisted index: {self._collection.count()} markets"
                )
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            t0 = time.monotonic()
            self._model = SentenceTransformer(MODEL_NAME)
            log.info(f"[index] Loaded {MODEL_NAME} in {time.monotonic() - t0:.1f}s")

    def warm(self) -> None:
        """Load the model and persisted collection (sets ready if data exists).
        Blocking — run in an executor at startup so the first news burst can
        use the previous run's index while the fresh sync is still underway."""
        try:
            self._ensure_loaded()
        except Exception as e:
            log.warning(f"[index] Warm-up failed: {e}")

    def pre_warm(self) -> None:
        """Load every persisted market embedding into the in-memory LRU cache so
        the first headlines don't pay a cold Chroma read per market. Blocking —
        run in an executor at startup. Safe no-op when the collection is empty or
        unavailable."""
        try:
            self._ensure_loaded()
            if self._collection is None or self._collection.count() == 0:
                return
            data = self._collection.get(include=["embeddings"])
            ids = data.get("ids") or []
            embeddings = data.get("embeddings") or []
            n = 0
            for cid, emb in zip(ids, embeddings):
                if emb is not None:
                    self._embed_cache.put(cid, emb)
                    n += 1
            log.info(f"[index] Pre-warmed {n} market embeddings")
        except Exception as e:
            log.warning(f"[index] Pre-warm failed: {e}")

    def get_market_embedding(self, condition_id: str):
        """Embedding vector for one market, cache-first. Returns the cached
        vector on a hit (skipping the Chroma disk lookup), else reads it from
        Chroma, caches it, and returns it. None if the market isn't indexed."""
        cached = self._embed_cache.get(condition_id)
        if cached is not None:
            return cached
        self._ensure_loaded()
        try:
            data = self._collection.get(ids=[condition_id], include=["embeddings"])
        except Exception as e:
            log.debug(f"[index] embedding lookup failed for {condition_id[:16]}: {e}")
            return None
        embeddings = data.get("embeddings") or []
        if not embeddings or embeddings[0] is None:
            return None
        self._embed_cache.put(condition_id, embeddings[0])
        return embeddings[0]

    def sync(self, markets: list[Market]) -> None:
        """Upsert changed/new markets, drop untracked ones. Blocking — run in
        an executor. Embeds only docs whose content hash changed, so the first
        call builds the full index and later calls touch a handful of rows.

        A corrupted store (e.g. hnsw segment damage after a process was
        killed mid-write) is healed automatically: drop the collection,
        re-embed everything, retry once."""
        self._ensure_loaded()
        try:
            self._sync_once(markets)
        except Exception as e:
            log.error(
                f"[index] SYNC FAILED ({type(e).__name__}: {e}) — "
                f"rebuilding the collection from scratch"
            )
            self._rebuild_collection()
            self._sync_once(markets)
            log.error("[index] Rebuild succeeded — index is healthy again")

    def _rebuild_collection(self) -> None:
        """Drop and recreate the collection (heals segment corruption)."""
        self.ready = False
        try:
            self._client.delete_collection(COLLECTION)
        except Exception as e:
            log.warning(f"[index] delete_collection failed ({e}); recreating anyway")
        self._collection = self._client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def _sync_once(self, markets: list[Market]) -> None:
        t0 = time.monotonic()

        docs = {m.condition_id: _doc_text(m) for m in markets if m.condition_id}
        existing = self._collection.get(include=["metadatas"])
        existing_hashes = {
            id_: (md or {}).get("doc_hash")
            for id_, md in zip(existing["ids"], existing["metadatas"])
        }

        stale = [id_ for id_ in existing_hashes if id_ not in docs]
        if stale:
            self._collection.delete(ids=stale)
            for id_ in stale:
                self._embed_cache._data.pop(id_, None)  # drop untracked from cache

        to_embed = [
            (id_, doc, h)
            for id_, doc in docs.items()
            if existing_hashes.get(id_) != (h := _doc_hash(doc))
        ]

        for i in range(0, len(to_embed), EMBED_BATCH):
            batch = to_embed[i : i + EMBED_BATCH]
            embeddings = self._model.encode(
                [doc for _, doc, _ in batch],
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            emb_list = embeddings.tolist()
            self._collection.upsert(
                ids=[id_ for id_, _, _ in batch],
                embeddings=emb_list,
                documents=[doc for _, doc, _ in batch],
                metadatas=[{"doc_hash": h} for _, _, h in batch],
            )
            # Keep the in-memory cache fresh with the just-embedded vectors.
            for (id_, _, _), emb in zip(batch, emb_list):
                self._embed_cache.put(id_, emb)
            done = min(i + EMBED_BATCH, len(to_embed))
            if len(to_embed) > EMBED_BATCH:
                log.info(f"[index] Embedded {done}/{len(to_embed)} markets")

        self.ready = True
        log.info(
            f"[index] Sync: {len(to_embed)} embedded, {len(stale)} removed, "
            f"{self._collection.count()} total in {time.monotonic() - t0:.1f}s"
        )

    def search(
        self,
        text: str,
        top_k: int | None = None,
        max_distance: float | None = None,
    ) -> dict[str, float]:
        """Return {condition_id: cosine distance} for markets semantically
        close to `text`. Empty until the index is warm. Blocking (~50ms)."""
        if not self.ready:
            return {}
        self._ensure_loaded()
        top_k = top_k or config.EMBED_TOP_K
        if max_distance is None:
            max_distance = config.EMBED_DISTANCE_THRESHOLD

        embedding = self._model.encode([text], normalize_embeddings=True)
        result = self._collection.query(
            query_embeddings=embedding.tolist(),
            n_results=min(top_k, max(self._collection.count(), 1)),
            include=["distances"],
        )
        return {
            id_: dist
            for id_, dist in zip(result["ids"][0], result["distances"][0])
            if dist <= max_distance
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from locus.markets.gamma import fetch_active_markets, filter_by_categories

    print("Fetching niche markets...")
    all_m = fetch_active_markets(
        limit=None, min_volume=config.MIN_VOLUME_USD, max_volume=config.MAX_VOLUME_USD
    )
    niche = [
        m for m in filter_by_categories(all_m)
        if config.MIN_VOLUME_USD <= m.volume <= config.MAX_VOLUME_USD
    ]
    print(f"Syncing index with {len(niche)} markets...")
    index = MarketIndex()
    index.sync(niche)

    for headline in (
        "Powell signals the Fed is ready to begin cutting rates this summer",
        "OpenAI announces GPT-5 general availability for enterprise customers",
        "Bitcoin falls below $60,000 as ETF outflows accelerate",
    ):
        t0 = time.monotonic()
        hits = index.search(headline)
        ms = (time.monotonic() - t0) * 1000
        by_id = {m.condition_id: m for m in niche}
        print(f"\n\"{headline[:60]}\" ({ms:.0f}ms)")
        for cid, dist in sorted(hits.items(), key=lambda kv: kv[1]):
            m = by_id.get(cid)
            if m:
                print(f"  {dist:.3f}  [{m.category}] {m.question[:60]}")
