"""LRU embedding cache: eviction + hit/miss accounting, plus MarketIndex
pre_warm() loading and get_market_embedding() being cache-first."""
import types

import pytest

from locus.core import market_index
from locus.core.market_index import LRUCache, MarketIndex


# --- LRUCache ----------------------------------------------------------------

def test_lru_hit_and_miss_counts():
    c = LRUCache(maxsize=10)
    assert c.get("a") is None and c.misses == 1      # miss
    c.put("a", [1.0])
    assert c.get("a") == [1.0] and c.hits == 1       # hit
    assert c.get("b") is None and c.misses == 2


def test_lru_evicts_least_recently_used():
    c = LRUCache(maxsize=2)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")            # touch 'a' so 'b' is now the LRU
    c.put("c", 3)         # over capacity -> evict 'b'
    assert "a" in c and "c" in c
    assert "b" not in c
    assert len(c) == 2


def test_lru_maxsize_floored_to_one():
    c = LRUCache(maxsize=0)
    c.put("a", 1)
    c.put("b", 2)
    assert len(c) == 1 and "b" in c


# --- MarketIndex pre_warm / get_market_embedding -----------------------------

class _FakeCollection:
    """Stand-in for a Chroma collection backed by an in-memory dict."""
    def __init__(self, store):
        self._store = store            # {id: embedding}
        self.get_calls = 0

    def count(self):
        return len(self._store)

    def get(self, ids=None, include=None):
        self.get_calls += 1
        if ids is None:
            keys = list(self._store)
            return {"ids": keys, "embeddings": [self._store[k] for k in keys]}
        return {"ids": ids, "embeddings": [self._store.get(i) for i in ids]}


@pytest.fixture
def index_with_fake(monkeypatch):
    idx = MarketIndex(path="/tmp/does-not-matter")
    store = {"m1": [0.1, 0.2], "m2": [0.3, 0.4], "m3": [0.5, 0.6]}
    fake = _FakeCollection(store)
    idx._collection = fake
    idx._model = object()           # non-None so _ensure_loaded() skips the heavy load
    idx.ready = True
    # _ensure_loaded is a no-op now that _collection/_model are set.
    monkeypatch.setattr(idx, "_ensure_loaded", lambda: None)
    return idx, fake


def test_pre_warm_loads_all_embeddings(index_with_fake, caplog):
    idx, fake = index_with_fake
    import logging
    with caplog.at_level(logging.INFO):
        idx.pre_warm()
    assert len(idx._embed_cache) == 3
    assert "Pre-warmed 3 market embeddings" in caplog.text
    # All three now resolve from cache without another Chroma read.
    calls_before = fake.get_calls
    assert idx.get_market_embedding("m1") == [0.1, 0.2]
    assert fake.get_calls == calls_before          # served from cache, no disk hit
    assert idx._embed_cache.hits >= 1


def test_get_market_embedding_falls_back_to_chroma_then_caches(index_with_fake):
    idx, fake = index_with_fake
    # Cold cache -> first lookup hits Chroma...
    assert idx.get_market_embedding("m2") == [0.3, 0.4]
    assert fake.get_calls == 1
    # ...second lookup is served from the cache (no further disk read).
    assert idx.get_market_embedding("m2") == [0.3, 0.4]
    assert fake.get_calls == 1


def test_get_market_embedding_unknown_returns_none(index_with_fake):
    idx, _ = index_with_fake
    assert idx.get_market_embedding("nope") is None
