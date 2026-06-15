"""LRU embedding cache: eviction + hit/miss accounting, plus MarketIndex
pre_warm() loading and get_market_embedding() being cache-first."""
import numpy as np
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
    """Stand-in for a Chroma collection backed by an in-memory dict.

    `as_numpy=True` returns embeddings as a numpy array (as real Chroma does),
    guarding against the `x or []` / truthiness bug on ndarrays."""
    def __init__(self, store, as_numpy=False):
        self._store = store            # {id: embedding}
        self.get_calls = 0
        self._as_numpy = as_numpy

    def count(self):
        return len(self._store)

    def _wrap(self, rows):
        return np.array(rows) if self._as_numpy else rows

    def get(self, ids=None, include=None):
        self.get_calls += 1
        if ids is None:
            keys = list(self._store)
            return {"ids": keys, "embeddings": self._wrap([self._store[k] for k in keys])}
        return {"ids": ids, "embeddings": self._wrap([self._store.get(i) for i in ids])}


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


def test_pre_warm_handles_numpy_embeddings(monkeypatch):
    # Regression: real Chroma returns embeddings as a numpy array, so any
    # truthiness/`or []` on them raises "truth value ambiguous". This must not.
    idx = MarketIndex(path="/tmp/x")
    fake = _FakeCollection({"m1": [0.1, 0.2], "m2": [0.3, 0.4]}, as_numpy=True)
    idx._collection = fake
    idx._model = object()
    idx.ready = True
    monkeypatch.setattr(idx, "_ensure_loaded", lambda: None)

    idx.pre_warm()                                 # must not raise
    assert len(idx._embed_cache) == 2
    emb = idx.get_market_embedding("m1")           # cached numpy row
    assert np.allclose(emb, [0.1, 0.2])
