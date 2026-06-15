"""Semantic (cosine) headline dedup: near-duplicates blocked, distinct ones
forwarded, exact repeats short-circuited without embedding, and TTL expiry."""
from datetime import datetime, timedelta, timezone

import numpy as np

from locus.sources.news_stream import SemanticDeduper

T0 = datetime(2026, 6, 16, 12, 0, 0, tzinfo=timezone.utc)


class FakeModel:
    """Maps each headline to a fixed (then normalized) vector, so cosine
    similarities in the tests are exactly controllable. Counts encode calls."""
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = 0

    def encode(self, texts, normalize_embeddings=True):
        self.calls += 1
        out = []
        for t in texts:
            v = np.array(self.mapping[t], dtype=float)
            if normalize_embeddings:
                v = v / np.linalg.norm(v)
            out.append(v)
        return np.array(out)


def _deduper(threshold=0.92, ttl_hours=2.0, mapping=None):
    dd = SemanticDeduper(threshold, maxlen=500, ttl_hours=ttl_hours)
    if mapping is not None:
        dd._model = FakeModel(mapping)
    return dd


def test_near_duplicate_is_blocked():
    dd = _deduper(mapping={"A": [1.0, 0.0], "A2": [0.99, 0.02]})
    assert dd.check("A", now=T0) is None              # first time seen -> recorded
    res = dd.check("A2", now=T0)                       # reworded near-dup
    assert res is not None
    entry, sim = res
    assert sim > 0.92
    assert entry["headline"] == "A"


def test_distinct_headline_is_forwarded():
    # Orthogonal vectors -> cosine 0, well under the 0.92 threshold.
    dd = _deduper(mapping={"A": [1.0, 0.0], "B": [0.0, 1.0]})
    assert dd.check("A", now=T0) is None
    assert dd.check("B", now=T0) is None


def test_below_threshold_not_blocked():
    # cosine([1,0],[1,1]) = 0.707 < 0.92 -> not a duplicate.
    dd = _deduper(mapping={"A": [1.0, 0.0], "C": [1.0, 1.0]})
    assert dd.check("A", now=T0) is None
    assert dd.check("C", now=T0) is None


def test_exact_repeat_blocked_without_embedding():
    dd = _deduper(mapping={"A": [1.0, 0.0]})
    assert dd.check("A", now=T0) is None
    calls = dd._model.calls
    res = dd.check("A", now=T0)                        # byte-identical repeat
    assert res is not None and res[1] == 1.0
    assert dd._model.calls == calls                   # exact-hash path skipped encoding


def test_ttl_expiry_allows_resend():
    dd = _deduper(ttl_hours=2.0, mapping={"A": [1.0, 0.0], "A2": [0.99, 0.02]})
    assert dd.check("A", now=T0) is None
    # 3h later the earlier embedding has aged out of the 2h window, so the
    # near-duplicate is no longer suppressed.
    later = T0 + timedelta(hours=3)
    assert dd.check("A2", now=later) is None
    assert len(dd._recent) == 1                        # only the fresh one remains


def test_fails_open_when_encode_errors():
    dd = _deduper(threshold=0.92)

    class Boom:
        def encode(self, *a, **k):
            raise RuntimeError("model down")

    dd._model = Boom()
    # A non-exact headline can't be embedded -> forwarded rather than dropped.
    assert dd.check("totally new headline", now=T0) is None
