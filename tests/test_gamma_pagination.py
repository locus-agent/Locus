"""Parallel tag-based market fetching in gamma.fetch_active_markets.

Covers the merge/dedupe/sort core, dedup across the many parallel streams, the
volume-range fallback when the /tags endpoint is down, and that the fan-out can
return far more markets than a single offset-capped stream (> 2100).
"""
import pytest

from locus.markets import gamma
from locus.markets.gamma import Market


@pytest.fixture(autouse=True)
def _reset_tags_cache(monkeypatch):
    # Never let a previous (or real) get_tags() cache leak between tests.
    monkeypatch.setattr(gamma, "_tags_cache", None)


def m(cid, vol=1000.0):
    return Market(cid, f"Q {cid}", "politics", 0.5, 0.5, vol, "2026-12-31", True, [])


def _all_tags():
    """A {slug: tag_id} map covering every tracked category."""
    return {slug: i for i, slug in enumerate(gamma._TAG_CATEGORIES)}


# --- _merge_dedupe_sort (pure) -------------------------------------------

def test_merge_dedupe_sort_dedupes_and_sorts():
    s1 = [m("a", 100), m("b", 300)]
    s2 = [m("b", 300), m("c", 200)]   # 'b' is a cross-stream duplicate
    out = gamma._merge_dedupe_sort([s1, s2], None, None, None)
    assert [x.condition_id for x in out] == ["b", "c", "a"]   # volume DESC, deduped


def test_merge_skips_exceptions_and_applies_bounds():
    s1 = [m("a", 100), m("b", 60000), m("c", 25000)]
    # return_exceptions=True means a failed stream arrives as an Exception.
    out = gamma._merge_dedupe_sort([s1, RuntimeError("boom")], 1000, 50000, None)
    # 'a' (<1000) and 'b' (>50000) are filtered; the exception is skipped.
    assert [x.condition_id for x in out] == ["c"]


def test_merge_applies_limit():
    out = gamma._merge_dedupe_sort([[m("x", 5), m("y", 4), m("z", 3)]], None, None, 2)
    assert [x.condition_id for x in out] == ["x", "y"]


# --- fetch_active_markets orchestration -----------------------------------

def test_fetch_dedupes_across_streams(monkeypatch):
    monkeypatch.setattr(gamma, "get_tags", _all_tags)

    async def fake_tag(client, tag_id, max_items=gamma._GAMMA_TAG_OFFSET_CAP):
        return [m("dup1", 100), m("dup2", 200)]   # every tag stream returns these

    async def fake_general(client, max_items=gamma._GAMMA_TAG_OFFSET_CAP):
        return [m("dup2", 200), m("gen", 500)]    # overlaps 'dup2'

    monkeypatch.setattr(gamma, "_fetch_with_tag", fake_tag)
    monkeypatch.setattr(gamma, "_fetch_general", fake_general)

    out = gamma.fetch_active_markets(limit=None)
    assert sorted(x.condition_id for x in out) == ["dup1", "dup2", "gen"]


def test_tag_failure_falls_back_to_volume_ranges(monkeypatch):
    def boom_tags():
        raise RuntimeError("tags endpoint down")

    monkeypatch.setattr(gamma, "get_tags", boom_tags)

    called = {"tag": 0, "ranges": []}

    async def fake_tag(client, tag_id, max_items=gamma._GAMMA_TAG_OFFSET_CAP):
        called["tag"] += 1
        return [m("should-not-appear")]

    async def fake_range(client, vmin, vmax, max_items=gamma._GAMMA_TAG_OFFSET_CAP):
        called["ranges"].append((vmin, vmax))
        return [m(f"r-{int(vmin)}-{int(vmax)}", vmin + 1)]

    monkeypatch.setattr(gamma, "_fetch_with_tag", fake_tag)
    monkeypatch.setattr(gamma, "_fetch_volume_range", fake_range)

    out = gamma.fetch_active_markets(limit=None)
    assert called["tag"] == 0                                  # tag path never used
    assert set(called["ranges"]) == set(gamma._FALLBACK_VOLUME_RANGES)
    assert {x.condition_id for x in out} == {"r-1000-50000", "r-50000-500000"}


def test_total_markets_exceeds_2100(monkeypatch):
    monkeypatch.setattr(gamma, "get_tags", _all_tags)

    counter = {"n": 0}

    def make_batch(n):
        out = []
        for _ in range(n):
            counter["n"] += 1
            out.append(m(f"mk-{counter['n']}", float(counter["n"])))
        return out

    async def fake_tag(client, tag_id, max_items=gamma._GAMMA_TAG_OFFSET_CAP):
        return make_batch(300)

    async def fake_general(client, max_items=gamma._GAMMA_TAG_OFFSET_CAP):
        return make_batch(300)

    monkeypatch.setattr(gamma, "_fetch_with_tag", fake_tag)
    monkeypatch.setattr(gamma, "_fetch_general", fake_general)

    out = gamma.fetch_active_markets(limit=None)
    # 8 tag streams + 1 general stream * 300 unique markets each = 2700 > 2100,
    # which a single offset-capped (2000) stream could never reach.
    assert len(out) > 2100


def test_get_tags_builds_slug_to_id_map_and_caches(monkeypatch):
    calls = {"n": 0}

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            calls["n"] += 1
            # One short page (< page size) ends pagination immediately.
            return FakeResp([
                {"id": 7, "slug": "Politics", "label": "Politics"},
                {"id": 9, "slug": "crypto", "label": "Crypto"},
            ])

    monkeypatch.setattr(gamma.httpx, "Client", FakeClient)
    tags = gamma.get_tags()
    assert tags == {"politics": 7, "crypto": 9}    # slugs lowercased
    # Second call is served from cache (no new HTTP client use).
    before = calls["n"]
    assert gamma.get_tags() == {"politics": 7, "crypto": 9}
    assert calls["n"] == before
