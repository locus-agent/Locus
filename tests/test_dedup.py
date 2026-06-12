"""Dedup cache must evict oldest-first, never the newest keys."""
from locus.sources.news_stream import RecentKeys


def test_basic_seen_semantics():
    rk = RecentKeys(max_size=10, keep=5)
    assert rk.seen("a") is False
    assert rk.seen("a") is True
    assert rk.seen("b") is False


def test_eviction_keeps_newest_drops_oldest():
    rk = RecentKeys(max_size=10, keep=5)
    for i in range(11):  # overflow triggers a trim to the 5 newest
        rk.seen(f"k{i}")
    assert len(rk) == 5
    assert "k10" in rk and "k9" in rk and "k6" in rk  # newest survived
    assert "k0" not in rk and "k1" not in rk           # oldest evicted


def test_eviction_is_deterministic_not_random():
    # The old set()-based trim kept an arbitrary subset; survivors must now
    # be exactly the most recent `keep` keys, every time.
    for _ in range(5):
        rk = RecentKeys(max_size=20, keep=10)
        for i in range(21):
            rk.seen(f"key{i}")
        survivors = {f"key{i}" for i in range(21) if f"key{i}" in rk}
        assert survivors == {f"key{i}" for i in range(11, 21)}
