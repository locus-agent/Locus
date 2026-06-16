"""Parallel candidate processing: the batching helper and the per-tier
concurrency semaphores in PipelineV2._classify_with_semaphores."""
import asyncio
import threading
import time

import pytest

from locus import config
from locus.core import pipeline as pl
from locus.core.classifier import Classification
from locus.markets.gamma import Market
from locus.sources.news_stream import NewsEvent
from datetime import datetime, timezone


def _mkt(i):
    return Market(f"c{i}", f"Will thing {i} happen?", "ai", 0.5, 0.5, 5000, "2026-12-31", True, [])


def _event():
    now = datetime.now(timezone.utc)
    return NewsEvent(headline="h", source="rss", url="", received_at=now,
                     published_at=now, latency_ms=0)


# --- batching helper ---------------------------------------------------------

def test_batches_splits_into_groups():
    assert list(pl._batches(list(range(10)), 4)) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert list(pl._batches([], 4)) == []
    assert list(pl._batches([1, 2, 3], 8)) == [[1, 2, 3]]      # one short batch
    assert list(pl._batches([1, 2], 0)) == [[1], [2]]          # size floored to 1


def _prefilter_rejection():
    return Classification(direction="neutral", materiality=0.0, reasoning="",
                          latency_ms=1, model="haiku", action="prefiltered_haiku")


def _concurrency_tracking_fn(state, lock, hold=0.02, ret=None):
    def fn(*args, **kwargs):
        with lock:
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
        time.sleep(hold)               # force overlap so concurrency is observable
        with lock:
            state["cur"] -= 1
        return ret if ret is not None else _prefilter_rejection()
    return fn


def test_haiku_semaphore_caps_parallel_prefilters(monkeypatch):
    monkeypatch.setattr(config, "TIERED_CLASSIFICATION_ENABLED", True)
    p = pl.PipelineV2()
    p._haiku_sem = asyncio.Semaphore(3)   # cap concurrency at 3

    state = {"cur": 0, "max": 0}
    lock = threading.Lock()
    monkeypatch.setattr(pl, "haiku_prefilter", _concurrency_tracking_fn(state, lock))

    async def main():
        return await asyncio.gather(*(
            p._classify_with_semaphores(_event(), _mkt(i)) for i in range(12)
        ))

    results = asyncio.run(main())
    assert all(r.action == "prefiltered_haiku" for r in results)  # all rejected, no Sonnet
    assert state["max"] <= 3                                      # never exceeded the semaphore
    assert state["max"] >= 2                                      # but did run in parallel


def test_sonnet_semaphore_caps_parallel_deep_calls(monkeypatch):
    monkeypatch.setattr(config, "TIERED_CLASSIFICATION_ENABLED", True)
    p = pl.PipelineV2()
    p._haiku_sem = asyncio.Semaphore(16)
    p._sonnet_sem = asyncio.Semaphore(2)   # tight Sonnet cap

    # Haiku always passes (returns None) -> every candidate reaches the Sonnet call.
    monkeypatch.setattr(pl, "haiku_prefilter", lambda *a, **k: None)

    state = {"cur": 0, "max": 0}
    lock = threading.Lock()
    deep = Classification(direction="bullish", materiality=0.7, reasoning="",
                          latency_ms=5, model="sonnet")
    monkeypatch.setattr(pl, "classify", _concurrency_tracking_fn(state, lock, ret=deep))

    async def main():
        return await asyncio.gather(*(
            p._classify_with_semaphores(_event(), _mkt(i)) for i in range(10)
        ))

    results = asyncio.run(main())
    assert all(r.model == "sonnet" for r in results)
    assert state["max"] <= 2     # Sonnet concurrency never exceeded its semaphore


# --- classification path selection (ensemble precedence / parking) -----------

def _calls_tracker(monkeypatch):
    """Patch the three classification entrypoints with markers and record which
    one ran. classify()/haiku_prefilter are sync (run in executor); ensemble is
    async (awaited directly)."""
    calls = {"haiku": 0, "classify": 0, "ensemble": 0}

    def haiku(*a, **k):
        calls["haiku"] += 1
        return None  # pass the prefilter so the deep call is reached

    def classify(headline, market, source, as_of=None, model=None):
        calls["classify"] += 1
        return Classification(direction="bullish", materiality=0.7, reasoning="",
                              latency_ms=1, model=model or "default")

    async def ensemble(*a, **k):
        calls["ensemble"] += 1
        return Classification(direction="bullish", materiality=0.7, reasoning="",
                              latency_ms=1, model="ensemble", ensemble_used=True)

    monkeypatch.setattr(pl, "haiku_prefilter", haiku)
    monkeypatch.setattr(pl, "classify", classify)
    monkeypatch.setattr(pl, "classify_ensemble", ensemble)
    return calls


def _run_classify(p):
    return asyncio.run(p._classify_with_semaphores(_event(), _mkt(1)))


def test_tiered_always_uses_haiku_then_sonnet_even_with_ensemble_on(monkeypatch):
    # Tiered wins outright: the ensemble must never run while tiering is on,
    # even if ENSEMBLE_ENABLED is (mistakenly) left true.
    monkeypatch.setattr(config, "TIERED_CLASSIFICATION_ENABLED", True)
    monkeypatch.setattr(config, "ENSEMBLE_ENABLED", True)
    calls = _calls_tracker(monkeypatch)
    result = _run_classify(pl.PipelineV2())
    assert calls == {"haiku": 1, "classify": 1, "ensemble": 0}
    assert result.model != "ensemble"


def test_tiering_off_ensemble_off_uses_single_classify(monkeypatch):
    # Default-style fallback (ensemble parked): a single deep classify(), Sonnet
    # model, no ensemble.
    monkeypatch.setattr(config, "TIERED_CLASSIFICATION_ENABLED", False)
    monkeypatch.setattr(config, "ENSEMBLE_ENABLED", False)
    monkeypatch.setattr(config, "SCORING_MODEL", "claude-sonnet-test")
    calls = _calls_tracker(monkeypatch)
    result = _run_classify(pl.PipelineV2())
    assert calls == {"haiku": 0, "classify": 1, "ensemble": 0}
    assert result.model == "claude-sonnet-test"  # deep classify uses SCORING_MODEL


def test_tiering_off_ensemble_on_uses_ensemble(monkeypatch):
    # The only way to reach the parked ensemble: tiering off AND ensemble on.
    monkeypatch.setattr(config, "TIERED_CLASSIFICATION_ENABLED", False)
    monkeypatch.setattr(config, "ENSEMBLE_ENABLED", True)
    calls = _calls_tracker(monkeypatch)
    result = _run_classify(pl.PipelineV2())
    assert calls == {"haiku": 0, "classify": 0, "ensemble": 1}
    assert result.ensemble_used is True


def test_ensemble_disabled_by_default_in_config():
    # The shipped config must keep the parked Grok ensemble off; flipping it back
    # on (in code or env) should trip this guard.
    assert config.ENSEMBLE_ENABLED is False
