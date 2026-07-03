"""Cost-reduction gates in front of the model spend.

FIX 1: market-type exclusions (price-target / coin-flip) are pure
question-pattern checks and run BEFORE any model call — a market destined for
the price_target_market/coinflip_market skip must never reach Haiku or Sonnet.
The skip is still logged as a classification row for funnel visibility.

FIX 2: per-market classification cooldown (CLASSIFY_COOLDOWN_MINUTES): a
near-duplicate headline on a market classified within the window skips the
model (action 'cooldown_skip'); a materially different headline bypasses the
cooldown (breaking-news exception).

The pipeline-level tests drive the REAL PipelineV2._process_news candidate
chain with the news externals faked (same pattern as test_headline_reservation).
"""
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from locus import config
from locus.core import pipeline as pl
from locus.core.classifier import Classification
from locus.markets.gamma import Market
from locus.sources.news_stream import NewsEvent, headline_similarity


@pytest.fixture(autouse=True)
def _pin_exclusions(monkeypatch):
    """Pin the market-type exclusion flags and cooldown to shipped defaults so
    a developer .env can't flip the behavior under test."""
    monkeypatch.setattr(config, "EXCLUDE_PRICE_TARGET_MARKETS", True)
    monkeypatch.setattr(config, "EXCLUDE_COINFLIP_MARKETS", True)
    monkeypatch.setattr(config, "CLASSIFY_COOLDOWN_MINUTES", 30.0)
    monkeypatch.setattr(config, "DEDUP_COSINE_THRESHOLD", 0.92)


def mkt(cid, question):
    return Market(cid, question, "crypto", 0.5, 0.5, 5000, "2026-12-31", True, [])


def ev(headline):
    now = datetime.now(timezone.utc)
    return NewsEvent(headline=headline, source="rss", url="", received_at=now,
                     published_at=now, latency_ms=0)


# --- market_type_skip: pure pattern checks, no model dependency --------------

def test_market_type_skip_price_target():
    assert pl.market_type_skip(mkt("c1", "Will Bitcoin reach $200k in July?")) == "price_target_market"


def test_market_type_skip_coinflip():
    assert pl.market_type_skip(mkt("c1", "Bitcoin Up or Down on July 4?")) == "coinflip_market"


def test_market_type_skip_normal_market():
    assert pl.market_type_skip(mkt("c1", "Will the Fed cut rates in July?")) is None


def test_market_type_skip_respects_flags(monkeypatch):
    monkeypatch.setattr(config, "EXCLUDE_PRICE_TARGET_MARKETS", False)
    monkeypatch.setattr(config, "EXCLUDE_COINFLIP_MARKETS", False)
    assert pl.market_type_skip(mkt("c1", "Will Bitcoin reach $200k in July?")) is None
    assert pl.market_type_skip(mkt("c1", "Bitcoin Up or Down on July 4?")) is None


# --- headline_similarity ------------------------------------------------------

def test_headline_similarity_exact_match_shortcircuits():
    # Identical (case/whitespace-insensitive) headlines score 1.0 without
    # touching the embedding model.
    assert headline_similarity("BTC surges", "  btc surges ") == 1.0


# --- cooldown_recent_classification (unit) ------------------------------------

def test_cooldown_disabled_never_reads_db(monkeypatch):
    monkeypatch.setattr(config, "CLASSIFY_COOLDOWN_MINUTES", 0.0)
    monkeypatch.setattr(
        pl.logger, "find_recent_market_classification",
        lambda *a, **k: pytest.fail("DB read despite disabled cooldown"),
    )
    assert pl.cooldown_recent_classification("h", "c1") is None


def test_cooldown_cold_market_classifies(monkeypatch):
    monkeypatch.setattr(pl.logger, "find_recent_market_classification",
                        lambda *a, **k: None)
    assert pl.cooldown_recent_classification("h", "c1") is None


def test_cooldown_near_duplicate_blocks(monkeypatch):
    recent = {"headline": "Bitcoin jumps 3% after ETF inflows",
              "direction": "bullish", "materiality": 0.5, "confidence": 0.6}
    monkeypatch.setattr(pl.logger, "find_recent_market_classification",
                        lambda *a, **k: recent)
    monkeypatch.setattr(pl, "headline_similarity", lambda a, b: 0.95)
    assert pl.cooldown_recent_classification(
        "Bitcoin surges 3% on ETF inflow news", "c1") == recent


def test_cooldown_materially_different_headline_bypasses(monkeypatch):
    recent = {"headline": "Bitcoin jumps 3% after ETF inflows",
              "direction": "bullish", "materiality": 0.5, "confidence": 0.6}
    monkeypatch.setattr(pl.logger, "find_recent_market_classification",
                        lambda *a, **k: recent)
    monkeypatch.setattr(pl, "headline_similarity", lambda a, b: 0.20)
    assert pl.cooldown_recent_classification(
        "SEC sues major exchange over custody violations", "c1") is None


# --- logger.find_recent_market_classification ---------------------------------

def _log(tmp_db, action, direction="bullish", cid="c1", headline="h1"):
    tmp_db.log_classification(
        market_question="q", headline=headline, news_source="rss",
        direction=direction, materiality=0.5, edge=None, action=action,
        condition_id=cid, yes_price=0.5,
    )


def test_find_recent_market_classification_real_rows_only(tmp_db):
    # Non-real rows never arm the cooldown: pre-model skips carry no direction,
    # and prefilter/cache/cooldown/error actions are excluded by name.
    _log(tmp_db, "prefiltered_haiku")
    _log(tmp_db, "cached")
    _log(tmp_db, "cooldown_skip")
    _log(tmp_db, "error")
    _log(tmp_db, "price_target_market", direction=None)
    assert tmp_db.find_recent_market_classification("c1", 30.0) is None

    _log(tmp_db, "skip", headline="the real one")
    row = tmp_db.find_recent_market_classification("c1", 30.0)
    assert row is not None and row["headline"] == "the real one"
    # Other markets stay cold.
    assert tmp_db.find_recent_market_classification("c2", 30.0) is None


def test_cooldown_skip_rows_excluded_from_exact_cache(tmp_db):
    # A cooldown_skip row must not be served as a reusable prior by the exact
    # (headline, market) dedup cache.
    _log(tmp_db, "cooldown_skip")
    assert tmp_db.find_recent_classification("h1", "c1", 0.5, 0.02, 24.0) is None


# --- pipeline integration: the real _process_news candidate chain -------------

def build_pipeline(monkeypatch, matched_markets):
    """PipelineV2 with faked news externals: every headline matches
    `matched_markets`, the model recorder counts classify calls, edge detection
    finds no edge (so surviving candidates log action 'skip'). Everything else
    — pre-model gates, gate_trade, logging — is real."""
    p = pl.PipelineV2()
    p.market_watcher = SimpleNamespace(tracked_markets=matched_markets,
                                       index=None, snapshots={})
    monkeypatch.setattr(
        pl, "match_news_to_markets_hybrid",
        lambda headline, tracked, index=None: [
            (m, "keyword", 0.9) for m in matched_markets
        ],
    )
    monkeypatch.setattr(pl, "prefilter_match", lambda *a, **k: False)
    # Exact (headline, market) cache off, so the cooldown path is what's tested.
    monkeypatch.setattr(pl.logger, "find_recent_classification",
                        lambda *a, **k: None)

    calls = {"classify": 0}

    async def classify(event, market):
        calls["classify"] += 1
        return Classification(direction="bullish", materiality=0.4, reasoning="",
                              latency_ms=1, model="test", confidence=0.6)

    monkeypatch.setattr(p, "_classify_with_semaphores", classify)
    monkeypatch.setattr(pl, "detect_edge_v2", lambda *a, **k: None)
    return p, calls


async def drive(p, event, done, settle=0.1, timeout=5.0):
    task = asyncio.create_task(p._process_news())
    await p.news_queue.put(event)

    async def _wait():
        while not done():
            await asyncio.sleep(0.02)

    try:
        await asyncio.wait_for(_wait(), timeout=timeout)
        await asyncio.sleep(settle)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def rows(tmp_db):
    conn = tmp_db._conn()
    out = conn.execute(
        "SELECT action, direction, materiality FROM classifications ORDER BY id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in out]


def test_price_target_market_never_reaches_model(tmp_db, monkeypatch):
    p, calls = build_pipeline(
        monkeypatch, [mkt("c1", "Will Bitcoin reach $200k in July?")]
    )
    asyncio.run(drive(p, ev("Bitcoin rallies on ETF news"),
                      lambda: len(rows(tmp_db)) >= 1))
    assert calls["classify"] == 0                       # no Haiku, no Sonnet
    logged = rows(tmp_db)
    assert [r["action"] for r in logged] == ["price_target_market"]
    assert logged[0]["direction"] is None               # no model output to log
    assert p.stats["market_type_skips"] == 1


def test_coinflip_market_never_reaches_model(tmp_db, monkeypatch):
    p, calls = build_pipeline(
        monkeypatch, [mkt("c1", "Bitcoin Up or Down on July 4?")]
    )
    asyncio.run(drive(p, ev("Bitcoin rallies on ETF news"),
                      lambda: len(rows(tmp_db)) >= 1))
    assert calls["classify"] == 0
    assert [r["action"] for r in rows(tmp_db)] == ["coinflip_market"]


def test_cooldown_blocks_rapid_reclassification(tmp_db, monkeypatch):
    market = mkt("c1", "Will the Fed cut rates in July?")
    p, calls = build_pipeline(monkeypatch, [market])

    async def main():
        # First headline classifies normally (cold market).
        await drive(p, ev("Fed signals rate cut"), lambda: len(rows(tmp_db)) >= 1)
        # A near-duplicate seconds later must not buy another model call.
        # (identical headline -> exact-match shortcut in headline_similarity;
        # the exact dedup cache is off in build_pipeline, so this exercises
        # the cooldown, not the cache)
        await drive(p, ev("Fed signals rate cut"), lambda: len(rows(tmp_db)) >= 2)

    asyncio.run(main())
    assert calls["classify"] == 1
    logged = rows(tmp_db)
    assert [r["action"] for r in logged] == ["skip", "cooldown_skip"]
    # The skip row carries the prior direction but a NULL materiality, so it
    # can never fake an independent confirming source (get_confirming_sources).
    assert logged[1]["direction"] == "bullish"
    assert logged[1]["materiality"] is None
    assert p.stats["cooldown_skips"] == 1


def test_materially_different_headline_bypasses_cooldown(tmp_db, monkeypatch):
    market = mkt("c1", "Will the Fed cut rates in July?")
    p, calls = build_pipeline(monkeypatch, [market])
    # Different stories: force the similarity check below the dup threshold.
    monkeypatch.setattr(pl, "headline_similarity", lambda a, b: 0.30)

    async def main():
        await drive(p, ev("Fed signals rate cut"), lambda: len(rows(tmp_db)) >= 1)
        await drive(p, ev("Powell resigns effective immediately"),
                    lambda: len(rows(tmp_db)) >= 2)

    asyncio.run(main())
    assert calls["classify"] == 2                       # breaking news reclassifies
    assert [r["action"] for r in rows(tmp_db)] == ["skip", "skip"]


def test_first_time_classification_unaffected(tmp_db, monkeypatch):
    market = mkt("c1", "Will the Fed cut rates in July?")
    p, calls = build_pipeline(monkeypatch, [market])
    asyncio.run(drive(p, ev("Fed signals rate cut"),
                      lambda: len(rows(tmp_db)) >= 1))
    assert calls["classify"] == 1
    logged = rows(tmp_db)
    assert [r["action"] for r in logged] == ["skip"]    # classified, no edge
    assert logged[0]["direction"] == "bullish"
