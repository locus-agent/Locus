"""One-position-per-headline cap under concurrency (LOGIC_REVIEW.md finding #1).

gate_trade check-and-RESERVES the headline synchronously, so concurrent
candidates for the same headline can never both pass; the reservation is
committed by an actual position open and released on every failure path
(late gate, execution skip/error, crash) so a failed candidate never burns
the headline. Policy: the cap is one POSITION per headline, not one attempt —
after a release, a later candidate may reserve and open.

The pipeline-level tests drive the REAL PipelineV2._process_news candidate
chain (real gate_trade, real gates, real logging) with the news-specific
externals faked: matching, classification, edge detection, and the orderbook.
"""
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from locus import config
from locus.core import pipeline as pl
from locus.core import positions
from locus.core.classifier import Classification
from locus.core.edge import Signal
from locus.markets.gamma import Market
from locus.sources.news_stream import NewsEvent


@pytest.fixture(autouse=True)
def _pin_gates(monkeypatch):
    """Pin every threshold the candidate chain reads, so the tests exercise
    only the reservation mechanics (same pattern as test_gates)."""
    monkeypatch.setattr(config, "MIN_MATERIALITY_BULLISH", 0.3)
    monkeypatch.setattr(config, "HIGH_MATERIALITY_THRESHOLD", 0.5)
    monkeypatch.setattr(config, "EDGE_THRESHOLD", 0.1)
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_ENABLED", False)
    monkeypatch.setattr(config, "MAX_POSITIONS_PER_EVENT", 1)
    monkeypatch.setattr(config, "DAILY_SPEND_LIMIT_USD", 1000.0)
    monkeypatch.setattr(config, "PARALLEL_BATCH_SIZE", 8)
    monkeypatch.setattr(config, "SPORTS_ENABLED", False)


def mkt(cid, event_id):
    return Market(cid, f"Will {cid} happen soon?", "ai", 0.5, 0.5, 5000,
                  "2026-12-31", True, [], event_id=event_id)


def ev(headline):
    now = datetime.now(timezone.utc)
    return NewsEvent(headline=headline, source="rss", url="", received_at=now,
                     published_at=now, latency_ms=0)


def _classification():
    return Classification(direction="bullish", materiality=0.4, reasoning="",
                          latency_ms=1, model="test", confidence=0.6)


def _signal(market, headline):
    return Signal(market=market, claude_score=0.4, market_price=market.yes_price,
                  edge=0.2, side="YES", bet_amount=10.0, reasoning="r",
                  headlines=headline, news_source="rss",
                  classification="bullish", materiality=0.4, confidence=0.6)


def build_pipeline(monkeypatch, matched_markets, classify_delays=None,
                   orderbook_scores=None):
    """A PipelineV2 whose news externals are faked: every headline matches
    `matched_markets` (a mutable list, so tests can vary it between events),
    classification is a canned bullish call (optionally delayed per market, to
    control interleaving), edge detection always produces a YES signal, and
    the orderbook returns `orderbook_scores` in call order (then None=allow).
    Everything else — gate_trade, the late gates, logging, execution — is real.
    """
    delays = classify_delays or {}
    scores = list(orderbook_scores or [])
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
    monkeypatch.setattr(pl.logger, "find_recent_classification",
                        lambda *a, **k: None)

    async def classify(event, market):
        await asyncio.sleep(delays.get(market.condition_id, 0.0))
        return _classification()

    monkeypatch.setattr(p, "_classify_with_semaphores", classify)

    def edge(market, classification, event):
        return SimpleNamespace(signal=_signal(market, event.headline))

    monkeypatch.setattr(pl, "detect_edge_v2", edge)

    def orderbook(token_id):
        return scores.pop(0) if scores else None

    monkeypatch.setattr(pl, "fetch_orderbook_imbalance", orderbook)
    return p


async def drive(p, event, done, settle=0.25, timeout=5.0):
    """Feed one event through the real _process_news task and wait until
    `done()` holds, plus a settle window for the enqueue/finally to finish."""
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


def actions(tmp_db):
    conn = tmp_db._conn()
    rows = conn.execute("SELECT action FROM classifications ORDER BY id").fetchall()
    conn.close()
    return [r["action"] for r in rows]


def drain(p):
    sigs = []
    while not p.signal_queue.empty():
        sigs.append(p.signal_queue.get_nowait())
    return sigs


# --- the race: concurrent candidates, one headline -------------------------

def test_concurrent_candidates_same_headline_only_one_opens(tmp_db, monkeypatch):
    # Two markets on DIFFERENT events matched by one headline, processed
    # concurrently in one batch — the event lock cannot save this pairing;
    # only the headline reservation can. Exactly ONE may open.
    a, b = mkt("a", "eA"), mkt("b", "eB")
    p = build_pipeline(monkeypatch, [a, b])
    headline = "big shared headline"

    async def main():
        await drive(p, ev(headline), lambda: len(actions(tmp_db)) >= 2)
        signals = drain(p)
        results = [await p._execute_with_lock(s) for s in signals]
        return signals, results

    signals, results = asyncio.run(main())

    assert sorted(actions(tmp_db)) == ["capped", "signal"]
    assert len(signals) == 1
    assert [r["status"] for r in results] == ["dry_run"]
    assert len(positions.get_open_positions()) == 1
    # The open committed the reservation permanently.
    assert headline in p._traded_headlines


def test_orderbook_block_releases_for_later_candidate(tmp_db, monkeypatch):
    # Candidate A classifies instantly, passes gate_trade (reserving), and is
    # blocked at the orderbook stage (-0.9 = strong opposing flow) -> its
    # reservation is RELEASED. Candidate B classifies 0.3s later — after the
    # release — so it may reserve and open. The cap is one POSITION per
    # headline, not one attempt.
    a, b = mkt("a", "eA"), mkt("b", "eB")
    p = build_pipeline(monkeypatch, [a, b],
                       classify_delays={"b": 0.3},
                       orderbook_scores=[-0.9])
    headline = "strong headline, thin book on A"

    async def main():
        await drive(p, ev(headline), lambda: len(actions(tmp_db)) >= 2)
        signals = drain(p)
        results = [await p._execute_with_lock(s) for s in signals]
        return signals, results

    signals, results = asyncio.run(main())

    assert sorted(actions(tmp_db)) == ["orderbook_skip", "signal"]
    assert len(signals) == 1
    assert signals[0].market.condition_id == "b"
    assert [r["status"] for r in results] == ["dry_run"]
    assert len(positions.get_open_positions()) == 1
    assert headline in p._traded_headlines  # B's committed reservation


def test_candidate_error_after_gate_releases_reservation(tmp_db, monkeypatch):
    # A candidate that crashes mid-chain (orderbook fetch raises) after
    # gate_trade reserved must release in its finally — no leak, no signal.
    a = mkt("a", "eA")
    p = build_pipeline(monkeypatch, [a])

    def boom(token_id):
        raise RuntimeError("orderbook down")

    monkeypatch.setattr(pl, "fetch_orderbook_imbalance", boom)
    headline = "headline that crashes the chain"

    async def main():
        await drive(p, ev(headline),
                    lambda: p.stats["news_processed"] >= 1, settle=0.4)

    asyncio.run(main())
    assert headline not in p._traded_headlines
    assert asyncio.run(_drain_empty(p))
    assert positions.get_open_positions() == []


async def _drain_empty(p):
    return p.signal_queue.empty()


def test_sequential_same_headline_capped_after_open(tmp_db, monkeypatch):
    # Non-concurrent behavior is unchanged: once a position opens (commit), a
    # LATER event with the same headline is capped for good.
    matched = [mkt("a", "eA")]
    p = build_pipeline(monkeypatch, matched)
    headline = "same headline twice"

    async def main():
        await drive(p, ev(headline), lambda: len(actions(tmp_db)) >= 1)
        [signal] = drain(p)
        result = await p._execute_with_lock(signal)
        assert result["status"] == "dry_run"

        matched[:] = [mkt("b", "eB")]  # second event matches a fresh market
        await drive(p, ev(headline), lambda: len(actions(tmp_db)) >= 2)
        return drain(p)

    later_signals = asyncio.run(main())
    assert actions(tmp_db) == ["signal", "capped"]
    assert later_signals == []
    assert len(positions.get_open_positions()) == 1


# --- executor-stage commit/release (both dry-run and live-shaped) -----------

def _executor_signal(p, headline="h"):
    """A signal whose reservation was taken at gate time and handed to the
    executor stage (as _process_news does at enqueue)."""
    p._traded_headlines.add(headline)
    return _signal(mkt("x", "eX"), headline)


def test_dry_run_open_commits_reservation(tmp_db):
    p = pl.PipelineV2()
    signal = _executor_signal(p)
    result = asyncio.run(p._execute_with_lock(signal))
    assert result["status"] == "dry_run"
    assert len(positions.get_open_positions()) == 1
    assert "h" in p._traded_headlines  # committed


def test_live_executed_commits_reservation(tmp_db, monkeypatch):
    # Live parity: a confirmed live fill commits exactly like a dry-run open.
    async def fake_exec(signal):
        return {"trade_id": 1, "market": signal.market.question,
                "side": signal.side, "amount": signal.bet_amount,
                "actual_cost_usd": 9.5, "actual_shares": 19.0, "edge": 0.2,
                "status": "executed", "order_id": "o1", "latency_ms": 0}

    monkeypatch.setattr(pl, "execute_trade_async", fake_exec)
    p = pl.PipelineV2()
    signal = _executor_signal(p)
    result = asyncio.run(p._execute_with_lock(signal))
    assert result["status"] == "executed"
    assert len(positions.get_open_positions()) == 1
    assert "h" in p._traded_headlines


@pytest.mark.parametrize("status", ["skipped_wide_spread", "skipped_thin_book",
                                    "resting", "rejected_daily_limit",
                                    "dust_fill", "error_ValueError"])
def test_failed_execution_releases_reservation(tmp_db, monkeypatch, status):
    # Every no-position execution outcome (live skips, resting GTC, daily
    # limit, a dust fill sold back, errors) must release so the headline
    # isn't burned.
    async def fake_exec(signal):
        return {"trade_id": 1, "market": signal.market.question,
                "side": signal.side, "amount": signal.bet_amount,
                "actual_cost_usd": None, "edge": 0.2, "status": status,
                "order_id": None, "latency_ms": 0}

    monkeypatch.setattr(pl, "execute_trade_async", fake_exec)
    p = pl.PipelineV2()
    signal = _executor_signal(p)
    asyncio.run(p._execute_with_lock(signal))
    assert positions.get_open_positions() == []
    assert "h" not in p._traded_headlines  # released


def test_sanity_downgraded_phantom_fill_releases_reservation(tmp_db, monkeypatch):
    # A live "executed" with no real fill cost is downgraded to resting and
    # opens nothing — the reservation must be released too.
    async def fake_exec(signal):
        return {"trade_id": 1, "market": signal.market.question,
                "side": signal.side, "amount": signal.bet_amount,
                "actual_cost_usd": None, "edge": 0.2, "status": "executed",
                "order_id": "o1", "latency_ms": 0}

    monkeypatch.setattr(pl, "execute_trade_async", fake_exec)
    p = pl.PipelineV2()
    signal = _executor_signal(p)
    result = asyncio.run(p._execute_with_lock(signal))
    assert result["status"] == "resting"
    assert positions.get_open_positions() == []
    assert "h" not in p._traded_headlines


def test_execution_exception_releases_reservation(tmp_db, monkeypatch):
    async def boom(signal):
        raise RuntimeError("clob down")

    monkeypatch.setattr(pl, "execute_trade_async", boom)
    p = pl.PipelineV2()
    signal = _executor_signal(p)
    result = asyncio.run(p._execute_with_lock(signal))
    assert result is None
    assert "h" not in p._traded_headlines


def test_lost_event_race_releases_reservation(tmp_db, monkeypatch):
    # The execute-time risk re-check blocking (lost race) also releases.
    monkeypatch.setattr(
        pl, "compute_circuit_breaker",
        lambda: {"triggered": True, "reason": "drawdown", "metrics": {}},
    )
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_ENABLED", True)
    p = pl.PipelineV2()
    signal = _executor_signal(p)
    result = asyncio.run(p._execute_with_lock(signal))
    assert result is None
    assert "h" not in p._traded_headlines


# --- whale path --------------------------------------------------------------

def _whale_setup(monkeypatch, market, headline, orderbook_score):
    monkeypatch.setattr(config, "WHALE_TRADING_ENABLED", True)
    monkeypatch.setattr(pl.whale_tracker, "decide_investigation",
                        lambda opp: "investigate")
    monkeypatch.setattr(pl.whale_tracker, "whale_headline", lambda opp: headline)
    monkeypatch.setattr(pl, "classify_fast",
                        lambda h, m, s: _classification())
    monkeypatch.setattr(pl, "detect_edge_v2",
                        lambda m, c, e: SimpleNamespace(signal=_signal(m, headline)))
    monkeypatch.setattr(pl, "fetch_orderbook_imbalance",
                        lambda token_id: orderbook_score)
    return {"market": market, "size_usd": 5000.0}


def test_whale_gate_block_releases_reservation(tmp_db, monkeypatch):
    market = mkt("w", "eW")
    headline = "whale moved on w"
    opp = _whale_setup(monkeypatch, market, headline, orderbook_score=-0.9)
    p = pl.PipelineV2()
    asyncio.run(p._investigate_whale_opportunity(opp))
    assert headline not in p._traded_headlines  # blocked -> released
    assert p.signal_queue.empty()


def test_whale_signal_keeps_reservation_until_execution(tmp_db, monkeypatch):
    market = mkt("w", "eW")
    headline = "whale moved on w"
    opp = _whale_setup(monkeypatch, market, headline, orderbook_score=None)
    p = pl.PipelineV2()
    asyncio.run(p._investigate_whale_opportunity(opp))
    assert headline in p._traded_headlines  # handed to the executor stage
    assert p.signal_queue.qsize() == 1
