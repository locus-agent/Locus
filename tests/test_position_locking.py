"""Event-level locking around position opening (PipelineV2).

Two signals on the same event are evaluated concurrently by _process_news
before either has opened, so both can clear the per-event exposure cap. The
event lock + fresh risk re-check in _execute_with_lock closes that race: the
loser sees the winner's just-opened position and backs off. Different events use
distinct locks and open in parallel.
"""
import asyncio

import pytest

from locus import config
from locus.core import pipeline as pl
from locus.core.edge import Signal
from locus.markets.gamma import Market


@pytest.fixture(autouse=True)
def _deterministic_config(monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", True)
    monkeypatch.setattr(config, "MAX_POSITIONS_PER_EVENT", 1)
    monkeypatch.setattr(config, "DAILY_SPEND_LIMIT_USD", 1000.0)
    # Keep the circuit breaker out of the way unless a test drives it.
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_ENABLED", False)


def mkt(cid, event_id="e1", category="politics", q=None):
    return Market(cid, q or f"Will candidate {cid} win the race?", category,
                  0.5, 0.5, 5000, "2026-12-31", True, [], event_id=event_id)


def sig(market, side="YES", amount=25.0):
    return Signal(market=market, claude_score=0.7, market_price=0.5, edge=0.2,
                  side=side, bet_amount=amount, reasoning="r", headlines="h",
                  news_source="rss", classification="bullish", materiality=0.6,
                  confidence=0.6)


def test_same_event_only_first_opens(tmp_db):
    # Two sibling markets on the SAME event, opened concurrently. The per-event
    # cap is 1, so only the first may open; the second must re-check out.
    p = pl.PipelineV2()
    a, b = mkt("a", event_id="e1"), mkt("b", event_id="e1")

    async def main():
        return await asyncio.gather(
            p._execute_with_lock(sig(a)), p._execute_with_lock(sig(b))
        )

    results = asyncio.run(main())
    opened = [r for r in results if r is not None]
    blocked = [r for r in results if r is None]
    assert len(opened) == 1 and len(blocked) == 1
    assert len(tmp_db_open_positions()) == 1
    assert p.stats.get("race_blocks") == 1


def test_different_events_open_in_parallel(tmp_db):
    # Distinct events -> distinct locks -> both open (no exposure conflict).
    p = pl.PipelineV2()
    a = mkt("a", event_id="e1", q="Will the senate bill pass?")
    b = mkt("b", event_id="e2", q="Will the mayor resign in Denver?")

    async def main():
        return await asyncio.gather(
            p._execute_with_lock(sig(a)), p._execute_with_lock(sig(b))
        )

    results = asyncio.run(main())
    assert all(r is not None for r in results)
    assert {r["status"] for r in results} == {"dry_run"}
    assert len(tmp_db_open_positions()) == 2
    assert p.stats.get("race_blocks", 0) == 0
    # Two distinct events created two distinct locks.
    assert set(p._position_locks) == {"e1", "e2"}


def test_circuit_breaker_blocks_on_recheck(tmp_db, monkeypatch):
    # A breaker that trips during the re-check blocks the open entirely.
    monkeypatch.setattr(
        pl, "compute_circuit_breaker",
        lambda: {"triggered": True, "reason": "drawdown", "metrics": {}},
    )
    p = pl.PipelineV2()
    result = asyncio.run(p._execute_with_lock(sig(mkt("a", event_id="e1"))))
    assert result is None
    assert tmp_db_open_positions() == []
    assert p.stats.get("race_blocks") == 1


def test_recheck_passes_on_clean_book(tmp_db):
    # Sanity: with an empty book and gates clear, the re-check approves.
    p = pl.PipelineV2()
    ok = asyncio.run(p._recheck_risk_gates(sig(mkt("a")), mkt("a")))
    assert ok is True


def test_daily_spend_limit_blocks_on_recheck(tmp_db, monkeypatch):
    # An already-deployed day at the limit blocks a further open on re-check.
    monkeypatch.setattr(config, "DAILY_SPEND_LIMIT_USD", 30.0)
    monkeypatch.setattr(pl.logger, "get_daily_pnl", lambda: -20.0)  # $20 deployed
    p = pl.PipelineV2()
    # 20 + 25 = 45 > 30 -> blocked.
    ok = asyncio.run(p._recheck_risk_gates(sig(mkt("a"), amount=25.0), mkt("a")))
    assert ok is False


def test_acquire_position_lock_is_stable_per_key(tmp_db):
    p = pl.PipelineV2()

    async def main():
        l1 = await p._acquire_position_lock("e1")
        l2 = await p._acquire_position_lock("e1")
        l3 = await p._acquire_position_lock("e2")
        return l1, l2, l3

    l1, l2, l3 = asyncio.run(main())
    assert l1 is l2          # same key -> same lock
    assert l1 is not l3      # different key -> different lock


def tmp_db_open_positions():
    """Helper: current open positions from the active (tmp) DB."""
    from locus.core import positions
    return positions.get_open_positions()
