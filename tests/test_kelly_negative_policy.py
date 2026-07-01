"""Negative-Kelly policy: zero/negative Kelly means DO NOT TRADE, uniformly.

When Claude's own win-probability (confidence) is below what the market price
implies, the Kelly fraction is zero/negative — the model's own math says -EV.
The old behavior floored those bets to KELLY_MIN_BET_USD; now they are skipped
with the distinct funnel action "kelly_negative" (classification still
logged). Positive-but-tiny Kelly is different: +EV, merely small — it still
floors to the min bet.
"""
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from locus import config
from locus.core import edge as edge_mod
from locus.core import event_context, pipeline as pl, positions
from locus.core.classifier import Classification
from locus.core.edge import EdgeMetrics, Signal, detect_edge_v2
from locus.markets.gamma import Market
from locus.sources.news_stream import NewsEvent


@pytest.fixture(autouse=True)
def _pinned(monkeypatch):
    monkeypatch.setattr(config, "KELLY_BANKROLL_USD", 100.0)
    monkeypatch.setattr(config, "MAX_BET_USD", 25.0)
    monkeypatch.setattr(config, "KELLY_MIN_BET_USD", 2.0)
    monkeypatch.setattr(config, "EDGE_THRESHOLD", 0.10)
    monkeypatch.setattr(config, "BULLISH_MIN_PRICE", 0.12)
    monkeypatch.setattr(config, "BULLISH_MAX_PRICE", 0.82)
    monkeypatch.setattr(edge_mod, "get_cached_winrate", lambda: 0.75)  # factor 1.0


def _mkt(price, cid="c1"):
    return Market(cid, f"Will {cid} happen soon?", "ai", price,
                  round(1 - price, 4), 5000, "2026-12-31", True, [],
                  event_id=f"e-{cid}")


def _cls(confidence, materiality=0.5):
    return Classification(direction="bullish", materiality=materiality,
                          confidence=confidence, reasoning="", latency_ms=1,
                          model="test")


EVENT = NewsEvent(headline="kelly test headline", source="rss", url="",
                  received_at=datetime.now(timezone.utc),
                  published_at=datetime.now(timezone.utc), latency_ms=0)


# --- detect_edge_v2 ------------------------------------------------------------

def test_negative_kelly_yields_skip_reason_no_signal():
    # YES at 0.70 with confidence 0.60 < the 0.70 the price implies: Kelly
    # negative. Edge guards pass (edge 0.15 >= 0.10), but no trade.
    m = detect_edge_v2(_mkt(0.70), _cls(confidence=0.60), EVENT)
    assert isinstance(m, EdgeMetrics)
    assert m.signal is None
    assert m.skip_reason == "kelly_negative"
    assert m.recommended_size == 0.0
    assert m.edge == pytest.approx(0.15)  # metrics still reported for the funnel


def test_zero_kelly_also_skips():
    # Confidence exactly at the implied probability -> Kelly 0 -> no trade.
    m = detect_edge_v2(_mkt(0.70), _cls(confidence=0.70), EVENT)
    assert m.signal is None and m.skip_reason == "kelly_negative"


def test_small_positive_kelly_floors_to_min_bet():
    # Confidence 0.505 at even odds: base $0.50 -> +EV but tiny -> floored to
    # KELLY_MIN_BET_USD, and a real Signal is built (no skip).
    m = detect_edge_v2(_mkt(0.50), _cls(confidence=0.505), EVENT)
    assert m.skip_reason is None
    assert m.signal is not None
    assert m.signal.bet_amount == config.KELLY_MIN_BET_USD


def test_positive_kelly_unaffected():
    # Base $20 (conf 0.70 at even odds) x edge factor 1.375 = $27.50 -> capped.
    m = detect_edge_v2(_mkt(0.50), _cls(confidence=0.70), EVENT)
    assert m.skip_reason is None
    assert m.signal is not None and m.signal.bet_amount == config.MAX_BET_USD


# --- event switch: a -EV sibling must not be traded either ---------------------

def test_switched_signal_sizes_zero_on_negative_kelly():
    # build_switched_signal sizes the sibling with size_position: implied NO on
    # a sibling at yes 0.60 with confidence 0.38 (< the 0.40 NO implies) is
    # negative Kelly -> bet_amount 0.0, which the pipeline treats as
    # "don't switch".
    original = Signal(market=_mkt(0.35, "a"), claude_score=0.5, market_price=0.35,
                      edge=0.2, side="YES", bet_amount=10.0, reasoning="r",
                      headlines="h", classification="bullish", materiality=0.5,
                      confidence=0.38)
    rec = {"recommended_market": _mkt(0.60, "b"), "recommended_side": "NO",
           "implied_edge": 0.30, "reason": "implied bearish on sibling"}
    switched = event_context.build_switched_signal(original, rec)
    assert switched.bet_amount == 0.0


# --- pipeline funnel: skipped with its own action, classification logged -------

def _run_candidate(tmp_db, monkeypatch, confidence):
    """Drive one candidate through the real _process_news chain with the news
    externals faked and the REAL detect_edge_v2/sizing in play."""
    market = _mkt(0.70, "c1")
    p = pl.PipelineV2()
    p.market_watcher = SimpleNamespace(tracked_markets=[market], index=None,
                                       snapshots={})
    monkeypatch.setattr(
        pl, "match_news_to_markets_hybrid",
        lambda headline, tracked, index=None: [(market, "keyword", 0.9)],
    )
    monkeypatch.setattr(pl, "prefilter_match", lambda *a, **k: False)
    monkeypatch.setattr(pl.logger, "find_recent_classification",
                        lambda *a, **k: None)
    monkeypatch.setattr(pl, "fetch_orderbook_imbalance", lambda token_id: None)
    monkeypatch.setattr(config, "MIN_MATERIALITY_BULLISH", 0.3)
    monkeypatch.setattr(config, "HIGH_MATERIALITY_THRESHOLD", 0.9)
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_ENABLED", False)

    async def classify(event, mkt):
        return _cls(confidence=confidence)

    monkeypatch.setattr(p, "_classify_with_semaphores", classify)

    async def main():
        task = asyncio.create_task(p._process_news())
        now = datetime.now(timezone.utc)
        await p.news_queue.put(NewsEvent(
            headline="kelly funnel headline", source="rss", url="",
            received_at=now, published_at=now, latency_ms=0,
        ))
        while not _actions(tmp_db):
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(asyncio.wait_for(main(), timeout=5))
    return p


def _actions(tmp_db):
    conn = tmp_db._conn()
    rows = conn.execute(
        "SELECT action, direction, materiality FROM classifications"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def test_pipeline_logs_kelly_negative_and_skips(tmp_db, monkeypatch):
    p = _run_candidate(tmp_db, monkeypatch, confidence=0.60)  # negative Kelly
    rows = _actions(tmp_db)
    # The classification itself is still logged, under the distinct action.
    assert [r["action"] for r in rows] == ["kelly_negative"]
    assert rows[0]["direction"] == "bullish"
    assert rows[0]["materiality"] == pytest.approx(0.5)
    # No trade: nothing enqueued, no position, headline not reserved.
    assert p.signal_queue.empty()
    assert positions.get_open_positions() == []
    assert "kelly funnel headline" not in p._traded_headlines


def test_pipeline_trades_normally_on_positive_kelly(tmp_db, monkeypatch):
    p = _run_candidate(tmp_db, monkeypatch, confidence=0.80)  # positive Kelly
    rows = _actions(tmp_db)
    assert [r["action"] for r in rows] == ["signal"]
    assert p.signal_queue.qsize() == 1
