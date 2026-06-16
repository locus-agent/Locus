"""Whale-path gates in PipelineV2._investigate_whale_opportunity: the
WHALE_TRADING_ENABLED master switch plus the circuit-breaker, category-exposure,
orderbook, and CoV gates (previously missing on the whale path).

The whale-specific gate chain is exercised end-to-end with gate_trade and the
gate dependencies mocked to PASS; each test flips one to verify it blocks the
trade while the whale_triggered classification is still logged."""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from locus import config
from locus.core import pipeline as pl
from locus.core.classifier import Classification
from locus.core.edge import Signal, EdgeMetrics
from locus.markets.gamma import Market


def _market():
    end = (datetime.now(timezone.utc) + timedelta(hours=48)).isoformat()
    return Market("cw", "Will whale market resolve YES?", "crypto", 0.5, 0.5, 5000,
                  end, True,
                  tokens=[{"token_id": "tYES", "outcome": "Yes"},
                          {"token_id": "tNO", "outcome": "No"}])


def _classification(materiality=0.7):
    return Classification(direction="bullish", materiality=materiality, reasoning="r",
                          latency_ms=1, model="sonnet", confidence=0.6)


def _signal(market):
    return Signal(market=market, claude_score=0.7, market_price=0.5, edge=0.2,
                  side="YES", bet_amount=10.0, reasoning="r", headlines="h",
                  news_source="whale", classification="bullish", materiality=0.7,
                  confidence=0.6)


@pytest.fixture
def whale_pipeline(tmp_db, monkeypatch):
    """A PipelineV2 whose whale-path gate dependencies all PASS. gate_trade is
    mocked to approve (its internals are tested in test_gates); each test flips
    one of the whale-specific gates."""
    monkeypatch.setattr(config, "WHALE_TRADING_ENABLED", True)
    monkeypatch.setattr(config, "MAX_POSITIONS_PER_EVENT", 1)

    p = pl.PipelineV2()
    market = _market()
    signal = _signal(market)

    monkeypatch.setattr(pl.whale_tracker, "decide_investigation",
                        lambda opp, now=None: "investigate")
    monkeypatch.setattr(pl, "classify_fast", lambda *a, **k: _classification())
    monkeypatch.setattr(pl, "detect_edge_v2",
                        lambda m, c, e: EdgeMetrics(edge=0.2, expected_edge=0.12,
                                                    vol_adj=1.0, recommended_size=10.0,
                                                    signal=signal))
    monkeypatch.setattr(pl, "gate_trade", lambda event, sig, traded, now=None: (sig, "signal"))
    monkeypatch.setattr(pl.positions, "get_open_positions", lambda *a, **k: [])
    monkeypatch.setattr(pl.positions, "check_correlation_risk",
                        lambda *a, **k: {"risk_level": "low", "correlated_positions": [],
                                         "total_exposure_usd": 0.0})
    monkeypatch.setattr(pl.positions, "check_category_exposure",
                        lambda *a, **k: {"allowed": True, "warning": False,
                                         "current_usd": 0, "limit_usd": 100, "pct": 0})
    monkeypatch.setattr(pl, "fetch_orderbook_imbalance", lambda *a, **k: None)
    monkeypatch.setattr(pl, "verify_novelty", lambda *a, **k: None)
    monkeypatch.setattr(pl, "compute_circuit_breaker",
                        lambda: {"triggered": False, "reason": "", "metrics": {}})
    monkeypatch.setattr(pl.event_context, "get_event_exposure",
                        lambda *a, **k: {"event_id": "", "position_count": 0,
                                         "total_exposure_usd": 0.0, "positions": []})

    opp = {"market": market, "size_usd": 5000.0, "wallet": "0xabc",
           "outcome": "Yes", "title": "t"}
    return p, opp


def _run(p, opp):
    asyncio.run(p._investigate_whale_opportunity(opp))


def _actions(tmp_db):
    return [c["action"] for c in tmp_db.get_recent_classifications(limit=10)]


def test_happy_path_queues_signal(whale_pipeline, tmp_db):
    p, opp = whale_pipeline
    _run(p, opp)
    assert p.signal_queue.qsize() == 1
    assert "whale_triggered" in _actions(tmp_db)


def test_disabled_blocks_trade_but_still_logs(whale_pipeline, tmp_db, monkeypatch):
    p, opp = whale_pipeline
    monkeypatch.setattr(config, "WHALE_TRADING_ENABLED", False)
    _run(p, opp)
    assert p.signal_queue.qsize() == 0                  # no trade queued
    assert "whale_triggered" in _actions(tmp_db)        # investigation still logged


def test_circuit_breaker_blocks_whale_trade(whale_pipeline, tmp_db, monkeypatch):
    p, opp = whale_pipeline
    monkeypatch.setattr(pl, "compute_circuit_breaker",
                        lambda: {"triggered": True, "reason": "7d drawdown", "metrics": {}})
    _run(p, opp)
    assert p.signal_queue.qsize() == 0
    assert "whale_triggered" in _actions(tmp_db)


def test_category_exposure_blocks_whale_trade(whale_pipeline, tmp_db, monkeypatch):
    p, opp = whale_pipeline
    monkeypatch.setattr(pl.positions, "check_category_exposure",
                        lambda *a, **k: {"allowed": False, "warning": True,
                                         "current_usd": 200, "limit_usd": 75, "pct": 2.6})
    _run(p, opp)
    assert p.signal_queue.qsize() == 0


def test_cov_blocks_whale_trade(whale_pipeline, tmp_db, monkeypatch):
    p, opp = whale_pipeline
    monkeypatch.setattr(config, "COV_ENABLED", True)
    monkeypatch.setattr(config, "COV_MATERIALITY_THRESHOLD", 0.65)
    monkeypatch.setattr(config, "COV_CONFIDENCE_THRESHOLD", 0.75)
    # materiality 0.7 >= 0.65 and price 0.5 in band -> CoV runs; high-confidence
    # "already priced" must block the whale trade.
    monkeypatch.setattr(pl, "verify_novelty",
                        lambda *a, **k: {"already_priced": True, "confidence": 0.9,
                                         "reason": "priced in"})
    _run(p, opp)
    assert p.signal_queue.qsize() == 0


def test_orderbook_imbalance_blocks_whale_trade(whale_pipeline, tmp_db, monkeypatch):
    p, opp = whale_pipeline
    # Strong opposing sell pressure on the YES book blocks a YES buy.
    monkeypatch.setattr(pl, "fetch_orderbook_imbalance", lambda *a, **k: -0.9)
    _run(p, opp)
    assert p.signal_queue.qsize() == 0


def test_uses_classify_fast_with_whale_source(whale_pipeline, tmp_db, monkeypatch):
    # The whale path must route through the tiered classify_fast (source="whale"),
    # and classify_async is no longer imported in the pipeline module.
    p, opp = whale_pipeline
    calls = []

    def fake(*a, **k):
        calls.append(a)
        return _classification()

    monkeypatch.setattr(pl, "classify_fast", fake)
    _run(p, opp)
    assert calls and calls[0][2] == "whale"          # classify_fast(headline, market, "whale")
    assert not hasattr(pl, "classify_async")         # import removed