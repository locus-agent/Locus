"""Event-switch target gating (LOGIC_REVIEW finding #8).

Every gate before the pipeline's event-context step runs against the ORIGINAL
market; a recommended sibling is a different market — different resolution
time, question shape, category, and order book — and previously switched in
without re-passing any of them. pipeline.switch_target_gate re-runs the
market-specific gates against the switch TARGET; a non-None return declines
the switch at the call site (the original signal, which cleared its own full
chain, stands — mirroring the negative-Kelly decline), and None lets the
switch proceed exactly as before.
"""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from locus import config
from locus.core import pipeline
from locus.core.edge import Signal
from locus.markets.gamma import Market


def _mkt(q="Will Alice win the nomination?", price=0.5, category="politics",
         end_date="", cid="sib1"):
    return Market(cid, q, category, price, round(1 - price, 4), 5000,
                  end_date, True, [], event_id="e1")


def _switched(market, side="NO", materiality=0.6):
    direction = "bearish" if side == "NO" else "bullish"
    return Signal(market=market, claude_score=materiality,
                  market_price=market.yes_price, edge=0.2, side=side,
                  bet_amount=10.0, reasoning="r", headlines="h",
                  news_source="rss", classification=direction,
                  materiality=materiality, confidence=0.6)


@pytest.fixture(autouse=True)
def _benign_defaults(monkeypatch):
    """Neutral gate inputs so each test flips exactly one gate: orderbook
    unavailable (fails open), CoV off, exclusion filters on, sports on,
    generous category limits, standard resolution floor."""
    monkeypatch.setattr(pipeline, "fetch_orderbook_imbalance", lambda token: None)
    monkeypatch.setattr(config, "COV_ENABLED", False)
    monkeypatch.setattr(config, "EXCLUDE_PRICE_TARGET_MARKETS", True)
    monkeypatch.setattr(config, "EXCLUDE_COINFLIP_MARKETS", True)
    monkeypatch.setattr(config, "SPORTS_ENABLED", True)
    monkeypatch.setattr(config, "MIN_HOURS_TO_RESOLUTION", 4.0)
    monkeypatch.setattr(config, "SPORTS_MIN_HOURS_TO_RESOLUTION", 4.0)
    monkeypatch.setattr(config, "MAX_EXPOSURE_PER_CATEGORY",
                        {"politics": 100.0, "sports": 100.0, "other": 100.0})


def _gate(switched, open_positions=None):
    return asyncio.run(
        pipeline.switch_target_gate(switched, "some headline", open_positions or [])
    )


def test_clean_target_passes():
    # A far-out, plain-question, low-exposure sibling clears every gate — the
    # switch proceeds exactly as before the fix.
    assert _gate(_switched(_mkt(end_date="2030-01-01T00:00:00Z"))) is None


def test_coinflip_target_declines_switch():
    m = _mkt(q="Bitcoin Up or Down on July 3?", category="crypto")
    assert _gate(_switched(m)) == "coinflip_market"


def test_price_target_target_declines_switch():
    m = _mkt(q="Will Bitcoin reach $150,000 by December 31?", category="crypto")
    assert _gate(_switched(m)) == "price_target_market"


def test_target_too_close_to_resolution_declines_switch():
    soon = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    assert _gate(_switched(_mkt(end_date=soon))) == "too_close_to_resolution"


def test_sports_target_declines_when_sports_disabled(monkeypatch):
    monkeypatch.setattr(config, "SPORTS_ENABLED", False)
    assert _gate(_switched(_mkt(category="sports"))) == "sports_disabled"


def test_orderbook_opposing_flow_declines_switch(monkeypatch):
    # Strong sell pressure on the sibling's book blocks a YES entry — the
    # original market's book was checked upstream, the sibling's never was.
    monkeypatch.setattr(pipeline, "fetch_orderbook_imbalance", lambda token: -0.9)
    assert _gate(_switched(_mkt(), side="YES")) == "orderbook_skip"
    # ...and the same book allows the side trading WITH the flow.
    assert _gate(_switched(_mkt(), side="NO")) is None


def test_category_exposure_of_target_declines_switch(monkeypatch):
    # The sibling can be a different inferred category than the original; its
    # category's book exposure must be re-checked.
    monkeypatch.setattr(config, "MAX_EXPOSURE_PER_CATEGORY",
                        {"politics": 10.0, "other": 10.0})
    book = [{"category": "politics", "amount_usd": 15.0,
             "market_question": "Will Bob win the nomination?"}]
    assert _gate(_switched(_mkt()), open_positions=book) == "category_limit"


def test_cov_already_priced_on_target_declines_switch(monkeypatch):
    # CoV must judge novelty against the SIBLING's price, not the original's.
    monkeypatch.setattr(config, "COV_ENABLED", True)
    monkeypatch.setattr(config, "COV_MATERIALITY_THRESHOLD", 0.65)
    monkeypatch.setattr(config, "COV_CONFIDENCE_THRESHOLD", 0.75)
    asked = []

    def fake_verify(headline, market, price):
        asked.append((market.condition_id, price))
        return {"already_priced": True, "confidence": 0.9}

    monkeypatch.setattr(pipeline, "verify_novelty", fake_verify)
    m = _mkt(price=0.5)
    assert _gate(_switched(m, materiality=0.7)) == "already_priced_in"
    assert asked == [("sib1", 0.5)]  # verified at the sibling's own price


def test_cov_not_priced_passes(monkeypatch):
    monkeypatch.setattr(config, "COV_ENABLED", True)
    monkeypatch.setattr(config, "COV_MATERIALITY_THRESHOLD", 0.65)
    monkeypatch.setattr(
        pipeline, "verify_novelty",
        lambda h, m, p: {"already_priced": False, "confidence": 0.9},
    )
    assert _gate(_switched(_mkt(price=0.5), materiality=0.7)) is None
