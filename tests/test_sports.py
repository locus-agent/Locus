"""Sports-category detection and the extra-strict sports trade gates."""
from datetime import datetime, timedelta, timezone

import pytest

from locus import config
from locus.core.pipeline import gate_trade
from locus.core.edge import Signal
from locus.markets.gamma import Market, _infer_category
from locus.sources.news_stream import NewsEvent

NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _pin_sports_config(monkeypatch):
    """Pin the sports thresholds so the tests don't depend on .env overrides."""
    monkeypatch.setattr(config, "SPORTS_ENABLED", True)
    monkeypatch.setattr(config, "SPORTS_MATERIALITY_THRESHOLD", 0.48)
    monkeypatch.setattr(config, "SPORTS_MIN_HOURS_TO_RESOLUTION", 10.0)
    monkeypatch.setattr(config, "MIN_HOURS_TO_RESOLUTION", 4.0)
    monkeypatch.setattr(config, "MAX_HEADLINES_PER_SPORTS_EVENT", 2)
    # Keep the standard floors well below the sports floor so a 0.45-materiality
    # signal would pass as non-sports but fail as sports.
    monkeypatch.setattr(config, "MATERIALITY_THRESHOLD_BULLISH", 0.3)
    monkeypatch.setattr(config, "MATERIALITY_THRESHOLD_BEARISH", 0.4)
    monkeypatch.setattr(config, "HIGH_MATERIALITY_THRESHOLD", 0.9)


def _market(category="sports", end_date="", event_id=""):
    return Market("c1", "Will Arsenal win the Premier League?", category,
                  0.5, 0.5, 5000, end_date, True, [], event_id=event_id)


def _sig(market, materiality=0.6, direction="bullish"):
    return Signal(market=market, claude_score=0.7, market_price=0.5, edge=0.2,
                  side="YES", bet_amount=25.0, reasoning="", headlines="h",
                  classification=direction, materiality=materiality)


def _ev(headline="Arsenal news", ago_s=60):
    pub = NOW - timedelta(seconds=ago_s)
    return NewsEvent(headline=headline, source="rss", url="",
                     received_at=NOW, published_at=pub, latency_ms=ago_s * 1000)


# --- 1. Keyword detection ---

@pytest.mark.parametrize("question", [
    "Will the Chiefs win the Super Bowl?",
    "Will LeBron score 30 in the NBA finals?",
    "Will Arsenal win the Premier League?",
    "Who wins the Champions League final?",
    "Will Djokovic win Wimbledon?",
    "Will there be a goal in the first half?",
])
def test_sports_keywords_detected(question):
    assert _infer_category(question, []) == "sports"


def test_non_sports_unaffected():
    assert _infer_category("Will Bitcoin hit $100k?", []) == "crypto"
    assert _infer_category("Will Trump win the election?", []) == "politics"


# --- 2. Materiality threshold ---

def test_sports_materiality_threshold_applied():
    # 0.45 clears the bullish floor (0.3) but not the sports floor (0.48).
    s, a = gate_trade(_ev(), _sig(_market(), materiality=0.45),
                      set(), now=NOW, sports_event_counts={})
    assert s is None and a == "low_materiality"


def test_sports_above_threshold_passes():
    s, a = gate_trade(_ev(), _sig(_market(), materiality=0.6),
                      set(), now=NOW, sports_event_counts={})
    assert a == "signal" and s is not None


# --- 3. Min hours to resolution ---

def test_sports_min_hours_enforced():
    # Closes in 6h: passes the standard 4h floor but not the sports 10h floor.
    end = (NOW + timedelta(hours=6)).isoformat()
    s, a = gate_trade(_ev(), _sig(_market(end_date=end)),
                      set(), now=NOW, sports_event_counts={})
    assert s is None and a == "too_close_to_resolution"


def test_non_sports_uses_standard_min_hours():
    # Same 6h close on a non-sports market clears the 4h standard floor.
    end = (NOW + timedelta(hours=6)).isoformat()
    s, a = gate_trade(_ev(), _sig(_market(category="crypto", end_date=end)),
                      set(), now=NOW, sports_event_counts={})
    assert a == "signal" and s is not None


# --- 4. Per-event headline cap ---

def test_sports_event_cap_blocks_over_limit():
    counts = {"e1": 2}  # already at MAX_HEADLINES_PER_SPORTS_EVENT
    s, a = gate_trade(_ev(), _sig(_market(event_id="e1")),
                      set(), now=NOW, sports_event_counts=counts)
    assert s is None and a == "sports_event_cap"


def test_sports_event_cap_allows_under_limit():
    counts = {"e1": 1}
    s, a = gate_trade(_ev(), _sig(_market(event_id="e1")),
                      set(), now=NOW, sports_event_counts=counts)
    assert a == "signal" and s is not None


def test_sports_event_cap_ignores_other_events():
    counts = {"e2": 5}
    s, a = gate_trade(_ev(), _sig(_market(event_id="e1")),
                      set(), now=NOW, sports_event_counts=counts)
    assert a == "signal" and s is not None


# --- 5. Disabled guard ---

def test_sports_disabled_guard(monkeypatch):
    monkeypatch.setattr(config, "SPORTS_ENABLED", False)
    s, a = gate_trade(_ev(), _sig(_market()), set(), now=NOW, sports_event_counts={})
    assert s is None and a == "sports_disabled"


def test_disabled_does_not_affect_non_sports(monkeypatch):
    monkeypatch.setattr(config, "SPORTS_ENABLED", False)
    s, a = gate_trade(_ev(), _sig(_market(category="crypto")),
                      set(), now=NOW, sports_event_counts={})
    assert a == "signal" and s is not None
