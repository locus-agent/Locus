"""Truth Social as a first-class fast news source: tighter freshness window
than generic RSS, plus the materiality-boost rule in the classification prompt."""
from datetime import datetime, timedelta, timezone

import pytest

from locus import config
from locus.core.classifier import CLASSIFICATION_PROMPT
from locus.core.pipeline import gate_trade
from locus.core.edge import Signal
from locus.markets.gamma import Market
from locus.sources.news_stream import NewsEvent

NOW = datetime(2026, 6, 17, 12, 0, 0, tzinfo=timezone.utc)
MKT = Market("c1", "Will X happen?", "politics", 0.5, 0.5, 5000, "", True, [])


@pytest.fixture(autouse=True)
def _pin_age_limits(monkeypatch):
    """Pin freshness windows to their defaults so the test doesn't depend on a
    developer's .env overrides."""
    monkeypatch.setattr(config, "MAX_NEWS_AGE_SECONDS_TRUTHSOCIAL", 1800.0)
    monkeypatch.setattr(config, "MAX_NEWS_AGE_SECONDS_RSS", 7200.0)
    monkeypatch.setattr(config, "MATERIALITY_THRESHOLD_BULLISH", 0.3)
    monkeypatch.setattr(config, "HIGH_MATERIALITY_THRESHOLD", 0.9)


def _src_sig(news_source):
    return Signal(market=MKT, claude_score=0.7, market_price=0.5, edge=0.2,
                  side="YES", bet_amount=25.0, reasoning="", headlines="h",
                  classification="bullish", materiality=0.35,
                  news_source=news_source)


def _ev(ago_s):
    pub = NOW - timedelta(seconds=ago_s)
    return NewsEvent(headline="Trump posts", source="truthsocial", url="",
                     received_at=NOW, published_at=pub, latency_ms=ago_s * 1000)


# --- Freshness windows -------------------------------------------------------

def test_truthsocial_age_limit_is_30min():
    assert config.get_max_age_seconds("truthsocial") == 1800.0


def test_rss_age_limit_is_2h():
    assert config.get_max_age_seconds("rss") == 7200.0


def test_truthsocial_gate_uses_30min_window():
    # 30 min (1800s) trades; just past it is stale.
    ts = _src_sig("truthsocial")
    assert gate_trade(_ev(1800), ts, set(), now=NOW)[1] == "signal"
    s, a = gate_trade(_ev(1801), ts, set(), now=NOW)
    assert s is None and a == "stale"


def test_truthsocial_stricter_than_rss():
    # 45 min old: still fresh for generic RSS (2h), stale for Truth Social (30m).
    assert gate_trade(_ev(45 * 60), _src_sig("rss"), set(), now=NOW)[1] == "signal"
    s, a = gate_trade(_ev(45 * 60), _src_sig("truthsocial"), set(), now=NOW)
    assert s is None and a == "stale"


# --- Prompt boost rule -------------------------------------------------------

def test_boost_rule_present_in_prompt():
    assert "TRUTH SOCIAL BOOST" in CLASSIFICATION_PROMPT
    assert "truthsocial" in CLASSIFICATION_PROMPT
    assert "+0.15" in CLASSIFICATION_PROMPT
