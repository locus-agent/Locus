"""Extended freshness window for long-horizon geopolitical markets.

is_geopolitical() detection (question / headline / both / u.s. normalization),
plus the gate_trade freshness behaviour: a geopolitical market resolving > 7
days out gets the 12h window, while short-term or non-geopolitical markets keep
the standard source window.
"""
from datetime import datetime, timedelta, timezone

import pytest

from locus import config
from locus.core.pipeline import gate_trade, is_geopolitical
from locus.core.edge import Signal
from locus.markets.gamma import Market
from locus.sources.news_stream import NewsEvent

NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=timezone.utc)

# Resolution dates relative to NOW: far out (> 7 days) gets the extended
# geopolitical window; soon (< 7 days) keeps the standard source window.
END_FAR = (NOW + timedelta(days=30)).isoformat()
END_SOON = (NOW + timedelta(days=3)).isoformat()


@pytest.fixture(autouse=True)
def _pin_thresholds(monkeypatch):
    monkeypatch.setattr(config, "MATERIALITY_THRESHOLD_BULLISH", 0.3)
    monkeypatch.setattr(config, "MATERIALITY_THRESHOLD_BEARISH", 0.4)
    monkeypatch.setattr(config, "HIGH_MATERIALITY_THRESHOLD", 0.5)
    monkeypatch.setattr(config, "MAX_NEWS_AGE_SECONDS_GEOPOLITICAL", 43200)


def mkt(question="Will X happen?", end_date=END_FAR):
    return Market("c1", question, "politics", 0.5, 0.5, 5000, end_date, True, [])


def sig(market, materiality=0.35, direction="bullish", news_source="rss"):
    return Signal(market=market, claude_score=0.7, market_price=0.5, edge=0.2,
                  side="YES", bet_amount=25.0, reasoning="", headlines="h",
                  classification=direction, materiality=materiality,
                  news_source=news_source)


def ev(headline, published_ago_s):
    pub = NOW - timedelta(seconds=published_ago_s)
    return NewsEvent(headline=headline, source="rss", url="",
                     received_at=NOW, published_at=pub,
                     latency_ms=int(published_ago_s * 1000))


# --- is_geopolitical detection -------------------------------------------

def test_detected_via_question():
    assert is_geopolitical(mkt("Will the Iran nuclear deal be signed?"), "") is True


def test_detected_via_headline():
    assert is_geopolitical(mkt("Will the deal close?"), "Russia escalates in Ukraine") is True


def test_detected_via_both():
    assert is_geopolitical(mkt("Will sanctions be lifted?"), "Putin meets Zelensky") is True


def test_non_geopolitical_not_detected():
    assert is_geopolitical(mkt("Will Bitcoin hit $100k?"), "ETF inflows surge") is False


def test_us_normalization():
    # "U.S. troops" -> "us troops"; "troops" is a geopolitical keyword. The
    # normalization keeps the keyword list free of punctuation variants.
    assert is_geopolitical(mkt("Will U.S. troops withdraw?"), "") is True
    assert is_geopolitical(mkt("Will United States sign the treaty?"), "") is True


# --- Extended freshness window in gate_trade -----------------------------

def test_geopolitical_long_horizon_uses_extended_window():
    # 6h old: stale for plain RSS (2h limit), but allowed for a long-horizon
    # geopolitical market (12h window).
    market = mkt("Will the Iran nuclear deal be signed?", end_date=END_FAR)
    s, a = gate_trade(ev("Iran talks resume", 6 * 3600), sig(market), set(), now=NOW)
    assert s is not None and a == "signal"


def test_geopolitical_extended_window_boundary():
    market = mkt("Will the Iran nuclear deal be signed?", end_date=END_FAR)
    # Just inside 12h trades; just past it is stale.
    assert gate_trade(ev("Iran", 43200), sig(market), set(), now=NOW)[1] == "signal"
    s, a = gate_trade(ev("Iran", 43201), sig(market), set(), now=NOW)
    assert s is None and a == "stale"


def test_geopolitical_short_horizon_uses_standard_window():
    # Geopolitical, but resolves in 3 days (< 7): standard RSS 2h window, so a
    # 6h-old headline is stale.
    market = mkt("Will the Iran nuclear deal be signed?", end_date=END_SOON)
    s, a = gate_trade(ev("Iran talks resume", 6 * 3600), sig(market), set(), now=NOW)
    assert s is None and a == "stale"


def test_non_geopolitical_long_horizon_uses_standard_window():
    # Long horizon but not geopolitical: standard RSS window, 6h-old is stale.
    market = mkt("Will Apple release a new iPhone?", end_date=END_FAR)
    s, a = gate_trade(ev("Apple rumor", 6 * 3600), sig(market), set(), now=NOW)
    assert s is None and a == "stale"
