"""Trade-time risk gates: freshness (incl. queue dwell + unknown dates) and headline cap."""
from datetime import datetime, timedelta, timezone

import pytest

from locus import config
from locus.core.pipeline import gate_trade, news_age_seconds
from locus.core.edge import Signal
from locus.markets.gamma import Market
from locus.sources.news_stream import NewsEvent

NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=timezone.utc)
LIMIT = config.MAX_NEWS_AGE_SECONDS


@pytest.fixture(autouse=True)
def _pin_materiality_thresholds(monkeypatch):
    """Pin the direction-specific materiality floors to their standard defaults
    so the tests don't depend on the developer's .env overrides (same pattern
    used for other configs). The floor tests below are written against these:
    bullish 0.3, bearish 0.4 (bearish gets a higher bar)."""
    monkeypatch.setattr(config, "MATERIALITY_THRESHOLD_BULLISH", 0.3)
    monkeypatch.setattr(config, "MATERIALITY_THRESHOLD_BEARISH", 0.4)

MKT = Market("c1", "Will X happen?", "ai", 0.5, 0.5, 5000, "", True, [])


def sig(materiality=0.35, direction="bullish"):
    """A would-be signal that clears the materiality floor by default, so the
    freshness/cap gates can be tested in isolation."""
    return Signal(market=MKT, claude_score=0.7, market_price=0.5, edge=0.2,
                  side="YES", bet_amount=25.0, reasoning="", headlines="h",
                  classification=direction, materiality=materiality)


# Default tradeable signal: bullish, materiality just above the bullish floor
# (0.3) and below the high-materiality confirmation gate (0.5).
SIG = sig()


def ev(headline, published_ago_s, latency_ms=None):
    pub = NOW - timedelta(seconds=published_ago_s)
    if latency_ms is None:
        latency_ms = int(published_ago_s * 1000)  # received just now
    return NewsEvent(headline=headline, source="rss", url="",
                     received_at=NOW, published_at=pub, latency_ms=latency_ms)


def test_fresh_news_trades():
    s, a = gate_trade(ev("fresh", 60), SIG, set(), now=NOW)
    assert s is SIG and a == "signal"


def test_old_news_is_stale():
    s, a = gate_trade(ev("old", 26 * 3600), SIG, set(), now=NOW)
    assert s is None and a == "stale"


def test_boundary_is_inclusive():
    assert gate_trade(ev("at limit", LIMIT), SIG, set(), now=NOW)[1] == "signal"
    assert gate_trade(ev("past limit", LIMIT + 1), SIG, set(), now=NOW)[1] == "stale"


def test_queue_dwell_counts():
    # Received quickly after publication (latency 1s), but it sat in the
    # queue: by decision time it is 20 minutes old -> stale.
    event = ev("dwelled", 20 * 60, latency_ms=1000)
    s, a = gate_trade(event, SIG, set(), now=NOW)
    assert s is None and a == "stale"


def test_unknown_publication_time_is_stale():
    event = ev("undated", 0, latency_ms=-1)
    assert news_age_seconds(event, NOW) is None
    s, a = gate_trade(event, SIG, set(), now=NOW)
    assert s is None and a == "stale"


def src_sig(news_source):
    """Tradeable signal tagged with a specific news source."""
    return Signal(market=MKT, claude_score=0.7, market_price=0.5, edge=0.2,
                  side="YES", bet_amount=25.0, reasoning="", headlines="h",
                  classification="bullish", materiality=0.35,
                  news_source=news_source)


def test_twitter_uses_15min_limit():
    # 15 min (900s) is the limit; just past it is stale, just under trades.
    twitter = src_sig("twitter")
    assert gate_trade(ev("t", 900), twitter, set(), now=NOW)[1] == "signal"
    s, a = gate_trade(ev("t", 901), twitter, set(), now=NOW)
    assert s is None and a == "stale"


def test_rss_uses_2h_limit():
    # 90 min old would be stale for Twitter, but RSS allows up to 2 hours.
    rss = src_sig("rss")
    assert gate_trade(ev("r", 90 * 60), rss, set(), now=NOW)[1] == "signal"
    s, a = gate_trade(ev("r", 2 * 3600 + 1), rss, set(), now=NOW)
    assert s is None and a == "stale"


def test_newsapi_uses_4h_limit():
    # 3 hours old would be stale for RSS, but NewsAPI allows up to 4 hours.
    newsapi = src_sig("newsapi")
    assert gate_trade(ev("n", 3 * 3600), newsapi, set(), now=NOW)[1] == "signal"
    s, a = gate_trade(ev("n", 4 * 3600 + 1), newsapi, set(), now=NOW)
    assert s is None and a == "stale"


def test_unknown_source_falls_back_to_15min():
    unknown = src_sig("mystery-wire")
    assert gate_trade(ev("u", 900), unknown, set(), now=NOW)[1] == "signal"
    s, a = gate_trade(ev("u", 901), unknown, set(), now=NOW)
    assert s is None and a == "stale"


def test_headline_cap_allows_one_trade():
    traded = set()
    assert gate_trade(ev("Hilton expands", 60), SIG, traded, now=NOW)[1] == "signal"
    s, a = gate_trade(ev("Hilton expands", 60), SIG, traded, now=NOW)
    assert s is None and a == "capped"


def test_stale_does_not_consume_headline_cap():
    traded = set()
    gate_trade(ev("old story", 26 * 3600), SIG, traded, now=NOW)
    assert "old story" not in traded


def test_no_edge_is_plain_skip():
    s, a = gate_trade(ev("whatever", 60), None, set(), now=NOW)
    assert s is None and a == "skip"


# --- Direction-specific materiality floors -------------------------------

def test_bullish_below_floor_is_low_materiality():
    s, a = gate_trade(ev("h", 60), sig(0.25, "bullish"), set(), now=NOW)
    assert s is None and a == "low_materiality"


def test_bullish_at_floor_trades():
    assert gate_trade(ev("h", 60), sig(0.3, "bullish"), set(), now=NOW)[1] == "signal"


def test_bearish_uses_higher_floor():
    # 0.35 clears the bullish floor (0.3) but not the bearish floor (0.4).
    s, a = gate_trade(ev("h", 60), sig(0.35, "bearish"), set(), now=NOW)
    assert s is None and a == "low_materiality"


def test_bearish_at_floor_trades():
    assert gate_trade(ev("h", 60), sig(0.4, "bearish"), set(), now=NOW)[1] == "signal"


def test_low_materiality_does_not_consume_headline_cap():
    traded = set()
    gate_trade(ev("weak news", 60), sig(0.1, "bullish"), traded, now=NOW)
    assert "weak news" not in traded


# --- High-materiality multi-source confirmation gate ---------------------

def _seed_classification(db, source, direction="bullish", condition_id="c1", ago_hours=0.5):
    """Insert a prior directional classification at NOW - ago_hours."""
    created_at = (NOW - timedelta(hours=ago_hours)).strftime("%Y-%m-%d %H:%M:%S")
    conn = db._conn()
    conn.execute(
        """INSERT INTO classifications
           (market_question, headline, news_source, direction, action,
            condition_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("Will X happen?", "prior", source, direction, "signal", condition_id, created_at),
    )
    conn.commit()
    conn.close()


def test_high_materiality_single_source_needs_confirmation(tmp_db):
    # No prior sources: only this event's source counts -> hold.
    s, a = gate_trade(ev("big news", 60), sig(0.6, "bullish"), set(), now=NOW)
    assert s is None and a == "needs_confirmation"


def test_high_materiality_second_source_confirms(tmp_db):
    _seed_classification(tmp_db, source="twitter", direction="bullish")
    # Event source is "rss" (see ev); twitter prior makes two distinct sources.
    s, a = gate_trade(ev("big news", 60), sig(0.6, "bullish"), set(), now=NOW)
    assert s is not None and a == "signal"


def test_high_materiality_same_source_not_enough(tmp_db):
    _seed_classification(tmp_db, source="rss", direction="bullish")
    s, a = gate_trade(ev("big news", 60), sig(0.6, "bullish"), set(), now=NOW)
    assert s is None and a == "needs_confirmation"


def test_high_materiality_other_direction_not_counted(tmp_db):
    _seed_classification(tmp_db, source="twitter", direction="bearish")
    s, a = gate_trade(ev("big news", 60), sig(0.6, "bullish"), set(), now=NOW)
    assert s is None and a == "needs_confirmation"


def test_high_materiality_stale_confirmation_not_counted(tmp_db):
    # Prior source agrees but is 3h old, outside the 2h window.
    _seed_classification(tmp_db, source="twitter", direction="bullish", ago_hours=3)
    s, a = gate_trade(ev("big news", 60), sig(0.6, "bullish"), set(), now=NOW)
    assert s is None and a == "needs_confirmation"


def test_needs_confirmation_does_not_consume_headline_cap(tmp_db):
    traded = set()
    gate_trade(ev("big news", 60), sig(0.6, "bullish"), traded, now=NOW)
    assert "big news" not in traded
