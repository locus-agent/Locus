"""Trade-time risk gates: freshness (incl. queue dwell + unknown dates) and headline cap."""
from datetime import datetime, timedelta, timezone

from locus import config
from locus.core.pipeline import gate_trade, news_age_seconds
from locus.core.edge import Signal
from locus.markets.gamma import Market
from locus.sources.news_stream import NewsEvent

NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=timezone.utc)
LIMIT = config.MAX_NEWS_AGE_SECONDS

MKT = Market("c1", "Will X happen?", "ai", 0.5, 0.5, 5000, "", True, [])
SIG = Signal(market=MKT, claude_score=0.7, market_price=0.5, edge=0.2,
             side="YES", bet_amount=25.0, reasoning="", headlines="h")


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
