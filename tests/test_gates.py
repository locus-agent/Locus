"""Trade-time risk gates: freshness (incl. queue dwell + unknown dates) and headline cap."""
from datetime import datetime, timedelta, timezone

import pytest

from locus import config
from locus.core.pipeline import gate_trade, news_age_seconds, consume_headline
from locus.core.edge import Signal
from locus.markets.gamma import Market
from locus.sources.news_stream import NewsEvent

NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=timezone.utc)
# Default (unknown-source) freshness window. A signal with no news_source falls
# back to this in get_max_age_seconds.
LIMIT = config.MAX_NEWS_AGE_SECONDS_DEFAULT


@pytest.fixture(autouse=True)
def _pin_materiality_thresholds(monkeypatch):
    """Pin the direction-specific materiality floors, the high-materiality
    confirmation gate, and the source freshness windows to their standard
    defaults so the tests don't depend on the developer's .env overrides (same
    pattern used for other configs). The floor tests below are written against
    these: bullish 0.3, bearish 0.4 (bearish gets a higher bar), and the
    confirmation gate at 0.5 needing 2 distinct sources."""
    monkeypatch.setattr(config, "MATERIALITY_THRESHOLD_BULLISH", 0.3)
    monkeypatch.setattr(config, "MATERIALITY_THRESHOLD_BEARISH", 0.4)
    monkeypatch.setattr(config, "HIGH_MATERIALITY_THRESHOLD", 0.5)
    monkeypatch.setattr(config, "MIN_CONFIRMING_SOURCES", 2)
    monkeypatch.setattr(config, "MAX_NEWS_AGE_SECONDS_DEFAULT", 14400)
    monkeypatch.setattr(config, "MAX_NEWS_AGE_SECONDS_TWITTER", 10800)
    monkeypatch.setattr(config, "MAX_NEWS_AGE_SECONDS_RSS", 21600)
    monkeypatch.setattr(config, "MAX_NEWS_AGE_SECONDS_NEWSAPI", 18000)

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
    # queue: by decision time it is 5 hours old -> stale (past the 4h default).
    event = ev("dwelled", 5 * 3600, latency_ms=1000)
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


def test_twitter_uses_3h_limit():
    # 3h (10800s) is the limit; just past it is stale, just under trades.
    twitter = src_sig("twitter")
    assert gate_trade(ev("t", 10800), twitter, set(), now=NOW)[1] == "signal"
    s, a = gate_trade(ev("t", 10801), twitter, set(), now=NOW)
    assert s is None and a == "stale"


def test_rss_uses_6h_limit():
    # 4 hours old would be stale for Twitter (3h), but RSS allows up to 6 hours.
    rss = src_sig("rss")
    assert gate_trade(ev("r", 4 * 3600), rss, set(), now=NOW)[1] == "signal"
    s, a = gate_trade(ev("r", 6 * 3600 + 1), rss, set(), now=NOW)
    assert s is None and a == "stale"


def test_newsapi_uses_5h_limit():
    # NewsAPI allows up to 5 hours; just past it is stale.
    newsapi = src_sig("newsapi")
    assert gate_trade(ev("n", 5 * 3600), newsapi, set(), now=NOW)[1] == "signal"
    s, a = gate_trade(ev("n", 5 * 3600 + 1), newsapi, set(), now=NOW)
    assert s is None and a == "stale"


def test_unknown_source_falls_back_to_default():
    # Unknown source uses the 4h default window.
    unknown = src_sig("mystery-wire")
    assert gate_trade(ev("u", 4 * 3600), unknown, set(), now=NOW)[1] == "signal"
    s, a = gate_trade(ev("u", 4 * 3600 + 1), unknown, set(), now=NOW)
    assert s is None and a == "stale"


def test_gate_trade_approves_without_consuming_headline():
    # gate_trade approves ("signal") but no longer records the headline — the
    # cap is consumed later, after every gate, in consume_headline.
    traded = set()
    s, a = gate_trade(ev("Hilton expands", 60), SIG, traded, now=NOW)
    assert s is SIG and a == "signal"
    assert "Hilton expands" not in traded


def test_capped_when_headline_already_consumed():
    # A headline already recorded (a prior open consumed it) makes the next
    # match on the same headline capped.
    traded = {"Hilton expands"}
    s, a = gate_trade(ev("Hilton expands", 60), SIG, traded, now=NOW)
    assert s is None and a == "capped"


def test_stale_does_not_consume_headline_cap():
    traded = set()
    gate_trade(ev("old story", 26 * 3600), SIG, traded, now=NOW)
    assert "old story" not in traded


# --- consume_headline: cap consumed only at the confirmed open ------------
# "open" == a signal that cleared EVERY gate and is being opened (signal is not
# None at the terminal step), distinct from the earlier "signal" gate_trade
# returns before the late gates (CoV, orderbook, ...) run.

def test_consume_headline_marks_on_open():
    traded = set()
    assert consume_headline(traded, "big news", SIG) is True
    assert "big news" in traded


def test_consume_headline_skips_when_late_gate_blocked():
    # CoV / orderbook (and every other late gate) null the signal -> the
    # headline must stay free for other markets matched to the same headline.
    traded = set()
    assert consume_headline(traded, "big news", None) is False
    assert "big news" not in traded


def test_late_gate_block_leaves_headline_for_next_market():
    # End-to-end of the timing fix. Market A is approved by gate_trade (no
    # consume), then a late gate blocks it (signal None). Market B on the SAME
    # headline must therefore still pass gate_trade and be able to open.
    traded = set()
    sA, aA = gate_trade(ev("shared headline", 60), SIG, traded, now=NOW)
    assert aA == "signal" and "shared headline" not in traded   # not consumed yet
    consume_headline(traded, "shared headline", None)            # A blocked late
    assert "shared headline" not in traded                       # still free

    sB, aB = gate_trade(ev("shared headline", 60), SIG, traded, now=NOW)
    assert aB == "signal"                                        # B not capped
    consume_headline(traded, "shared headline", sB)              # B opens
    assert "shared headline" in traded                           # now consumed


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
