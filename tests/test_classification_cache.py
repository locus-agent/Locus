"""Dedup memory: reuse stored classifications for repeat (headline, market) pairs."""
from locus.memory import logger as L


def _log(tmp_db, headline="h1", cid="c1", price=0.50, action="skip", direction="bullish"):
    return tmp_db.log_classification(
        market_question="Will X happen?", headline=headline, news_source="rss",
        direction=direction, materiality=0.5, edge=None, action=action,
        match_source="keyword", condition_id=cid, yes_price=price,
        yes_token_id="tok",
    )


def find(price=0.50, headline="h1", cid="c1"):
    return L.find_recent_classification(headline, cid, price, 0.02, 24.0)


def test_hit_on_same_pair_and_price(tmp_db):
    _log(tmp_db)
    prior = find(price=0.51)  # within +-0.02
    assert prior is not None
    assert prior["direction"] == "bullish"


def test_miss_when_price_moved(tmp_db):
    _log(tmp_db, price=0.50)
    assert find(price=0.60) is None  # 0.10 move: market changed, re-classify


def test_miss_on_different_market_or_headline(tmp_db):
    _log(tmp_db)
    assert find(cid="other-market") is None
    assert find(headline="different headline") is None


def test_derived_rows_are_not_reused(tmp_db):
    # cached/prefiltered/error rows must never seed further cache hits
    _log(tmp_db, action="cached")
    _log(tmp_db, action="prefiltered", direction=None)
    _log(tmp_db, action="error")
    assert find() is None


def test_most_recent_real_row_wins(tmp_db):
    _log(tmp_db, direction="bullish")
    _log(tmp_db, direction="bearish")
    assert find()["direction"] == "bearish"
