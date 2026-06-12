"""Keyword matching must match whole tokens, not substrings."""
from locus.core.matcher import match_news_to_markets, match_news_to_markets_hybrid, prefilter_match, tokenize
from locus.markets.gamma import Market


def _mkt(question, cid="c1"):
    return Market(cid, question, "crypto", 0.5, 0.5, 5000, "", True, [])


def test_tokenize_strips_punctuation():
    assert "bitcoin" in tokenize("Bitcoin falls below $60,000!")
    assert "$60,000" in tokenize("Bitcoin falls below $60,000")


def test_substring_junk_no_longer_matches():
    cap_market = _mkt("Will Cap launch a token by June 30, 2026?")
    matches = match_news_to_markets(
        "4 Artificial Intelligence (AI) Companies Are Planning to Raise More Capital",
        [cap_market],
    )
    assert matches == [], '"cap" must not match inside "Capital"'


def test_whole_token_still_matches():
    btc = _mkt("Will the price of Bitcoin be above $60,000 on June 12?")
    matches = match_news_to_markets(
        "Bitcoin falls below $60,000 as ETF outflows accelerate", [btc]
    )
    assert matches == [btc]


def test_max_matches_respected():
    markets = [_mkt(f"Will Bitcoin reach ${i}000 in June?", cid=f"c{i}") for i in range(10)]
    matches = match_news_to_markets("Bitcoin surges in June", markets, max_matches=5)
    assert len(matches) == 5


def test_hybrid_without_index_is_keyword_only():
    btc = _mkt("Will the price of Bitcoin be above $60,000 on June 12?")
    result = match_news_to_markets_hybrid("Bitcoin falls below $60,000", [btc], index=None)
    assert len(result) == 1
    market, source, score = result[0]
    assert market is btc and source == "keyword"
    assert 0 < score <= 1


def test_prefilter_skips_weak_offtopic_keyword_matches(monkeypatch):
    from locus import config
    monkeypatch.setattr(config, "PREFILTER_KEYWORD_SCORE", 0.25)
    crypto_mkt = _mkt("Will Cap launch a token by June 30, 2026?")  # category crypto
    # weak keyword match (score 0.1), headline topic "other" -> prefiltered
    assert prefilter_match("Blink camera kit is a great deal", crypto_mkt, "keyword", 0.10) is True
    # same weak score but headline is on-topic for crypto -> kept
    assert prefilter_match("Bitcoin token launch rumors", crypto_mkt, "keyword", 0.10) is False
    # strong keyword match -> kept regardless of topic
    assert prefilter_match("Blink camera kit is a great deal", crypto_mkt, "keyword", 0.50) is False
    # embedding-backed matches always pass
    assert prefilter_match("Blink camera kit", crypto_mkt, "embedding", 0.10) is False
    assert prefilter_match("Blink camera kit", crypto_mkt, "both", 0.10) is False
