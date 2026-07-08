"""Market-structure gates in gate_trade: time-to-resolution and price-target
exclusion, plus the is_price_target_market helper."""
import logging
from datetime import datetime, timedelta, timezone

import pytest

from locus import config
from locus.core.pipeline import gate_trade, is_price_target_market
from locus.core.edge import Signal
from locus.markets.gamma import Market, is_coinflip_market
from locus.sources.news_stream import NewsEvent

NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _pin_filter_config(monkeypatch):
    """Pin the new filter knobs to their documented defaults so the tests don't
    depend on the developer's .env. Also pin the materiality floors so a passing
    signal reaches the market-structure gates."""
    monkeypatch.setattr(config, "MIN_HOURS_TO_RESOLUTION", 4.0)
    monkeypatch.setattr(config, "EXCLUDE_PRICE_TARGET_MARKETS", True)
    monkeypatch.setattr(config, "PRICE_TARGET_KEYWORDS", [
        "reach $", "hit $", "above $", "below $", "exceed $", "surpass $",
        "dip to $", "dips to $", "fall to $", "falls to $",
        "drop to $", "drops to $", "sink to $", "sinks to $",
        "crash to $", "crashes to $",
        "new all-time high", "new ath", "all time high",
        "will bitcoin reach", "will ethereum hit",
    ])
    monkeypatch.setattr(config, "EXCLUDE_COINFLIP_MARKETS", True)
    monkeypatch.setattr(config, "COINFLIP_PATTERNS", ["up or down"])
    monkeypatch.setattr(config, "MIN_MATERIALITY_DEFAULT", 0.33)
    monkeypatch.setattr(config, "MIN_MATERIALITY_BULLISH", 0.3)
    monkeypatch.setattr(config, "MIN_MATERIALITY_BEARISH", 0.4)
    monkeypatch.setattr(config, "HIGH_MATERIALITY_THRESHOLD", 0.5)


def mkt(question="Will X happen?", end_date="", slug="some-market"):
    return Market("c1", question, "crypto", 0.5, 0.5, 5000, end_date, True, [],
                  slug=slug)


def sig(market):
    """A would-be signal that clears the materiality floor by default."""
    return Signal(market=market, claude_score=0.7, market_price=0.5, edge=0.2,
                  side="YES", bet_amount=25.0, reasoning="", headlines="h",
                  classification="bullish", materiality=0.35)


def ev(headline="fresh", published_ago_s=60):
    pub = NOW - timedelta(seconds=published_ago_s)
    return NewsEvent(headline=headline, source="rss", url="",
                     received_at=NOW, published_at=pub,
                     latency_ms=int(published_ago_s * 1000))


def end_in_hours(h):
    return (NOW + timedelta(hours=h)).isoformat()


# --- Time-to-resolution gate ---------------------------------------------

def test_too_close_blocks_under_4h():
    s, a = gate_trade(ev(), sig(mkt(end_date=end_in_hours(3.9))), set(), now=NOW)
    assert s is None and a == "too_close_to_resolution"


def test_allows_over_4h():
    s, a = gate_trade(ev(), sig(mkt(end_date=end_in_hours(5))), set(), now=NOW)
    assert s is not None and a == "signal"


def test_boundary_at_4h_allows():
    # Exactly at the floor is not "< MIN_HOURS_TO_RESOLUTION" -> allowed.
    s, a = gate_trade(ev(), sig(mkt(end_date=end_in_hours(4))), set(), now=NOW)
    assert s is not None and a == "signal"


def test_unknown_end_date_does_not_block():
    # No close time -> can't compute -> the gate doesn't fire.
    s, a = gate_trade(ev(), sig(mkt(end_date="")), set(), now=NOW)
    assert s is not None and a == "signal"


def test_too_close_does_not_consume_headline_cap():
    traded = set()
    gate_trade(ev("soon"), sig(mkt(end_date=end_in_hours(1))), traded, now=NOW)
    assert "soon" not in traded


def test_too_close_logging_format(caplog):
    with caplog.at_level(logging.INFO, logger="locus.core.pipeline"):
        gate_trade(ev(), sig(mkt(end_date=end_in_hours(2), slug="btc-100k")),
                   set(), now=NOW)
    assert "Filtered: too_close_to_resolution | btc-100k | 2.0h left" in caplog.text


# --- Price-target gate ----------------------------------------------------

PRICE_TARGET_QUESTIONS = [
    "Will Bitcoin reach $100,000 by July?",
    "Will ETH hit $5,000?",
    "Bitcoin above $90k in 2026?",
    "Will the price fall below $50?",
    "Will Solana exceed $300?",
    "Will gold surpass $3,000?",
    "Bitcoin new all-time high this year?",
    "Will BTC set a new ATH?",
    "Will Ethereum reach an all time high?",
    "Will Bitcoin reach the moon?",
    "Will Ethereum hit a record?",
    # Downside verb + threshold phrasings (position 56 regression).
    "Will Bitcoin dip to $60,000 in July?",
    "Bitcoin dips to $58,000 this week?",
    "Will ETH fall to $2,000?",
    "Bitcoin falls to $50k by August?",
    "Will Solana drop to $100?",
    "Dogecoin drops to $0.10 in 2026?",
    "Will Bitcoin drop below $55,000?",   # covered by the existing "below $"
    "Will Bitcoin sink to $40,000?",
    "Will ETH crash to $1,000?",
]


NON_PRICE_TARGET_QUESTIONS = [
    "Will the Fed cut rates in July?",
    # A "$" threshold in a non-asset-price context must not be swept in.
    "Will X raise $60,000 for charity?",
    # "rise to $" is deliberately NOT a keyword (policy/salary markets).
    "Will the federal minimum wage rise to $15 per hour?",
    "Will the average salary rise to $60,000?",
    "Will Ethereum launch sharding by 2027?",
]


@pytest.mark.parametrize("question", NON_PRICE_TARGET_QUESTIONS)
def test_non_price_threshold_dollar_mentions_not_flagged(question):
    assert not is_price_target_market(question)


@pytest.mark.parametrize("question", PRICE_TARGET_QUESTIONS)
def test_price_target_catches_all_keywords(question):
    # Case-insensitive: upper-case the whole question, the gate still catches it.
    assert is_price_target_market(question)
    assert is_price_target_market(question.upper())


def test_price_target_gate_blocks_in_gate_trade():
    s, a = gate_trade(ev(), sig(mkt(question="Will Bitcoin reach $100k?",
                                    end_date=end_in_hours(48))), set(), now=NOW)
    assert s is None and a == "price_target_market"


def test_non_price_target_market_allowed():
    assert not is_price_target_market("Will the Fed cut rates in July?")
    s, a = gate_trade(ev(), sig(mkt(question="Will the Fed cut rates in July?",
                                    end_date=end_in_hours(48))), set(), now=NOW)
    assert s is not None and a == "signal"


def test_disabled_bypasses_filter(monkeypatch):
    monkeypatch.setattr(config, "EXCLUDE_PRICE_TARGET_MARKETS", False)
    assert not is_price_target_market("Will Bitcoin reach $100k?")
    s, a = gate_trade(ev(), sig(mkt(question="Will Bitcoin reach $100k?",
                                    end_date=end_in_hours(48))), set(), now=NOW)
    assert s is not None and a == "signal"


def test_keyword_list_is_configurable(monkeypatch):
    monkeypatch.setattr(config, "PRICE_TARGET_KEYWORDS", ["custom phrase"])
    assert is_price_target_market("Some market with a custom phrase in it")
    # A default keyword no longer matches once the list is overridden.
    assert not is_price_target_market("Will Bitcoin reach $100k?")


def test_price_target_does_not_consume_headline_cap():
    traded = set()
    gate_trade(ev("btc target"),
               sig(mkt(question="Will Bitcoin reach $100k?", end_date=end_in_hours(48))),
               traded, now=NOW)
    assert "btc target" not in traded


def test_price_target_logging_format(caplog):
    q = "Will Bitcoin reach $100,000 by the end of the year and beyond all limits?"
    with caplog.at_level(logging.INFO, logger="locus.core.pipeline"):
        gate_trade(ev(), sig(mkt(question=q, end_date=end_in_hours(48),
                                 slug="btc-reach-100k")), set(), now=NOW)
    assert f"Filtered: price_target_market | btc-reach-100k | question: {q[:70]}..." in caplog.text


# --- Coinflip gate (safety net; primary filter is market_watcher) ---------

COINFLIP_QUESTIONS = [
    "Bitcoin Up or Down on June 29?",
    "Ethereum Up or Down today?",
    "Bitcoin UP OR DOWN on June 30?",   # case-insensitive
]


@pytest.mark.parametrize("question", COINFLIP_QUESTIONS)
def test_coinflip_helper_matches_case_insensitively(question):
    assert is_coinflip_market(question)
    assert is_coinflip_market(question.upper())
    assert is_coinflip_market(question.lower())


def test_coinflip_gate_blocks_in_gate_trade():
    s, a = gate_trade(ev(), sig(mkt(question="Bitcoin Up or Down on June 29?",
                                    end_date=end_in_hours(48))), set(), now=NOW)
    assert s is None and a == "coinflip_market"


def test_non_coinflip_market_allowed():
    assert not is_coinflip_market("Will the Fed cut rates in July?")
    s, a = gate_trade(ev(), sig(mkt(question="Will the Fed cut rates in July?",
                                    end_date=end_in_hours(48))), set(), now=NOW)
    assert s is not None and a == "signal"


def test_coinflip_disabled_bypasses_filter(monkeypatch):
    monkeypatch.setattr(config, "EXCLUDE_COINFLIP_MARKETS", False)
    assert not is_coinflip_market("Bitcoin Up or Down on June 29?")
    s, a = gate_trade(ev(), sig(mkt(question="Bitcoin Up or Down on June 29?",
                                    end_date=end_in_hours(48))), set(), now=NOW)
    assert s is not None and a == "signal"


def test_coinflip_does_not_consume_headline_cap():
    traded = set()
    gate_trade(ev("flip"),
               sig(mkt(question="Bitcoin Up or Down on June 29?", end_date=end_in_hours(48))),
               traded, now=NOW)
    assert "flip" not in traded


def test_coinflip_logging_format(caplog):
    q = "Bitcoin Up or Down on June 29 at exactly noon eastern time and onward?"
    with caplog.at_level(logging.INFO, logger="locus.core.pipeline"):
        gate_trade(ev(), sig(mkt(question=q, end_date=end_in_hours(48),
                                 slug="btc-up-down")), set(), now=NOW)
    assert f"Filtered: coinflip_market | btc-up-down | question: {q[:70]}..." in caplog.text
