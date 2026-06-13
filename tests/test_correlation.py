"""Correlation tracker: topic extraction, risk tiers, and the pipeline gate."""
from datetime import datetime, timezone

from locus import config
from locus.core import positions
from locus.core.pipeline import gate_trade
from locus.core.edge import Signal
from locus.markets.gamma import Market
from locus.sources.news_stream import NewsEvent

NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=timezone.utc)


def _pos(question, side="YES", amount=25.0):
    return {"market_question": question, "side": side, "amount_usd": amount}


# --- entity / topic extraction ---

def test_extract_capitalized_entities():
    topics = positions.extract_topics("Will SpaceX list on the NYSE in 2026?")
    assert "spacex" in topics
    assert "nyse" in topics


def test_extract_acronym_and_lowercase_topic():
    topics = positions.extract_topics("Will an AI model trigger a recession?")
    assert "ai" in topics            # two-letter acronym
    assert "recession" in topics     # lowercase keyword list


def test_scaffolding_words_are_not_topics():
    topics = positions.extract_topics("Will the next president win?")
    assert "will" not in topics
    assert "the" not in topics
    assert "win" not in topics
    assert "president" in topics


def test_shared_topic_intersects():
    a = positions.extract_topics("Will Trump win the 2026 primary?")
    b = positions.extract_topics("Will Trump pardon someone before July?")
    assert "trump" in (a & b)


# --- risk tiers ---

def test_low_risk_no_overlap():
    book = [_pos("Will SpaceX list on the NYSE?"), _pos("Will the Fed cut rates?")]
    r = positions.check_correlation_risk("Will Trump win the primary?", "YES", book)
    assert r["risk_level"] == "low"
    assert r["correlated_positions"] == []
    assert r["total_exposure_usd"] == 0.0


def test_medium_two_shared_positions():
    book = [
        _pos("Will Trump win the GOP primary?", amount=10.0),
        _pos("Will Trump be indicted again?", amount=10.0),
    ]
    r = positions.check_correlation_risk("Will Trump debate in 2026?", "YES", book)
    assert len(r["correlated_positions"]) == 2
    assert r["total_exposure_usd"] == 20.0   # exposure below $50
    assert r["risk_level"] == "medium"       # tier driven by count == 2


def test_medium_by_exposure_single_position():
    book = [_pos("Will Trump win the primary?", amount=60.0)]
    r = positions.check_correlation_risk("Will Trump debate?", "YES", book)
    assert r["risk_level"] == "medium"       # one position, but >$50 exposure


def test_high_three_shared_positions():
    book = [
        _pos("Will Trump win the primary?", amount=5.0),
        _pos("Will Trump be indicted?", amount=5.0),
        _pos("Will Trump debate?", amount=5.0),
    ]
    r = positions.check_correlation_risk("Will Trump pardon someone?", "YES", book)
    assert len(r["correlated_positions"]) == 3
    assert r["risk_level"] == "high"         # 3+ positions


def test_high_by_exposure():
    book = [_pos("Will Trump win the primary?", amount=80.0)]
    r = positions.check_correlation_risk("Will Trump debate?", "YES", book)
    assert r["total_exposure_usd"] == 80.0   # >$75
    assert r["risk_level"] == "high"


def test_empty_book_is_low():
    r = positions.check_correlation_risk("Will Trump win?", "YES", [])
    assert r["risk_level"] == "low"


# --- pipeline gate ---

MKT = Market("c1", "Will Trump win the 2026 primary?", "election", 0.5, 0.5,
             5000, "", True, [])
SIG = Signal(market=MKT, claude_score=0.7, market_price=0.5, edge=0.2,
             side="YES", bet_amount=25.0, reasoning="", headlines="h")


def _ev(headline):
    return NewsEvent(headline=headline, source="rss", url="",
                     received_at=NOW, published_at=NOW, latency_ms=0)


def _gate_with_correlation(book):
    """Mirror the pipeline's gate_trade -> correlation sequence."""
    signal, action = gate_trade(_ev("Trump news"), SIG, set(), now=NOW)
    if signal is not None:
        corr = positions.check_correlation_risk(MKT.question, signal.side, book)
        if corr["risk_level"] == "high":
            signal, action = None, "correlation_block"
    return signal, action


def test_gate_blocks_high_correlation():
    book = [
        _pos("Will Trump be indicted?"),
        _pos("Will Trump debate?"),
        _pos("Will Trump pardon someone?"),
    ]
    signal, action = _gate_with_correlation(book)
    assert signal is None
    assert action == "correlation_block"


def test_gate_allows_medium_correlation():
    book = [
        _pos("Will Trump be indicted?", amount=10.0),
        _pos("Will Trump debate?", amount=10.0),
    ]
    signal, action = _gate_with_correlation(book)
    assert signal is SIG          # 2 related positions, $20 -> medium, allowed
    assert action == "signal"


def test_gate_allows_when_uncorrelated():
    book = [_pos("Will SpaceX list on the NYSE?"), _pos("Will the Fed cut rates?")]
    signal, action = _gate_with_correlation(book)
    assert signal is SIG
    assert action == "signal"
