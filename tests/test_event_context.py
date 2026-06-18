"""Event context awareness: grouping, implied opportunities, best-outcome
selection, and per-event exposure."""
import pytest

from locus import config
from locus.core import event_context
from locus.core.edge import Signal
from locus.markets.gamma import Market


@pytest.fixture(autouse=True)
def deterministic_config(monkeypatch):
    monkeypatch.setattr(config, "EDGE_THRESHOLD", 0.10)
    monkeypatch.setattr(config, "KELLY_BANKROLL_USD", 100.0)
    monkeypatch.setattr(config, "MAX_BET_USD", 25.0)


def mkt(cid, price, event_id="e1", q=None):
    return Market(cid, q or f"Will {cid} win?", "politics", price, round(1 - price, 4),
                  5000, "", True, [], event_id=event_id)


def sig(market, side="YES", direction="bullish", materiality=0.6, confidence=0.6):
    edge = event_context._edge_for(side, market.yes_price, materiality)
    return Signal(market=market, claude_score=materiality, market_price=market.yes_price,
                  edge=edge, side=side, bet_amount=10.0, reasoning="r", headlines="h",
                  news_source="rss", classification=direction, materiality=materiality,
                  confidence=confidence)


def pos(condition_id, event_id="e1", amount=10.0):
    return {"condition_id": condition_id, "event_id": event_id, "amount_usd": amount}


# --- get_event_markets ---------------------------------------------------

def test_event_grouping_returns_same_event_members():
    tracked = [mkt("a", 0.4, "e1"), mkt("b", 0.3, "e1"),
               mkt("c", 0.5, "e2"), mkt("d", 0.6, "")]
    members = event_context.get_event_markets("e1", tracked)
    assert {m.condition_id for m in members} == {"a", "b"}


def test_event_grouping_empty_event_id_returns_nothing():
    tracked = [mkt("a", 0.4, "e1"), mkt("b", 0.3, "")]
    assert event_context.get_event_markets("", tracked) == []


# --- _has_room price guards (must read config, not hardcoded bands) ------

@pytest.fixture
def pinned_price_bands(monkeypatch):
    """Pin the price-room bands to the documented edge.detect_edge_v2 defaults."""
    monkeypatch.setattr(config, "BULLISH_MIN_PRICE", 0.12)
    monkeypatch.setattr(config, "BULLISH_MAX_PRICE", 0.82)
    monkeypatch.setattr(config, "BEARISH_MIN_PRICE", 0.18)
    monkeypatch.setattr(config, "BEARISH_MAX_PRICE", 0.88)


def test_has_room_uses_config_bullish_band(pinned_price_bands):
    # The old hardcoded YES band was 0.05-0.85; with the config band (0.12-0.82)
    # a 0.10 longshot and a 0.84 favourite are now correctly rejected.
    assert event_context._has_room("YES", 0.10) is False   # was allowed under 0.05
    assert event_context._has_room("YES", 0.12) is True    # at config min
    assert event_context._has_room("YES", 0.82) is True    # at config max
    assert event_context._has_room("YES", 0.84) is False   # was allowed under 0.85


def test_has_room_uses_config_bearish_band(pinned_price_bands):
    # The old hardcoded NO band was 0.15-0.95; config band is 0.18-0.88.
    assert event_context._has_room("NO", 0.16) is False    # was allowed under 0.15
    assert event_context._has_room("NO", 0.18) is True     # at config min
    assert event_context._has_room("NO", 0.88) is True     # at config max
    assert event_context._has_room("NO", 0.92) is False    # was allowed under 0.95


def test_has_room_tracks_config_overrides(monkeypatch):
    # Proves the guard reads config at call time rather than hardcoded numbers.
    monkeypatch.setattr(config, "BULLISH_MIN_PRICE", 0.30)
    monkeypatch.setattr(config, "BULLISH_MAX_PRICE", 0.70)
    assert event_context._has_room("YES", 0.25) is False
    assert event_context._has_room("YES", 0.50) is True
    assert event_context._has_room("YES", 0.75) is False


def test_find_best_outcome_respects_config_band(monkeypatch):
    # A sibling priced just outside the bullish band must not be recommended as
    # an implied NO/YES play, even with strong materiality.
    monkeypatch.setattr(config, "EDGE_THRESHOLD", 0.05)
    monkeypatch.setattr(config, "BEARISH_MIN_PRICE", 0.18)
    monkeypatch.setattr(config, "BEARISH_MAX_PRICE", 0.88)
    # Bearish on A implies a YES play on its siblings. Give the sibling a price
    # below the bullish band so the implied YES has no room.
    monkeypatch.setattr(config, "BULLISH_MIN_PRICE", 0.12)
    monkeypatch.setattr(config, "BULLISH_MAX_PRICE", 0.82)
    a = mkt("a", 0.6, q="Will A win?")
    sibling = mkt("b", 0.08, q="Will B win?")  # 0.08 < BULLISH_MIN_PRICE
    rec = event_context.find_best_outcome(
        sig(a, side="NO", direction="bearish", materiality=0.9), [a, sibling], []
    )
    # The only candidate that can clear is the direct signal (not the sibling).
    assert rec is None or rec["recommended_market"].condition_id == "a"


# --- is_categorical ------------------------------------------------------

def test_categorical_when_prices_sum_to_one():
    assert event_context.is_categorical([mkt("a", 0.5), mkt("b", 0.3), mkt("c", 0.2)])


def test_not_categorical_when_prices_far_from_one():
    assert not event_context.is_categorical([mkt("a", 0.5), mkt("b", 0.9)])


def test_not_categorical_single_market():
    assert not event_context.is_categorical([mkt("a", 0.5)])


# --- find_best_outcome ---------------------------------------------------

def test_bullish_switches_to_higher_edge_sibling():
    # A+B sum ~1.15 (categorical). Bullish on A (edge 0.5*0.4=0.20); the implied
    # NO on B has more room (edge 0.5*0.55=0.275) -> switch to NO on B.
    a, b = mkt("a", 0.6), mkt("b", 0.55)
    rec = event_context.find_best_outcome(sig(a, materiality=0.5), [a, b], [])
    assert rec["recommended_market"].condition_id == "b"
    assert rec["recommended_side"] == "NO"
    assert rec["implied_edge"] == pytest.approx(0.275, abs=1e-6)
    assert "implied bearish" in rec["reason"]


def test_no_switch_when_direct_signal_is_best():
    a, b, c = mkt("a", 0.5), mkt("b", 0.3), mkt("c", 0.2)
    rec = event_context.find_best_outcome(sig(a, materiality=0.6), [a, b, c], [])
    assert rec["recommended_market"].condition_id == "a"  # direct YES, edge 0.30
    assert rec["recommended_side"] == "YES"


def test_bearish_implies_bullish_on_siblings():
    # Bearish on A -> siblings become more likely -> implied YES. With A+B<1,
    # the implied YES on B (edge 0.5*0.55=0.275) beats direct NO on A (0.225).
    a, b = mkt("a", 0.45), mkt("b", 0.45)
    rec = event_context.find_best_outcome(
        sig(a, side="NO", direction="bearish", materiality=0.5), [a, b], []
    )
    assert rec["recommended_market"].condition_id == "b"
    assert rec["recommended_side"] == "YES"
    assert "implied bullish" in rec["reason"]


def test_non_categorical_event_has_no_implied_plays():
    # Prices sum to 1.4 -> not categorical; only the direct signal is a candidate.
    a, b = mkt("a", 0.5), mkt("b", 0.9)
    rec = event_context.find_best_outcome(sig(a, materiality=0.6), [a, b], [])
    assert rec["recommended_market"].condition_id == "a"


def test_held_sibling_is_excluded_from_recommendation():
    a, b = mkt("a", 0.6), mkt("b", 0.55)
    # We already hold B, so the best alternative falls back to the direct signal.
    rec = event_context.find_best_outcome(
        sig(a, materiality=0.5), [a, b], [pos("b")]
    )
    assert rec["recommended_market"].condition_id == "a"


def test_returns_none_when_nothing_clears_edge_threshold():
    # Tiny materiality -> all edges below EDGE_THRESHOLD.
    a, b = mkt("a", 0.6), mkt("b", 0.55)
    assert event_context.find_best_outcome(sig(a, materiality=0.05), [a, b], []) is None


def test_implied_play_skips_outcome_without_price_room():
    # Bullish on A (direct YES edge 0.6*0.2=0.12); sibling B at 0.10 has no NO
    # room (needs >= 0.15) -> skipped, so the direct signal wins.
    a, b = mkt("a", 0.80), mkt("b", 0.10)
    rec = event_context.find_best_outcome(sig(a, materiality=0.6), [a, b], [])
    assert rec["recommended_market"].condition_id == "a"


# --- correlation gate on switched sibling outcomes -----------------------

def test_correlation_high_drops_switched_sibling(monkeypatch):
    # Bullish on A makes the implied NO on sibling B the highest-edge play
    # (see test_bullish_switches_to_higher_edge_sibling). But we already hold a
    # large same-topic position, so the correlation gate (HIGH risk) must drop B
    # and fall back to the direct signal on A.
    monkeypatch.setattr(config, "EDGE_THRESHOLD", 0.10)
    a = mkt("a", 0.6, q="Will Hilton win?")
    b = mkt("b", 0.55, q="Will Biden win?")
    book = [{"condition_id": "z", "event_id": "e2",
             "market_question": "Will Biden resign?", "side": "NO",
             "amount_usd": 80.0}]  # shares 'biden', >$75 -> high risk
    rec = event_context.find_best_outcome(sig(a, materiality=0.5), [a, b], book)
    assert rec["recommended_market"].condition_id == "a"
    assert rec["recommended_side"] == "YES"


def test_correlation_medium_does_not_drop_sibling(monkeypatch):
    # Medium correlation risk only warns (in the pipeline) — it must not drop the
    # higher-edge sibling switch.
    monkeypatch.setattr(config, "EDGE_THRESHOLD", 0.10)
    a = mkt("a", 0.6, q="Will Hilton win?")
    b = mkt("b", 0.55, q="Will Biden win?")
    book = [{"condition_id": "z", "event_id": "e2",
             "market_question": "Will Biden resign?", "side": "NO",
             "amount_usd": 60.0}]  # shares 'biden', $50-75 -> medium risk
    rec = event_context.find_best_outcome(sig(a, materiality=0.5), [a, b], book)
    assert rec["recommended_market"].condition_id == "b"
    assert rec["recommended_side"] == "NO"


def test_correlation_high_on_all_candidates_returns_none(monkeypatch):
    # Both the direct signal and the sibling are HIGH-risk correlated with the
    # open book -> nothing clears, so no switch/open at all.
    monkeypatch.setattr(config, "EDGE_THRESHOLD", 0.10)
    a = mkt("a", 0.6, q="Will Biden win?")
    b = mkt("b", 0.55, q="Will Biden lose?")
    book = [{"condition_id": "z", "event_id": "e2",
             "market_question": "Will Biden resign?", "side": "NO",
             "amount_usd": 80.0}]  # shares 'biden' with both -> high risk
    assert event_context.find_best_outcome(sig(a, materiality=0.5), [a, b], book) is None


# --- build_switched_signal ----------------------------------------------

def test_build_switched_signal_resizes_and_relabels():
    a, b = mkt("a", 0.6), mkt("b", 0.55)
    original = sig(a, materiality=0.5, confidence=0.6)
    rec = event_context.find_best_outcome(original, [a, b], [])
    switched = event_context.build_switched_signal(original, rec)
    assert switched.market.condition_id == "b"
    assert switched.side == "NO"
    assert switched.classification == "bearish"
    assert switched.market_price == 0.55
    assert switched.materiality == 0.5 and switched.confidence == 0.6
    assert "[event-switch]" in switched.reasoning
    assert switched.bet_amount >= 1.0


# --- get_event_exposure --------------------------------------------------

def test_event_exposure_sums_related_positions():
    book = [pos("a", "e1", 10.0), pos("b", "e1", 15.0), pos("c", "e2", 20.0)]
    exp = event_context.get_event_exposure("e1", book)
    assert exp["position_count"] == 2
    assert exp["total_exposure_usd"] == 25.0


def test_event_exposure_zero_for_unknown_event():
    book = [pos("a", "e1", 10.0)]
    exp = event_context.get_event_exposure("e2", book)
    assert exp["position_count"] == 0 and exp["total_exposure_usd"] == 0.0


def test_event_exposure_blocks_at_max_positions_per_event():
    # The pipeline blocks when count >= MAX_POSITIONS_PER_EVENT (default 1).
    book = [pos("a", "e1", 10.0)]
    exp = event_context.get_event_exposure("e1", book)
    assert exp["position_count"] >= config.MAX_POSITIONS_PER_EVENT
