"""Orderbook imbalance gate: score math, allow/block logic, and fail-open fetch."""
import types

import pytest

from locus.core import orderbook
from locus.core.orderbook import (
    compute_imbalance,
    orderbook_allows,
    fetch_orderbook_imbalance,
    IMBALANCE_THRESHOLD,
)


# --- compute_imbalance ---

def test_balanced_book_is_zero():
    bids = [(0.49, 100.0)]
    asks = [(0.51, 100.0)]
    assert compute_imbalance(bids, asks) == 0.0


def test_all_bids_is_plus_one():
    assert compute_imbalance([(0.5, 50.0)], []) == 1.0


def test_all_asks_is_minus_one():
    assert compute_imbalance([], [(0.5, 50.0)]) == -1.0


def test_empty_book_is_none():
    assert compute_imbalance([], []) is None


def test_score_formula():
    # bid 300 vs ask 100 -> (300-100)/400 = 0.5
    bids = [(0.49, 200.0), (0.48, 100.0)]
    asks = [(0.51, 100.0)]
    assert compute_imbalance(bids, asks) == 0.5


def test_only_top_five_levels_count():
    # 6 bid levels of 10 each, but only the best 5 (=50) are summed; one ask 50.
    bids = [(0.40 + i / 100, 10.0) for i in range(6)]
    asks = [(0.60, 50.0)]
    # bid_volume capped at 50 -> (50-50)/100 = 0
    assert compute_imbalance(bids, asks) == 0.0


def test_best_levels_selected_not_first_seen():
    # Unsorted input: best 5 bids by price, best 5 asks by price.
    bids = [(0.10, 1.0), (0.49, 100.0)]   # best bid is 0.49/100
    asks = [(0.90, 1.0), (0.51, 100.0)]   # best ask is 0.51/100
    # with depth 5 both fit; volumes 101 vs 101 -> 0
    assert compute_imbalance(bids, asks) == 0.0


# --- orderbook_allows ---

def test_yes_blocked_by_strong_sell_pressure():
    assert orderbook_allows("YES", -0.5) is False     # below -0.3
    assert orderbook_allows("YES", -0.3) is True       # boundary inclusive
    assert orderbook_allows("YES", 0.5) is True        # buy pressure is fine


def test_no_blocked_by_strong_buy_pressure():
    assert orderbook_allows("NO", 0.5) is False        # above +0.3
    assert orderbook_allows("NO", 0.3) is True         # boundary inclusive
    assert orderbook_allows("NO", -0.5) is True         # sell pressure is fine


def test_none_score_blocks_by_default():
    # Fail closed: no book -> no trade (slippage and flow are unknowable).
    assert orderbook_allows("YES", None) is False
    assert orderbook_allows("NO", None) is False


def test_none_score_allows_when_fail_open_configured(monkeypatch):
    from locus import config
    monkeypatch.setattr(config, "ORDERBOOK_FAIL_OPEN", True)
    assert orderbook_allows("YES", None) is True
    assert orderbook_allows("NO", None) is True


def test_threshold_constant_is_point_three():
    assert IMBALANCE_THRESHOLD == 0.3


# --- fetch_orderbook_imbalance (no network) ---

def _fake_book(bids, asks):
    mk = lambda p, s: types.SimpleNamespace(price=str(p), size=str(s))
    return types.SimpleNamespace(
        bids=[mk(p, s) for p, s in bids],
        asks=[mk(p, s) for p, s in asks],
    )


def test_fetch_no_token_returns_none():
    assert fetch_orderbook_imbalance(None) is None
    assert fetch_orderbook_imbalance("") is None


def test_fetch_parses_book(monkeypatch):
    monkeypatch.setattr(orderbook, "_http_get_book", lambda t: {
        "bids": [{"price": "0.49", "size": "300"}],
        "asks": [{"price": "0.51", "size": "100"}]})
    assert fetch_orderbook_imbalance("tok") == 0.5


def test_fetch_error_returns_none_and_gate_blocks(monkeypatch):
    def boom(t):
        raise RuntimeError("CLOB unreachable")
    monkeypatch.setattr(orderbook, "_http_get_book", boom)
    assert fetch_orderbook_imbalance("tok") is None
    assert orderbook_allows("YES", fetch_orderbook_imbalance("tok")) is False


def test_empty_book_returns_none(monkeypatch):
    monkeypatch.setattr(orderbook, "_http_get_book", lambda t: {"bids": [], "asks": []})
    assert fetch_orderbook_imbalance("tok") is None


def test_fetch_book_sorts_and_filters(monkeypatch):
    monkeypatch.setattr(orderbook, "_http_get_book", lambda t: {
        "bids": [{"price": "0.40", "size": "5"}, {"price": "0.45", "size": "7"},
                 {"price": "bad", "size": "1"}],
        "asks": [{"price": "0.60", "size": "3"}, {"price": "0.55", "size": "2"},
                 {"price": "0.70", "size": "0"}]})
    book = orderbook.fetch_book("tok")
    assert book["bids"] == [(0.45, 7.0), (0.40, 5.0)]
    assert book["asks"] == [(0.55, 2.0), (0.60, 3.0)]


def test_side_book_mirrors_no():
    yes = {"bids": [(0.45, 7.0)], "asks": [(0.55, 2.0)]}
    no = orderbook.side_book(yes, "NO")
    assert no["asks"] == [(0.55, 7.0)]
    assert no["bids"] == [(0.45, 2.0)]
    assert orderbook.side_book(yes, "YES") is yes
    assert orderbook.side_book(None, "NO") is None


def test_walk_asks_for_usd_crosses_levels():
    asks = [(0.50, 10.0), (0.60, 100.0)]
    shares, vwap, spent = orderbook.walk_asks_for_usd(asks, 11.0)
    # 10 sh @0.50 = $5, then $6 @0.60 = 10 sh -> 20 sh for $11
    assert shares == pytest.approx(20.0)
    assert spent == pytest.approx(11.0)
    assert vwap == pytest.approx(0.55)


def test_walk_asks_limited_by_depth():
    shares, vwap, spent = orderbook.walk_asks_for_usd([(0.5, 4.0)], 100.0)
    assert shares == 4.0 and spent == pytest.approx(2.0) and vwap == 0.5


def test_walk_bids_respects_floor():
    bids = [(0.50, 10.0), (0.45, 10.0), (0.30, 100.0)]
    filled, vwap = orderbook.walk_bids_for_shares(bids, 50.0, floor_price=0.44)
    assert filled == 20.0
    assert vwap == pytest.approx(0.475)
