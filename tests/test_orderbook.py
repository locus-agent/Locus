"""Orderbook imbalance gate: score math, allow/block logic, and fail-open fetch."""
import types

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


def test_none_score_always_allows():
    # Fail open: an unavailable book must never block a trade.
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
    book = _fake_book([(0.49, 300.0)], [(0.51, 100.0)])
    monkeypatch.setattr(orderbook, "_clob_client",
                        lambda: types.SimpleNamespace(get_order_book=lambda t: book))
    assert fetch_orderbook_imbalance("tok") == 0.5


def test_fetch_fails_open_on_error(monkeypatch):
    def boom():
        raise RuntimeError("CLOB unreachable")
    monkeypatch.setattr(orderbook, "_clob_client", boom)
    assert fetch_orderbook_imbalance("tok") is None
