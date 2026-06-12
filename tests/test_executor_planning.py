"""Live-order planning: spread guard, depth downsizing, share sizing."""
import types

from locus.core.executor import plan_live_order, _best_levels


def test_normal_order_priced_at_best_ask_sized_in_shares():
    price, shares, status = plan_live_order(
        bet_usd=25.0, best_bid=0.48, best_ask=0.50, ask_size_shares=1000.0,
        max_spread=0.05,
    )
    assert status == "ok"
    assert price == 0.50
    assert shares == 50.0  # $25 at $0.50/share — shares, not dollars


def test_wide_spread_is_skipped():
    _, _, status = plan_live_order(25.0, 0.40, 0.50, 1000.0, max_spread=0.05)
    assert status == "skipped_wide_spread"


def test_thin_book_is_skipped():
    # only 1 share available at the ask -> $0.50 notional, below the $1 floor
    _, _, status = plan_live_order(25.0, 0.49, 0.50, 1.0, max_spread=0.05)
    assert status == "skipped_thin_book"


def test_partial_depth_downsizes():
    price, shares, status = plan_live_order(25.0, 0.49, 0.50, 20.0, max_spread=0.05)
    assert status == "ok"
    assert shares == 20.0  # wanted 50, book offers 20


def test_empty_book_is_skipped():
    _, _, status = plan_live_order(25.0, None, None, None, max_spread=0.05)
    assert status == "skipped_empty_book"


def test_missing_bid_side_still_trades():
    # no bids (one-sided book) — spread unknown, ask present: allowed
    price, shares, status = plan_live_order(25.0, None, 0.50, 100.0, max_spread=0.05)
    assert status == "ok" and price == 0.50


def _level(p, s):
    return types.SimpleNamespace(price=str(p), size=str(s))


def test_best_levels_extraction():
    book = types.SimpleNamespace(
        bids=[_level(0.01, 600), _level(0.48, 100), _level(0.45, 50)],
        asks=[_level(0.99, 100), _level(0.52, 30), _level(0.60, 10)],
    )
    best_bid, best_ask, ask_size = _best_levels(book)
    assert best_bid == 0.48
    assert best_ask == 0.52
    assert ask_size == 30.0


def test_best_levels_empty_book():
    book = types.SimpleNamespace(bids=[], asks=[])
    assert _best_levels(book) == (None, None, None)
