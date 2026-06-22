"""Live-order planning: spread guard, depth downsizing, share sizing."""
import types

from locus.core.executor import (
    plan_live_order,
    _best_levels,
    round_to_tick,
    _diagnose_order_error,
)


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


def test_round_to_tick_penny_market():
    # 0.01 tick: off-grid prices snap to the nearest penny
    assert round_to_tick(0.523, "0.01") == 0.52
    assert round_to_tick(0.527, "0.01") == 0.53
    assert round_to_tick(0.50, "0.01") == 0.50  # already on grid


def test_round_to_tick_fine_market():
    # 0.001 tick keeps the third decimal
    assert round_to_tick(0.5237, "0.001") == 0.524
    assert round_to_tick(0.5231, "0.001") == 0.523


def test_round_to_tick_no_float_dust():
    # the classic 0.1+0.2 style artefact must not leak into the price
    assert round_to_tick(0.30000000004, "0.01") == 0.30


def test_round_to_tick_clamps_into_valid_band():
    # price_valid requires tick <= price <= 1 - tick
    assert round_to_tick(0.999, "0.01") == 0.99
    assert round_to_tick(0.001, "0.01") == 0.01
    assert round_to_tick(1.5, "0.01") == 0.99


def test_round_to_tick_accepts_float_tick():
    assert round_to_tick(0.527, 0.01) == 0.53


def test_diagnose_flags_tick_size():
    exc = types.SimpleNamespace(
        status_code=400, error_msg={"error": "price (0.523) not a valid tick"}
    )
    reasons = _diagnose_order_error(exc, "tok-1", 0.523, 50.0, "BUY", 0.01)
    assert "tick_size" in reasons


def test_diagnose_flags_min_order_size():
    exc = types.SimpleNamespace(
        status_code=400, error_msg="order amount below the minimum order size"
    )
    reasons = _diagnose_order_error(exc, "tok-1", 0.52, 1.0, "BUY", 0.01)
    assert "min_order_size" in reasons


def test_diagnose_flags_signature_type_on_auth_status():
    exc = types.SimpleNamespace(status_code=401, error_msg="unauthorized")
    reasons = _diagnose_order_error(exc, "tok-1", 0.52, 50.0, "BUY", 0.01)
    assert "signature_type" in reasons


def test_diagnose_handles_plain_exception():
    # no status_code / error_msg attributes — must not raise, returns a list
    reasons = _diagnose_order_error(ValueError("boom"), "tok-1", 0.5, 50.0, "BUY")
    assert isinstance(reasons, list)
