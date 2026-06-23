"""Live-order planning: spread guard, depth downsizing, share sizing."""
import types

from locus.core.executor import (
    plan_live_order,
    plan_live_sell,
    valid_size_step,
    round_size_for_clob,
    _best_levels,
    round_to_tick,
    _diagnose_order_error,
)


def _maker_amount(price, size):
    """Scaled USDC maker amount the CLOB signs for a BUY (must be a whole 1e4)."""
    return round(price * size * 1e6)


def _taker_amount(size):
    """Scaled outcome-token amount (must be a whole multiple of 1e4)."""
    return round(size * 1e6)


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


def test_regression_invalid_order_inputs_case():
    # The exact order that the CLOB rejected as "Invalid order inputs":
    # price=0.081, naive size 25.4 -> maker_amount 2_057_400, not a 1e4 multiple.
    assert _maker_amount(0.081, 25.4) % 10_000 != 0  # the bug
    # ~$2.06 bet at $0.081 is the 25.4-share order that was rejected.
    price, shares, status = plan_live_order(
        bet_usd=2.06, best_bid=0.080, best_ask=0.081, ask_size_shares=10_000.0,
        max_spread=0.05,
    )
    assert status == "ok"
    assert price == 0.081
    # 25.4 raw shares snapped down to a valid step (10 shares at this price)
    assert shares == 20.0
    assert _maker_amount(price, shares) % 10_000 == 0
    assert _taker_amount(shares) % 10_000 == 0


def test_valid_size_step_makes_amounts_whole_1e4_multiples():
    # across a range of tick-aligned prices, snapping any size must yield a
    # maker AND taker amount divisible by 10_000 (the CLOB's hidden rule).
    for price in (0.081, 0.123, 0.50, 0.52, 0.337, 0.9, 0.015):
        for raw in (25.4, 6.25, 313.0, 7.77, 50.0):
            snapped = round_size_for_clob(price, raw)
            if snapped <= 0:
                continue
            assert snapped <= raw + 1e-9          # only ever rounds down
            assert _maker_amount(price, snapped) % 10_000 == 0, (price, raw, snapped)
            assert _taker_amount(snapped) % 10_000 == 0, (price, raw, snapped)


def test_valid_size_step_known_values():
    assert valid_size_step(0.081) == 10.0   # coarse step on a low-price 0.001 mkt
    assert valid_size_step(0.50) == 0.02     # fine step on a clean half


def test_below_five_share_minimum_is_skipped():
    # 4 shares clears $1 notional at this price but is under the 5-share floor.
    _, _, status = plan_live_order(2.0, 0.49, 0.50, 4.0, max_spread=0.05)
    assert status == "skipped_thin_book"


def test_sell_snaps_to_valid_step():
    price, shares, status = plan_live_sell(
        25.4, best_bid=0.081, best_ask=0.082, bid_size_shares=10_000.0,
        max_spread=0.05,
    )
    assert status == "ok"
    assert shares == 20.0
    assert _maker_amount(price, shares) % 10_000 == 0


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


def test_best_levels_dict_book():
    # py_clob_client_v2 may return a plain dict instead of an OrderBookSummary,
    # with each level itself a dict — both shapes must extract identically.
    book = {
        "bids": [{"price": "0.01", "size": "600"}, {"price": "0.48", "size": "100"}],
        "asks": [{"price": "0.99", "size": "100"}, {"price": "0.52", "size": "30"}],
    }
    best_bid, best_ask, ask_size = _best_levels(book)
    assert best_bid == 0.48
    assert best_ask == 0.52
    assert ask_size == 30.0


def test_best_levels_dict_book_missing_side():
    # a dict with no 'asks' key must not raise — empty side, no best ask
    book = {"bids": [{"price": "0.48", "size": "100"}]}
    assert _best_levels(book) == (0.48, None, None)


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
