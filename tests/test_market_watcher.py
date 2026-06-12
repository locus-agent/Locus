"""WS price updates: O(1) token lookup, momentum, unknown assets ignored."""
from datetime import datetime, timedelta, timezone

from locus.markets.gamma import Market
from locus.markets.market_watcher import MarketWatcher, MarketSnapshot


def _watcher_with(market):
    w = MarketWatcher()
    w.tracked_markets = [market]
    w.snapshots[market.condition_id] = MarketSnapshot(
        market=market, last_price=market.yes_price, prev_price=market.yes_price,
        last_update=datetime.now(timezone.utc) - timedelta(seconds=60),
    )
    w._token_to_cid = {
        t["token_id"]: market.condition_id for t in market.tokens
    }
    return w


MKT = Market(
    "cond1", "Will X happen?", "ai", 0.50, 0.50, 5000, "", True,
    tokens=[{"token_id": "tokYES", "outcome": "Yes"}, {"token_id": "tokNO", "outcome": "No"}],
)


def test_price_update_via_token_index():
    w = _watcher_with(MKT)
    w._apply_price_update("tokYES", {"best_bid": "0.58", "best_ask": "0.62"})
    snap = w.snapshots["cond1"]
    assert snap.last_price == 0.60
    assert snap.prev_price == 0.50
    assert snap.momentum > 0
    assert w.stats["price_updates"] == 1


def test_unknown_asset_is_ignored():
    w = _watcher_with(MKT)
    w._apply_price_update("not-a-token", {"price": "0.99"})
    assert w.snapshots["cond1"].last_price == 0.50
    assert w.stats["price_updates"] == 0


def test_price_fallback_field():
    w = _watcher_with(MKT)
    w._apply_price_update("tokNO", {"price": "0.45"})
    assert w.snapshots["cond1"].last_price == 0.45


def test_refresh_rebuilds_token_map():
    # the map used by _apply_price_update is derived from tracked markets
    w = _watcher_with(MKT)
    assert w._token_to_cid == {"tokYES": "cond1", "tokNO": "cond1"}
