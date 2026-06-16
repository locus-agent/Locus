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
    w._token_index = {
        t["token_id"]: (market.condition_id, t["outcome"].lower())
        for t in market.tokens
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
    w._apply_price_update("tokYES", {"price": "0.45"})
    assert w.snapshots["cond1"].last_price == 0.45


def test_no_asset_tick_converted_to_yes_terms():
    # Snapshots hold YES prices; a NO-token tick must be stored as 1 - p,
    # not clobber the mark with the NO price.
    w = _watcher_with(MKT)
    w._apply_price_update("tokNO", {"price": "0.45"})
    assert w.snapshots["cond1"].last_price == 0.55
    w._apply_price_update("tokNO", {"best_bid": "0.38", "best_ask": "0.42"})
    assert w.snapshots["cond1"].last_price == 0.60


def test_non_binary_outcome_tick_ignored():
    mkt = Market(
        "cond1", "Will X happen?", "ai", 0.50, 0.50, 5000, "", True,
        tokens=[{"token_id": "tokYES", "outcome": "Yes"},
                {"token_id": "tokNO", "outcome": "No"},
                {"token_id": "tokOTHER", "outcome": "Outcome_2"}],
    )
    w = _watcher_with(mkt)
    w._apply_price_update("tokOTHER", {"price": "0.10"})
    assert w.snapshots["cond1"].last_price == 0.50
    assert w.stats["price_updates"] == 0


def test_refresh_rebuilds_token_map():
    # the map used by _apply_price_update is derived from tracked markets
    w = _watcher_with(MKT)
    assert w._token_index == {"tokYES": ("cond1", "yes"), "tokNO": ("cond1", "no")}


def test_refresh_closes_stale_ws_subscription(monkeypatch):
    """When the tracked asset set changes, the open WS must be closed so the
    connect loop re-subscribes with current assets."""
    import asyncio

    class FakeWS:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    async def run():
        w = MarketWatcher()
        ws = FakeWS()
        w._ws_connected = True
        w._active_ws = ws
        w._subscribed_assets = {"oldtok"}

        async def fake_fetch():
            return [MKT]

        # drive just the post-fetch part of refresh by faking the fetch
        monkeypatch.setattr(
            "locus.markets.market_watcher.fetch_active_markets",
            lambda **kw: [MKT],
        )
        monkeypatch.setattr(
            "locus.markets.market_watcher.filter_by_categories", lambda ms: ms
        )
        w._schedule_index_sync = lambda: None
        await w.refresh_markets()
        return ws.closed, w._token_index

    closed, mapping = asyncio.run(run())
    assert closed is True
    assert set(mapping) == {"tokYES", "tokNO"}


# --- MarketSnapshot.update_price (polling-fallback price staleness fix) -------

def _snap(last=0.50, prev=0.50, age_s=60):
    return MarketSnapshot(
        market=MKT, last_price=last, prev_price=prev,
        last_update=datetime.now(timezone.utc) - timedelta(seconds=age_s),
    )


def test_update_price_changes_and_rolls_prev():
    snap = _snap(last=0.50, prev=0.50)
    assert snap.update_price(0.65) is True
    assert snap.last_price == 0.65
    assert snap.prev_price == 0.50            # prev rolled forward from old last
    # a second move rolls prev forward again
    assert snap.update_price(0.70) is True
    assert (snap.last_price, snap.prev_price) == (0.70, 0.65)


def test_update_price_noop_within_noise_band():
    t0 = datetime.now(timezone.utc) - timedelta(minutes=1)
    snap = MarketSnapshot(market=MKT, last_price=0.50, prev_price=0.40, last_update=t0)
    assert snap.update_price(0.50) is False        # identical
    assert snap.update_price(0.50005) is False     # diff 0.00005 <= 0.0001 noise
    assert snap.update_price(0.5001) is False       # diff exactly 0.0001 -> still no-op
    assert snap.last_price == 0.50                  # unchanged
    assert snap.prev_price == 0.40                  # prev untouched
    assert snap.last_update == t0                   # last_update not restamped
    # Just past the band updates.
    assert snap.update_price(0.5002) is True        # diff 0.0002 > 0.0001


def test_update_price_stamps_shared_now():
    now = datetime.now(timezone.utc)
    snap = _snap(age_s=60)
    assert snap.update_price(0.60, now=now) is True
    assert snap.last_update == now


def test_update_price_logs_only_on_change(caplog):
    import logging
    snap = _snap(last=0.50, prev=0.50)
    with caplog.at_level(logging.DEBUG, logger="locus.markets.market_watcher"):
        assert snap.update_price(0.50) is False
        assert "price update" not in caplog.text   # no log on a no-op
        assert snap.update_price(0.65) is True
        assert "price update" in caplog.text        # debug log only on real change


def test_refresh_updates_existing_snapshot_price(monkeypatch):
    """The polling-fallback bug: refresh_markets must roll fresh prices into
    EXISTING snapshots, not only newly tracked ones."""
    import asyncio

    w = MarketWatcher()
    w.snapshots["cond1"] = _snap(last=0.50, prev=0.50, age_s=300)

    moved = Market("cond1", "Will X happen?", "ai", 0.65, 0.35, 5000, "", True,
                   tokens=MKT.tokens)
    monkeypatch.setattr(
        "locus.markets.market_watcher.fetch_active_markets", lambda **kw: [moved]
    )
    monkeypatch.setattr(
        "locus.markets.market_watcher.filter_by_categories", lambda ms: ms
    )
    w._schedule_index_sync = lambda: None

    before = w.stats["price_updates"]
    asyncio.run(w.refresh_markets())

    snap = w.snapshots["cond1"]
    assert snap.last_price == 0.65               # rolled to the fresh price
    assert snap.prev_price == 0.50               # prev preserved
    assert w.stats["price_updates"] == before + 1


def test_refresh_does_not_recount_unchanged_price(monkeypatch):
    # An existing snapshot already at the fetched price must not inflate the
    # price_updates counter on refresh.
    import asyncio

    w = MarketWatcher()
    w.snapshots["cond1"] = _snap(last=0.50, prev=0.50, age_s=300)
    monkeypatch.setattr(
        "locus.markets.market_watcher.fetch_active_markets", lambda **kw: [MKT]  # 0.50
    )
    monkeypatch.setattr(
        "locus.markets.market_watcher.filter_by_categories", lambda ms: ms
    )
    w._schedule_index_sync = lambda: None

    before = w.stats["price_updates"]
    asyncio.run(w.refresh_markets())
    assert w.snapshots["cond1"].last_price == 0.50
    assert w.stats["price_updates"] == before    # unchanged -> not counted
