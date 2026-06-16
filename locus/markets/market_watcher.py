"""
Polymarket WebSocket subscriber — live price feed + niche market filtering.
Maintains a live snapshot of tracked markets and detects momentum shifts.
"""
from __future__ import annotations

import asyncio
import json
import ssl
import time
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field

import certifi

from locus import config
from locus.core.market_index import MarketIndex
from locus.markets.gamma import Market, fetch_active_markets, filter_by_categories

log = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    market: Market
    last_price: float
    prev_price: float
    last_update: datetime
    momentum: float = 0.0  # price change per minute

    @property
    def price_change(self) -> float:
        return self.last_price - self.prev_price

    def update_price(self, new_price: float, now: datetime | None = None) -> bool:
        """Roll a fresh price into the snapshot, guarding against sub-tick noise.

        Stamps prev_price/last_price/last_update and returns True only when the
        price actually moved (> 0.0001); an unchanged price is a silent no-op
        (no stamp, no log) so a quiet poll doesn't churn last_update or spam the
        debug log. `now` is shared by the caller so a refresh can count exactly
        which snapshots it touched this cycle."""
        if abs(self.last_price - new_price) <= 0.0001:
            return False
        self.prev_price = self.last_price
        self.last_price = new_price
        self.last_update = now or datetime.now(timezone.utc)
        log.debug(
            f"[watcher] price update {self.market.condition_id[:12]} "
            f"{self.prev_price:.4f} -> {self.last_price:.4f}"
        )
        return True


class MarketWatcher:
    """Watches niche Polymarket markets via WebSocket + periodic Gamma API refresh."""

    def __init__(self):
        self.snapshots: dict[str, MarketSnapshot] = {}
        self.tracked_markets: list[Market] = []
        self.index = MarketIndex()  # semantic index; synced after each refresh
        # WS asset id -> (condition id, outcome). Both YES and NO assets are
        # subscribed; snapshots store YES-terms prices, so NO ticks need 1-p.
        self._token_index: dict[str, tuple[str, str]] = {}
        self._subscribed_assets: set[str] = set()  # assets the live WS covers
        self._active_ws = None
        self._index_syncing = False
        self._refresh_interval = 300  # refresh market list every 5 min
        self._ws_connected = False
        self.stats = {
            "ws_messages": 0,
            "price_updates": 0,
            "market_refreshes": 0,
        }

    def get_niche_markets(self, markets: list[Market]) -> list[Market]:
        """Filter to niche markets within volume bounds."""
        return [
            m for m in markets
            if config.MIN_VOLUME_USD <= m.volume <= config.MAX_VOLUME_USD
            and m.active
        ]

    async def refresh_markets(self):
        """Fetch and filter markets from Gamma API."""
        try:
            # Full paginated scan of the niche volume band (Gamma-side filter),
            # not a top-N-by-volume slice — niche markets live at the tail.
            fetch_start = time.monotonic()
            all_markets = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: fetch_active_markets(
                    limit=None,
                    min_volume=config.MIN_VOLUME_USD,
                    max_volume=config.MAX_VOLUME_USD,
                ),
            )
            fetch_secs = time.monotonic() - fetch_start
            categorized = filter_by_categories(all_markets)
            self.tracked_markets = self.get_niche_markets(categorized)

            # Update snapshots
            now = datetime.now(timezone.utc)
            existing_ids = set(self.snapshots.keys())
            new_ids = set()

            for m in self.tracked_markets:
                new_ids.add(m.condition_id)
                if m.condition_id not in self.snapshots:
                    self.snapshots[m.condition_id] = MarketSnapshot(
                        market=m,
                        last_price=m.yes_price,
                        prev_price=m.yes_price,
                        last_update=now,
                    )
                else:
                    snap = self.snapshots[m.condition_id]
                    snap.market = m  # update metadata
                    # Roll the freshly fetched price into the EXISTING snapshot.
                    # Previously only brand-new snapshots got a price, so during
                    # a WebSocket outage the polling refresh left existing marks
                    # stale (positions marked at the last WS tick, not live).
                    snap.update_price(m.yes_price, now)

            # Remove stale snapshots
            for stale_id in existing_ids - new_ids:
                del self.snapshots[stale_id]

            # Markets refreshed this cycle (new or price-moved) carry this
            # refresh's `now` stamp; unchanged snapshots keep their older stamp.
            refreshed = sum(
                1 for snap in self.snapshots.values() if snap.last_update == now
            )
            self.stats["price_updates"] += refreshed

            # O(1) lookup for WS price ticks (was an O(markets) scan per tick)
            self._token_index = {
                token_id: (m.condition_id, (t.get("outcome") or "").lower())
                for m in self.tracked_markets
                for t in m.tokens
                if (token_id := t.get("token_id"))
            }

            # If the tracked set changed, the open WS is subscribed to a stale
            # asset list — newly tracked markets would get no live prices
            # until an accidental reconnect. Close it; the connect loop
            # re-subscribes with the current assets.
            current_assets = set(self._token_index)
            if (
                self._ws_connected
                and self._active_ws is not None
                and current_assets != self._subscribed_assets
            ):
                log.info(
                    f"[watcher] Tracked set changed "
                    f"(+{len(current_assets - self._subscribed_assets)}/"
                    f"-{len(self._subscribed_assets - current_assets)} assets) — "
                    f"reconnecting WebSocket"
                )
                try:
                    await self._active_ws.close()
                except Exception:
                    pass

            self.stats["market_refreshes"] += 1
            log.info(
                f"[watcher] Fetched {len(all_markets)} markets in volume band "
                f"${config.MIN_VOLUME_USD:,.0f}-${config.MAX_VOLUME_USD:,.0f} "
                f"in {fetch_secs:.1f}s -> {len(categorized)} in target categories "
                f"-> tracking {len(self.tracked_markets)} niche markets"
            )

            self._schedule_index_sync()

        except Exception as e:
            log.warning(f"[watcher] Market refresh error: {e}")

    def _schedule_index_sync(self):
        """Sync the semantic index in a worker thread (first build embeds all
        ~3k markets and takes minutes; later syncs touch only changed rows).
        News matching falls back to keyword-only until index.ready."""
        if self._index_syncing:
            return
        self._index_syncing = True
        markets_snapshot = list(self.tracked_markets)

        def _sync():
            try:
                self.index.sync(markets_snapshot)
            except Exception as e:
                log.warning(f"[watcher] Index sync error: {e}")
            finally:
                self._index_syncing = False

        asyncio.get_event_loop().run_in_executor(None, _sync)

    async def _connect_websocket(self):
        """Connect to Polymarket WebSocket for live price updates."""
        try:
            import websockets
        except ImportError:
            log.warning("[watcher] websockets not installed — using polling fallback")
            return

        ssl_context = ssl.create_default_context(cafile=certifi.where())

        while True:
            try:
                async with websockets.connect(config.POLYMARKET_WS_HOST, ssl=ssl_context) as ws:
                    self._ws_connected = True
                    self._active_ws = ws
                    log.info("[watcher] WebSocket connected")

                    # Subscribe to the market channel: one message with all asset (token) ids
                    asset_ids = [
                        tid
                        for market in self.tracked_markets
                        for token in market.tokens
                        if (tid := token.get("token_id"))
                    ]
                    self._subscribed_assets = set(asset_ids)
                    if asset_ids:
                        await ws.send(json.dumps({"assets_ids": asset_ids, "type": "market"}))

                    # Listen for updates
                    while True:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=10)
                        except asyncio.TimeoutError:
                            # Send ping
                            await ws.ping()
                            continue

                        self.stats["ws_messages"] += 1

                        if not msg:
                            continue

                        try:
                            data = json.loads(msg)
                        except json.JSONDecodeError:
                            log.debug(f"[watcher] Skipping non-JSON WebSocket message: {msg!r}")
                            continue

                        self._handle_ws_message(data)

            except Exception as e:
                self._ws_connected = False
                self._active_ws = None
                log.warning(f"[watcher] WebSocket error: {e}, reconnecting in 5s")
                await asyncio.sleep(5)

    def _handle_ws_message(self, data: dict | list):
        """Process a market-channel WebSocket message (book snapshot or price change)."""
        if isinstance(data, list):
            # Initial book snapshots sent on subscribe — no single "current price" to
            # extract here; live prices arrive via subsequent price_change events.
            return

        event_type = data.get("event_type", "")

        if event_type == "price_change":
            for change in data.get("price_changes", []):
                self._apply_price_update(change.get("asset_id", ""), change)
        elif event_type == "last_trade_price":
            self._apply_price_update(data.get("asset_id", ""), data)

    def _apply_price_update(self, asset_id: str, change: dict):
        """Update the snapshot for asset_id using best_bid/best_ask (or price as fallback)."""
        if not asset_id:
            return

        best_bid = change.get("best_bid")
        best_ask = change.get("best_ask")
        if best_bid is not None and best_ask is not None:
            price = (float(best_bid) + float(best_ask)) / 2
        elif change.get("price") is not None:
            price = float(change["price"])
        else:
            return

        cid, outcome = self._token_index.get(asset_id, ("", ""))
        snap = self.snapshots.get(cid)
        if snap is None:
            return
        # Snapshots hold YES-terms prices. A NO-asset tick carries the NO
        # token's price: convert, don't clobber. Other outcomes (multi-outcome
        # markets) have no YES-terms equivalent — ignore those ticks.
        if outcome == "no":
            price = 1.0 - price
        elif outcome != "yes":
            return
        now = datetime.now(timezone.utc)
        elapsed = (now - snap.last_update).total_seconds()
        snap.prev_price = snap.last_price
        snap.last_price = price
        snap.last_update = now
        if elapsed > 0:
            snap.momentum = (snap.last_price - snap.prev_price) / (elapsed / 60)
        self.stats["price_updates"] += 1

    async def _polling_fallback(self):
        """Poll Gamma API for price updates when WebSocket unavailable.

        refresh_markets() now rolls fresh prices into existing snapshots (not
        just newly tracked markets), so positions stay marked at live prices
        during a WS outage instead of freezing at the last WS tick."""
        while True:
            await asyncio.sleep(30)
            if self._ws_connected:
                continue
            before = self.stats["price_updates"]
            await self.refresh_markets()
            refreshed = self.stats["price_updates"] - before
            log.info(f"Polling fallback: refreshed {refreshed} market prices")

    async def run(self):
        """Start the market watcher — refresh + WebSocket + polling fallback."""
        # Warm the persisted semantic index right away: the RSS source bursts
        # headlines in the first seconds, before the first refresh+sync lands.
        asyncio.get_event_loop().run_in_executor(None, self.index.warm)
        await self.refresh_markets()

        async def refresh_loop():
            while True:
                await asyncio.sleep(self._refresh_interval)
                await self.refresh_markets()

        await asyncio.gather(
            refresh_loop(),
            self._connect_websocket(),
            self._polling_fallback(),
            return_exceptions=True,
        )

    def get_snapshot(self, condition_id: str) -> MarketSnapshot | None:
        return self.snapshots.get(condition_id)


if __name__ == "__main__":
    async def _test():
        watcher = MarketWatcher()
        await watcher.refresh_markets()
        print(f"Tracking {len(watcher.tracked_markets)} niche markets:")
        for m in watcher.tracked_markets[:10]:
            print(f"  [{m.category}] ${m.volume:,.0f} | YES:{m.yes_price:.2f} | {m.question[:60]}")

    asyncio.run(_test())
