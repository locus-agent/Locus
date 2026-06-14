#!/usr/bin/env python3
"""
Polymarket Pipeline — V1 (synchronous) and V2 (async event-driven).
V1: Scrape → Score → Edge → Trade (loop-based)
V2: News stream → Match → Classify → Edge → Trade (event-driven)
"""
from __future__ import annotations

import asyncio
import time
import logging
from datetime import datetime, timedelta, timezone

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from locus import config
from locus.memory import logger
from locus.core.export_status import export_status
from locus.sources.scraper import scrape_all
from locus.markets.gamma import fetch_active_markets, filter_by_categories, get_token_id
from locus.core.scorer import score_market, filter_news_for_market
from locus.core.edge import detect_edge, detect_edge_v2, Signal
from locus.core.executor import execute_trade, execute_trade_async
from locus.supervisor import supervise
from locus.sources.news_stream import NewsAggregator, NewsEvent
from locus.markets.market_watcher import MarketWatcher
from locus.core.matcher import match_news_to_markets, match_news_to_markets_hybrid, prefilter_match
from locus.core.classifier import classify_async, classify_edge_type
from locus.core import event_context
from locus.core import reentry
from locus.core import whale_tracker
from locus.core.orderbook import fetch_orderbook_imbalance, orderbook_allows
from locus.core.journal import maybe_write_journal
from locus.core import positions

console = Console()
log = logging.getLogger(__name__)


# ============================================================
# V2: Event-Driven Pipeline
# ============================================================

def news_age_seconds(event: NewsEvent, now: datetime | None = None) -> float | None:
    """Age of the news at this moment (publication -> now), or None when the
    publication time is unknown (latency_ms == -1 sentinel). Computed at
    decision time, not receipt time, so queue dwell counts against it."""
    if event.latency_ms is not None and event.latency_ms < 0:
        return None
    now = now or datetime.now(timezone.utc)
    return (now - event.published_at).total_seconds()


def gate_trade(event: NewsEvent, signal, traded_headlines: set[str], now: datetime | None = None):
    """Trade-time risk gates, applied after classification so every
    classification is still logged and calibrated.

    Returns (signal_or_none, action):
      "skip"               — no edge detected
      "stale"              — would-be signal, but the headline is older than
                             MAX_NEWS_AGE_SECONDS as of *now* (including time
                             spent in the queue), or its publication time is
                             unknown; we classify old news for calibration but
                             never trade on it
      "capped"             — this headline already produced a trade; one
                             headline matching N markets must not open N
                             correlated positions
      "low_materiality"    — materiality below the direction-specific floor
                             (bearish calls need a higher bar than bullish)
      "needs_confirmation" — high-materiality (>= HIGH_MATERIALITY_THRESHOLD)
                             call seen from fewer than MIN_CONFIRMING_SOURCES
                             distinct sources within CONFIRMATION_WINDOW_HOURS;
                             obvious news is the least accurate, so hold until
                             a second source agrees
      "signal"             — trade approved; headline recorded against the cap
    """
    if signal is None:
        return None, "skip"
    age = news_age_seconds(event, now)
    if age is None or age > config.MAX_NEWS_AGE_SECONDS:
        return None, "stale"
    if event.headline in traded_headlines:
        return None, "capped"

    # Direction-specific materiality floor (calibration: bearish accuracy is
    # far worse than bullish, so bearish needs a higher bar).
    direction = signal.classification
    if signal.materiality < config.materiality_threshold(direction):
        return None, "low_materiality"

    # High-materiality confirmation gate: the most "obvious" news grades worst
    # (likely already priced in), so require the same directional read from
    # >= MIN_CONFIRMING_SOURCES distinct sources within the recent window.
    if signal.materiality >= config.HIGH_MATERIALITY_THRESHOLD:
        since = (
            (now or datetime.now(timezone.utc))
            - timedelta(hours=config.CONFIRMATION_WINDOW_HOURS)
        ).strftime("%Y-%m-%d %H:%M:%S")
        sources = logger.get_confirming_sources(
            signal.market.condition_id, direction, since
        )
        sources.add(event.source)
        if len(sources) < config.MIN_CONFIRMING_SOURCES:
            return None, "needs_confirmation"

    traded_headlines.add(event.headline)
    return signal, "signal"


CALIBRATION_STARTUP_DELAY_SECONDS = 300.0


async def run_calibration_schedule(
    startup_delay: float,
    interval_seconds: float,
    runner=None,
) -> None:
    """Run a full calibration cycle on a fixed schedule, off the event loop.
    Errors are logged loudly and never crash the loop (the supervisor above
    is the second net). Standalone so the schedule is unit-testable."""
    from locus.memory import calibrator

    runner = runner or calibrator.run_cycle
    await asyncio.sleep(startup_delay)
    while True:
        try:
            summary = await asyncio.get_event_loop().run_in_executor(None, runner)
            acc = (
                f", track record {summary['total']} @ {summary['accuracy']:.1f}%"
                if summary.get("total") else ""
            )
            console.print(
                f"  [dim]calibration: {summary['resolved']} resolved, "
                f"{summary['graded']} graded{acc}[/dim]"
            )
        except Exception:
            log.exception("[pipeline] CALIBRATION CYCLE FAILED (will retry next interval)")
        await asyncio.sleep(interval_seconds)


class PipelineV2:
    """Async event-driven pipeline. Runs indefinitely."""

    def __init__(self):
        self.news_queue: asyncio.Queue = asyncio.Queue()
        self.signal_queue: asyncio.Queue = asyncio.Queue()
        self.news_aggregator = NewsAggregator(self.news_queue)
        self.market_watcher = MarketWatcher()
        self._traded_headlines: set[str] = set()
        self.running = False
        self.stats = {
            "news_processed": 0,
            "markets_matched": 0,
            "signals_found": 0,
            "trades_executed": 0,
            "whale_last_check": None,
        }

    async def run(self):
        """Start all pipeline components concurrently."""
        self.running = True
        mode = "[red bold]LIVE[/red bold]" if not config.DRY_RUN else "[yellow]DRY RUN[/yellow]"
        console.print(Panel(f"Pipeline V2 Starting  |  Mode: {mode}", style="bright_green"))
        console.print(f"  Niche filter: ${config.MIN_VOLUME_USD:,.0f} - ${config.MAX_VOLUME_USD:,.0f} volume")
        console.print(
            f"  Materiality threshold: bullish {config.MATERIALITY_THRESHOLD_BULLISH} / "
            f"bearish {config.MATERIALITY_THRESHOLD_BEARISH} "
            f"(confirm >= {config.HIGH_MATERIALITY_THRESHOLD} from "
            f"{config.MIN_CONFIRMING_SOURCES} sources)"
        )
        console.print(f"  Speed target: {config.SPEED_TARGET_SECONDS}s")
        console.print()

        await asyncio.get_event_loop().run_in_executor(None, positions.backfill_positions)

        try:
            await asyncio.gather(
                supervise("news_aggregator", self.news_aggregator.run, self.stats),
                supervise("market_watcher", self.market_watcher.run, self.stats),
                supervise("process_news", self._process_news, self.stats),
                supervise("execute_signals", self._execute_signals, self.stats),
                supervise("status_printer", self._status_printer, self.stats),
                supervise(
                    "calibration",
                    lambda: run_calibration_schedule(
                        CALIBRATION_STARTUP_DELAY_SECONDS,
                        config.CALIBRATION_INTERVAL_HOURS * 3600,
                    ),
                    self.stats,
                ),
                supervise("whale_tracker", self._whale_check_loop, self.stats),
            )
        except asyncio.CancelledError:
            self.running = False

    async def _process_news(self):
        """Process each news event: match → classify → detect edge."""
        # Hold consumption until the first market refresh: news sources burst
        # headlines at startup, and matching them against an empty market list
        # silently discards the whole burst. They buffer in the queue meanwhile.
        while not self.market_watcher.tracked_markets:
            await asyncio.sleep(1)

        while True:
            event: NewsEvent = await self.news_queue.get()
            self.stats["news_processed"] += 1

            # Log the news event
            logger.log_news_event(
                headline=event.headline,
                source=event.source,
                received_at=event.received_at.isoformat(),
                latency_ms=event.latency_ms,
            )

            # Match to niche markets: keyword pre-filter + embedding index
            # (search blocks ~50ms, so run it off the event loop)
            matched = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: match_news_to_markets_hybrid(
                    event.headline,
                    self.market_watcher.tracked_markets,
                    index=self.market_watcher.index,
                ),
            )

            if not matched:
                continue

            self.stats["markets_matched"] += len(matched)

            # Classify against each matched market
            for market, match_source, match_score in matched:
                try:
                    # Cheap pre-classification gate: weak keyword-only matches
                    # with no topic overlap aren't worth a Claude call.
                    if prefilter_match(event.headline, market, match_source, match_score):
                        self.stats["prefiltered"] = self.stats.get("prefiltered", 0) + 1
                        logger.log_classification(
                            market_question=market.question,
                            headline=event.headline,
                            news_source=event.source,
                            direction=None,
                            materiality=None,
                            edge=None,
                            action="prefiltered",
                            match_source=match_source,
                            condition_id=market.condition_id,
                            yes_price=market.yes_price,
                            yes_token_id=get_token_id(market, "YES"),
                        )
                        continue

                    # Dedup memory: same headline against the same market at
                    # ~the same price was already classified — reuse it
                    # instead of buying the same answer from Claude again.
                    prior = logger.find_recent_classification(
                        event.headline,
                        market.condition_id,
                        market.yes_price,
                        config.CLASSIFY_CACHE_PRICE_TOLERANCE,
                        config.CLASSIFY_CACHE_HOURS,
                    )
                    if prior:
                        self.stats["cache_hits"] = self.stats.get("cache_hits", 0) + 1
                        logger.log_classification(
                            market_question=market.question,
                            headline=event.headline,
                            news_source=event.source,
                            direction=prior["direction"],
                            materiality=prior["materiality"],
                            edge=None,
                            action="cached",
                            match_source=match_source,
                            condition_id=market.condition_id,
                            yes_price=market.yes_price,
                            yes_token_id=get_token_id(market, "YES"),
                            confidence=prior.get("confidence"),
                        )
                        continue

                    classification = await classify_async(
                        event.headline, market, event.source
                    )

                    if classification.error:
                        self.stats["classify_error_streak"] = (
                            self.stats.get("classify_error_streak", 0) + 1
                        )
                        raw_signal, signal, action, edge_type = None, None, "error", None
                    else:
                        self.stats["classify_error_streak"] = 0
                        raw_signal = detect_edge_v2(market, classification, event)
                        signal, action = gate_trade(event, raw_signal, self._traded_headlines)

                        # Re-entry gate: if this market was recently closed and
                        # is still being watched, a fresh signal only gets back
                        # in when it clears the re-entry bar for why we exited.
                        reentry_active = False
                        edge_type_override = None
                        if signal is not None:
                            decision = await asyncio.get_event_loop().run_in_executor(
                                None, lambda: self._check_reentry(market, classification)
                            )
                            if decision is not None and not decision["should_reenter"]:
                                self.stats["reentry_blocked"] = (
                                    self.stats.get("reentry_blocked", 0) + 1
                                )
                                signal, action = None, "reentry_blocked"
                                console.print(
                                    f"  [dim]REENTRY BLOCK: {decision['reason']} "
                                    f"on \"{market.question[:40]}\"[/dim]"
                                )
                            elif decision is not None:
                                reentry_active = True
                                edge_type_override = "reentry"
                                self.stats["reentry_triggered"] = (
                                    self.stats.get("reentry_triggered", 0) + 1
                                )
                                console.print(
                                    f"  [bright_green]REENTRY[/bright_green]: "
                                    f"{decision['reason']} on \"{market.question[:40]}\""
                                )

                        # Correlation gate: don't stack the book into one
                        # subject. HIGH risk blocks; MEDIUM warns but allows.
                        if signal is not None:
                            corr = positions.check_correlation_risk(
                                market.question, signal.side, positions.get_open_positions()
                            )
                            if corr["risk_level"] == "high":
                                self.stats["correlation_blocks"] = (
                                    self.stats.get("correlation_blocks", 0) + 1
                                )
                                signal, action = None, "correlation_block"
                                console.print(
                                    f"  [red]CORRELATION BLOCK[/red]: "
                                    f"{len(corr['correlated_positions'])} related "
                                    f"positions (${corr['total_exposure_usd']:.0f} exposure) "
                                    f"on \"{market.question[:40]}\""
                                )
                            elif corr["risk_level"] == "medium":
                                console.print(
                                    f"  [yellow]CORRELATION WARNING[/yellow]: "
                                    f"{len(corr['correlated_positions'])} related "
                                    f"positions (${corr['total_exposure_usd']:.0f} exposure) "
                                    f"on \"{market.question[:40]}\" (allowed)"
                                )

                        # Orderbook imbalance gate: don't trade into strong
                        # opposing flow on the live YES book. Fails open when
                        # the CLOB is unreachable (imbalance is None -> allowed).
                        if signal is not None:
                            imbalance = await asyncio.get_event_loop().run_in_executor(
                                None, fetch_orderbook_imbalance, get_token_id(market, "YES")
                            )
                            if not orderbook_allows(signal.side, imbalance):
                                self.stats["orderbook_skips"] = (
                                    self.stats.get("orderbook_skips", 0) + 1
                                )
                                console.print(
                                    f"  [yellow]ORDERBOOK SKIP[/yellow]: {signal.side} "
                                    f"into imbalance {imbalance:+.2f} "
                                    f"on \"{market.question[:40]}\""
                                )
                                signal, action = None, "orderbook_skip"

                        # Attribute a surviving signal to an edge type, for the
                        # trade record and per-type calibration. A re-entry keeps
                        # its 'reentry' label over the inferred news/momentum one.
                        edge_type = edge_type_override or (
                            classify_edge_type(market) if signal is not None else None
                        )
                        if signal is not None:
                            signal.edge_type = edge_type

                        # Event context: once a signal clears the gates, look
                        # across the whole event. Cap per-event exposure (sibling
                        # outcomes are highly correlated), then prefer the
                        # highest-edge outcome — often an implied play on a
                        # sibling rather than the market the news named.
                        if signal is not None:
                            open_positions = await asyncio.get_event_loop().run_in_executor(
                                None, positions.get_open_positions
                            )
                            event_id = getattr(market, "event_id", "") or ""
                            exposure = event_context.get_event_exposure(event_id, open_positions)
                            if event_id and exposure["position_count"] >= config.MAX_POSITIONS_PER_EVENT:
                                self.stats["event_exposure_blocks"] = (
                                    self.stats.get("event_exposure_blocks", 0) + 1
                                )
                                signal, action = None, "event_exposure_block"
                                console.print(
                                    f"  [red]EVENT EXPOSURE BLOCK[/red]: "
                                    f"{exposure['position_count']} position(s) "
                                    f"(${exposure['total_exposure_usd']:.0f}) already on event "
                                    f"{event_id} for \"{market.question[:40]}\""
                                )
                            else:
                                event_markets = event_context.get_event_markets(
                                    event_id, self.market_watcher.tracked_markets
                                )
                                best = event_context.find_best_outcome(
                                    signal, event_markets, open_positions
                                )
                                if best and best["recommended_market"].condition_id != market.condition_id:
                                    self.stats["event_switches"] = (
                                        self.stats.get("event_switches", 0) + 1
                                    )
                                    switched = event_context.build_switched_signal(signal, best)
                                    console.print(
                                        f"  [cyan]EVENT SWITCH[/cyan]: "
                                        f"original=\"{market.question[:40]}\" ({signal.side}) -> "
                                        f"switched_to=\"{switched.market.question[:40]}\" ({switched.side}) "
                                        f"edge {signal.edge:.1%} -> {switched.edge:.1%} "
                                        f"[{best['reason']}]"
                                    )
                                    signal = switched

                        # A triggered re-entry is logged as the opportunity it is
                        # (green on the dashboard), even if a later gate blocked
                        # the trade. Only consume the re-entry budget when we
                        # actually re-enter (the signal survived the gate chain).
                        if reentry_active:
                            action = "reentry_triggered"
                            if signal is not None:
                                await asyncio.get_event_loop().run_in_executor(
                                    None, lambda: self._record_reentry(market.condition_id)
                                )

                    logger.log_classification(
                        market_question=market.question,
                        headline=event.headline,
                        news_source=event.source,
                        direction=classification.direction,
                        materiality=classification.materiality,
                        edge=raw_signal.edge if raw_signal else None,
                        action=action,
                        match_source=match_source,
                        condition_id=market.condition_id,
                        yes_price=market.yes_price,
                        yes_token_id=get_token_id(market, "YES"),
                        edge_type=edge_type,
                        confidence=classification.confidence if not classification.error else None,
                        event_id=getattr(market, "event_id", "") or None,
                    )

                    if action in ("stale", "capped", "needs_confirmation"):
                        age = news_age_seconds(event)
                        age_min = age / 60 if age is not None else -1
                        console.print(
                            f"  [dim]{action.upper()}: suppressed would-be signal "
                            f"({event.source}, news age {'unknown' if age is None else f'{age_min:.0f}m'}) "
                            f"on \"{market.question[:40]}\"[/dim]"
                        )

                    if classification.direction in ("bullish", "bearish") and not classification.error:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: positions.trigger_news_reeval(
                                market.condition_id, classification.direction,
                                classification.materiality, event.headline,
                            ),
                        )

                    if signal:
                        self.stats["signals_found"] += 1
                        await self.signal_queue.put(signal)
                        console.print(
                            f"  [bright_green]SIGNAL[/bright_green] "
                            f"[{event.source}] {classification.direction.upper()} "
                            f"mat:{classification.materiality:.2f} "
                            f"→ {signal.side} ${signal.bet_amount} "
                            f"on \"{market.question[:40]}...\" "
                            f"({signal.total_latency_ms}ms)"
                        )
                except Exception as e:
                    log.warning(f"[pipeline] Classification error: {e}")

    def _check_reentry(self, market, classification):
        """Re-entry decision for a watched market, or None when the market isn't
        being watched. Read-only; runs off the event loop (own DB conn)."""
        conn = logger._conn()
        try:
            watched = reentry.find_watched_market(conn, market.condition_id)
            if watched is None:
                return None
            since = (
                datetime.now(timezone.utc)
                - timedelta(hours=config.CONFIRMATION_WINDOW_HOURS)
            ).strftime("%Y-%m-%d %H:%M:%S")
            sources = logger.get_confirming_sources(
                market.condition_id, classification.direction, since
            )
            return reentry.check_reentry_opportunity(
                watched, classification, confirming_source_count=len(sources)
            )
        finally:
            conn.close()

    def _record_reentry(self, condition_id: str):
        """Consume one re-entry from the market's watch window."""
        conn = logger._conn()
        try:
            reentry.record_reentry(conn, condition_id)
            conn.commit()
        finally:
            conn.close()

    async def _whale_check_loop(self):
        """Periodically shadow the watched wallets and investigate the niche
        markets they move that we missed. Clean return (no wallets configured)
        ends supervision intentionally — whale tracking is simply disabled."""
        if not config.WHALE_WALLETS:
            console.print("  [dim]whale tracking disabled (no WHALE_WALLETS configured)[/dim]")
            return
        # Don't poll before the first market refresh: we match whale trades
        # against tracked markets, and that list is empty at startup.
        while not self.market_watcher.tracked_markets:
            await asyncio.sleep(1)
        console.print(
            f"  [dim]whale tracking: {len(config.WHALE_WALLETS)} wallet(s), "
            f"every {config.WHALE_CHECK_INTERVAL_MINUTES:.0f}m[/dim]"
        )
        interval = config.WHALE_CHECK_INTERVAL_MINUTES * 60.0
        while True:
            try:
                await self._run_whale_check()
            except Exception:
                log.exception("[pipeline] WHALE CHECK FAILED (will retry next interval)")
            await asyncio.sleep(interval)

    async def _run_whale_check(self):
        """One whale poll: fetch recent whale trades, find missed opportunities,
        and investigate each. All blocking work runs off the event loop."""
        loop = asyncio.get_event_loop()
        trades = await loop.run_in_executor(
            None,
            lambda: whale_tracker.fetch_recent_whale_trades(config.WHALE_LOOKBACK_MINUTES),
        )
        self.stats["whale_last_check"] = datetime.now(timezone.utc).isoformat()
        self.stats["whale_checks"] = self.stats.get("whale_checks", 0) + 1
        if not trades:
            return

        def _find():
            conn = logger._conn()
            try:
                return whale_tracker.find_missed_opportunities(
                    trades, self.market_watcher.tracked_markets, conn,
                    min_trade_usd=config.WHALE_MIN_TRADE_USD,
                )
            finally:
                conn.close()

        missed = await loop.run_in_executor(None, _find)
        if not missed:
            return
        console.print(
            f"  [cyan]WHALE[/cyan] {len(trades)} recent trade(s), "
            f"{len(missed)} missed opportunit(ies)"
        )
        for opp in missed:
            await self._investigate_whale_opportunity(opp)

    async def _investigate_whale_opportunity(self, opp: dict):
        """One Claude call decides hold/investigate; if investigate, run the
        market through the normal classify -> gates path with edge_type='whale'.
        A whale_triggered classification is always logged (opportunity record +
        6h cooldown marker)."""
        market = opp["market"]
        loop = asyncio.get_event_loop()

        decision = await loop.run_in_executor(
            None, lambda: whale_tracker.decide_investigation(opp)
        )
        if decision != "investigate":
            console.print(
                f"  [dim]WHALE HOLD: \"{market.question[:40]}\" "
                f"(whale ${opp['size_usd']:,.0f})[/dim]"
            )
            return

        self.stats["whale_investigations"] = self.stats.get("whale_investigations", 0) + 1
        headline = whale_tracker.whale_headline(opp)
        now = datetime.now(timezone.utc)
        whale_event = NewsEvent(
            headline=headline, source="whale", url="",
            received_at=now, published_at=now, latency_ms=0,
        )

        classification = await classify_async(headline, market, "whale")

        signal = None
        if not classification.error:
            raw_signal = detect_edge_v2(market, classification, whale_event)
            signal, _ = gate_trade(whale_event, raw_signal, self._traded_headlines)

            if signal is not None:
                open_positions = await loop.run_in_executor(None, positions.get_open_positions)
                corr = positions.check_correlation_risk(
                    market.question, signal.side, open_positions
                )
                if corr["risk_level"] == "high":
                    signal = None
                else:
                    imbalance = await loop.run_in_executor(
                        None, fetch_orderbook_imbalance, get_token_id(market, "YES")
                    )
                    if not orderbook_allows(signal.side, imbalance):
                        signal = None
                    else:
                        event_id = getattr(market, "event_id", "") or ""
                        exposure = event_context.get_event_exposure(event_id, open_positions)
                        if event_id and exposure["position_count"] >= config.MAX_POSITIONS_PER_EVENT:
                            signal = None
                        else:
                            signal.edge_type = "whale"

        logger.log_classification(
            market_question=market.question,
            headline=headline,
            news_source="whale",
            direction=classification.direction,
            materiality=classification.materiality,
            edge=signal.edge if signal else None,
            action="whale_triggered",
            match_source="whale",
            condition_id=market.condition_id,
            yes_price=market.yes_price,
            yes_token_id=get_token_id(market, "YES"),
            edge_type="whale",
            confidence=classification.confidence if not classification.error else None,
            event_id=getattr(market, "event_id", "") or None,
        )

        if signal is not None:
            self.stats["signals_found"] += 1
            await self.signal_queue.put(signal)
            console.print(
                f"  [bright_green]WHALE SIGNAL[/bright_green] "
                f"{classification.direction.upper()} mat:{classification.materiality:.2f} "
                f"→ {signal.side} ${signal.bet_amount} on \"{market.question[:40]}\""
            )
        else:
            console.print(
                f"  [cyan]WHALE TRIGGERED[/cyan] (no trade after gates) "
                f"\"{market.question[:40]}\" "
                f"[{classification.direction} mat:{classification.materiality:.2f}]"
            )

    async def _execute_signals(self):
        """Execute trades from the signal queue."""
        while True:
            signal: Signal = await self.signal_queue.get()
            result = await execute_trade_async(signal)
            self.stats["trades_executed"] += 1

            if result["status"] in ("dry_run", "executed"):
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: positions.open_position(
                        result["trade_id"], signal.market, signal.side,
                        signal.bet_amount, headline=signal.headlines,
                        reasoning=signal.reasoning,
                    ),
                )

            status_color = "bright_green" if result["status"] in ("dry_run", "executed") else "red"
            console.print(
                f"  [{status_color}]{result['status']}[/{status_color}] "
                f"{result['side']} ${result['amount']:.2f} "
                f"on \"{result['market'][:40]}\" "
                f"(edge:{result['edge']:.1%} latency:{result.get('latency_ms', 0)}ms)"
            )

    async def _status_printer(self):
        """Print periodic status updates and export the public dashboard snapshot."""
        last_news_processed = 0
        while True:
            await asyncio.sleep(30)
            ns = self.news_aggregator.stats
            console.print(
                f"\n  [dim]Status: "
                f"news={self.stats['news_processed']} "
                f"(tw:{ns.get('twitter', 0)} tg:{ns.get('telegram', 0)} rss:{ns.get('rss', 0)} na:{ns.get('newsapi', 0)}) "
                f"matched={self.stats['markets_matched']} "
                f"signals={self.stats['signals_found']} "
                f"trades={self.stats['trades_executed']} "
                f"markets={len(self.market_watcher.tracked_markets)}"
                + (f" [red]restarts={self.stats['task_restarts']}[/red]" if self.stats.get("task_restarts") else "")
                + (f" [red]err={self.stats['classify_error_streak']}[/red]" if self.stats.get("classify_error_streak") else "")
                + "[/dim]\n"
            )

            headlines_last_cycle = self.stats["news_processed"] - last_news_processed
            last_news_processed = self.stats["news_processed"]

            # Mark open positions to live prices; hard SL + rule-triggered
            # re-evaluations happen inside (off the event loop).
            try:
                prices = {
                    cid: snap.last_price
                    for cid, snap in self.market_watcher.snapshots.items()
                }
                pos_stats = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: positions.update_and_manage(prices)
                )
                if pos_stats.get("stop_losses") or pos_stats.get("reevals"):
                    console.print(
                        f"  [dim]positions: {pos_stats['updated']} marked, "
                        f"{pos_stats['stop_losses']} stop-loss, "
                        f"{pos_stats['reevals']} re-evals[/dim]"
                    )
            except Exception as e:
                log.warning(f"[pipeline] Position management error: {e}")

            # Daily journal: first cycle after 21:00 UTC (no-op otherwise).
            try:
                entry = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: maybe_write_journal(self.stats)
                )
                if entry:
                    console.print(f"  [dim]journal entry written ({len(entry.split())} words)[/dim]")
            except Exception as e:
                log.warning(f"[pipeline] Journal error: {e}")

            try:
                export_status(
                    headlines_last_cycle=headlines_last_cycle,
                    markets_tracked=len(self.market_watcher.tracked_markets),
                    classify_error_streak=self.stats.get("classify_error_streak", 0),
                    current_prices={
                        cid: snap.last_price
                        for cid, snap in self.market_watcher.snapshots.items()
                    },
                    whale_last_check=self.stats.get("whale_last_check"),
                )
            except Exception as e:
                log.warning(f"[pipeline] Status export error: {e}")


def run_pipeline_v2():
    """Entry point for V2 event-driven pipeline."""
    pipeline = PipelineV2()
    try:
        asyncio.run(pipeline.run())
    except KeyboardInterrupt:
        console.print(f"\n[bright_green]Pipeline stopped. {pipeline.stats}[/bright_green]")


# ============================================================
# V1: Synchronous Loop Pipeline (preserved for backward compat)
# ============================================================

def run_pipeline(
    max_markets: int = 10,
    lookback_hours: int | None = None,
    categories: list[str] | None = None,
) -> list[dict]:
    """V1: Run the full pipeline once. Returns list of trade results."""

    run_id = logger.log_run_start()
    results = []
    signals: list[Signal] = []

    mode = "[yellow]DRY RUN[/yellow]" if config.DRY_RUN else "[red bold]LIVE[/red bold]"
    console.print(Panel(f"Pipeline V1 Run #{run_id}  |  Mode: {mode}", style="cyan"))

    # Step 1: Scrape News
    console.print("\n[bold]1. Scraping news...[/bold]")
    news = scrape_all(lookback_hours)
    console.print(f"   Found {len(news)} unique headlines")

    if not news:
        console.print("[yellow]   No news found. Aborting run.[/yellow]")
        logger.log_run_end(run_id, 0, 0, 0, "no_news")
        export_status(headlines_last_cycle=0)
        return results

    # Step 2: Fetch Markets
    console.print("\n[bold]2. Fetching Polymarket markets...[/bold]")
    all_markets = fetch_active_markets(limit=100)
    markets = filter_by_categories(all_markets, categories)[:max_markets]
    console.print(f"   {len(markets)} markets in target categories (of {len(all_markets)} total)")

    if not markets:
        console.print("[yellow]   No markets found. Aborting run.[/yellow]")
        logger.log_run_end(run_id, 0, 0, 0, "no_markets")
        export_status(headlines_last_cycle=len(news))
        return results

    # Step 3: Score Each Market
    console.print(f"\n[bold]3. Scoring {len(markets)} markets against news...[/bold]")

    for i, market in enumerate(markets):
        console.print(f"\n   [{i+1}/{len(markets)}] {market.question[:80]}")
        console.print(f"   Market price: YES={market.yes_price:.2f} NO={market.no_price:.2f}")

        relevant_news = filter_news_for_market(market, news)
        console.print(f"   Relevant headlines: {len(relevant_news)}")

        score_result = score_market(market, relevant_news)
        claude_score = score_result["confidence"]
        reasoning = score_result["reasoning"]
        console.print(f"   Claude score: {claude_score:.2f}  (market: {market.yes_price:.2f})")

        headlines_str = "\n".join(n.headline for n in relevant_news[:5])
        signal = detect_edge(market, claude_score, reasoning, headlines_str)

        if signal:
            edge_pct = signal.edge * 100
            console.print(f"   [green bold]SIGNAL: {signal.side} | Edge: {edge_pct:.1f}% | Size: ${signal.bet_amount}[/green bold]")
            signals.append(signal)
        else:
            edge = abs(claude_score - market.yes_price)
            console.print(f"   [dim]No edge (diff: {edge:.2f}, threshold: {config.EDGE_THRESHOLD})[/dim]")

        time.sleep(0.5)

    # Step 4: Execute Trades
    if signals:
        console.print(f"\n[bold]4. Executing {len(signals)} trades...[/bold]")
        for signal in signals:
            result = execute_trade(signal)
            results.append(result)
            status_color = "green" if result["status"] in ("dry_run", "executed") else "red"
            console.print(f"   [{status_color}]{result['status']}[/{status_color}] {result['market'][:60]} | {result['side']} ${result['amount']}")
    else:
        console.print("\n[bold]4. No signals — nothing to execute.[/bold]")

    logger.log_run_end(run_id, len(markets), len(signals), len(results))
    _print_summary(results, len(markets), len(signals))
    export_status(headlines_last_cycle=len(news))
    return results


def _print_summary(results: list[dict], markets_scanned: int, signals_found: int):
    table = Table(title="Pipeline Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Markets scanned", str(markets_scanned))
    table.add_row("Signals found", str(signals_found))
    table.add_row("Trades placed", str(len(results)))
    table.add_row("Mode", "DRY RUN" if config.DRY_RUN else "LIVE")
    console.print(table)


if __name__ == "__main__":
    run_pipeline()
