#!/usr/bin/env python3
"""
Polymarket Pipeline — V2 (async, event-driven).
News stream → Match → Classify → Edge → Trade.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from rich.console import Console
from rich.panel import Panel

from locus import config
from locus.memory import logger
from locus.core.export_status import export_status
from locus.core.performance import compute_circuit_breaker
from locus.markets.gamma import get_token_id
from locus.core.edge import detect_edge_v2, Signal
from locus.core.executor import execute_trade_async
from locus.supervisor import supervise
from locus.sources.news_stream import NewsAggregator, NewsEvent
from locus.markets.market_watcher import MarketWatcher
from locus.core.matcher import match_news_to_markets, match_news_to_markets_hybrid, prefilter_match
from locus.core.classifier import (
    classify_async, classify_fast, classify_edge_type, haiku_prefilter, classify,
)
from locus.core.multi_classifier import classify_ensemble, is_low_consensus
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

def _batches(items, size):
    """Yield successive `size`-length slices of `items` (size floored to 1)."""
    size = max(1, size)
    for i in range(0, len(items), size):
        yield items[i:i + size]


def news_age_seconds(event: NewsEvent, now: datetime | None = None) -> float | None:
    """Age of the news at this moment (publication -> now), or None when the
    publication time is unknown (latency_ms == -1 sentinel). Computed at
    decision time, not receipt time, so queue dwell counts against it."""
    if event.latency_ms is not None and event.latency_ms < 0:
        return None
    now = now or datetime.now(timezone.utc)
    return (now - event.published_at).total_seconds()


def is_price_target_market(question: str) -> bool:
    """True when a market question is a price-target market (e.g. "Will Bitcoin
    reach $100k") — these resolve on a price threshold that news rarely moves
    cleanly in our favor, so they're excluded when EXCLUDE_PRICE_TARGET_MARKETS
    is set. Matched case-insensitively against config.PRICE_TARGET_KEYWORDS."""
    if not config.EXCLUDE_PRICE_TARGET_MARKETS:
        return False
    q = (question or "").lower()
    return any(kw.lower() in q for kw in config.PRICE_TARGET_KEYWORDS)


def gate_trade(event: NewsEvent, signal, traded_headlines: set[str], now: datetime | None = None):
    """Trade-time risk gates, applied after classification so every
    classification is still logged and calibrated.

    Returns (signal_or_none, action):
      "skip"               — no edge detected
      "stale"              — would-be signal, but the headline is older than
                             the source-specific freshness limit
                             (config.get_max_age_seconds) as of *now*
                             (including time spent in the queue), or its
                             publication time is unknown; we classify old news
                             for calibration but never trade on it
      "capped"             — this headline already produced a trade; one
                             headline matching N markets must not open N
                             correlated positions
      "too_close_to_resolution" — market resolves within
                             config.MIN_HOURS_TO_RESOLUTION hours; the thesis
                             has too little time to play out
      "price_target_market" — market is a price-target market (e.g. "Will
                             Bitcoin reach $100k"); excluded when
                             config.EXCLUDE_PRICE_TARGET_MARKETS is set
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
    if age is None or age > config.get_max_age_seconds(signal.news_source):
        return None, "stale"
    if event.headline in traded_headlines:
        return None, "capped"

    # Market-structure gates (independent of the news): never open a position
    # in a market about to resolve (no time for the thesis to play out) or a
    # price-target market (resolves on a price threshold news rarely moves
    # cleanly). Neither consumes the headline cap.
    market = signal.market
    hours_left = positions.hours_to_close(market.end_date, now)
    if hours_left is not None and hours_left < config.MIN_HOURS_TO_RESOLUTION:
        log.info(
            f"Filtered: too_close_to_resolution | {market.slug} | {hours_left:.1f}h left"
        )
        return None, "too_close_to_resolution"
    if is_price_target_market(market.question):
        log.info(
            f"Filtered: price_target_market | {market.slug} | "
            f"question: {market.question[:70]}..."
        )
        return None, "price_target_market"

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
        # Separate concurrency limits for the two model tiers: Haiku is cheap and
        # fast (allow many in flight), Sonnet is expensive (keep it tight).
        self._haiku_sem = asyncio.Semaphore(config.HAIKU_SEMAPHORE_SIZE)
        self._sonnet_sem = asyncio.Semaphore(config.SONNET_SEMAPHORE_SIZE)
        # Latest circuit-breaker read (refreshed each status cycle and whenever a
        # signal is evaluated); drives the trade-time gate and the status line.
        self._circuit_breaker: dict = {"triggered": False, "reason": "", "metrics": {}}
        self.running = False
        self.stats = {
            "news_processed": 0,
            "markets_matched": 0,
            "signals_found": 0,
            "trades_executed": 0,
            "whale_last_check": None,
            # Running totals for the average model-call latency exported to the
            # dashboard (sum / count over classifications that hit a model).
            "classification_latency_sum_ms": 0,
            "classification_count": 0,
        }

    def avg_classification_latency_ms(self) -> float:
        """Mean latency of model-backed classifications so far (0.0 if none)."""
        n = self.stats.get("classification_count", 0)
        return self.stats["classification_latency_sum_ms"] / n if n else 0.0

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
        console.print(
            f"  News sources: {len(config.RSS_FEEDS)} RSS feeds, "
            f"{len(config.TWITTER_KEYWORDS)} Twitter keywords"
        )
        console.print()

        await asyncio.get_event_loop().run_in_executor(None, positions.backfill_positions)
        # Pre-warm the market-embedding LRU cache from the persisted Chroma
        # collection so the first headline burst doesn't pay a cold read per
        # market. Off the event loop; safe no-op on a fresh (empty) store.
        await asyncio.get_event_loop().run_in_executor(
            None, self.market_watcher.index.pre_warm
        )

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

            # Classify each matched market as its own coroutine, so a headline
            # matching N markets doesn't serialize N model calls. The per-
            # candidate gates are synchronous and never yield mid-check, so the
            # one-position-per-headline cap in gate_trade still holds under the
            # concurrent gather below.
            async def process_candidate(market, match_source, match_score):
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
                        return

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
                        return

                    # Classify under the per-tier concurrency limits (Haiku
                    # prefilter -> Sonnet deep when tiered; ensemble/single-model
                    # otherwise). See _classify_with_semaphores.
                    classification = await self._classify_with_semaphores(event, market)

                    # Feed the dashboard's average model-call latency.
                    self.stats["classification_latency_sum_ms"] += classification.latency_ms
                    self.stats["classification_count"] += 1

                    if classification.action == "prefiltered_haiku":
                        # Haiku triaged this out before any Sonnet call — log the
                        # action and skip the edge/gate chain entirely.
                        self.stats["prefiltered_haiku"] = (
                            self.stats.get("prefiltered_haiku", 0) + 1
                        )
                        raw_signal, signal, action, edge_type = None, None, "prefiltered_haiku", None
                    elif classification.error:
                        self.stats["classify_error_streak"] = (
                            self.stats.get("classify_error_streak", 0) + 1
                        )
                        raw_signal, signal, action, edge_type = None, None, "error", None
                    else:
                        self.stats["classify_error_streak"] = 0
                        edge_metrics = detect_edge_v2(market, classification, event)
                        raw_signal = edge_metrics.signal if edge_metrics else None
                        signal, action = gate_trade(event, raw_signal, self._traded_headlines)

                        # Multi-LLM consensus gate: when the two models disagree
                        # (consensus below the floor), don't trust the call —
                        # suppress the would-be signal. Single-model/non-ensemble
                        # results (consensus_score None) skip this gate.
                        if signal is not None and is_low_consensus(classification):
                            self.stats["low_consensus"] = (
                                self.stats.get("low_consensus", 0) + 1
                            )
                            signal, action = None, "low_consensus"
                            console.print(
                                f"  [yellow]LOW CONSENSUS[/yellow]: "
                                f"{classification.consensus_score:.2f} < "
                                f"{config.ENSEMBLE_MIN_CONSENSUS} on "
                                f"\"{market.question[:40]}\""
                            )

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

                        # Category exposure gate: cap combined open exposure per
                        # market category. Soft band (>= CATEGORY_SOFT_LIMIT_PCT
                        # of the hard limit) warns but allows; over the hard
                        # limit blocks (action 'category_limit').
                        if signal is not None:
                            cat_exp = positions.check_category_exposure(
                                getattr(market, "category", "") or "other",
                                positions.get_open_positions(),
                            )
                            if not cat_exp["allowed"]:
                                self.stats["category_limits"] = (
                                    self.stats.get("category_limits", 0) + 1
                                )
                                signal, action = None, "category_limit"
                                console.print(
                                    f"  [red]CATEGORY LIMIT[/red]: {market.category} "
                                    f"exposure ${cat_exp['current_usd']:.0f}/"
                                    f"${cat_exp['limit_usd']:.0f} ({cat_exp['pct']:.0%}) "
                                    f"on \"{market.question[:40]}\""
                                )
                            elif cat_exp["warning"]:
                                console.print(
                                    f"  [yellow]CATEGORY WARNING[/yellow]: {market.category} "
                                    f"exposure ${cat_exp['current_usd']:.0f}/"
                                    f"${cat_exp['limit_usd']:.0f} ({cat_exp['pct']:.0%}) "
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

                        # Circuit breaker: when recent realized performance has
                        # deteriorated past the configured limits, hold every
                        # would-be trade until it recovers. Final say over every
                        # earlier gate (including re-entry), so it runs last.
                        # Signals are rare, so the extra DB read here is cheap.
                        if signal is not None:
                            breaker = await asyncio.get_event_loop().run_in_executor(
                                None, compute_circuit_breaker
                            )
                            self._circuit_breaker = breaker
                            if breaker["triggered"]:
                                self.stats["circuit_breaker_blocks"] = (
                                    self.stats.get("circuit_breaker_blocks", 0) + 1
                                )
                                signal, action = None, "circuit_breaker"
                                reentry_active = False
                                console.print(
                                    f"  [red]CIRCUIT BREAKER ACTIVE[/red]: "
                                    f"{breaker['reason']} — holding trade on "
                                    f"\"{market.question[:40]}\""
                                )

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
                        expected_edge=raw_signal.expected_edge if raw_signal else None,
                        vol_adj=raw_signal.vol_adj if raw_signal else None,
                        action=action,
                        match_source=match_source,
                        condition_id=market.condition_id,
                        yes_price=market.yes_price,
                        yes_token_id=get_token_id(market, "YES"),
                        edge_type=edge_type,
                        confidence=classification.confidence if not classification.error else None,
                        event_id=getattr(market, "event_id", "") or None,
                        consensus_score=classification.consensus_score,
                        ensemble_used=classification.ensemble_used,
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

            # Run the candidates concurrently in batches of PARALLEL_BATCH_SIZE.
            batch_size = max(1, config.PARALLEL_BATCH_SIZE)
            log.info(
                f"Processing {len(matched)} candidates in parallel (batch {batch_size})"
            )
            for batch in _batches(matched, batch_size):
                await asyncio.gather(*(
                    process_candidate(market, match_source, match_score)
                    for market, match_source, match_score in batch
                ))

    async def _classify_with_semaphores(self, event, market):
        """Classify one headline/market under the per-tier concurrency limits.

        Tiered: the Haiku prefilter runs under the Haiku semaphore, then — for
        survivors only — the deep Sonnet call runs under the tighter Sonnet
        semaphore. Non-tiered ensemble/single-model calls run under the Sonnet
        (expensive-tier) semaphore."""
        loop = asyncio.get_event_loop()
        if config.TIERED_CLASSIFICATION_ENABLED:
            async with self._haiku_sem:
                rejected = await loop.run_in_executor(
                    None, haiku_prefilter, event.headline, market, event.source
                )
            if rejected is not None:
                return rejected
            async with self._sonnet_sem:
                return await loop.run_in_executor(
                    None, classify, event.headline, market, event.source, None,
                    config.SCORING_MODEL,
                )
        if config.ENSEMBLE_ENABLED:
            async with self._sonnet_sem:
                return await classify_ensemble(event.headline, market, event.source)
        async with self._sonnet_sem:
            return await classify_async(event.headline, market, event.source)

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
            edge_metrics = detect_edge_v2(market, classification, whale_event)
            raw_signal = edge_metrics.signal if edge_metrics else None
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
            expected_edge=signal.expected_edge if signal else None,
            vol_adj=signal.vol_adj if signal else None,
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

            # Refresh the circuit-breaker read each cycle so the status line
            # reflects it even between signals (off the event loop — DB read).
            try:
                self._circuit_breaker = await asyncio.get_event_loop().run_in_executor(
                    None, compute_circuit_breaker
                )
            except Exception as e:
                log.warning(f"[pipeline] Circuit breaker check error: {e}")

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
            if self._circuit_breaker.get("triggered"):
                console.print(
                    f"  [red bold]CIRCUIT BREAKER ACTIVE[/red bold] "
                    f"[red]({self._circuit_breaker.get('reason', '')}) — trading paused[/red]"
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
                if (pos_stats.get("stop_losses") or pos_stats.get("reevals")
                        or pos_stats.get("time_pressure_exits")
                        or pos_stats.get("near_certain_exits")):
                    console.print(
                        f"  [dim]positions: {pos_stats['updated']} marked, "
                        f"{pos_stats['stop_losses']} stop-loss, "
                        f"{pos_stats.get('time_pressure_exits', 0)} time-pressure, "
                        f"{pos_stats.get('near_certain_exits', 0)} near-certain, "
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
                    avg_classification_latency_ms=self.avg_classification_latency_ms(),
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
