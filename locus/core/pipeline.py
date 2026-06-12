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
from datetime import datetime, timezone

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
from locus.core.matcher import match_news_to_markets, match_news_to_markets_hybrid
from locus.core.classifier import classify_async
from locus.core.journal import maybe_write_journal

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
      "skip"   — no edge detected
      "stale"  — would-be signal, but the headline is older than
                 MAX_NEWS_AGE_SECONDS as of *now* (including time spent in
                 the queue), or its publication time is unknown; we classify
                 old news for calibration but never trade on it
      "capped" — this headline already produced a trade; one headline
                 matching N markets must not open N correlated positions
      "signal" — trade approved; headline recorded against the cap
    """
    if signal is None:
        return None, "skip"
    age = news_age_seconds(event, now)
    if age is None or age > config.MAX_NEWS_AGE_SECONDS:
        return None, "stale"
    if event.headline in traded_headlines:
        return None, "capped"
    traded_headlines.add(event.headline)
    return signal, "signal"


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
        }

    async def run(self):
        """Start all pipeline components concurrently."""
        self.running = True
        mode = "[red bold]LIVE[/red bold]" if not config.DRY_RUN else "[yellow]DRY RUN[/yellow]"
        console.print(Panel(f"Pipeline V2 Starting  |  Mode: {mode}", style="bright_green"))
        console.print(f"  Niche filter: ${config.MIN_VOLUME_USD:,.0f} - ${config.MAX_VOLUME_USD:,.0f} volume")
        console.print(f"  Materiality threshold: {config.MATERIALITY_THRESHOLD}")
        console.print(f"  Speed target: {config.SPEED_TARGET_SECONDS}s")
        console.print()

        try:
            await asyncio.gather(
                supervise("news_aggregator", self.news_aggregator.run, self.stats),
                supervise("market_watcher", self.market_watcher.run, self.stats),
                supervise("process_news", self._process_news, self.stats),
                supervise("execute_signals", self._execute_signals, self.stats),
                supervise("status_printer", self._status_printer, self.stats),
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
            for market, match_source in matched:
                try:
                    classification = await classify_async(
                        event.headline, market, event.source
                    )

                    if classification.error:
                        self.stats["classify_error_streak"] = (
                            self.stats.get("classify_error_streak", 0) + 1
                        )
                        raw_signal, signal, action = None, None, "error"
                    else:
                        self.stats["classify_error_streak"] = 0
                        raw_signal = detect_edge_v2(market, classification, event)
                        signal, action = gate_trade(event, raw_signal, self._traded_headlines)

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
                    )

                    if action in ("stale", "capped"):
                        age = news_age_seconds(event)
                        age_min = age / 60 if age is not None else -1
                        console.print(
                            f"  [dim]{action.upper()}: suppressed would-be signal "
                            f"({event.source}, news age {'unknown' if age is None else f'{age_min:.0f}m'}) "
                            f"on \"{market.question[:40]}\"[/dim]"
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

    async def _execute_signals(self):
        """Execute trades from the signal queue."""
        while True:
            signal: Signal = await self.signal_queue.get()
            result = await execute_trade_async(signal)
            self.stats["trades_executed"] += 1

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
