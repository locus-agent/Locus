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
from locus.markets.gamma import get_token_id, is_coinflip_market
from locus.core.edge import detect_edge_v2, Signal
from locus.core.executor import execute_trade_async
from locus.supervisor import supervise
from locus.sources.news_stream import NewsAggregator, NewsEvent
from locus.markets.market_watcher import MarketWatcher
from locus.core.matcher import match_news_to_markets, match_news_to_markets_hybrid, prefilter_match
from locus.core.classifier import (
    classify_fast, classify_edge_type, haiku_prefilter, classify,
    verify_novelty,
)
from locus.core.multi_classifier import classify_ensemble, is_low_consensus
from locus.core import event_context
from locus.core import passive
from locus.core import whale_tracker
from locus.core.orderbook import fetch_orderbook_imbalance, orderbook_allows
from locus.core.journal import maybe_write_journal, maybe_check_missed_opportunities
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


def _hours_since(ts: str | None, now: datetime | None = None) -> float:
    """Hours elapsed since a stored timestamp ("YYYY-MM-DD HH:MM:SS", UTC).
    Returns a large number when the timestamp is missing/unparseable, so a
    cooldown check treats unknown ages as 'long ago' (never blocks on bad data)."""
    if not ts:
        return float("inf")
    try:
        dt = datetime.fromisoformat(ts.replace(" ", "T"))
    except (ValueError, AttributeError):
        return float("inf")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return (now - dt).total_seconds() / 3600.0


def news_age(event: NewsEvent, now: datetime | None = None) -> tuple[float, str]:
    """Age of the news at this moment and the timestamp basis it was measured
    from. Prefers the publication time (published_at); falls back to the receipt
    time (received_at) when the source gave no usable publication time — either a
    None published_at or the latency_ms == -1 sentinel (e.g. an RSS item with no
    parseable <pubDate>). Computed at decision time, not receipt time, so queue
    dwell counts against it.

    Returns (age_seconds, "published_at" | "received_at")."""
    now = now or datetime.now(timezone.utc)
    published = getattr(event, "published_at", None)
    unknown_pub = published is None or (event.latency_ms is not None and event.latency_ms < 0)
    if unknown_pub:
        return (now - event.received_at).total_seconds(), "received_at"
    return (now - published).total_seconds(), "published_at"


def known_published_at(event: NewsEvent) -> str | None:
    """The event's publication time as an ISO 8601 string when the source gave a
    usable one, else None — so the classifications.published_at column records a
    real publication time and stays NULL when we only have receipt time (the
    None published_at / latency_ms == -1 sentinel; see news_age)."""
    published = getattr(event, "published_at", None)
    if published is None or (event.latency_ms is not None and event.latency_ms < 0):
        return None
    return published.isoformat()


def is_price_target_market(question: str) -> bool:
    """True when a market question is a price-target market (e.g. "Will Bitcoin
    reach $100k") — these resolve on a price threshold that news rarely moves
    cleanly in our favor, so they're excluded when EXCLUDE_PRICE_TARGET_MARKETS
    is set. Matched case-insensitively against config.PRICE_TARGET_KEYWORDS."""
    if not config.EXCLUDE_PRICE_TARGET_MARKETS:
        return False
    q = (question or "").lower()
    return any(kw.lower() in q for kw in config.PRICE_TARGET_KEYWORDS)


def is_geopolitical(market, headline: str = "") -> bool:
    """True when a market (and optional headline) is about long-horizon
    geopolitics — Iran deals, the Ukraine/Russia war, Taiwan tensions,
    sanctions, treaties, etc. Matched case-insensitively against
    config.GEOPOLITICAL_KEYWORDS over the market question + headline.
    'u.s.'/'united states' are normalized to 'us' so the keyword list can stay
    simple. Used to widen the freshness window for slow-moving geopolitical
    markets (see gate_trade)."""
    text = (getattr(market, "question", "") + " " + (headline or "")).lower()
    text = text.replace("u.s.", "us").replace("united states", "us")
    return any(kw in text for kw in config.GEOPOLITICAL_KEYWORDS)


def should_verify_novelty(materiality: float, yes_price: float) -> bool:
    """Whether a signal warrants a Chain-of-Verification novelty check: only
    high-materiality calls (>= COV_MATERIALITY_THRESHOLD) at non-extreme prices
    (0.15 < price < 0.85). Extreme prices are skipped — there the move is
    structural, not a question of whether the news is already priced in.
    Disabled wholesale when COV_ENABLED is false."""
    return (
        config.COV_ENABLED
        and materiality >= config.COV_MATERIALITY_THRESHOLD
        and 0.15 < yes_price < 0.85
    )


def cov_blocks(cov: dict | None) -> bool:
    """True when a CoV result confidently says the news is already priced in
    (already_priced AND confidence >= COV_CONFIDENCE_THRESHOLD). A None result
    (skipped or failed open) never blocks."""
    return bool(
        cov is not None
        and cov.get("already_priced")
        and cov.get("confidence", 0.0) >= config.COV_CONFIDENCE_THRESHOLD
    )


async def switch_target_gate(switched, headline: str,
                             open_positions: list[dict]) -> str | None:
    """Re-run the market-specific gates against an event-switch TARGET
    (LOGIC_REVIEW finding #8). Every gate before the event-context step was
    evaluated against the ORIGINAL market; a recommended sibling is a different
    market — different resolution time, question shape, category, and order
    book — and must clear the same market-specific bars before capital moves:

      - sports_disabled + the (sports-aware) resolution-time floor
      - price-target and coin-flip question filters
      - category exposure (the sibling can be a different inferred category)
      - orderbook imbalance (different token, different book)
      - CoV already-priced check at the SIBLING's price (when warranted)

    Signal-level gates are NOT redone: freshness, the headline cap, materiality
    floors, and confirmation judge the news itself, which is unchanged —
    and price-room, the edge threshold, held-markets, and correlation are
    already re-checked per candidate inside event_context.find_best_outcome.

    Returns None when the target passes, else the failing gate's action name;
    the caller declines the switch and keeps the original signal (which
    cleared its own full gate chain), mirroring the negative-Kelly decline."""
    market = switched.market
    is_sports = (getattr(market, "category", "") or "") == "sports"
    if is_sports and not config.SPORTS_ENABLED:
        return "sports_disabled"
    min_hours = (
        config.SPORTS_MIN_HOURS_TO_RESOLUTION if is_sports
        else config.MIN_HOURS_TO_RESOLUTION
    )
    hours_left = positions.hours_to_close(market.end_date)
    if hours_left is not None and hours_left < min_hours:
        return "too_close_to_resolution"
    if is_price_target_market(market.question):
        return "price_target_market"
    if is_coinflip_market(market.question):
        return "coinflip_market"
    cat_exp = positions.check_category_exposure(
        getattr(market, "category", "") or "other", open_positions or []
    )
    if not cat_exp["allowed"]:
        return "category_limit"

    loop = asyncio.get_event_loop()
    imbalance = await loop.run_in_executor(
        None, fetch_orderbook_imbalance, get_token_id(market, "YES")
    )
    if not orderbook_allows(switched.side, imbalance):
        return "orderbook_skip"

    if should_verify_novelty(switched.materiality, market.yes_price):
        cov = await loop.run_in_executor(
            None, verify_novelty, headline, market, market.yes_price
        )
        if cov_blocks(cov):
            return "already_priced_in"
    return None


def get_materiality_threshold(direction: str, category: str) -> float:
    """Materiality floor for a would-be signal, chosen by category then direction.

    Priority order (first match wins):
      - category == "geopolitical" -> MIN_MATERIALITY_GEOPOLITICAL
      - category == "sports"       -> MIN_MATERIALITY_SPORTS
      - direction == "bearish"     -> MIN_MATERIALITY_BEARISH
      - direction == "bullish"     -> MIN_MATERIALITY_BULLISH
      - otherwise                  -> MIN_MATERIALITY_DEFAULT

    Category takes precedence over direction: sports headlines are noisy and
    geopolitical markets move slowly, so those floors apply regardless of the
    call's direction. Reads config.MIN_MATERIALITY_* at call time so runtime
    overrides (cli --threshold) are honored."""
    if category == "geopolitical":
        return config.MIN_MATERIALITY_GEOPOLITICAL
    if category == "sports":
        return config.MIN_MATERIALITY_SPORTS
    if direction == "bearish":
        return config.MIN_MATERIALITY_BEARISH
    if direction == "bullish":
        return config.MIN_MATERIALITY_BULLISH
    return config.MIN_MATERIALITY_DEFAULT


def effective_materiality(signal) -> float:
    """The materiality the gates judge a signal by: the time-horizon-penalized
    adjusted_materiality when detect_edge_v2 set it, else the raw materiality
    (signals built outside the edge path — e.g. tests — leave it None)."""
    adj = getattr(signal, "adjusted_materiality", None)
    return adj if adj is not None else signal.materiality


def multi_source_adjust(signal, confirmed: bool) -> str:
    """Multi-source confirmation for high-materiality signals (size, not block).

    Only acts when effective materiality >= MULTI_SOURCE_CONFIRM_THRESHOLD. When
    an independent second source already vouched for the call (`confirmed`), the
    size is untouched; otherwise it is shaved by MULTI_SOURCE_SIZE_REDUCTION.
    Returns 'confirmed', 'reduced', or 'n/a' (below the threshold) for logging.
    """
    if effective_materiality(signal) < config.MULTI_SOURCE_CONFIRM_THRESHOLD:
        return "n/a"
    if confirmed:
        return "confirmed"
    signal.bet_amount = round(signal.bet_amount * (1.0 - config.MULTI_SOURCE_SIZE_REDUCTION), 2)
    return "reduced"


def gate_trade(event: NewsEvent, signal, traded_headlines: set[str], now: datetime | None = None,
               sports_event_counts: dict[str, int] | None = None):
    """Trade-time risk gates, applied after classification so every
    classification is still logged and calibrated.

    Sports markets are gated extra strictly: when config.SPORTS_ENABLED is
    false they are dropped entirely (action 'sports_disabled'); otherwise they
    use config.MIN_MATERIALITY_SPORTS (the sports floor from
    get_materiality_threshold, not a direction floor),
    config.SPORTS_MIN_HOURS_TO_RESOLUTION (not MIN_HOURS_TO_RESOLUTION), and a
    per-event headline cap — once sports_event_counts[event_id] reaches
    config.MAX_HEADLINES_PER_SPORTS_EVENT the market is skipped (action
    'sports_event_cap').

    Returns (signal_or_none, action):
      "skip"               — no edge detected
      "sports_disabled"    — a sports market while config.SPORTS_ENABLED is off
      "sports_event_cap"   — sports market whose event already hit
                             config.MAX_HEADLINES_PER_SPORTS_EVENT traded headlines
      "stale"              — would-be signal, but the headline is older than
                             the source-specific freshness limit
                             (config.get_max_age_seconds) as of *now*
                             (including time spent in the queue). Age is measured
                             from published_at when the source gave a usable
                             publication time, else from received_at (see
                             news_age); we classify old news for calibration but
                             never trade on it
      "capped"             — this headline already produced a trade, or another
                             in-flight candidate currently holds its
                             reservation (see "signal" below); one headline
                             matching N markets must not open N correlated
                             positions
      "too_close_to_resolution" — market resolves within
                             config.MIN_HOURS_TO_RESOLUTION hours; the thesis
                             has too little time to play out
      "price_target_market" — market is a price-target market (e.g. "Will
                             Bitcoin reach $100k"); excluded when
                             config.EXCLUDE_PRICE_TARGET_MARKETS is set
      "coinflip_market"    — short-term "Up or Down" coin-flip market (e.g.
                             "Bitcoin Up or Down on June 29?"); normally dropped
                             from the niche set, blocked here as a safety net when
                             config.EXCLUDE_COINFLIP_MARKETS is set
      "low_materiality"    — materiality below the direction- and category-aware
                             floor (get_materiality_threshold: geopolitical/sports
                             categories first, then bearish > bullish)
      "needs_confirmation" — high-materiality (>= HIGH_MATERIALITY_THRESHOLD)
                             call seen from fewer than MIN_CONFIRMING_SOURCES
                             distinct sources within CONFIRMATION_WINDOW_HOURS;
                             obvious news is the least accurate, so hold until
                             a second source agrees. A confirming row must be
                             independent: a different source AND a
                             non-identical headline (exact match, the dedup
                             cache's key) with materiality >=
                             CONFIRMATION_MIN_MATERIALITY
      "signal"             — approved by these gates. The headline is RESERVED
                             into traded_headlines here, in the same synchronous
                             step as the "capped" check above (gate_trade never
                             awaits, so check-and-reserve is atomic under the
                             event loop): candidates for the same headline
                             evaluated concurrently can never both pass. The
                             reservation becomes permanent only when a position
                             actually opens; every later failure — a late gate
                             (CoV, orderbook, exposure, ...), execution
                             (resting/skipped/error), or a crash — must release
                             it via release_headline so the headline stays
                             available to other candidates. Policy: the cap is
                             one POSITION per headline, not one attempt.
    """
    if signal is None:
        return None, "skip"
    market = signal.market
    age, age_basis = news_age(event, now)
    if age_basis == "published_at":
        log.info(f"Using published_at | {event.source} | age {age / 60:.0f}m | {market.slug}")
    else:
        log.info(f"Using received_at (no pubDate) | {event.source} | age {age / 60:.0f}m | {market.slug}")
    # Geopolitical markets move slowly: a long-horizon (resolution > 7 days out)
    # diplomatic/military/political story stays tradeable for far longer than a
    # typical breaking headline, so flag it as the 'geopolitical' category to
    # get the much wider freshness window from get_max_age_seconds.
    hours_to_resolution = positions.hours_to_close(market.end_date, now)
    if (
        is_geopolitical(market, event.headline)
        and hours_to_resolution is not None
        and hours_to_resolution > 7 * 24
    ):
        category = "geopolitical"
        log.info(
            f"Geopolitical market: extended freshness window | {market.slug}"
        )
    else:
        category = getattr(market, "category", "") or ""
    max_age = config.get_max_age_seconds(
        signal.news_source, category, hours_to_resolution
    )
    if age > max_age:
        return None, "stale"
    if event.headline in traded_headlines:
        return None, "capped"

    # Market-structure gates (independent of the news): never open a position
    # in a market about to resolve (no time for the thesis to play out) or a
    # price-target market (resolves on a price threshold news rarely moves
    # cleanly). Neither consumes the headline cap.
    is_sports = (getattr(market, "category", "") or "") == "sports"

    # Sports markets are off entirely unless the feature is enabled.
    if is_sports and not config.SPORTS_ENABLED:
        return None, "sports_disabled"

    # Per-event headline cap for sports: one event (e.g. a single match) can
    # spawn many headlines; only trade the first MAX_HEADLINES_PER_SPORTS_EVENT.
    if is_sports:
        event_id = getattr(market, "event_id", "") or ""
        counts = sports_event_counts or {}
        if event_id and counts.get(event_id, 0) >= config.MAX_HEADLINES_PER_SPORTS_EVENT:
            log.info(f"Filtered: sports_event_cap | {market.slug} | event {event_id}")
            return None, "sports_event_cap"

    # Sports get a tighter resolution-time floor than the standard one.
    min_hours = (
        config.SPORTS_MIN_HOURS_TO_RESOLUTION if is_sports
        else config.MIN_HOURS_TO_RESOLUTION
    )
    hours_left = positions.hours_to_close(market.end_date, now)
    if hours_left is not None and hours_left < min_hours:
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
    # Safety net: short-term "Up or Down" coin-flips are normally dropped from the
    # niche set (market_watcher.get_niche_markets), but if one reaches here it's
    # still classified for calibration yet never traded.
    if is_coinflip_market(market.question):
        log.info(
            f"Filtered: coinflip_market | {market.slug} | "
            f"question: {market.question[:70]}..."
        )
        return None, "coinflip_market"

    # Materiality floor, direction- and category-aware (get_materiality_threshold):
    # category wins first (geopolitical markets move slowly; sports are noisy and
    # get a higher bar), then direction (calibration: bearish accuracy is worse
    # than bullish). `category` is the freshness category resolved above, so a
    # long-horizon geopolitical market gets the geopolitical floor and a sports
    # market gets the sports floor.
    direction = signal.classification
    floor = get_materiality_threshold(direction, category)
    # Judge against the time-horizon-penalized materiality (long-horizon
    # resolutions need a stronger raw read to clear the floor).
    mat = effective_materiality(signal)
    if mat < floor:
        return None, "low_materiality"

    # High-materiality confirmation gate: the most "obvious" news grades worst
    # (likely already priced in), so require the same directional read from
    # >= MIN_CONFIRMING_SOURCES distinct sources within the recent window.
    # A confirmation must be genuinely independent: a different source with a
    # NON-identical headline (exact match — the dedup cache's key; the same
    # wire story cross-posted through a second feed is logged under that
    # feed's source via the 'cached' path and is one story, not two) whose own
    # materiality cleared CONFIRMATION_MIN_MATERIALITY.
    if mat >= config.HIGH_MATERIALITY_THRESHOLD:
        since = (
            (now or datetime.now(timezone.utc))
            - timedelta(hours=config.CONFIRMATION_WINDOW_HOURS)
        ).strftime("%Y-%m-%d %H:%M:%S")
        sources = logger.get_confirming_sources(
            signal.market.condition_id, direction, since,
            exclude_headline=event.headline,
            min_materiality=config.CONFIRMATION_MIN_MATERIALITY,
        )
        sources.add(event.source)
        if len(sources) < config.MIN_CONFIRMING_SOURCES:
            return None, "needs_confirmation"

    # Reserve the headline NOW, atomically with the "capped" check at the top:
    # gate_trade is fully synchronous (no awaits), so under asyncio's
    # single-threaded loop no other candidate can run between that check and
    # this add — concurrent candidates for the same headline can never both
    # pass. The caller owns the reservation from here: it is committed by an
    # actual position open, and MUST be released (release_headline) on every
    # other outcome so a failed candidate doesn't burn the headline.
    traded_headlines.add(event.headline)
    return signal, "signal"


def release_headline(traded_headlines: set[str], headline: str) -> None:
    """Release a headline reservation taken by gate_trade.

    Called on every path where a gate-approved candidate does NOT end in an
    open position — a later gate block (low consensus, re-entry, correlation,
    category, orderbook, CoV, event exposure, circuit breaker), a lost
    execute-time race, an unfilled/skipped/errored order, or an exception —
    so the headline stays available: the cap is one POSITION per headline, not
    one attempt, and a later candidate may then reserve and open. A confirmed
    open never releases, making the reservation permanent. Discard-based, so
    releasing an unreserved headline is a harmless no-op."""
    traded_headlines.discard(headline)


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
        # Per-event traded-headline counts for sports markets, enforcing
        # config.MAX_HEADLINES_PER_SPORTS_EVENT (event_id -> count).
        self._sports_event_counts: dict[str, int] = {}
        # Separate concurrency limits for the two model tiers: Haiku is cheap and
        # fast (allow many in flight), Sonnet is expensive (keep it tight).
        self._haiku_sem = asyncio.Semaphore(config.HAIKU_SEMAPHORE_SIZE)
        self._sonnet_sem = asyncio.Semaphore(config.SONNET_SEMAPHORE_SIZE)
        # Per-event locks serializing the risk-recheck -> open critical section
        # so two signals on the same event (evaluated concurrently, before
        # either opened) can't both clear the exposure cap and double-open.
        # Keyed by event_id (or category fallback); see _acquire_position_lock.
        self._position_locks: dict[str, asyncio.Lock] = {}
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
            f"  Materiality floors: default {config.MIN_MATERIALITY_DEFAULT} / "
            f"bullish {config.MIN_MATERIALITY_BULLISH} / bearish {config.MIN_MATERIALITY_BEARISH} / "
            f"geo {config.MIN_MATERIALITY_GEOPOLITICAL} / sports {config.MIN_MATERIALITY_SPORTS} "
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
        # Passive-entry crash recovery (core/passive.py): reconcile persisted
        # pending limit orders against the CLOB — a fill that happened while we
        # were down opens its position late with the real fill data, a vanished
        # order releases — and re-reserve the headlines of orders still resting
        # so a second candidate can't trade them into this fresh session.
        try:
            recon = await asyncio.get_event_loop().run_in_executor(
                None, passive.reconcile_on_startup
            )
            for headline in recon.get("reserved_headlines", []):
                self._traded_headlines.add(headline)
        except Exception as e:
            log.warning(f"[pipeline] Passive startup reconcile error: {e}")
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
            # matching N markets doesn't serialize N model calls. All of these
            # candidates share ONE headline, so the one-position-per-headline
            # cap must hold under the concurrent gather below: gate_trade
            # check-and-RESERVES the headline synchronously (no await between
            # check and reserve), so only one candidate can hold it at a time.
            # The reservation is committed by an actual position open
            # (_execute_with_lock) and released on every other outcome — see
            # the finally below and release_headline.
            async def process_candidate(market, match_source, match_score):
                # Reservation bookkeeping: set once gate_trade approves (it has
                # reserved the headline), cleared from this candidate's hands
                # when the signal is enqueued (ownership passes to
                # _execute_with_lock). The finally releases a reservation this
                # candidate still owns, so no blocked/crashed path leaks one.
                reserved_headline = None
                enqueued = False
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
                        if edge_metrics is not None and raw_signal is None and edge_metrics.skip_reason:
                            # Sizing declined the trade (kelly_negative: the
                            # model's own win-probability says -EV at these
                            # odds). No trade in any mode; logged under its own
                            # action so the funnel shows it distinctly from a
                            # generic no-edge skip.
                            signal, action = None, edge_metrics.skip_reason
                        else:
                            signal, action = gate_trade(
                                event, raw_signal, self._traded_headlines,
                                sports_event_counts=self._sports_event_counts,
                            )
                        # gate_trade reserved the headline iff it approved; this
                        # candidate owns that reservation until it enqueues.
                        if signal is not None:
                            reserved_headline = event.headline

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
                                None,
                                lambda: self._check_reentry(
                                    market, classification, signal.bet_amount
                                ),
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
                                # Re-entries are sized down (REENTRY_SIZE_FACTOR).
                                signal.bet_amount = decision["size_usd"]
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

                        # Multi-source confirmation: a high-materiality signal
                        # (adjusted_materiality >= MULTI_SOURCE_CONFIRM_THRESHOLD)
                        # keeps full size only if a second, independent source
                        # already made a matching directional call on this market
                        # recently; otherwise it bets smaller (not blocked).
                        if signal is not None and effective_materiality(signal) >= config.MULTI_SOURCE_CONFIRM_THRESHOLD:
                            since = (
                                datetime.now(timezone.utc)
                                - timedelta(hours=config.MULTI_SOURCE_CONFIRM_WINDOW_HOURS)
                            ).strftime("%Y-%m-%d %H:%M:%S")
                            confirmed = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: logger.has_multi_source_confirmation(
                                    market.condition_id, signal.classification, since,
                                    event.source, config.MULTI_SOURCE_CONFIRM_MIN_MATERIALITY,
                                ),
                            )
                            before = signal.bet_amount
                            result = multi_source_adjust(signal, confirmed)
                            if result == "confirmed":
                                log.info(f"Multi-source confirmed: {market.slug}")
                                console.print(
                                    f"  [green]MULTI-SOURCE CONFIRMED[/green]: "
                                    f"\"{market.question[:40]}\""
                                )
                            elif result == "reduced":
                                console.print(
                                    f"  [yellow]UNCONFIRMED HIGH-MATERIALITY[/yellow]: "
                                    f"size ${before:.0f} -> ${signal.bet_amount:.0f} "
                                    f"on \"{market.question[:40]}\""
                                )

                        # Orderbook imbalance gate + Chain-of-Verification (CoV).
                        # Both are independent network calls, so fetch them
                        # concurrently. Orderbook: don't trade into strong
                        # opposing flow on the live YES book (fails open when the
                        # CLOB is unreachable -> imbalance None -> allowed). CoV:
                        # for high-materiality signals at non-extreme prices, a
                        # cheap Haiku call verifies the news is genuinely new and
                        # not already reflected in the price (else no edge left);
                        # skipped (cov None) otherwise, and fails open on error.
                        if signal is not None:
                            loop = asyncio.get_event_loop()
                            run_cov = should_verify_novelty(
                                classification.materiality, market.yes_price
                            )

                            async def _orderbook():
                                return await loop.run_in_executor(
                                    None, fetch_orderbook_imbalance, get_token_id(market, "YES")
                                )

                            async def _cov():
                                if not run_cov:
                                    return None
                                return await loop.run_in_executor(
                                    None, verify_novelty, event.headline, market, market.yes_price
                                )

                            imbalance, cov = await asyncio.gather(_orderbook(), _cov())

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

                            # CoV: confidently-already-priced news has no edge.
                            if signal is not None and cov_blocks(cov):
                                self.stats["already_priced_in"] = (
                                    self.stats.get("already_priced_in", 0) + 1
                                )
                                log.info(
                                    f"CoV blocked: already priced in | {market.slug} | "
                                    f"price={market.yes_price:.3f} | {cov.get('reason', '')}"
                                )
                                console.print(
                                    f"  [orange3]CoV BLOCKED[/orange3]: already priced in "
                                    f"(conf {cov.get('confidence', 0.0):.2f}) "
                                    f"on \"{market.question[:40]}\""
                                )
                                signal, action = None, "already_priced_in"

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
                                    switched = event_context.build_switched_signal(signal, best)
                                    if switched.bet_amount <= 0:
                                        # Kelly prices the sibling's odds as
                                        # -EV (zero/negative Kelly sizes to 0):
                                        # don't switch into a no-trade — keep
                                        # the original signal, which cleared
                                        # its own sizing.
                                        log.info(
                                            "[pipeline] Event switch declined: "
                                            "kelly_negative on sibling %s",
                                            switched.market.slug
                                            or switched.market.condition_id,
                                        )
                                    elif (blocked := await switch_target_gate(
                                            switched, event.headline, open_positions)):
                                        # The sibling fails a market-specific
                                        # gate the original already cleared —
                                        # decline the switch, keep the original
                                        # (mirrors the kelly_negative decline).
                                        self.stats["event_switch_declines"] = (
                                            self.stats.get("event_switch_declines", 0) + 1
                                        )
                                        log.info(
                                            "[pipeline] Event switch declined: %s "
                                            "on sibling %s — keeping original signal",
                                            blocked,
                                            switched.market.slug
                                            or switched.market.condition_id,
                                        )
                                        console.print(
                                            f"  [yellow]EVENT SWITCH DECLINED[/yellow]: "
                                            f"sibling \"{switched.market.question[:40]}\" "
                                            f"fails {blocked} — keeping original"
                                        )
                                    else:
                                        self.stats["event_switches"] = (
                                            self.stats.get("event_switches", 0) + 1
                                        )
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
                        fee_cost=raw_signal.fee_cost if raw_signal else None,
                        time_horizon=classification.time_horizon,
                        adjusted_materiality=classification.adjusted_materiality,
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
                        published_at=known_published_at(event),
                    )

                    if action in ("stale", "capped", "needs_confirmation"):
                        age, age_basis = news_age(event)
                        basis_note = "" if age_basis == "published_at" else ", no pubDate"
                        console.print(
                            f"  [dim]{action.upper()}: suppressed would-be signal "
                            f"({event.source}, news age {age / 60:.0f}m{basis_note}) "
                            f"on \"{market.question[:40]}\"[/dim]"
                        )

                    if classification.direction in ("bullish", "bearish") and not classification.error:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: positions.trigger_news_reeval(
                                market.condition_id, classification.direction,
                                classification.materiality, event.headline,
                                event.source,
                            ),
                        )

                    if signal:
                        # Count this traded headline against the sports per-event
                        # cap (the switched signal's market is the one we open).
                        if (getattr(signal.market, "category", "") or "") == "sports":
                            ev_id = getattr(signal.market, "event_id", "") or ""
                            if ev_id:
                                self._sports_event_counts[ev_id] = (
                                    self._sports_event_counts.get(ev_id, 0) + 1
                                )
                        self.stats["signals_found"] += 1
                        # Ownership of the headline reservation passes to
                        # _execute_with_lock: it commits on a confirmed open and
                        # releases on any execution failure.
                        await self.signal_queue.put(signal)
                        enqueued = True
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
                finally:
                    # A reservation this candidate still owns (approved by
                    # gate_trade but blocked by a later gate, or crashed before
                    # enqueueing) is released so the headline isn't burned by a
                    # candidate that never opened.
                    if reserved_headline is not None and not enqueued:
                        release_headline(self._traded_headlines, reserved_headline)

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

        Default (TIERED_CLASSIFICATION_ENABLED): the Haiku prefilter runs under
        the Haiku semaphore, then — for survivors only — the deep Sonnet call
        runs under the tighter Sonnet semaphore.

        With tiering off, a single Sonnet classify() runs under the Sonnet
        semaphore. The Claude+Grok ensemble path only runs if ENSEMBLE_ENABLED
        is explicitly turned on (off by default — Grok is parked; see
        multi_classifier)."""
        loop = asyncio.get_event_loop()
        if config.TIERED_CLASSIFICATION_ENABLED:
            async with self._haiku_sem:
                rejected = await loop.run_in_executor(
                    None, haiku_prefilter, event.headline, market, event.source,
                    getattr(market, "category", "") or "",
                )
            if rejected is not None:
                return rejected
            async with self._sonnet_sem:
                return await loop.run_in_executor(
                    None, classify, event.headline, market, event.source, None,
                    config.SCORING_MODEL,
                )
        # Tiering off: ensemble only when explicitly enabled (parked), else a
        # single deep classify() under the expensive-tier semaphore.
        if config.ENSEMBLE_ENABLED:
            async with self._sonnet_sem:
                return await classify_ensemble(event.headline, market, event.source)
        async with self._sonnet_sem:
            return await loop.run_in_executor(
                None, classify, event.headline, market, event.source, None,
                config.SCORING_MODEL,
            )

    def _check_reentry(self, market, classification, base_size_usd):
        """Re-entry decision for a watched market via the Re-entry 2.0 gate
        (positions.check_reentry_opportunity), or None when the market isn't
        being watched. Read-only; runs off the event loop (own DB conn).

        Returns None (not watched), {"should_reenter": False, "reason": ...}
        (watched but blocked), or {"should_reenter": True, "size_usd": float,
        "reason": ...} (re-enter at the reduced size)."""
        conn = logger._conn()
        try:
            watched = logger.find_watched_market(conn, market.condition_id)
            if watched is None:
                return None
            exit_reason = watched.get("exit_reason") or ""
            closed_at = watched.get("closed_at")
            hours_since_close = _hours_since(closed_at)
            hours_to_resolution = positions.hours_to_close(
                getattr(market, "end_date", None)
            )
            result = positions.check_reentry_opportunity(
                exit_reason=exit_reason,
                materiality=classification.materiality,
                base_size_usd=base_size_usd,
                hours_since_close=hours_since_close,
                hours_to_resolution=hours_to_resolution,
                event_id=getattr(market, "event_id", "") or None,
                conn=conn,
            )
            if result is None:
                return {
                    "should_reenter": False,
                    "reason": f"re-entry gate blocked (exit '{exit_reason or 'unknown'}')",
                }
            return {"should_reenter": True, **result}
        finally:
            conn.close()

    def _record_reentry(self, condition_id: str):
        """Consume one re-entry from the market's watch window."""
        conn = logger._conn()
        try:
            logger.record_reentry(conn, condition_id)
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

        # Tiered Haiku->Sonnet, same as the main pipeline (was a single Haiku
        # classify_async). classify_fast is synchronous -> run off the loop.
        classification = await loop.run_in_executor(
            None, classify_fast, headline, market, "whale"
        )

        signal = None
        # Headline-reservation bookkeeping, mirroring process_candidate:
        # gate_trade reserves the whale headline when it approves; the
        # reservation passes to _execute_with_lock at enqueue, and any other
        # outcome (a blocked gate below, or an exception) releases it in the
        # finally so a failed investigation doesn't burn the headline for a
        # later look at the same opportunity.
        reserved = False
        enqueued = False
        # WHALE_TRADING_ENABLED is the master safety switch: investigations still
        # run and are logged below, but no trade is queued when it's off.
        try:
            if config.WHALE_TRADING_ENABLED and not classification.error:
                # Circuit breaker first: if recent realized performance has
                # deteriorated, hold whale trades too — and skip the rest of the
                # gate work entirely.
                breaker = await loop.run_in_executor(None, compute_circuit_breaker)
                self._circuit_breaker = breaker
                if breaker["triggered"]:
                    console.print(
                        f"  [red]CIRCUIT BREAKER ACTIVE[/red]: {breaker['reason']} — "
                        f"holding whale trade on \"{market.question[:40]}\""
                    )
                else:
                    edge_metrics = detect_edge_v2(market, classification, whale_event)
                    raw_signal = edge_metrics.signal if edge_metrics else None
                    signal, _ = gate_trade(whale_event, raw_signal, self._traded_headlines)
                    reserved = signal is not None

                    if signal is not None:
                        open_positions = await loop.run_in_executor(None, positions.get_open_positions)
                        # Correlation + category-exposure gates (category exposure
                        # was previously missing on the whale path).
                        corr = positions.check_correlation_risk(
                            market.question, signal.side, open_positions
                        )
                        cat_exp = positions.check_category_exposure(
                            getattr(market, "category", "") or "other", open_positions
                        )
                        if corr["risk_level"] == "high":
                            signal = None
                        elif not cat_exp["allowed"]:
                            signal = None
                        else:
                            # CoV novelty + orderbook imbalance in parallel (same
                            # pattern as the main pipeline). CoV was missing here.
                            run_cov = should_verify_novelty(
                                classification.materiality, market.yes_price
                            )

                            async def _orderbook():
                                return await loop.run_in_executor(
                                    None, fetch_orderbook_imbalance, get_token_id(market, "YES")
                                )

                            async def _cov():
                                if not run_cov:
                                    return None
                                return await loop.run_in_executor(
                                    None, verify_novelty, headline, market, market.yes_price
                                )

                            imbalance, cov = await asyncio.gather(_orderbook(), _cov())
                            if not orderbook_allows(signal.side, imbalance) or cov_blocks(cov):
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
                fee_cost=signal.fee_cost if signal else None,
                time_horizon=classification.time_horizon,
                adjusted_materiality=classification.adjusted_materiality,
                action="whale_triggered",
                match_source="whale",
                condition_id=market.condition_id,
                yes_price=market.yes_price,
                yes_token_id=get_token_id(market, "YES"),
                edge_type="whale",
                confidence=classification.confidence if not classification.error else None,
                event_id=getattr(market, "event_id", "") or None,
                published_at=known_published_at(whale_event),
            )

            if signal is not None:
                self.stats["signals_found"] += 1
                # Ownership of the headline reservation passes to
                # _execute_with_lock (commit on open, release on failure).
                await self.signal_queue.put(signal)
                enqueued = True
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
        finally:
            if reserved and not enqueued:
                release_headline(self._traded_headlines, headline)

    async def _acquire_position_lock(self, event_key: str) -> asyncio.Lock:
        """The asyncio.Lock guarding position-opening for one event (its
        event_id, or category as a fallback), created on first use. Serializes
        the risk-recheck -> open critical section per event; distinct events get
        distinct locks and open concurrently. Lazy creation is safe under
        asyncio's single-threaded loop — there is no await between the lookup
        and the insert, so two coroutines can't create competing locks."""
        lock = self._position_locks.get(event_key)
        if lock is None:
            lock = asyncio.Lock()
            self._position_locks[event_key] = lock
        return lock

    async def _recheck_risk_gates(self, signal: Signal, market) -> bool:
        """Re-check the position-opening risk gates against FRESH DB state, held
        inside the event lock immediately before execution.

        The per-candidate gates in _process_news run concurrently across a batch
        and read the book *before* any of that batch's signals have opened, so
        they can approve two same-event signals at once. This second,
        authoritative pass sees the freshly-opened book, so the loser of the race
        backs off here. Returns False (logging which gate) when correlation
        (HIGH risk), category exposure, the per-event exposure cap, the circuit
        breaker, or the daily spend limit would block; True when all pass."""
        loop = asyncio.get_event_loop()
        ident = market.slug or market.condition_id
        open_positions = await loop.run_in_executor(None, positions.get_open_positions)

        corr = positions.check_correlation_risk(market.question, signal.side, open_positions)
        if corr["risk_level"] == "high":
            log.info(f"[pipeline] Risk re-check failed: correlation_block on {ident}")
            return False

        cat_exp = positions.check_category_exposure(
            getattr(market, "category", "") or "other", open_positions
        )
        if not cat_exp["allowed"]:
            log.info(f"[pipeline] Risk re-check failed: category_limit on {ident}")
            return False

        # The race target: re-read the per-event position cap so the second
        # same-event signal now sees the first's just-opened position.
        event_id = getattr(market, "event_id", "") or ""
        exposure = event_context.get_event_exposure(event_id, open_positions)
        if event_id and exposure["position_count"] >= config.MAX_POSITIONS_PER_EVENT:
            log.info(f"[pipeline] Risk re-check failed: event_exposure_block on {ident}")
            return False

        breaker = await loop.run_in_executor(None, compute_circuit_breaker)
        self._circuit_breaker = breaker
        if breaker["triggered"]:
            log.info(f"[pipeline] Risk re-check failed: circuit_breaker on {ident}")
            return False

        daily_spent = abs(await loop.run_in_executor(None, logger.get_daily_pnl))
        if daily_spent + signal.bet_amount > config.DAILY_SPEND_LIMIT_USD:
            log.info(f"[pipeline] Risk re-check failed: daily_spend_limit on {ident}")
            return False

        return True

    async def _execute_with_lock(self, signal: Signal) -> dict | None:
        """Open one signal's position under its event-level lock, re-checking the
        risk gates against fresh state first (see _recheck_risk_gates). Returns
        the execute result, or None when the re-check blocked it (a lost race) or
        the open errored.

        Owns the signal's headline reservation (taken by gate_trade, handed over
        at enqueue): a confirmed position open commits it permanently; every
        other outcome — lost race, rejected/skipped/resting/errored execution,
        or an exception — releases it in the finally, so a failed candidate
        never burns the headline for later ones. A 'passive_pending' result (a
        resting passive limit entry, core/passive.py) is the one non-open
        outcome that KEEPS the reservation: the order may still fill, so the
        headline must stay unavailable to other candidates until the passive
        lifecycle resolves it (commit on fill, release on expiry/cancel)."""
        market = signal.market
        event_key = (getattr(market, "event_id", "") or "") or (
            getattr(market, "category", "") or "other"
        )
        opened = False
        passive_pending = False
        try:
            async with await self._acquire_position_lock(event_key):
                if not await self._recheck_risk_gates(signal, market):
                    self.stats["race_blocks"] = self.stats.get("race_blocks", 0) + 1
                    log.info("[pipeline] Race condition blocked: risk re-check failed")
                    console.print(
                        f"  [red]RACE BLOCKED[/red]: risk re-check failed on "
                        f"\"{market.question[:40]}\""
                    )
                    return None

                result = await execute_trade_async(signal)
                self.stats["trades_executed"] += 1

                # Sanity check: a live "executed" with no real fill cost is a
                # phantom fill (the order rested but was mis-reported). Downgrade
                # it to "resting" so we never open a position with no money behind
                # it. dry_run legitimately has no fill cost, so it's exempt.
                if result["status"] == "executed" and not result.get("actual_cost_usd"):
                    log.warning(
                        "[pipeline] Sanity check failed: executed but no fill cost "
                        "— treating as resting (order %s, \"%s\")",
                        result.get("order_id"), market.question[:40],
                    )
                    result["status"] = "resting"

                if result["status"] in ("dry_run", "executed"):
                    # Only a confirmed fill (or a dry-run simulation) opens a
                    # local position.
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: positions.open_position(
                            result["trade_id"], signal.market, signal.side,
                            signal.bet_amount, headline=signal.headlines,
                            reasoning=signal.reasoning,
                            actual_cost_usd=result.get("actual_cost_usd"),
                            token_count=result.get("actual_shares"),
                        ),
                    )
                    # Position opened: the headline reservation is committed.
                    opened = True
                elif result["status"] == "passive_pending":
                    # Passive limit entry resting on the book BY DESIGN
                    # (core/passive.py): no position yet, but the headline
                    # reservation stays held while the pending order lives —
                    # the lifecycle commits it on fill or releases it on
                    # expiry/cancel via the management cycle.
                    passive_pending = True
                    log.info(
                        "[pipeline] passive entry resting: order %s @ %.4f on "
                        "\"%s\" — reservation held, awaiting fill",
                        result.get("order_id"), result.get("limit_price") or 0.0,
                        market.question[:40],
                    )
                    console.print(
                        f"  [cyan]PASSIVE[/cyan] {result['side']} "
                        f"${result['amount']:.2f} resting @ "
                        f"{result.get('limit_price') or 0.0:.3f} "
                        f"on \"{result['market'][:40]}\""
                    )
                elif result["status"] == "resting":
                    # Order is live on the book but unfilled — do NOT open a
                    # position yet; a later fill is reconciled on the exchange.
                    log.warning(
                        "[pipeline] order %s resting (unfilled) on \"%s\" — "
                        "position NOT opened",
                        result.get("order_id"), market.question[:40],
                    )
                elif result["status"] == "dust_fill":
                    # Fill below MIN_FILL_USD: the executor already sold the
                    # dust back (best-effort) — no position to manage, and the
                    # finally below releases the headline reservation.
                    log.warning(
                        "[pipeline] dust fill on order %s (\"%s\") — position "
                        "NOT opened",
                        result.get("order_id"), market.question[:40],
                    )

                if result["status"] in ("dry_run", "executed"):
                    status_color = "bright_green"
                elif result["status"] == "passive_pending":
                    status_color = "cyan"
                else:
                    status_color = "red"
                console.print(
                    f"  [{status_color}]{result['status']}[/{status_color}] "
                    f"{result['side']} ${result['amount']:.2f} "
                    f"on \"{result['market'][:40]}\" "
                    f"(edge:{result['edge']:.1%} latency:{result.get('latency_ms', 0)}ms)"
                )
                return result
        except Exception:
            log.exception("[pipeline] _execute_with_lock failed")
            return None
        finally:
            # No position -> release the headline reservation so another
            # candidate for the same headline may still open (discard-based:
            # a no-op for signals whose headline was never reserved). A resting
            # passive entry is the exception: its reservation stays held until
            # the pending-order lifecycle resolves it (fill commits,
            # expiry/cancel releases — see passive.check_pending_orders).
            if not opened and not passive_pending:
                release_headline(self._traded_headlines, signal.headlines)

    async def _execute_signals(self):
        """Execute trades from the signal queue. Each signal opens under its
        event-level lock as its own task, so opens on the same event serialize
        (and the loser re-checks out) while different events open in parallel."""
        pending: set[asyncio.Task] = set()
        while True:
            signal: Signal = await self.signal_queue.get()
            task = asyncio.create_task(self._execute_with_lock(signal))
            pending.add(task)
            task.add_done_callback(pending.discard)

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

            # Passive pending-order lifecycle (core/passive.py): fills open
            # their positions, timeouts/chase-aways cancel and hand back their
            # headlines. A single cheap SELECT and out when nothing is pending
            # (always, with the flag off), so flag-off behavior is unchanged.
            try:
                psummary = await asyncio.get_event_loop().run_in_executor(
                    None, passive.check_pending_orders
                )
                # Release on the loop thread — this set belongs to the pipeline.
                for headline in psummary.get("released_headlines", []):
                    release_headline(self._traded_headlines, headline)
                if psummary.get("checked"):
                    console.print(
                        f"  [dim]passive orders: {psummary['checked']} checked, "
                        f"{psummary['filled']} filled, "
                        f"{psummary['expired']} expired, "
                        f"{psummary['chased_away']} chased away, "
                        f"{psummary['dust']} dust[/dim]"
                    )
            except Exception as e:
                log.warning(f"[pipeline] Passive order lifecycle error: {e}")

            # Daily journal: first cycle after 21:00 UTC (no-op otherwise).
            try:
                entry = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: maybe_write_journal(self.stats)
                )
                if entry:
                    console.print(f"  [dim]journal entry written ({len(entry.split())} words)[/dim]")
            except Exception as e:
                log.warning(f"[pipeline] Journal error: {e}")

            # Daily missed-opportunity sweep: first cycle after 22:00 UTC.
            try:
                missed = await asyncio.get_event_loop().run_in_executor(
                    None, maybe_check_missed_opportunities
                )
                if missed:
                    console.print(f"  [dim]{missed} missed-opportunity lesson(s) logged[/dim]")
            except Exception as e:
                log.warning(f"[pipeline] Missed-opportunity check error: {e}")

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
