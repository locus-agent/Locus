"""
Claude classification engine — replaces probability estimation with direction classification.
Asks "does this news confirm or deny the market question?" instead of "what's the probability?"
"""
from __future__ import annotations

import json
import re
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import anthropic

from locus import config
from locus.memory import logger
from locus import memory
from locus.markets.gamma import Market

log = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

RETRY_BACKOFF_SECONDS = 2.0

CLASSIFICATION_PROMPT = """You are a news classifier for prediction markets.

## Market Question
{question}

## Market Context
{threshold_line}Current YES price: {yes_price:.3f} (implied probability: {yes_price:.1%})
Time remaining until market close: {time_remaining}

## Breaking News
{headline}
Source: {source}

## Your Track Record
{track_record}

## Task
Does this news make the market question MORE likely to resolve YES, MORE likely to resolve NO, or is it NOT RELEVANT?

Also rate the MATERIALITY — how much should this move the price? 0.0 means no impact, 1.0 means this is definitive evidence.

Judge materiality for THIS SPECIFIC market: does the news materially change the probability that
this specific threshold is crossed by this specific deadline? Direction alone is not materiality.
- If the market is already nearly decided (implied probability above ~95% or below ~5%) and the
  news is not dramatic enough to flip that outcome before the close, materiality is LOW (0.0-0.2)
  even if the news points in the "right" direction.
- A generic directional headline (analyst commentary, "looms"/"could"/"might" speculation,
  predictions without a concrete new event) should get LOW materiality for thresholds that are
  far from where the price currently implies the asset is trading.
- The less time remaining, the larger and more concrete the news must be to move the outcome.
Reserve materiality above 0.4 for concrete new developments plausibly large enough to push this
particular threshold across the line within the remaining time.

Use your track record above to calibrate: if a category or source has been unreliable, or a
past lesson applies to this situation, factor that into your materiality and reasoning.

Respond with ONLY valid JSON:
{{
  "direction": "bullish" | "bearish" | "neutral",
  "materiality": <float 0.0 to 1.0>,
  "reasoning": "<1 sentence>"
}}"""


# Dollar amount in the question, e.g. "$1,700" or "$0.50" — the market's price threshold.
_THRESHOLD_RE = re.compile(r"\$\s?[\d,]+(?:\.\d+)?")


def _extract_threshold(question: str) -> str | None:
    m = _THRESHOLD_RE.search(question)
    return m.group(0).replace(" ", "") if m else None


def _format_time_remaining(end_date: str, as_of: datetime) -> str:
    try:
        end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    except (ValueError, AttributeError, TypeError):
        return "unknown"
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    remaining = (end - as_of).total_seconds()
    if remaining <= 0:
        return "market is closing now"
    days, rem = divmod(int(remaining), 86400)
    hours = rem // 3600
    if days > 0:
        return f"{days} day{'s' if days != 1 else ''}, {hours} hour{'s' if hours != 1 else ''}"
    if hours > 0:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    return "less than 1 hour"


def _format_track_record() -> str:
    """Build the 'Your track record' prompt section from calibration history + past lessons."""
    record = memory.get_track_record()

    if record["total"] == 0:
        return "No resolved classifications yet."

    lines = [f"Overall accuracy: {record['accuracy']}% across {record['total']} resolved classifications."]

    if record["by_category"]:
        cat_str = ", ".join(f"{cat} {pct}%" for cat, pct in record["by_category"].items())
        lines.append(f"By category: {cat_str}")

    if record["by_source"]:
        src_str = ", ".join(f"{src} {pct}%" for src, pct in record["by_source"].items())
        lines.append(f"By source: {src_str}")

    lessons = logger.get_recent_lessons(limit=5)
    if lessons:
        lines.append("Recent lessons from past mistakes:")
        for l in lessons:
            lines.append(f"- {l['lesson']}")

    return "\n".join(lines)


@dataclass
class Classification:
    direction: str  # "bullish", "bearish", "neutral"
    materiality: float  # 0.0-1.0
    reasoning: str
    latency_ms: int
    model: str
    error: bool = False  # True when classification failed (API/parse error)


def classify(
    headline: str, market: Market, source: str = "unknown", as_of: datetime | None = None
) -> Classification:
    """Classify a news headline against a market question. Synchronous.

    as_of: the moment the news broke — defaults to now. Backtests pass the
    historical headline time so time-remaining is computed as of that moment.
    """
    start = time.time()

    if as_of is None:
        as_of = datetime.now(timezone.utc)
    elif as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    threshold = _extract_threshold(market.question)
    threshold_line = f"Price threshold in question: {threshold}\n" if threshold else ""

    prompt = CLASSIFICATION_PROMPT.format(
        question=market.question,
        threshold_line=threshold_line,
        time_remaining=_format_time_remaining(market.end_date, as_of),
        yes_price=market.yes_price,
        headline=headline,
        source=source,
        track_record=_format_track_record(),
    )

    last_err: Exception | None = None
    for attempt in range(2):  # one retry with backoff, no retry storms
        try:
            response = client.messages.create(
                model=config.CLASSIFICATION_MODEL,
                max_tokens=200,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            # Extract JSON
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            result = json.loads(text)
            latency = int((time.time() - start) * 1000)

            direction = result.get("direction", "neutral")
            if direction not in ("bullish", "bearish", "neutral"):
                direction = "neutral"

            materiality = max(0.0, min(1.0, float(result.get("materiality", 0))))

            return Classification(
                direction=direction,
                materiality=materiality,
                reasoning=result.get("reasoning", ""),
                latency_ms=latency,
                model=config.CLASSIFICATION_MODEL,
            )

        except Exception as e:
            last_err = e
            log.warning(f"[classifier] Error (attempt {attempt + 1}/2): {e}")
            if attempt == 0:
                time.sleep(RETRY_BACKOFF_SECONDS)

    # Both attempts failed: flag it instead of disguising the outage as a
    # confident "neutral" — the pipeline logs these as action="error".
    latency = int((time.time() - start) * 1000)
    return Classification(
        direction="neutral",
        materiality=0.0,
        reasoning=f"Classification error: {type(last_err).__name__}",
        latency_ms=latency,
        model=config.CLASSIFICATION_MODEL,
        error=True,
    )


# Edge types a signal can be attributed to. 'arbitrage' is defined for a
# future cross-platform arbitrage strategy and is not emitted yet.
EDGE_TYPES = ("news", "momentum", "arbitrage")

# A YES-price move larger than this (%) over the lookback window marks the
# signal as riding momentum rather than fresh news.
MOMENTUM_MOVE_PCT = 10.0


def classify_edge_type(market: Market, lookback_hours: float = 24.0) -> str:
    """Attribute a signal to an edge type from market context.

    - 'momentum' when this market's YES price has moved more than
      MOMENTUM_MOVE_PCT (relative) over the last `lookback_hours`, inferred
      from the earliest stored classification price for the market vs its
      current price.
    - 'news' otherwise — the default for our news-driven pipeline.
    - 'arbitrage' is reserved for future cross-platform arbitrage; never
      returned here yet.
    """
    prior_price = logger.get_earliest_classification_price(
        market.condition_id, lookback_hours
    )
    if prior_price and prior_price > 0:
        move_pct = abs(market.yes_price - prior_price) / prior_price * 100.0
        if move_pct > MOMENTUM_MOVE_PCT:
            return "momentum"
    return "news"


async def classify_async(
    headline: str, market: Market, source: str = "unknown", as_of: datetime | None = None
) -> Classification:
    """Async wrapper around classify()."""
    import asyncio
    return await asyncio.get_event_loop().run_in_executor(
        None, classify, headline, market, source, as_of
    )


if __name__ == "__main__":
    test_market = Market(
        condition_id="test",
        question="Will OpenAI release GPT-5 before August 2026?",
        category="ai",
        yes_price=0.62,
        no_price=0.38,
        volume=500000,
        end_date="2026-08-01",
        active=True,
        tokens=[],
    )

    result = classify(
        headline="OpenAI reportedly testing GPT-5 internally with select partners",
        market=test_market,
        source="The Information",
    )
    print(f"Direction: {result.direction}")
    print(f"Materiality: {result.materiality}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Latency: {result.latency_ms}ms")
