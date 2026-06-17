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

CRITICAL RULE — Direct Evidence Only:
Only classify as bullish or bearish if the news provides DIRECT, immediate evidence that the market will resolve YES or NO.

- One logical step maximum.
- No implied consequences, second-order effects, or chains of inference.

Examples of INVALID (must be neutral):
- News: 'Iran deal signed' → Market: 'Will Trump personally attend the next Iran meeting?' (indirect)
- News: 'Company bought 5000 ETH' → Market: 'Will ETH reach $2500 by July?' (implied price target)
- News: 'Diplomat said positive things' → Market: 'Will country X do action Y before date Z?' (inference chain)

Examples of VALID direct evidence:
- News: 'Trump just announced he will attend the Iran meeting on March 15' → Market: 'Will Trump attend next Iran meeting?' → bullish
- News: 'SEC officially approved ETH ETF' → Market: 'Will ETH ETF be approved before June?' → bullish

If you need more than one logical step to connect the headline to the market resolution → direction MUST be neutral.
Ask yourself: Does this headline alone almost prove the market outcome? If not — neutral.

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

Also estimate your CONFIDENCE: the probability, between 0.5 and 1.0, that the market actually
resolves in the direction you predicted. This is DISTINCT from materiality — materiality is how
much the news should move the price, confidence is how sure you are about the resulting outcome.
0.5 means a coin flip (no real conviction); 1.0 means near-certain. Position size is scaled by
this number, so be honest: reserve confidence above ~0.7 for situations where the evidence
genuinely makes the predicted outcome much more likely than not. For "neutral", return 0.5.

Respond with ONLY valid JSON:
{{
  "direction": "bullish" | "bearish" | "neutral",
  "materiality": <float 0.0 to 1.0>,
  "confidence": <float 0.5 to 1.0>,
  "reasoning": "<1 sentence>"
}}"""


# Simplified, JSON-only prompt for the Grok second opinion. Deliberately leaner
# than CLASSIFICATION_PROMPT (no track record / calibration guidance) so the two
# models reason independently rather than anchoring on the same context.
GROK_CLASSIFICATION_PROMPT = """You are a news classifier for prediction markets.

Market question: {question}
Current YES price: {yes_price:.3f} (implied probability {yes_price:.1%})
Time remaining until close: {time_remaining}

Breaking news: {headline}
Source: {source}

Does this news make the market MORE likely to resolve YES (bullish), MORE likely \
to resolve NO (bearish), or is it NOT RELEVANT (neutral)?
Rate materiality (0.0 = no impact, 1.0 = definitive evidence) and your confidence \
(0.5 = coin flip, 1.0 = near-certain) that the market resolves in your predicted direction.

Respond with ONLY valid JSON, no other text:
{{"direction": "bullish|bearish|neutral", "materiality": <0.0-1.0>, "confidence": <0.5-1.0>, "reasoning": "<1 sentence>"}}"""


# Strict JSON-only prompt for the cheap Haiku prefilter tier (see classify_fast).
# Deliberately tiny: it only triages relevance + rough materiality, not the full
# calibrated analysis the Sonnet tier does.
FAST_CLASSIFICATION_SYSTEM_PROMPT = """You are a fast news filter for prediction markets. Return ONLY valid JSON, no explanations.
Example: {"relevant": true, "direction": "bullish", "materiality": 0.65, "reason": "direct positive evidence"}
Fields: relevant (bool), direction (bullish/bearish/neutral), materiality (0.0-1.0), reason (1 sentence)"""


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


def get_active_prompt() -> str:
    """The classification prompt currently in force.

    Returns the latest evolved prompt from the prompt_versions table when one
    exists (meta-prompt evolution, see memory/meta_evolver.py), otherwise the
    hardcoded CLASSIFICATION_PROMPT. Any DB/load failure falls back to the
    hardcoded prompt — classification must never break because evolution did."""
    try:
        latest = logger.get_latest_prompt_version()
        if latest and latest.get("prompt_text"):
            return latest["prompt_text"]
    except Exception as e:
        log.warning(f"[classifier] Active-prompt load failed, using hardcoded: {e}")
    return CLASSIFICATION_PROMPT


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
    materiality: float  # 0.0-1.0: how much the news should move the price
    reasoning: str
    latency_ms: int
    model: str
    confidence: float = 0.5  # 0.5-1.0: P(market resolves in predicted direction)
    error: bool = False  # True when classification failed (API/parse error)
    raw_response: str = ""  # The model's raw text response (for debugging/audit)
    # Set to "prefiltered_haiku" by classify_fast when the cheap Haiku tier
    # rejected the headline before any Sonnet call; the pipeline logs this action
    # and skips the edge/gate chain. None for a normal full classification.
    action: str | None = None
    # Multi-LLM ensemble fields (set by multi_classifier.classify_ensemble);
    # left at defaults for plain single-model classify() results.
    consensus_score: float | None = None  # 0.0-1.0 agreement between models
    ensemble_used: bool = False  # True when two models were blended


# Direction synonyms normalized to our three canonical labels. Different models
# (and prompt drift) phrase direction differently; map them all to one vocab.
_BULLISH_SYNONYMS = {"bullish", "up", "positive", "buy", "long", "yes"}
_BEARISH_SYNONYMS = {"bearish", "down", "negative", "sell", "short", "no"}


def _normalize_direction(value: object) -> str:
    """Map a model's direction label (incl. synonyms) to bullish/bearish/neutral."""
    v = str(value or "").strip().lower()
    if v in _BULLISH_SYNONYMS:
        return "bullish"
    if v in _BEARISH_SYNONYMS:
        return "bearish"
    return "neutral"


def _clamp(value: object, lo: float, hi: float) -> float:
    """Coerce to float and clamp to [lo, hi]; returns lo on bad input."""
    try:
        return max(lo, min(hi, float(value)))
    except (ValueError, TypeError):
        return lo


# JSON object anywhere in the text (greedy, spans newlines).
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
# Loose key/value extraction straight from prose, for when JSON parsing fails.
_DIRECTION_RE = re.compile(r'direction["\s:=]+([a-zA-Z]+)', re.IGNORECASE)
_MATERIALITY_RE = re.compile(r'materiality["\s:=]+(-?[0-9]*\.?[0-9]+)', re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r'confidence["\s:=]+(-?[0-9]*\.?[0-9]+)', re.IGNORECASE)
_REASONING_RE = re.compile(r'reasoning["\s:=]+"([^"]*)"', re.IGNORECASE)


def parse_classification(text: str) -> dict:
    """Universal, defensive parser for any classifier model's text response.

    Three escalating strategies so a malformed or chatty response still yields
    usable fields instead of raising:
      1. Extract the first {...} block and json.loads it.
      2. Regex direction / materiality / confidence / reasoning from free text.
      3. Fall back to neutral / 0.0 / 0.5.

    Returns a dict with normalized, clamped keys: direction (bullish/bearish/
    neutral), materiality (0-1), confidence (0.5-1), reasoning. Direction
    synonyms (up/positive/buy, down/negative/sell) are normalized.
    """
    text = (text or "").strip()

    # Strip a Markdown code fence if present (```json ... ```).
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            candidate = parts[1]
            if candidate.lstrip().lower().startswith("json"):
                candidate = candidate.lstrip()[4:]
            text = candidate.strip()

    # Try 1: a JSON object somewhere in the text.
    m = _JSON_OBJ_RE.search(text)
    if m:
        try:
            result = json.loads(m.group(0))
            if isinstance(result, dict):
                return {
                    "direction": _normalize_direction(result.get("direction", "neutral")),
                    "materiality": _clamp(result.get("materiality", 0.0), 0.0, 1.0),
                    "confidence": _clamp(result.get("confidence", 0.5), 0.5, 1.0),
                    "reasoning": str(result.get("reasoning", "")),
                }
        except (ValueError, TypeError):
            pass

    # Try 2: regex the fields straight out of the prose.
    d = _DIRECTION_RE.search(text)
    mat = _MATERIALITY_RE.search(text)
    conf = _CONFIDENCE_RE.search(text)
    reason = _REASONING_RE.search(text)
    if d or mat or conf:
        return {
            "direction": _normalize_direction(d.group(1)) if d else "neutral",
            "materiality": _clamp(mat.group(1), 0.0, 1.0) if mat else 0.0,
            "confidence": _clamp(conf.group(1), 0.5, 1.0) if conf else 0.5,
            "reasoning": reason.group(1) if reason else "",
        }

    # Try 3: nothing parseable — neutral fallback.
    return {"direction": "neutral", "materiality": 0.0, "confidence": 0.5, "reasoning": ""}


_RELEVANT_RE = re.compile(r'relevant["\s:=]+(true|false)', re.IGNORECASE)


def _parse_relevant(text: str, default: bool = True) -> bool:
    """Extract the Haiku prefilter's `relevant` boolean from its JSON response.

    Defaults True (fail open) when the field is absent or unparseable, so a
    missing flag leaves the relevance decision to the materiality floor rather
    than silently dropping the headline."""
    text = text or ""
    m = _JSON_OBJ_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "relevant" in obj:
                return bool(obj["relevant"])
        except (ValueError, TypeError):
            pass
    rm = _RELEVANT_RE.search(text)
    if rm:
        return rm.group(1).lower() == "true"
    return default


def classify(
    headline: str, market: Market, source: str = "unknown", as_of: datetime | None = None,
    model: str | None = None,
) -> Classification:
    """Classify a news headline against a market question. Synchronous.

    as_of: the moment the news broke — defaults to now. Backtests pass the
    historical headline time so time-remaining is computed as of that moment.
    model: the Claude model to use — defaults to config.CLASSIFICATION_MODEL.
    classify_fast passes config.SCORING_MODEL (Sonnet) for the deep tier.
    """
    start = time.time()
    model = model or config.CLASSIFICATION_MODEL

    if as_of is None:
        as_of = datetime.now(timezone.utc)
    elif as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    threshold = _extract_threshold(market.question)
    threshold_line = f"Price threshold in question: {threshold}\n" if threshold else ""

    fmt_kwargs = dict(
        question=market.question,
        threshold_line=threshold_line,
        time_remaining=_format_time_remaining(market.end_date, as_of),
        yes_price=market.yes_price,
        headline=headline,
        source=source,
        track_record=_format_track_record(),
    )
    # Use the active (possibly evolved) prompt; if a stored prompt can't be
    # formatted (a bad placeholder slipped through), fall back to hardcoded so
    # classification keeps working.
    try:
        prompt = get_active_prompt().format(**fmt_kwargs)
    except (KeyError, IndexError, ValueError) as e:
        log.warning(f"[classifier] Evolved prompt failed to format ({e}); using hardcoded")
        prompt = CLASSIFICATION_PROMPT.format(**fmt_kwargs)

    last_err: Exception | None = None
    for attempt in range(2):  # one retry with backoff, no retry storms
        try:
            response = client.messages.create(
                model=model,
                max_tokens=200,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            # Universal parser: JSON -> regex -> neutral fallback. Never raises,
            # so a chatty/malformed response degrades to neutral rather than
            # being counted as an outage and retried.
            parsed = parse_classification(text)
            latency = int((time.time() - start) * 1000)

            return Classification(
                direction=parsed["direction"],
                materiality=parsed["materiality"],
                confidence=parsed["confidence"],
                reasoning=parsed["reasoning"],
                latency_ms=latency,
                model=model,
                raw_response=text,
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
        model=model,
        error=True,
    )


def haiku_prefilter(
    headline: str, market: Market, source: str = "unknown"
) -> Classification | None:
    """The cheap Haiku triage half of tiered classification.

    Returns a Classification with action='prefiltered_haiku' when Haiku rejects
    the headline (irrelevant, or materiality below HAIKU_MATERIALITY_THRESHOLD),
    or None when it passes and deserves the deep Sonnet analysis.

    Fails OPEN: any Haiku error (API or parse) returns None so the caller still
    runs the full Sonnet classify() rather than silently dropping tradable news.
    Isolated from the Sonnet call so the pipeline can gate the two tiers with
    separate concurrency semaphores.
    """
    start = time.time()
    user_prompt = (
        f"Market question: {market.question}\n"
        f"Current YES price: {market.yes_price:.3f}\n"
        f"Breaking news: {headline}\n"
        f"Source: {source}"
    )

    try:
        response = client.messages.create(
            model=config.HAIKU_MODEL,
            max_tokens=150,
            temperature=0.0,
            system=FAST_CLASSIFICATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = response.content[0].text.strip()
    except Exception as e:
        # Fail open: let the full Sonnet tier handle it rather than drop the news.
        log.warning(
            f"[classifier] Haiku prefilter failed ({type(e).__name__}: {e}); "
            f"falling through to full classify"
        )
        return None

    # Reuse the universal parser for direction/materiality/reasoning, and pull
    # the prefilter-only `relevant` flag separately.
    parsed = parse_classification(text)
    relevant = _parse_relevant(text)
    direction = parsed["direction"]
    materiality = parsed["materiality"]

    # Haiku tends to overestimate materiality; pull an inflated score back so it
    # can't wave a marginal headline through to the expensive tier on hype.
    if materiality > 0.85:
        materiality = 0.75

    if not relevant or materiality < config.HAIKU_MATERIALITY_THRESHOLD:
        log.info(
            f"haiku_prefilter: {market.slug} | relevant={relevant} | "
            f"mat={materiality:.2f} | action=prefiltered_haiku"
        )
        return Classification(
            direction=direction,
            materiality=materiality,
            confidence=parsed["confidence"],
            reasoning=parsed["reasoning"],
            latency_ms=int((time.time() - start) * 1000),
            model=config.HAIKU_MODEL,
            raw_response=text,
            action="prefiltered_haiku",
        )

    return None  # survived the prefilter — caller spends the deep Sonnet call


FAST_COV_SYSTEM_PROMPT = """You are a strict market novelty verifier.
Answer ONLY with valid JSON. No explanations.
Example: {"already_priced": false, "confidence": 0.82, "reason": "deal not yet confirmed by official sources"}
"""


def verify_novelty(headline: str, market: Market, current_price: float) -> dict | None:
    """Chain-of-Verification novelty check: a cheap Haiku call that asks whether
    the news is genuinely NEW information not yet reflected in the price.

    Returns {"already_priced": bool, "confidence": float, "reason": str}, or
    None on any error/parse failure. Fails OPEN (None) so a flaky verifier never
    silently drops a tradable signal — the caller treats None as "don't block"."""
    user_prompt = (
        f"Market: {market.question}\n"
        f"Current Yes price: {current_price:.3f}\n"
        f"News: {headline}\n"
        f"Is this news genuinely NEW information not yet reflected in the current price?"
    )

    try:
        response = client.messages.create(
            model=config.COV_MODEL,
            max_tokens=200,
            temperature=0.0,
            system=FAST_COV_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = response.content[0].text.strip()
        m = _JSON_OBJ_RE.search(text)
        if not m:
            return None
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            return None
        return {
            "already_priced": bool(obj.get("already_priced", False)),
            "confidence": _clamp(obj.get("confidence", 0.0), 0.0, 1.0),
            "reason": str(obj.get("reason", "")),
        }
    except Exception as e:
        # Fail open: a verifier outage must not block trading.
        log.warning(f"[classifier] CoV verify_novelty failed ({type(e).__name__}: {e})")
        return None


def classify_fast(
    headline: str, market: Market, source: str = "unknown", as_of: datetime | None = None
) -> Classification:
    """Tiered classification: a cheap Haiku prefilter in front of the full
    Sonnet classify(). Haiku triages relevance + rough materiality; only the
    headlines that survive pay for the deep Sonnet analysis.

    # Expected cost reduction: ~55% (Haiku ~4x cheaper + filters ~60-70% of low-value headlines)

    Fails OPEN: any Haiku error (API or parse) falls through to the full
    classify() so a flaky prefilter never silently drops tradable news.
    """
    rejected = haiku_prefilter(headline, market, source)
    if rejected is not None:
        return rejected
    return classify(headline, market, source, as_of, model=config.SCORING_MODEL)


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
