"""
Multi-LLM consensus classification.

Claude (weight 0.6) and Grok (weight 0.4) classify the same headline in
parallel; the two answers are blended into one Classification carrying a
consensus_score that measures how much the models agree on direction,
materiality, and confidence. The pipeline gates trades when consensus is low.

Degradation is graceful:
  - Grok unavailable (no key, not installed, API/parse error) -> Claude-only,
    consensus_score 0.85 (a lone model isn't fully corroborated, so < 1.0).
  - Both fail -> neutral / 0.0, flagged as an error.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

from locus import config
from locus.core.classifier import (
    Classification,
    classify_async,
    parse_classification,
    GROK_CLASSIFICATION_PROMPT,
    _format_time_remaining,
)
from locus.markets.gamma import Market

log = logging.getLogger(__name__)

CLAUDE_WEIGHT = 0.6
GROK_WEIGHT = 0.4
# Single-model fallback consensus: corroboration by one model only, so capped
# below the 1.0 reserved for "two models fully agree".
SINGLE_MODEL_CONSENSUS = 0.85

GROK_BASE_URL = "https://api.x.ai/v1"


def consensus_score(claude: Classification, grok: Classification) -> float:
    """How much the two classifications agree, on [0, 1].

    direction_match: 1.0 same direction, 0.0 opposite, 0.5 if exactly one is
                     neutral. mat_similarity: 1 - |Δmateriality|. conf_avg: mean
                     confidence. Weighted 0.5 / 0.3 / 0.2.
    """
    if claude.direction == grok.direction:
        direction_match = 1.0
    elif "neutral" in (claude.direction, grok.direction):
        direction_match = 0.5
    else:
        direction_match = 0.0

    mat_similarity = 1.0 - abs(claude.materiality - grok.materiality)
    conf_avg = (claude.confidence + grok.confidence) / 2.0
    return 0.5 * direction_match + 0.3 * mat_similarity + 0.2 * conf_avg


def blend(claude: Classification, grok: Classification) -> Classification:
    """Blend two successful classifications into one ensemble Classification.

    Materiality and confidence are weighted averages (0.6 Claude / 0.4 Grok);
    direction follows Claude (the higher-weighted model). consensus_score
    captures agreement; the low-consensus gate handles disagreement downstream.
    """
    materiality = CLAUDE_WEIGHT * claude.materiality + GROK_WEIGHT * grok.materiality
    confidence = CLAUDE_WEIGHT * claude.confidence + GROK_WEIGHT * grok.confidence
    return Classification(
        direction=claude.direction,
        materiality=round(materiality, 4),
        confidence=round(confidence, 4),
        reasoning=f"[ensemble] Claude: {claude.reasoning} || Grok: {grok.reasoning}",
        latency_ms=max(claude.latency_ms, grok.latency_ms),
        model=f"{config.CLASSIFICATION_MODEL}+{config.GROK_MODEL}",
        error=False,
        raw_response=grok.raw_response,
        consensus_score=round(consensus_score(claude, grok), 4),
        ensemble_used=True,
    )


def is_low_consensus(classification: Classification) -> bool:
    """True when an ensemble classification's two models disagreed enough to
    block a trade — consensus_score below config.ENSEMBLE_MIN_CONSENSUS.
    Single-model / non-ensemble results (consensus_score None) never block."""
    cscore = classification.consensus_score
    return cscore is not None and cscore < config.ENSEMBLE_MIN_CONSENSUS


async def _classify_grok(
    headline: str, market: Market, source: str = "unknown", as_of: datetime | None = None
) -> Classification | None:
    """Classify via Grok (xAI) over its OpenAI-compatible API. Returns None on
    any failure (missing key, package not installed, network, parse) so the
    ensemble can fall back to Claude-only."""
    if not config.GROK_API_KEY:
        return None
    try:
        from openai import AsyncOpenAI
    except ImportError:
        log.warning("[ensemble] openai package not installed; Grok disabled")
        return None

    if as_of is None:
        as_of = datetime.now(timezone.utc)
    elif as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    prompt = GROK_CLASSIFICATION_PROMPT.format(
        question=market.question,
        yes_price=market.yes_price,
        time_remaining=_format_time_remaining(market.end_date, as_of),
        headline=headline,
        source=source,
    )

    start = time.time()
    try:
        client = AsyncOpenAI(api_key=config.GROK_API_KEY, base_url=GROK_BASE_URL)
        resp = await client.chat.completions.create(
            model=config.GROK_MODEL,
            max_tokens=200,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (resp.choices[0].message.content or "").strip()
        parsed = parse_classification(text)
        return Classification(
            direction=parsed["direction"],
            materiality=parsed["materiality"],
            confidence=parsed["confidence"],
            reasoning=parsed["reasoning"],
            latency_ms=int((time.time() - start) * 1000),
            model=config.GROK_MODEL,
            raw_response=text,
        )
    except Exception as e:
        log.warning(f"[ensemble] Grok classification failed: {e}")
        return None


async def classify_ensemble(
    headline: str, market: Market, source: str = "unknown", as_of: datetime | None = None
) -> Classification:
    """Classify a headline with Claude + Grok in parallel and blend the result.

    Always returns a Classification. Falls back to Claude-only (consensus 0.85)
    when Grok is unavailable, and to a neutral error when both models fail.
    """
    claude, grok = await asyncio.gather(
        classify_async(headline, market, source, as_of),
        _classify_grok(headline, market, source, as_of),
    )

    claude_ok = claude is not None and not claude.error
    grok_ok = grok is not None and not grok.error

    # Both failed: neutral, flagged as an error (routes to the pipeline's error path).
    if not claude_ok and not grok_ok:
        return Classification(
            direction="neutral",
            materiality=0.0,
            confidence=0.5,
            reasoning="Both classifiers failed",
            latency_ms=claude.latency_ms if claude else 0,
            model=f"{config.CLASSIFICATION_MODEL}+{config.GROK_MODEL}",
            error=True,
            consensus_score=0.0,
            ensemble_used=False,
        )

    # Grok unavailable: Claude-only, not fully corroborated.
    if claude_ok and not grok_ok:
        log.warning("[ensemble] Grok unavailable; falling back to Claude-only")
        claude.consensus_score = SINGLE_MODEL_CONSENSUS
        claude.ensemble_used = False
        return claude

    # Claude failed but Grok succeeded: use Grok alone, same single-model bar.
    if grok_ok and not claude_ok:
        log.warning("[ensemble] Claude failed; falling back to Grok-only")
        grok.consensus_score = SINGLE_MODEL_CONSENSUS
        grok.ensemble_used = False
        return grok

    # Both succeeded: blend.
    return blend(claude, grok)
