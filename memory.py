"""
Memory system — summarizes past classification accuracy and extracts
short lessons from incorrect calls so the classifier can calibrate itself.
"""
from __future__ import annotations

import logging

import anthropic

import config
import logger
from markets import _infer_category

log = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

LESSON_PROMPT = """A prediction market classification turned out to be wrong.

## Market Question
{question}

## News Headline
{headline}

## Classification (at the time)
Direction: {classification} | Materiality: {materiality:.2f}
Reasoning: {reasoning}

## Actual Outcome
Market price moved from {entry_price:.2f} to {exit_price:.2f} ({actual_direction}).

## Task
In 1-2 sentences, explain why this classification was likely wrong. Be specific
and actionable so a future classifier can avoid the same mistake.

Respond with ONLY the lesson text — no preamble, no JSON."""


def get_track_record() -> dict:
    """
    Summarize the `calibration` table: total resolved classifications,
    overall accuracy, and accuracy broken down by market category and news source.
    """
    rows = logger.get_calibration_with_trades()

    by_category: dict[str, list[int]] = {}
    by_source: dict[str, list[int]] = {}

    for row in rows:
        category = _infer_category(row["market_question"], [])
        source = row["news_source"] or "unknown"
        by_category.setdefault(category, []).append(row["correct"])
        by_source.setdefault(source, []).append(row["correct"])

    def _pct(values: list[int]) -> float:
        return round(sum(values) / len(values) * 100, 1) if values else 0.0

    return {
        "total": len(rows),
        "accuracy": _pct([r["correct"] for r in rows]),
        "by_category": {cat: _pct(vals) for cat, vals in by_category.items()},
        "by_source": {src: _pct(vals) for src, vals in by_source.items()},
    }


def record_lesson(trade: dict, actual_direction: str, entry_price: float, exit_price: float) -> str:
    """Generate (via Claude) and store a short lesson for an incorrect classification."""
    headlines = trade.get("headlines") or ""
    headline = headlines.splitlines()[0] if headlines else "(no headline recorded)"

    prompt = LESSON_PROMPT.format(
        question=trade["market_question"],
        headline=headline,
        classification=trade.get("classification") or "neutral",
        materiality=trade.get("materiality") or 0.0,
        reasoning=trade.get("reasoning") or "",
        actual_direction=actual_direction,
        entry_price=entry_price,
        exit_price=exit_price,
    )

    try:
        response = client.messages.create(
            model=config.CLASSIFICATION_MODEL,
            max_tokens=150,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        lesson = response.content[0].text.strip()
    except Exception as e:
        log.warning(f"[memory] Lesson generation error: {e}")
        lesson = f"(lesson generation failed: {type(e).__name__})"

    logger.log_lesson(
        trade_id=trade["id"],
        market_question=trade["market_question"],
        classification=trade.get("classification") or "neutral",
        actual_direction=actual_direction,
        lesson=lesson,
    )
    return lesson


if __name__ == "__main__":
    record = get_track_record()
    print(f"Total resolved: {record['total']}")
    print(f"Overall accuracy: {record['accuracy']}%")
    print(f"By category: {record['by_category']}")
    print(f"By source: {record['by_source']}")

    lessons = logger.get_recent_lessons(limit=5)
    print(f"\nRecent lessons ({len(lessons)}):")
    for l in lessons:
        print(f"  - {l['lesson']}")
