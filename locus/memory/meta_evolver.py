"""
Meta-Prompt Evolution — once a week, Claude rewrites its own classification
prompt from accumulated lessons and accuracy stats.

After the daily journal entry, the pipeline asks: has it been a week since the
last evolution (or seven days of lessons, for the very first one)? If so,
evolve_prompt() feeds the current prompt, every lesson learned from a mistake,
and accuracy broken down by category and direction to Sonnet, asks for an
improved prompt, validates it still formats, and versions it:
  - a file under docs/prompts/classification_prompt_v{N}.txt (never overwritten)
  - a row in the prompt_versions table

The classifier then loads the latest version at runtime (classifier.get_active_prompt).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import anthropic

from locus import config
from locus.memory import logger
from locus.markets.gamma import _infer_category

log = logging.getLogger(__name__)

PROMPTS_DIR = config.PROJECT_ROOT / "docs" / "prompts"

# Sample kwargs used to verify an evolved prompt still .format()s cleanly before
# we trust it. Must cover every placeholder classifier.classify() fills in.
_VALIDATION_KWARGS = dict(
    question="Will X happen by 2027?",
    threshold_line="",
    time_remaining="3 days",
    yes_price=0.5,
    headline="Some breaking news",
    source="rss",
    track_record="No resolved classifications yet.",
)


def _parse_dt(value: str) -> datetime:
    """Parse a stored 'YYYY-MM-DD HH:MM:SS' timestamp as UTC."""
    dt = datetime.fromisoformat(value.replace(" ", "T"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def should_evolve(now: datetime | None = None) -> bool:
    """True when a prompt evolution is due.

    - If a prior version exists: due when PROMPT_EVOLUTION_INTERVAL_DAYS have
      passed since it was created.
    - If no version exists yet (first evolution): due once the oldest lesson is
      at least PROMPT_EVOLUTION_INTERVAL_DAYS old — i.e. a week of data to learn
      from. With no lessons at all there is nothing to improve, so not due.

    Disabled entirely via config.PROMPT_EVOLUTION_ENABLED.
    """
    if not config.PROMPT_EVOLUTION_ENABLED:
        return False
    now = now or datetime.now(timezone.utc)
    interval = timedelta(days=config.PROMPT_EVOLUTION_INTERVAL_DAYS)

    latest = logger.get_latest_prompt_version()
    if latest:
        return (now - _parse_dt(latest["created_at"])) >= interval

    earliest_lesson = logger.get_earliest_lesson_date()
    if not earliest_lesson:
        return False
    return (now - _parse_dt(earliest_lesson)) >= interval


def compute_accuracy_stats() -> dict:
    """Accuracy from graded classifications, broken down by category and
    direction. Returns {total, overall, by_category, by_direction} with
    percentages (0-100)."""
    rows = logger.get_classification_grades_with_meta()

    by_category: dict[str, list[int]] = {}
    by_direction: dict[str, list[int]] = {}
    for r in rows:
        cat = _infer_category(r.get("market_question") or "", [])
        direction = r.get("classification") or "neutral"
        by_category.setdefault(cat, []).append(r["correct"])
        by_direction.setdefault(direction, []).append(r["correct"])

    def _pct(values: list[int]) -> float:
        return round(sum(values) / len(values) * 100, 1) if values else 0.0

    return {
        "total": len(rows),
        "overall": _pct([r["correct"] for r in rows]),
        "by_category": {k: _pct(v) for k, v in by_category.items()},
        "by_direction": {k: _pct(v) for k, v in by_direction.items()},
    }


def build_meta_prompt(current_prompt: str, lessons: list[dict], stats: dict) -> str:
    """Construct the instruction sent to Sonnet to improve the prompt."""
    lesson_lines = "\n".join(f"- {l['lesson']}" for l in lessons) or "(none yet)"

    stat_parts = [f"{cat} {pct}%" for cat, pct in stats["by_category"].items()]
    stat_parts += [f"{d} {pct}%" for d, pct in stats["by_direction"].items()]
    stat_str = ", ".join(stat_parts) if stat_parts else "no graded data yet"

    return (
        "You are improving an AI trading agent's classification prompt. "
        "Here is the current prompt:\n\n"
        f"{current_prompt}\n\n"
        f"Here are {len(lessons)} lessons learned from mistakes:\n{lesson_lines}\n\n"
        f"Accuracy stats: {stat_str}.\n\n"
        "Improve the prompt to: 1) avoid the patterns that led to these mistakes, "
        "2) be more calibrated for categories with low accuracy, "
        "3) keep the JSON output format unchanged. "
        "You MUST keep every curly-brace template placeholder exactly as-is "
        "({question}, {threshold_line}, {yes_price}, {time_remaining}, {headline}, "
        "{source}, {track_record}) and must not introduce any new curly-brace "
        "tokens. Return ONLY the improved prompt text, nothing else."
    )


def _strip_fences(text: str) -> str:
    """Strip a wrapping Markdown code fence if the model added one."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def prompt_is_valid(prompt: str) -> bool:
    """A candidate prompt is usable only if it still formats with the real
    kwargs (every placeholder present, no stray braces) and keeps the JSON
    output contract."""
    if not prompt or '"direction"' not in prompt or '"materiality"' not in prompt:
        return False
    try:
        prompt.format(**_VALIDATION_KWARGS)
    except (KeyError, IndexError, ValueError):
        return False
    return True


async def _generate_improved_prompt(meta_prompt: str) -> str:
    """Call Sonnet for the improved prompt (off the event loop)."""
    def _call() -> str:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=config.SCORING_MODEL,
            max_tokens=2048,
            temperature=0.4,
            messages=[{"role": "user", "content": meta_prompt}],
        )
        return resp.content[0].text.strip()

    return await asyncio.get_event_loop().run_in_executor(None, _call)


async def evolve_prompt() -> dict | None:
    """Evolve the classification prompt from accumulated lessons + accuracy.

    Returns a summary dict on success, or None when nothing was saved (the
    model's output failed validation, or generation errored)."""
    lessons = logger.get_all_lessons()
    stats = compute_accuracy_stats()
    current_prompt = _current_prompt()
    meta_prompt = build_meta_prompt(current_prompt, lessons, stats)

    try:
        candidate = _strip_fences(await _generate_improved_prompt(meta_prompt))
    except Exception as e:
        log.warning(f"[meta_evolver] Prompt generation failed: {e}")
        return None

    if not prompt_is_valid(candidate):
        log.warning(
            "[meta_evolver] Evolved prompt failed validation (placeholders/JSON); not saving"
        )
        return None

    latest = logger.get_latest_prompt_version()
    version = (latest["version"] + 1) if latest else 1

    # Never overwrite: bump until the versioned file name is free, keeping the
    # DB version in lockstep with the file on disk.
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PROMPTS_DIR / f"classification_prompt_v{version}.txt"
    while path.exists():
        version += 1
        path = PROMPTS_DIR / f"classification_prompt_v{version}.txt"
    path.write_text(candidate)

    logger.save_prompt_version(
        version=version,
        prompt_text=candidate,
        lessons_count=len(lessons),
        accuracy_at_creation=stats["overall"],
    )

    summary = {
        "version": version,
        "path": str(path),
        "lessons_count": len(lessons),
        "accuracy_at_creation": stats["overall"],
        "chars": len(candidate),
        "prev_chars": len(current_prompt),
    }
    log.info(
        f"[meta_evolver] Evolved classification prompt -> v{version} "
        f"({len(lessons)} lessons, accuracy {stats['overall']}%, "
        f"{summary['prev_chars']} -> {summary['chars']} chars)"
    )
    return summary


def evolve_prompt_sync() -> dict | None:
    """Blocking wrapper around evolve_prompt() for sync callers (the journal
    trigger runs in a thread-pool executor with no event loop)."""
    return asyncio.run(evolve_prompt())


def _current_prompt() -> str:
    """The prompt to improve upon — the active one (latest evolved or hardcoded).
    Imported lazily to avoid a circular import with the classifier."""
    from locus.core import classifier
    return classifier.get_active_prompt()
