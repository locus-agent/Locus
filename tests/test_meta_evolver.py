"""Meta-prompt evolution: 7-day trigger, version saving/never-overwrite,
dynamic prompt loading + fallback, the CLI command, and validation."""
import argparse
import asyncio
import types
from datetime import datetime, timedelta, timezone

import pytest

from locus import config
from locus.core import classifier
from locus.memory import meta_evolver

NOW = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _pin_evolution_config(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROMPT_EVOLUTION_ENABLED", True)
    monkeypatch.setattr(config, "PROMPT_EVOLUTION_INTERVAL_DAYS", 7.0)
    monkeypatch.setattr(meta_evolver, "PROMPTS_DIR", tmp_path / "prompts")


def _ts(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")


def _add_lesson(db, created_at, lesson="Avoid chasing momentum headlines."):
    conn = db._conn()
    conn.execute(
        """INSERT INTO lessons (trade_id, market_question, classification,
           actual_direction, lesson, created_at) VALUES (?, ?, ?, ?, ?, ?)""",
        (1, "Will BTC hit $100k?", "bullish", "bearish", lesson, created_at),
    )
    conn.commit()
    conn.close()


def _add_version(db, version, created_at, prompt="P {question} {materiality}"):
    conn = db._conn()
    conn.execute(
        """INSERT INTO prompt_versions (version, prompt_text, created_at,
           lessons_count, accuracy_at_creation) VALUES (?, ?, ?, ?, ?)""",
        (version, prompt, created_at, 3, 12.5),
    )
    conn.commit()
    conn.close()


# --- should_evolve: 7-day trigger logic ----------------------------------

def test_no_data_no_evolution(tmp_db):
    assert meta_evolver.should_evolve(NOW) is False


def test_first_evolution_waits_for_seven_days_of_lessons(tmp_db):
    _add_lesson(tmp_db, _ts(3))  # only 3 days of lessons
    assert meta_evolver.should_evolve(NOW) is False


def test_first_evolution_fires_after_seven_days_of_lessons(tmp_db):
    _add_lesson(tmp_db, _ts(8))  # oldest lesson is 8 days old
    assert meta_evolver.should_evolve(NOW) is True


def test_recent_version_blocks_evolution(tmp_db):
    _add_lesson(tmp_db, _ts(30))
    _add_version(tmp_db, 1, _ts(3))  # last evolved 3 days ago
    assert meta_evolver.should_evolve(NOW) is False


def test_week_old_version_triggers_evolution(tmp_db):
    _add_version(tmp_db, 1, _ts(7))  # exactly a week ago
    assert meta_evolver.should_evolve(NOW) is True


def test_disabled_flag_blocks_evolution(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "PROMPT_EVOLUTION_ENABLED", False)
    _add_version(tmp_db, 1, _ts(30))
    assert meta_evolver.should_evolve(NOW) is False


def test_interval_is_configurable(tmp_db, monkeypatch):
    _add_lesson(tmp_db, _ts(4))
    monkeypatch.setattr(config, "PROMPT_EVOLUTION_INTERVAL_DAYS", 3.0)
    assert meta_evolver.should_evolve(NOW) is True  # 4 days > 3-day interval


# --- version saving + never-overwrite ------------------------------------

def _patch_llm(monkeypatch, text):
    async def fake(meta_prompt, system=None):
        return text
    monkeypatch.setattr(meta_evolver, "_generate_improved_prompt", fake)


GOOD_PROMPT = classifier.CLASSIFICATION_PROMPT + "\n\n(Evolved: be calibrated.)"


def test_evolve_saves_version_file_and_row(tmp_db, monkeypatch):
    _add_lesson(tmp_db, _ts(10))
    _patch_llm(monkeypatch, GOOD_PROMPT)

    result = asyncio.run(meta_evolver.evolve_prompt())
    assert result["version"] == 1
    assert result["lessons_count"] == 1
    assert (meta_evolver.PROMPTS_DIR / "classification_prompt_v1.txt").read_text() == GOOD_PROMPT

    row = tmp_db.get_latest_prompt_version()
    assert row["version"] == 1 and row["prompt_text"] == GOOD_PROMPT


def test_evolve_increments_version(tmp_db, monkeypatch):
    _patch_llm(monkeypatch, GOOD_PROMPT)
    asyncio.run(meta_evolver.evolve_prompt())
    asyncio.run(meta_evolver.evolve_prompt())
    assert tmp_db.get_latest_prompt_version()["version"] == 2
    assert (meta_evolver.PROMPTS_DIR / "classification_prompt_v2.txt").exists()


def test_evolve_never_overwrites_existing_file(tmp_db, monkeypatch):
    # A stray v1 file with no DB row (e.g. a prior crash) must not be clobbered.
    meta_evolver.PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    (meta_evolver.PROMPTS_DIR / "classification_prompt_v1.txt").write_text("ORIGINAL")
    _patch_llm(monkeypatch, GOOD_PROMPT)

    result = asyncio.run(meta_evolver.evolve_prompt())
    assert result["version"] == 2  # bumped past the existing file
    assert (meta_evolver.PROMPTS_DIR / "classification_prompt_v1.txt").read_text() == "ORIGINAL"


def test_evolve_rejects_invalid_prompt(tmp_db, monkeypatch):
    _patch_llm(monkeypatch, "A broken prompt with no placeholders and no JSON.")
    assert asyncio.run(meta_evolver.evolve_prompt()) is None
    assert tmp_db.get_latest_prompt_version() is None


def test_evolve_strips_code_fences(tmp_db, monkeypatch):
    _patch_llm(monkeypatch, "```\n" + GOOD_PROMPT + "\n```")
    asyncio.run(meta_evolver.evolve_prompt())
    assert tmp_db.get_latest_prompt_version()["prompt_text"] == GOOD_PROMPT


def test_evolve_returns_none_on_generation_error(tmp_db, monkeypatch):
    async def boom(meta_prompt, system=None):
        raise RuntimeError("api down")
    monkeypatch.setattr(meta_evolver, "_generate_improved_prompt", boom)
    assert asyncio.run(meta_evolver.evolve_prompt()) is None


def test_evolve_retries_once_with_stricter_system(tmp_db, monkeypatch):
    # First call returns a prompt missing {headline}; the retry returns a good one.
    bad = GOOD_PROMPT.replace("{headline}", "the news")
    calls = []

    async def fake(meta_prompt, system=None):
        calls.append(system)
        return bad if len(calls) == 1 else GOOD_PROMPT
    monkeypatch.setattr(meta_evolver, "_generate_improved_prompt", fake)

    result = asyncio.run(meta_evolver.evolve_prompt())
    assert result["version"] == 1
    assert calls[0] is None  # first attempt: no system prompt
    assert calls[1] == meta_evolver._STRICT_RETRY_SYSTEM  # retry tightens the contract
    assert tmp_db.get_latest_prompt_version()["prompt_text"] == GOOD_PROMPT


def test_evolve_gives_up_after_max_retries(tmp_db, monkeypatch):
    _add_lesson(tmp_db, _ts(10))
    _patch_llm(monkeypatch, "still broken, no placeholders or JSON")
    monkeypatch.setattr(config, "EVOLVE_MAX_RETRIES", 2)
    assert asyncio.run(meta_evolver.evolve_prompt()) is None
    assert tmp_db.get_latest_prompt_version() is None


# --- prompt validation ---------------------------------------------------

def test_hardcoded_prompt_is_valid():
    assert meta_evolver.prompt_is_valid(classifier.CLASSIFICATION_PROMPT) is True


def test_validation_rejects_missing_placeholder():
    # Drops {headline} -> can't format -> invalid.
    bad = classifier.CLASSIFICATION_PROMPT.replace("{headline}", "the news")
    # still has JSON contract but a bogus stray brace would break; here it's fine,
    # so instead introduce an unknown placeholder:
    bad = classifier.CLASSIFICATION_PROMPT + " {unknown_token}"
    assert meta_evolver.prompt_is_valid(bad) is False


def test_validation_rejects_dropped_json_contract():
    no_json = "Classify {question} {threshold_line}{yes_price} {time_remaining} {headline} {source} {track_record}"
    assert meta_evolver.prompt_is_valid(no_json) is False


# A minimal prompt that satisfies every requirement: all placeholders + JSON contract.
ALL_PLACEHOLDERS_PROMPT = (
    "Classify {question}. {threshold_line}Price {yes_price:.3f}. "
    "Time {time_remaining}. News {headline} from {source}. {track_record} "
    'Return {{"direction": "...", "materiality": 0.0}}.'
)


def test_validate_prompt_passes_with_all_placeholders():
    assert meta_evolver.validate_prompt(ALL_PLACEHOLDERS_PROMPT) == []
    assert meta_evolver.prompt_is_valid(ALL_PLACEHOLDERS_PROMPT) is True


def test_validate_prompt_names_missing_placeholders():
    dropped = ALL_PLACEHOLDERS_PROMPT.replace("{headline}", "the news").replace(
        "{track_record}", "no record")
    problems = meta_evolver.validate_prompt(dropped)
    assert len(problems) == 1
    assert "missing required placeholders" in problems[0]
    assert "{headline}" in problems[0] and "{track_record}" in problems[0]
    # placeholders that ARE present must not be reported as missing
    assert "{question}" not in problems[0]


def test_validate_prompt_yes_price_with_format_spec_counts():
    # {yes_price:.3f} must satisfy the {yes_price} requirement.
    assert "yes_price" not in " ".join(
        meta_evolver.validate_prompt(ALL_PLACEHOLDERS_PROMPT))


def test_validate_prompt_reports_missing_json_fields():
    no_json = ALL_PLACEHOLDERS_PROMPT.replace(
        'Return {{"direction": "...", "materiality": 0.0}}.', "Return a verdict.")
    problems = meta_evolver.validate_prompt(no_json)
    assert any('"direction"' in p for p in problems)
    assert any('"materiality"' in p for p in problems)


def test_validate_prompt_reports_unknown_placeholder():
    problems = meta_evolver.validate_prompt(ALL_PLACEHOLDERS_PROMPT + " {mystery}")
    assert any("prompt.format() failed" in p for p in problems)


def test_validate_prompt_empty():
    assert meta_evolver.validate_prompt("   ") == ["prompt is empty"]


@pytest.mark.parametrize("fence", ["```", "```text", "```markdown"])
def test_validate_strips_markdown_fences(fence):
    wrapped = f"{fence}\n{ALL_PLACEHOLDERS_PROMPT}\n```"
    assert meta_evolver._strip_fences(wrapped) == ALL_PLACEHOLDERS_PROMPT
    assert meta_evolver.prompt_is_valid(meta_evolver._strip_fences(wrapped)) is True


# --- dynamic prompt loading + fallback -----------------------------------

def test_get_active_prompt_defaults_to_hardcoded(tmp_db):
    assert classifier.get_active_prompt() == classifier.CLASSIFICATION_PROMPT


def test_get_active_prompt_loads_latest_version(tmp_db, monkeypatch):
    _patch_llm(monkeypatch, GOOD_PROMPT)
    asyncio.run(meta_evolver.evolve_prompt())
    assert classifier.get_active_prompt() == GOOD_PROMPT


def test_get_active_prompt_falls_back_on_load_error(tmp_db, monkeypatch):
    def boom():
        raise RuntimeError("db error")
    monkeypatch.setattr(classifier.logger, "get_latest_prompt_version", boom)
    assert classifier.get_active_prompt() == classifier.CLASSIFICATION_PROMPT


def test_classify_falls_back_when_active_prompt_unformattable(tmp_db, monkeypatch):
    # An active prompt with an unknown placeholder must not break classify():
    # it falls back to the hardcoded prompt and still returns a clean result.
    monkeypatch.setattr(classifier, "get_active_prompt", lambda: "Bad {nope} prompt")
    monkeypatch.setattr(classifier.time, "sleep", lambda s: None)

    def create(**kwargs):
        return types.SimpleNamespace(content=[types.SimpleNamespace(
            text='{"direction":"bullish","materiality":0.6,"confidence":0.7,"reasoning":"r"}')])
    monkeypatch.setattr(classifier, "client",
                        types.SimpleNamespace(messages=types.SimpleNamespace(create=create)))

    from locus.markets.gamma import Market
    mkt = Market("c1", "Will X happen?", "ai", 0.5, 0.5, 5000, "2026-12-31", True, [])
    result = classifier.classify("headline", mkt)
    assert result.error is False and result.direction == "bullish"


# --- accuracy stats + meta-prompt content --------------------------------

def test_compute_accuracy_stats(tmp_db):
    # Seed two graded classifications: one correct crypto/bullish, one wrong.
    conn = tmp_db._conn()
    conn.execute(
        """INSERT INTO classifications (id, market_question, action, direction)
           VALUES (1, 'Will Bitcoin hit $100k?', 'signal', 'bullish'),
                  (2, 'Will OpenAI release GPT-6?', 'signal', 'bearish')"""
    )
    conn.execute(
        """INSERT INTO classification_grades (classification_id, direction, correct)
           VALUES (1, 'bullish', 1), (2, 'bearish', 0)"""
    )
    conn.commit()
    conn.close()

    stats = meta_evolver.compute_accuracy_stats()
    assert stats["total"] == 2
    assert stats["overall"] == 50.0
    assert stats["by_category"]["crypto"] == 100.0
    assert stats["by_category"]["ai"] == 0.0
    assert stats["by_direction"]["bullish"] == 100.0
    assert stats["by_direction"]["bearish"] == 0.0


def test_build_meta_prompt_includes_lessons_and_stats():
    lessons = [{"lesson": "Don't chase momentum."}, {"lesson": "Bearish calls are weak."}]
    stats = {"total": 5, "overall": 30.0,
             "by_category": {"crypto": 31.6, "politics": 18.8},
             "by_direction": {"bullish": 18.1, "bearish": 11.1}}
    mp = meta_evolver.build_meta_prompt("CURRENT_PROMPT_HERE", lessons, stats)
    assert "CURRENT_PROMPT_HERE" in mp
    assert "2 lessons" in mp
    assert "Don't chase momentum." in mp
    assert "crypto 31.6%" in mp and "bearish 11.1%" in mp
    assert "keep the JSON output format unchanged" in mp
    assert "Return ONLY the improved prompt text" in mp


# --- CLI command ---------------------------------------------------------

def test_cli_evolve_runs(monkeypatch):
    import cli
    called = []

    async def fake_evolve():
        called.append(True)
        return {"version": 4, "lessons_count": 9, "accuracy_at_creation": 22.0,
                "prev_chars": 100, "chars": 130, "path": "/tmp/p_v4.txt"}
    monkeypatch.setattr(meta_evolver, "evolve_prompt", fake_evolve)

    cli.cmd_evolve(argparse.Namespace())  # must not raise
    assert called == [True]


def test_cli_evolve_handles_no_result(monkeypatch):
    import cli

    async def fake_evolve():
        return None
    monkeypatch.setattr(meta_evolver, "evolve_prompt", fake_evolve)
    cli.cmd_evolve(argparse.Namespace())  # must not raise on None
