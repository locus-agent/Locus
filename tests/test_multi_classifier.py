"""Universal parser, smart consensus formula, and ensemble fallback behavior."""
import asyncio

import pytest

from locus import config
from locus.core import multi_classifier
from locus.core.classifier import Classification, parse_classification
from locus.markets.gamma import Market

MKT = Market("c1", "Will X happen?", "ai", 0.5, 0.5, 5000, "2026-12-31", True, [])


def _cls(direction, mat, conf, *, error=False, model="m", raw="r", reasoning="why"):
    return Classification(
        direction=direction, materiality=mat, confidence=conf,
        reasoning=reasoning, latency_ms=10, model=model, error=error, raw_response=raw,
    )


# --- Universal parser ---

def test_parser_json():
    out = parse_classification(
        '{"direction":"bullish","materiality":0.7,"confidence":0.82,"reasoning":"r"}'
    )
    assert out == {"direction": "bullish", "materiality": 0.7,
                   "confidence": 0.82, "time_horizon": "medium", "reasoning": "r"}


def test_parser_json_in_code_fence_with_prose():
    text = 'Sure, here is my answer:\n```json\n{"direction": "bearish", "materiality": 0.5, "confidence": 0.6}\n```\nHope that helps!'
    out = parse_classification(text)
    assert out["direction"] == "bearish"
    assert out["materiality"] == 0.5 and out["confidence"] == 0.6


def test_parser_json_embedded_in_text():
    text = 'The classification is {"direction": "neutral", "materiality": 0.1} based on the news.'
    out = parse_classification(text)
    assert out["direction"] == "neutral" and out["materiality"] == 0.1


def test_parser_text_regex_fallback():
    text = "direction: bearish, materiality: 0.4, confidence: 0.65 — my reasoning here"
    out = parse_classification(text)
    assert out["direction"] == "bearish"
    assert out["materiality"] == 0.4 and out["confidence"] == 0.65


def test_parser_pure_fallback():
    out = parse_classification("totally unparseable prose with no fields at all")
    assert out == {"direction": "neutral", "materiality": 0.0,
                   "confidence": 0.5, "time_horizon": "medium", "reasoning": ""}


def test_parser_empty_and_none():
    assert parse_classification("")["direction"] == "neutral"
    assert parse_classification(None)["materiality"] == 0.0


@pytest.mark.parametrize("synonym,expected", [
    ("up", "bullish"), ("positive", "bullish"), ("buy", "bullish"), ("yes", "bullish"),
    ("down", "bearish"), ("negative", "bearish"), ("sell", "bearish"), ("no", "bearish"),
    ("BULLISH", "bullish"), ("Bearish", "bearish"), ("sideways", "neutral"),
])
def test_parser_direction_synonyms(synonym, expected):
    out = parse_classification('{"direction":"%s","materiality":0.5}' % synonym)
    assert out["direction"] == expected


def test_parser_synonym_in_text_mode():
    out = parse_classification("direction = UP materiality = 0.6")
    assert out["direction"] == "bullish" and out["materiality"] == 0.6


def test_parser_clamps_out_of_range():
    out = parse_classification(
        '{"direction":"bullish","materiality":1.7,"confidence":0.2}'
    )
    assert out["materiality"] == 1.0  # clamped to [0,1]
    assert out["confidence"] == 0.5   # clamped up to floor 0.5


def test_parser_handles_non_numeric_fields():
    out = parse_classification('{"direction":"bullish","materiality":"high"}')
    assert out["materiality"] == 0.0  # bad float -> lo


# --- Smart consensus formula ---

def test_consensus_full_agreement():
    claude = _cls("bullish", 0.8, 0.9)
    grok = _cls("bullish", 0.7, 0.8)
    # 0.5*1.0 + 0.3*(1-0.1) + 0.2*0.85 = 0.5 + 0.27 + 0.17 = 0.94
    assert multi_classifier.consensus_score(claude, grok) == pytest.approx(0.94)


def test_consensus_opposite_directions():
    claude = _cls("bullish", 0.8, 0.9)
    grok = _cls("bearish", 0.8, 0.9)
    # 0.5*0.0 + 0.3*1.0 + 0.2*0.9 = 0.48
    assert multi_classifier.consensus_score(claude, grok) == pytest.approx(0.48)


def test_consensus_one_neutral_is_half():
    claude = _cls("bullish", 0.5, 0.7)
    grok = _cls("neutral", 0.5, 0.7)
    # 0.5*0.5 + 0.3*1.0 + 0.2*0.7 = 0.25 + 0.3 + 0.14 = 0.69
    assert multi_classifier.consensus_score(claude, grok) == pytest.approx(0.69)


# --- Weighted averaging in blend ---

def test_blend_weighted_average():
    claude = _cls("bullish", 0.8, 0.9)
    grok = _cls("bullish", 0.3, 0.6)
    blended = multi_classifier.blend(claude, grok)
    # materiality = 0.6*0.8 + 0.4*0.3 = 0.60 ; confidence = 0.6*0.9 + 0.4*0.6 = 0.78
    assert blended.materiality == pytest.approx(0.60)
    assert blended.confidence == pytest.approx(0.78)
    assert blended.direction == "bullish"  # follows higher-weighted Claude
    assert blended.ensemble_used is True
    assert blended.consensus_score is not None


# --- classify_ensemble orchestration ---

def _patch(monkeypatch, claude_result, grok_result):
    async def fake_claude(headline, market, source="unknown", as_of=None):
        return claude_result
    async def fake_grok(headline, market, source="unknown", as_of=None):
        return grok_result
    monkeypatch.setattr(multi_classifier, "classify_async", fake_claude)
    monkeypatch.setattr(multi_classifier, "_classify_grok", fake_grok)


def test_ensemble_both_succeed_sets_flag(monkeypatch):
    _patch(monkeypatch, _cls("bullish", 0.8, 0.9), _cls("bullish", 0.7, 0.8))
    out = asyncio.run(multi_classifier.classify_ensemble("h", MKT))
    assert out.ensemble_used is True
    assert out.consensus_score == pytest.approx(0.94)
    assert out.materiality == pytest.approx(0.6 * 0.8 + 0.4 * 0.7)
    assert out.error is False


def test_ensemble_grok_failure_falls_back_to_claude(monkeypatch):
    claude = _cls("bullish", 0.8, 0.9)
    _patch(monkeypatch, claude, None)  # Grok unavailable
    out = asyncio.run(multi_classifier.classify_ensemble("h", MKT))
    assert out.direction == "bullish" and out.materiality == 0.8
    assert out.consensus_score == 0.85  # single model, not overconfident
    assert out.ensemble_used is False
    assert out.error is False


def test_ensemble_claude_failure_falls_back_to_grok(monkeypatch):
    grok = _cls("bearish", 0.6, 0.7)
    _patch(monkeypatch, _cls("neutral", 0.0, 0.5, error=True), grok)
    out = asyncio.run(multi_classifier.classify_ensemble("h", MKT))
    assert out.direction == "bearish" and out.consensus_score == 0.85
    assert out.ensemble_used is False


def test_ensemble_both_fail_is_neutral_error(monkeypatch):
    _patch(monkeypatch, _cls("neutral", 0.0, 0.5, error=True), None)
    out = asyncio.run(multi_classifier.classify_ensemble("h", MKT))
    assert out.direction == "neutral" and out.materiality == 0.0
    assert out.error is True and out.consensus_score == 0.0
    assert out.ensemble_used is False


def test_ensemble_runs_in_parallel(monkeypatch):
    # Both coroutines must be awaited concurrently (gather), not sequentially.
    order = []

    async def fake_claude(headline, market, source="unknown", as_of=None):
        order.append("claude_start")
        await asyncio.sleep(0.02)
        order.append("claude_end")
        return _cls("bullish", 0.8, 0.9)

    async def fake_grok(headline, market, source="unknown", as_of=None):
        order.append("grok_start")
        await asyncio.sleep(0.02)
        order.append("grok_end")
        return _cls("bullish", 0.7, 0.8)

    monkeypatch.setattr(multi_classifier, "classify_async", fake_claude)
    monkeypatch.setattr(multi_classifier, "_classify_grok", fake_grok)
    asyncio.run(multi_classifier.classify_ensemble("h", MKT))
    # Both started before either finished -> truly parallel.
    assert order[:2] == ["claude_start", "grok_start"]


# --- Low-consensus gate ---

def test_is_low_consensus_blocks_disagreement(monkeypatch):
    monkeypatch.setattr(config, "ENSEMBLE_MIN_CONSENSUS", 0.5)
    blended = multi_classifier.blend(_cls("bullish", 0.8, 0.9), _cls("bearish", 0.8, 0.9))
    assert blended.consensus_score == pytest.approx(0.48)
    assert multi_classifier.is_low_consensus(blended) is True


def test_is_low_consensus_passes_agreement(monkeypatch):
    monkeypatch.setattr(config, "ENSEMBLE_MIN_CONSENSUS", 0.5)
    blended = multi_classifier.blend(_cls("bullish", 0.8, 0.9), _cls("bullish", 0.7, 0.8))
    assert multi_classifier.is_low_consensus(blended) is False


def test_is_low_consensus_ignores_single_model():
    # Single-model fallback (consensus 0.85) and non-ensemble (None) never block.
    assert multi_classifier.is_low_consensus(_cls("bullish", 0.8, 0.9)) is False  # None
    single = _cls("bullish", 0.8, 0.9)
    single.consensus_score = 0.85
    assert multi_classifier.is_low_consensus(single) is False


# --- Persistence of the new columns ---

def test_log_classification_persists_consensus_columns(tmp_db):
    cid = tmp_db.log_classification(
        market_question="Q?", headline="h", news_source="rss",
        direction="bullish", materiality=0.6, edge=0.2, action="signal",
        consensus_score=0.94, ensemble_used=True,
    )
    row = next(c for c in tmp_db.get_recent_classifications(limit=10) if c["id"] == cid)
    assert row["consensus_score"] == pytest.approx(0.94)
    assert row["ensemble_used"] == 1  # stored as INTEGER


def test_log_classification_consensus_defaults_null(tmp_db):
    cid = tmp_db.log_classification(
        market_question="Q?", headline="h", news_source="rss",
        direction="neutral", materiality=0.1, edge=None, action="skip",
    )
    row = next(c for c in tmp_db.get_recent_classifications(limit=10) if c["id"] == cid)
    assert row["consensus_score"] is None
    assert row["ensemble_used"] is None
