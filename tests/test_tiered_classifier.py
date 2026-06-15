"""Tiered classification: a cheap Haiku prefilter in front of the full Sonnet
classify(). Verifies the prefilter rejects low-value headlines, passes good ones
through to Sonnet, caps inflated Haiku materiality, fails open on Haiku errors,
and that the disabled path is unaffected."""
import types

import pytest

from locus import config
from locus.core import classifier
from locus.markets.gamma import Market

MKT = Market("c1", "Will X happen?", "ai", 0.5, 0.5, 5000, "2026-12-31", True, [], slug="will-x")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(classifier.time, "sleep", lambda s: None)


def _response(text):
    return types.SimpleNamespace(content=[types.SimpleNamespace(text=text)])


def _fake_client(behaviors):
    """behaviors: list per .create() call — an Exception to raise or a str to return."""
    calls = {"n": 0, "models": []}

    def create(**kwargs):
        calls["models"].append(kwargs.get("model"))
        b = behaviors[min(calls["n"], len(behaviors) - 1)]
        calls["n"] += 1
        if isinstance(b, Exception):
            raise b
        return _response(b)

    return types.SimpleNamespace(messages=types.SimpleNamespace(create=create)), calls


# Haiku prefilter JSON shapes.
HAIKU_PASS = '{"relevant": true, "direction": "bullish", "materiality": 0.6, "reason": "strong"}'
HAIKU_LOWMAT = '{"relevant": true, "direction": "bullish", "materiality": 0.10, "reason": "weak"}'
HAIKU_IRRELEVANT = '{"relevant": false, "direction": "neutral", "materiality": 0.0, "reason": "off-topic"}'
HAIKU_INFLATED = '{"relevant": true, "direction": "bullish", "materiality": 0.95, "reason": "huge"}'
# Sonnet deep-analysis JSON shape (full schema with confidence).
SONNET = '{"direction": "bullish", "materiality": 0.7, "confidence": 0.8, "reasoning": "deep"}'


@pytest.fixture(autouse=True)
def pinned_models(monkeypatch):
    monkeypatch.setattr(config, "HAIKU_MODEL", "haiku-test")
    monkeypatch.setattr(config, "SCORING_MODEL", "sonnet-test")
    monkeypatch.setattr(config, "HAIKU_MATERIALITY_THRESHOLD", 0.25)


def test_haiku_rejects_low_materiality(monkeypatch):
    client, calls = _fake_client([HAIKU_LOWMAT])
    monkeypatch.setattr(classifier, "client", client)
    c = classifier.classify_fast("headline", MKT)
    assert c.action == "prefiltered_haiku"
    assert c.model == "haiku-test"
    assert calls["n"] == 1                     # Haiku only — no Sonnet call
    assert calls["models"] == ["haiku-test"]


def test_haiku_rejects_irrelevant(monkeypatch):
    client, calls = _fake_client([HAIKU_IRRELEVANT])
    monkeypatch.setattr(classifier, "client", client)
    c = classifier.classify_fast("headline", MKT)
    assert c.action == "prefiltered_haiku"
    assert calls["n"] == 1                      # short-circuits on relevant=false


def test_haiku_passes_high_to_sonnet(monkeypatch):
    client, calls = _fake_client([HAIKU_PASS, SONNET])
    monkeypatch.setattr(classifier, "client", client)
    c = classifier.classify_fast("headline", MKT)
    assert c.action is None                      # full result, not prefiltered
    assert c.model == "sonnet-test"              # deep tier ran on Sonnet
    assert (c.direction, c.materiality, c.confidence) == ("bullish", 0.7, 0.8)
    assert calls["models"] == ["haiku-test", "sonnet-test"]


def test_materiality_cap_085_to_075(monkeypatch):
    # Haiku returns 0.95 (> 0.85). It must be capped to 0.75, which still clears
    # the 0.25 floor, so it passes to Sonnet — but the cap is what we assert by
    # rejecting the Sonnet call and checking the prefiltered value instead.
    monkeypatch.setattr(config, "HAIKU_MATERIALITY_THRESHOLD", 0.80)  # force a reject after the cap
    client, calls = _fake_client([HAIKU_INFLATED])
    monkeypatch.setattr(classifier, "client", client)
    c = classifier.classify_fast("headline", MKT)
    # 0.95 -> capped 0.75 -> below the 0.80 floor -> prefiltered, materiality 0.75.
    assert c.action == "prefiltered_haiku"
    assert c.materiality == 0.75
    assert calls["n"] == 1


def test_materiality_exactly_085_not_capped(monkeypatch):
    # Cap triggers only for > 0.85; exactly 0.85 is left alone.
    at_cap = '{"relevant": true, "direction": "bullish", "materiality": 0.85, "reason": "x"}'
    monkeypatch.setattr(config, "HAIKU_MATERIALITY_THRESHOLD", 0.80)
    client, calls = _fake_client([at_cap])
    monkeypatch.setattr(classifier, "client", client)
    c = classifier.classify_fast("headline", MKT)
    # 0.85 is not capped and clears the 0.80 floor -> passes to Sonnet.
    assert calls["models"] == ["haiku-test", "sonnet-test"]


def test_haiku_error_falls_through_to_sonnet(monkeypatch):
    # First call (Haiku) raises; classify_fast must fail open to the full Sonnet
    # classify() rather than dropping the headline.
    client, calls = _fake_client([RuntimeError("haiku down"), SONNET])
    monkeypatch.setattr(classifier, "client", client)
    c = classifier.classify_fast("headline", MKT)
    assert c.error is False
    assert c.action is None
    assert c.model == "sonnet-test"
    assert calls["models"] == ["haiku-test", "sonnet-test"]


def test_disabled_path_uses_sonnet_directly_via_classify(monkeypatch):
    # When tiered is off the pipeline calls classify() directly; with no model
    # override it uses CLASSIFICATION_MODEL (unchanged behavior, single call).
    monkeypatch.setattr(config, "CLASSIFICATION_MODEL", "default-model")
    client, calls = _fake_client([SONNET])
    monkeypatch.setattr(classifier, "client", client)
    c = classifier.classify("headline", MKT)
    assert c.model == "default-model"
    assert calls["models"] == ["default-model"]
    assert c.action is None
