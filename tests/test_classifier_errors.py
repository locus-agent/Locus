"""classify() must retry once, then flag failures instead of faking neutral."""
import types

import pytest

from locus.core import classifier
from locus.markets.gamma import Market

MKT = Market("c1", "Will X happen?", "ai", 0.5, 0.5, 5000, "2026-12-31", True, [])


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(classifier.time, "sleep", lambda s: None)


def _response(text):
    return types.SimpleNamespace(content=[types.SimpleNamespace(text=text)])


def _fake_client(behaviors):
    """behaviors: list per call — an Exception to raise or a str to return."""
    calls = {"n": 0}

    def create(**kwargs):
        b = behaviors[min(calls["n"], len(behaviors) - 1)]
        calls["n"] += 1
        if isinstance(b, Exception):
            raise b
        return _response(b)

    client = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    return client, calls


GOOD = '{"direction": "bullish", "materiality": 0.7, "reasoning": "r"}'


def test_success_has_no_error_flag(monkeypatch):
    client, calls = _fake_client([GOOD])
    monkeypatch.setattr(classifier, "client", client)
    c = classifier.classify("headline", MKT)
    assert (c.direction, c.materiality, c.error) == ("bullish", 0.7, False)
    assert calls["n"] == 1


def test_transient_failure_recovers_on_retry(monkeypatch):
    client, calls = _fake_client([RuntimeError("api down"), GOOD])
    monkeypatch.setattr(classifier, "client", client)
    c = classifier.classify("headline", MKT)
    assert c.error is False and c.direction == "bullish"
    assert calls["n"] == 2  # exactly one retry


def test_persistent_failure_is_flagged_not_neutralized(monkeypatch):
    client, calls = _fake_client([RuntimeError("api down")])
    monkeypatch.setattr(classifier, "client", client)
    c = classifier.classify("headline", MKT)
    assert c.error is True
    assert c.direction == "neutral" and c.materiality == 0.0
    assert "RuntimeError" in c.reasoning
    assert calls["n"] == 2  # one retry max — no storm


def test_garbage_response_falls_back_to_neutral(monkeypatch):
    # The universal parser never raises: an unparseable response degrades to a
    # neutral (non-error) classification rather than being retried as an outage.
    client, calls = _fake_client(["not json at all"])
    monkeypatch.setattr(classifier, "client", client)
    c = classifier.classify("headline", MKT)
    assert c.error is False
    assert c.direction == "neutral" and c.materiality == 0.0 and c.confidence == 0.5
    assert calls["n"] == 1  # no retry — parsing succeeded (as a fallback)
