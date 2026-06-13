"""classify() parses Claude's confidence (win probability) and clamps it to [0.5, 1.0]."""
import types

import pytest

from locus.core import classifier
from locus.markets.gamma import Market

MKT = Market("c1", "Will X happen?", "ai", 0.5, 0.5, 5000, "2026-12-31", True, [])


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(classifier.time, "sleep", lambda s: None)


def _client(text):
    def create(**kwargs):
        return types.SimpleNamespace(content=[types.SimpleNamespace(text=text)])
    return types.SimpleNamespace(messages=types.SimpleNamespace(create=create))


def _set(monkeypatch, text):
    monkeypatch.setattr(classifier, "client", _client(text))


def test_parses_confidence(monkeypatch):
    _set(monkeypatch, '{"direction":"bullish","materiality":0.7,"confidence":0.82,"reasoning":"r"}')
    assert classifier.classify("h", MKT).confidence == 0.82


def test_missing_confidence_defaults_to_half(monkeypatch):
    _set(monkeypatch, '{"direction":"bullish","materiality":0.7,"reasoning":"r"}')
    assert classifier.classify("h", MKT).confidence == 0.5


def test_confidence_above_one_clamped_down(monkeypatch):
    _set(monkeypatch, '{"direction":"bullish","materiality":0.7,"confidence":1.5,"reasoning":"r"}')
    assert classifier.classify("h", MKT).confidence == 1.0


def test_confidence_below_half_clamped_up(monkeypatch):
    # "less than a coin flip in your own prediction" is incoherent -> floor 0.5.
    _set(monkeypatch, '{"direction":"bearish","materiality":0.7,"confidence":0.2,"reasoning":"r"}')
    assert classifier.classify("h", MKT).confidence == 0.5


def test_error_classification_has_default_confidence(monkeypatch):
    def create(**kwargs):
        raise RuntimeError("api down")
    monkeypatch.setattr(classifier, "client",
                        types.SimpleNamespace(messages=types.SimpleNamespace(create=create)))
    c = classifier.classify("h", MKT)
    assert c.error is True and c.confidence == 0.5
