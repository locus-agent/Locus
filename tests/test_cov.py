"""Chain-of-Verification (CoV) novelty check: the verify_novelty model call and
the pipeline guard/block decision helpers."""
import json
import types

import pytest

from locus import config
from locus.core import classifier
from locus.core.pipeline import should_verify_novelty, cov_blocks
from locus.markets.gamma import Market

MKT = Market("c1", "Will the merger close by Q3?", "ai", 0.50, 0.50, 5000, "",
             True, [], slug="merger-q3")


@pytest.fixture(autouse=True)
def _pin_cov_config(monkeypatch):
    monkeypatch.setattr(config, "COV_ENABLED", True)
    monkeypatch.setattr(config, "COV_MATERIALITY_THRESHOLD", 0.65)
    monkeypatch.setattr(config, "COV_CONFIDENCE_THRESHOLD", 0.75)
    monkeypatch.setattr(config, "COV_MODEL", "claude-haiku-4-5-20251001")


def _fake_client(monkeypatch, *, text=None, raise_exc=None):
    """Point classifier.client at a fake that returns `text` or raises."""
    def create(**kwargs):
        if raise_exc is not None:
            raise raise_exc
        return types.SimpleNamespace(content=[types.SimpleNamespace(text=text)])
    fake = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    monkeypatch.setattr(classifier, "client", fake)


# --- verify_novelty model call -------------------------------------------

def test_verify_novelty_parses_valid_json(monkeypatch):
    _fake_client(monkeypatch, text=json.dumps({
        "already_priced": True, "confidence": 0.82, "reason": "deal confirmed days ago"
    }))
    out = classifier.verify_novelty("Merger approved", MKT, 0.50)
    assert out == {"already_priced": True, "confidence": 0.82,
                   "reason": "deal confirmed days ago"}


def test_verify_novelty_handles_prose_around_json(monkeypatch):
    _fake_client(monkeypatch, text='Sure: {"already_priced": false, "confidence": 0.9}')
    out = classifier.verify_novelty("Merger approved", MKT, 0.50)
    assert out["already_priced"] is False
    assert out["confidence"] == pytest.approx(0.9)
    assert out["reason"] == ""  # missing field defaults cleanly


def test_verify_novelty_clamps_confidence(monkeypatch):
    _fake_client(monkeypatch, text=json.dumps({"already_priced": True, "confidence": 1.7}))
    out = classifier.verify_novelty("x", MKT, 0.50)
    assert out["confidence"] == 1.0


def test_verify_novelty_fails_open_on_exception(monkeypatch):
    _fake_client(monkeypatch, raise_exc=RuntimeError("api down"))
    assert classifier.verify_novelty("x", MKT, 0.50) is None


def test_verify_novelty_returns_none_on_unparseable(monkeypatch):
    _fake_client(monkeypatch, text="no json here at all")
    assert classifier.verify_novelty("x", MKT, 0.50) is None


# --- should_verify_novelty guard -----------------------------------------

def test_guard_runs_for_high_materiality_mid_price():
    assert should_verify_novelty(0.70, 0.50) is True
    assert should_verify_novelty(0.65, 0.50) is True  # at the threshold


def test_guard_skips_low_materiality():
    assert should_verify_novelty(0.64, 0.50) is False


def test_guard_skips_extreme_prices():
    assert should_verify_novelty(0.90, 0.05) is False   # too low
    assert should_verify_novelty(0.90, 0.95) is False   # too high
    assert should_verify_novelty(0.90, 0.15) is False   # boundary exclusive
    assert should_verify_novelty(0.90, 0.85) is False   # boundary exclusive
    assert should_verify_novelty(0.90, 0.16) is True


def test_guard_disabled_bypasses(monkeypatch):
    monkeypatch.setattr(config, "COV_ENABLED", False)
    assert should_verify_novelty(0.90, 0.50) is False


# --- cov_blocks decision --------------------------------------------------

def test_blocks_high_confidence_already_priced():
    assert cov_blocks({"already_priced": True, "confidence": 0.80}) is True
    assert cov_blocks({"already_priced": True, "confidence": 0.75}) is True  # at threshold


def test_allows_novel_news():
    assert cov_blocks({"already_priced": False, "confidence": 0.99}) is False


def test_low_confidence_passes():
    assert cov_blocks({"already_priced": True, "confidence": 0.74}) is False


def test_none_never_blocks():
    # None = skipped or failed open -> never blocks (fail open).
    assert cov_blocks(None) is False
