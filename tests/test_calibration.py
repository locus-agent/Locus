"""Calibration math and the resolve-each-trade-exactly-once invariant."""
import httpx

from locus.memory.calibrator import grade_direction, check_resolutions
from locus.memory import calibrator


def test_grade_direction():
    assert grade_direction(0.5, 1.0) == "bullish"
    assert grade_direction(0.5, 0.0) == "bearish"
    assert grade_direction(0.5, 0.5) == "neutral"
    assert grade_direction(0.01, 1.0) == "bullish"
    assert grade_direction(0.99, 1.0) == "bullish"


def _fake_gamma(market_payloads):
    """httpx.get replacement returning a canned Gamma response keyed by condition_ids."""
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_get(url, params=None, timeout=None):
        assert "condition_ids" in (params or {}), (
            "calibrator must query Gamma with condition_ids (plural); "
            "the singular form is silently ignored and returns random markets"
        )
        cid = params["condition_ids"]
        return FakeResponse(market_payloads.get(cid, []))

    return fake_get


def test_resolves_once_and_grades_correctly(tmp_db, monkeypatch):
    trade_id = tmp_db.log_trade(
        market_id="0xabc", market_question="Will X happen?",
        claude_score=0.8, market_price=0.40, edge=0.2, side="YES",
        amount_usd=10.0, status="dry_run", classification="bullish",
        materiality=0.7,
    )
    monkeypatch.setattr(
        calibrator.httpx, "get",
        _fake_gamma({"0xabc": [{"closed": True, "outcomePrices": '["1", "0"]'}]}),
    )
    lessons = []
    monkeypatch.setattr(calibrator.memory, "record_lesson", lambda *a, **k: lessons.append(a))

    assert check_resolutions() == 1
    rows = tmp_db.get_calibration_with_trades()
    assert len(rows) == 1
    assert rows[0]["correct"] == 1  # bullish call, market resolved YES from 0.40
    assert not lessons  # correct call -> no lesson

    # Second run must skip the already-graded trade entirely.
    assert check_resolutions() == 0
    assert len(tmp_db.get_calibration_with_trades()) == 1


def test_wrong_call_generates_one_lesson_total(tmp_db, monkeypatch):
    tmp_db.log_trade(
        market_id="0xdef", market_question="Will Y happen?",
        claude_score=0.8, market_price=0.60, edge=0.2, side="YES",
        amount_usd=10.0, status="dry_run", classification="bullish",
        materiality=0.7,
    )
    monkeypatch.setattr(
        calibrator.httpx, "get",
        _fake_gamma({"0xdef": [{"closed": True, "outcomePrices": '["0", "1"]'}]}),
    )
    lessons = []
    monkeypatch.setattr(calibrator.memory, "record_lesson", lambda *a, **k: lessons.append(a))

    assert check_resolutions() == 1
    assert len(lessons) == 1
    check_resolutions()
    check_resolutions()
    assert len(lessons) == 1  # re-runs must not regenerate lessons


def test_open_market_is_left_alone(tmp_db, monkeypatch):
    tmp_db.log_trade(
        market_id="0xopen", market_question="Still open?",
        claude_score=0.8, market_price=0.5, edge=0.2, side="YES",
        amount_usd=10.0, status="dry_run", classification="bullish",
        materiality=0.7,
    )
    monkeypatch.setattr(
        calibrator.httpx, "get",
        _fake_gamma({"0xopen": [{"closed": False, "outcomePrices": '["0.6", "0.4"]'}]}),
    )
    assert check_resolutions() == 0
    assert tmp_db.get_calibration_with_trades() == []
