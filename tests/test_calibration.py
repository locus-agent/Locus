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


def _log_simple_trade(tmp_db, market_id, classification="bullish"):
    return tmp_db.log_trade(
        market_id=market_id, market_question=f"Q {market_id}?", claude_score=0.8,
        market_price=0.40, edge=0.2, side="YES", amount_usd=10.0, status="dry_run",
        classification=classification, materiality=0.7,
    )


def test_get_recent_trades_unresolved_only_excludes_calibrated(tmp_db):
    t1 = _log_simple_trade(tmp_db, "0x1")
    t2 = _log_simple_trade(tmp_db, "0x2")
    t3 = _log_simple_trade(tmp_db, "0x3")
    # Mark t1 and t3 as resolved (present in the calibration table).
    tmp_db.log_calibration(trade_id=t1, classification="bullish", materiality=0.7,
                           entry_price=0.4, exit_price=1.0, correct=True)
    tmp_db.log_calibration(trade_id=t3, classification="bullish", materiality=0.7,
                           entry_price=0.4, exit_price=0.0, correct=False)

    unresolved = tmp_db.get_recent_trades(limit=2000, unresolved_only=True)
    ids = {t["id"] for t in unresolved}
    assert ids == {t2}                                   # only the uncalibrated trade
    # Without the filter, all three are returned.
    assert len(tmp_db.get_recent_trades(limit=2000)) == 3


def test_check_resolutions_finds_old_trade_buried_under_many_resolved(tmp_db, monkeypatch):
    # An old unresolved trade, then 120 newer trades that are already resolved.
    # unresolved_only must surface only the open trade regardless of how many
    # resolved rows sit on top, and the limit (2000) must accommodate the lot.
    old_id = _log_simple_trade(tmp_db, "0xold")
    for i in range(120):
        tid = _log_simple_trade(tmp_db, f"0xnew{i}")
        tmp_db.log_calibration(trade_id=tid, classification="bullish", materiality=0.7,
                               entry_price=0.4, exit_price=1.0, correct=True)

    # 121 trades total, but only the one open trade is unresolved.
    assert len(tmp_db.get_recent_trades(limit=2000)) == 121
    unresolved = tmp_db.get_recent_trades(limit=2000, unresolved_only=True)
    assert {t["id"] for t in unresolved} == {old_id}

    monkeypatch.setattr(
        calibrator.httpx, "get",
        _fake_gamma({"0xold": [{"closed": True, "outcomePrices": '["1", "0"]'}]}),
    )
    monkeypatch.setattr(calibrator.memory, "record_lesson", lambda *a, **k: None)

    assert check_resolutions() == 1                      # buried trade still resolved
    resolved_ids = {r["trade_id"] for r in tmp_db.get_calibration_with_trades()}
    assert old_id in resolved_ids


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
