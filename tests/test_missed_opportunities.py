"""Missed-opportunity tracking: relative-move math, candidate filters, the
top-50 cap, single batched price fetch, resolved-market exclusion, lessons."""
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from locus import config
from locus.memory import calibrator


class _FakeMessages:
    def __init__(self, text):
        self._text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("R", (), {"content": [type("C", (), {"text": self._text})()]})()


class _FakeClient:
    def __init__(self, text):
        self.messages = _FakeMessages(text)


@pytest.fixture(autouse=True)
def _stub_reflection_client(monkeypatch):
    """Stub the Anthropic client so check_missed_opportunities never touches the
    network: _generate_reflection runs for real but the model call is faked."""
    fake = _FakeClient("I saw the signal but held back; I should have trusted it.")
    monkeypatch.setattr(calibrator.memory, "client", fake)
    return fake


def _insert(
    db,
    *,
    action="skip",
    direction="bullish",
    materiality=0.7,
    condition_id="cid",
    yes_price=0.20,
    question=None,
    hours_ago=24,
):
    created = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    conn = sqlite3.connect(db.DB_PATH)
    conn.execute(
        """INSERT INTO classifications
           (market_question, headline, direction, materiality, action,
            condition_id, yes_price, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            question or f"Will {condition_id} happen?",
            "headline",
            direction,
            materiality,
            action,
            condition_id,
            yes_price,
            created,
        ),
    )
    conn.commit()
    conn.close()


def _lessons(db):
    conn = sqlite3.connect(db.DB_PATH)
    rows = [r[0] for r in conn.execute("SELECT lesson FROM lessons ORDER BY id").fetchall()]
    conn.close()
    return rows


def _fake_markets(monkeypatch, prices, closed=None):
    """Stub gamma.fetch_markets_by_condition_ids; record each call's id list."""
    closed = closed or {}
    calls = []

    def fake(ids):
        calls.append(list(ids))
        return {
            cid: {
                "condition_id": cid,
                "yes_price": prices[cid],
                "closed": closed.get(cid, False),
                "question": "",
                "slug": f"{cid}-slug",
            }
            for cid in ids
            if cid in prices
        }

    monkeypatch.setattr(calibrator.gamma, "fetch_markets_by_condition_ids", fake)
    return calls


def test_relative_threshold_bullish_and_bearish(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "MISSED_OPPORTUNITY_ENABLED", True)
    monkeypatch.setattr(config, "MISSED_OPPORTUNITY_THRESHOLD", 0.12)
    # bullish: 0.20 -> 0.24 = +20% (flagged); 0.20 -> 0.21 = +5% (not)
    _insert(tmp_db, direction="bullish", condition_id="bull_hit", yes_price=0.20)
    _insert(tmp_db, direction="bullish", condition_id="bull_miss", yes_price=0.20)
    # bearish: 0.50 -> 0.40 = +20% in our favor (flagged); 0.50 -> 0.49 (not)
    _insert(tmp_db, direction="bearish", condition_id="bear_hit", yes_price=0.50)
    _insert(tmp_db, direction="bearish", condition_id="bear_miss", yes_price=0.50)

    _fake_markets(monkeypatch, {
        "bull_hit": 0.24, "bull_miss": 0.21,
        "bear_hit": 0.40, "bear_miss": 0.49,
    })

    assert calibrator.check_missed_opportunities() == 2
    lessons = _lessons(tmp_db)
    assert any("bullish" in l for l in lessons)
    assert any("bearish" in l for l in lessons)


def test_materiality_filter(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "MISSED_OPPORTUNITY_ENABLED", True)
    monkeypatch.setattr(config, "MISSED_OPPORTUNITY_THRESHOLD", 0.12)
    # Below the 0.35 materiality floor: never a candidate, even though it moved.
    _insert(tmp_db, materiality=0.30, condition_id="weak", yes_price=0.20)
    _fake_markets(monkeypatch, {"weak": 0.30})  # +50% move
    assert calibrator.check_missed_opportunities() == 0
    assert _lessons(tmp_db) == []


def test_min_entry_price_filter(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "MISSED_OPPORTUNITY_ENABLED", True)
    monkeypatch.setattr(config, "MISSED_OPPORTUNITY_THRESHOLD", 0.12)
    monkeypatch.setattr(config, "MISSED_MIN_ENTRY_PRICE", 0.08)
    # Entry below MISSED_MIN_ENTRY_PRICE (dust longshot): excluded by the query.
    _insert(tmp_db, condition_id="dust", yes_price=0.05)
    _fake_markets(monkeypatch, {"dust": 0.50})  # huge move, but filtered out
    assert calibrator.check_missed_opportunities() == 0
    assert _lessons(tmp_db) == []


def test_top_50_cap_and_single_batch_fetch(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "MISSED_OPPORTUNITY_ENABLED", True)
    monkeypatch.setattr(config, "MISSED_OPPORTUNITY_THRESHOLD", 0.12)
    # 60 qualifying candidates; only the top 50 by materiality should be fetched.
    for i in range(60):
        _insert(
            tmp_db,
            materiality=0.40 + i * 0.001,
            condition_id=f"c{i:02d}",
            yes_price=0.20,
        )
    prices = {f"c{i:02d}": 0.20 for i in range(60)}  # flat: no lessons logged
    calls = _fake_markets(monkeypatch, prices)

    assert calibrator.check_missed_opportunities() == 0
    # Exactly one batched fetch, carrying at most the 50-row cap.
    assert len(calls) == 1
    assert len(calls[0]) == 50


def test_resolved_market_excluded(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "MISSED_OPPORTUNITY_ENABLED", True)
    monkeypatch.setattr(config, "MISSED_OPPORTUNITY_THRESHOLD", 0.12)
    _insert(tmp_db, condition_id="resolved", yes_price=0.20)
    _insert(tmp_db, condition_id="gone", yes_price=0.20)
    _insert(tmp_db, condition_id="live", yes_price=0.20)
    # resolved -> closed=True; gone -> absent from Gamma's response entirely.
    _fake_markets(
        monkeypatch,
        {"resolved": 0.90, "live": 0.30},  # both moved hard
        closed={"resolved": True},
    )
    assert calibrator.check_missed_opportunities() == 1
    # Only the "live" market yields a missed-opportunity lesson (resolved/gone
    # are excluded). The skip+high-materiality move also adds a Direct-Evidence
    # pattern lesson for the same market, so filter to the missed-opportunity one.
    missed = [l for l in _lessons(tmp_db) if l.startswith("Missed strong")]
    assert len(missed) == 1
    assert "live" in missed[0]


def test_lesson_format(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "MISSED_OPPORTUNITY_ENABLED", True)
    monkeypatch.setattr(config, "MISSED_OPPORTUNITY_THRESHOLD", 0.12)
    _insert(
        tmp_db,
        direction="bullish",
        materiality=0.70,
        condition_id="ai1",
        yes_price=0.20,
        question="Will Acme launch AI model?",
    )
    _fake_markets(monkeypatch, {"ai1": 0.24})  # +20%

    assert calibrator.check_missed_opportunities() == 1
    lesson = _lessons(tmp_db)[0]
    assert lesson == (
        "Missed strong bullish signal (mat=0.70) on 'Will Acme launch AI model?' "
        "— price moved +20%. Consider lowering min_materiality for ai category."
    )


def test_disabled_is_noop(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "MISSED_OPPORTUNITY_ENABLED", False)
    _insert(tmp_db, condition_id="x", yes_price=0.20)

    def boom(ids):  # must never be reached when disabled
        raise AssertionError("fetch should not run when disabled")

    monkeypatch.setattr(calibrator.gamma, "fetch_markets_by_condition_ids", boom)
    assert calibrator.check_missed_opportunities() == 0


# --- Narrative reflections ---------------------------------------------------

def _missed_rows(db):
    conn = sqlite3.connect(db.DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM lessons WHERE reflection IS NOT NULL ORDER BY id"
    ).fetchall()]
    conn.close()
    return rows


def test_reflection_generated(tmp_db, monkeypatch, _stub_reflection_client):
    """One Haiku call per missed opportunity; the returned text is the reflection."""
    monkeypatch.setattr(config, "MISSED_OPPORTUNITY_ENABLED", True)
    monkeypatch.setattr(config, "MISSED_OPPORTUNITY_THRESHOLD", 0.12)
    _insert(tmp_db, direction="bullish", condition_id="ai1", yes_price=0.20,
            question="Will Acme launch AI model?")
    _fake_markets(monkeypatch, {"ai1": 0.24})  # +20%

    assert calibrator.check_missed_opportunities() == 1
    # Exactly one model call, made with the Haiku model.
    assert len(_stub_reflection_client.messages.calls) == 1
    assert _stub_reflection_client.messages.calls[0]["model"] == config.HAIKU_MODEL


def test_reflection_stored_with_metadata(tmp_db, monkeypatch):
    """Reflection + skip metadata land in the lessons table alongside the lesson."""
    monkeypatch.setattr(config, "MISSED_OPPORTUNITY_ENABLED", True)
    monkeypatch.setattr(config, "MISSED_OPPORTUNITY_THRESHOLD", 0.12)
    _insert(tmp_db, direction="bullish", action="low_materiality", materiality=0.70,
            condition_id="ai1", yes_price=0.20, question="Will Acme launch AI model?")
    _fake_markets(monkeypatch, {"ai1": 0.24})  # +20%

    assert calibrator.check_missed_opportunities() == 1
    rows = _missed_rows(tmp_db)
    assert len(rows) == 1
    r = rows[0]
    assert r["reflection"] == "I saw the signal but held back; I should have trusted it."
    assert r["classification"] == "bullish"
    assert r["materiality"] == 0.70
    assert r["action"] == "low_materiality"
    assert r["pct_move"] == pytest.approx(20.0)
    assert r["slug"] == "ai1-slug"


def test_reflection_exported_shape(tmp_db, monkeypatch):
    """get_missed_opportunity_lessons + _missed_opportunity_row -> dashboard shape."""
    from locus.core import export_status

    monkeypatch.setattr(config, "MISSED_OPPORTUNITY_ENABLED", True)
    monkeypatch.setattr(config, "MISSED_OPPORTUNITY_THRESHOLD", 0.12)
    _insert(tmp_db, direction="bullish", action="skip", materiality=0.70,
            condition_id="ai1", yes_price=0.20, question="Will Acme launch AI model?")
    _fake_markets(monkeypatch, {"ai1": 0.24})  # +20%
    assert calibrator.check_missed_opportunities() == 1

    lessons = tmp_db.get_missed_opportunity_lessons(limit=5)
    assert len(lessons) == 1
    row = export_status._missed_opportunity_row(lessons[0])
    assert row["market_question"] == "Will Acme launch AI model?"
    assert row["slug"] == "ai1-slug"
    assert row["direction"] == "bullish"
    assert row["action"] == "skip"
    assert row["materiality"] == 0.70
    assert row["pct_move"] == pytest.approx(20.0)
    assert row["reflection"] == "I saw the signal but held back; I should have trusted it."
    assert "time" in row
