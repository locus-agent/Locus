"""Multi-source confirmation for high-materiality signals: the logger lookup
that finds an independent confirming source, and the size-reduction it drives."""
from datetime import datetime, timedelta, timezone

import pytest

from locus import config
from locus.core import pipeline
from locus.core.edge import Signal
from locus.markets.gamma import Market

NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=timezone.utc)
MKT = Market("c1", "Will X happen?", "ai", 0.5, 0.5, 5000, "", True, [])


def _seed(db, source, direction="bullish", materiality=0.5, condition_id="c1", ago_hours=0.5):
    created_at = (NOW - timedelta(hours=ago_hours)).strftime("%Y-%m-%d %H:%M:%S")
    conn = db._conn()
    conn.execute(
        """INSERT INTO classifications
           (market_question, headline, news_source, direction, materiality, action,
            condition_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ("Will X happen?", "prior", source, direction, materiality, "signal",
         condition_id, created_at),
    )
    conn.commit()
    conn.close()


def _since(hours=3):
    return (NOW - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")


# --- logger.has_multi_source_confirmation ------------------------------------

def test_confirmed_by_different_source(tmp_db):
    _seed(tmp_db, source="twitter", direction="bullish", materiality=0.5)
    assert tmp_db.has_multi_source_confirmation("c1", "bullish", _since(), "rss") is True


def test_same_source_does_not_confirm(tmp_db):
    _seed(tmp_db, source="rss", direction="bullish", materiality=0.5)
    assert tmp_db.has_multi_source_confirmation("c1", "bullish", _since(), "rss") is False


def test_other_direction_does_not_confirm(tmp_db):
    _seed(tmp_db, source="twitter", direction="bearish", materiality=0.5)
    assert tmp_db.has_multi_source_confirmation("c1", "bullish", _since(), "rss") is False


def test_low_materiality_does_not_confirm(tmp_db):
    _seed(tmp_db, source="twitter", direction="bullish", materiality=0.30)
    assert tmp_db.has_multi_source_confirmation(
        "c1", "bullish", _since(), "rss", min_materiality=0.35) is False


def test_stale_prior_outside_window_does_not_confirm(tmp_db):
    _seed(tmp_db, source="twitter", direction="bullish", materiality=0.5, ago_hours=5)
    assert tmp_db.has_multi_source_confirmation("c1", "bullish", _since(3), "rss") is False


# --- pipeline.multi_source_adjust (size, not block) --------------------------

def _sig(materiality=0.7, bet=25.0):
    return Signal(market=MKT, claude_score=0.7, market_price=0.5, edge=0.3,
                  side="YES", bet_amount=bet, reasoning="", headlines="h",
                  classification="bullish", materiality=materiality,
                  adjusted_materiality=materiality)


@pytest.fixture(autouse=True)
def _pin_multi_source(monkeypatch):
    monkeypatch.setattr(config, "MULTI_SOURCE_CONFIRM_THRESHOLD", 0.65)
    monkeypatch.setattr(config, "MULTI_SOURCE_SIZE_REDUCTION", 0.30)


def test_high_materiality_confirmed_keeps_full_size():
    s = _sig(materiality=0.7, bet=25.0)
    assert pipeline.multi_source_adjust(s, confirmed=True) == "confirmed"
    assert s.bet_amount == 25.0


def test_high_materiality_unconfirmed_is_reduced():
    s = _sig(materiality=0.7, bet=25.0)
    assert pipeline.multi_source_adjust(s, confirmed=False) == "reduced"
    assert s.bet_amount == pytest.approx(25.0 * 0.70)  # 30% reduction


def test_below_threshold_is_noop():
    # adjusted_materiality 0.5 < 0.65 -> the gate doesn't apply, size untouched.
    s = _sig(materiality=0.5, bet=25.0)
    assert pipeline.multi_source_adjust(s, confirmed=False) == "n/a"
    assert s.bet_amount == 25.0


def test_effective_materiality_falls_back_to_raw():
    # A signal with no adjusted_materiality set (built outside detect_edge_v2)
    # is judged on its raw materiality.
    s = Signal(market=MKT, claude_score=0.7, market_price=0.5, edge=0.3,
               side="YES", bet_amount=25.0, reasoning="", headlines="h",
               classification="bullish", materiality=0.7)
    assert pipeline.effective_materiality(s) == 0.7
    assert pipeline.multi_source_adjust(s, confirmed=False) == "reduced"
