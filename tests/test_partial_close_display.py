"""close_half partial realizations surfaced in the dashboard's Closed Positions
display (export_status._closed_positions_display)."""
import json
import types

import pytest

from locus import config
from locus.core import positions, export_status
from locus.markets.gamma import Market

MKT = Market("cond1", "Will X happen?", "ai", 0.50, 0.50, 5000, "", True, [],
             slug="will-x-happen")


@pytest.fixture(autouse=True)
def _dry_run(monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", True)
    monkeypatch.setattr(config, "DASHBOARD_POSITIONS_START_DATE", "")


def _open(tmp_db, side="YES", entry=0.50, amount=25.0):
    trade_id = tmp_db.log_trade(
        market_id="cond1", market_question="Will X happen?", claude_score=0.7,
        market_price=entry, edge=0.2, side=side, amount_usd=amount,
        status="dry_run", classification="bullish", materiality=0.7,
    )
    positions.open_position(trade_id, MKT, side, amount,
                            headline="orig headline", reasoning="orig reasoning")
    return positions.get_open_positions()[0]


def _fake_claude(monkeypatch, decision):
    def create(**kwargs):
        text = json.dumps({"decision": decision, "reasoning": "r"})
        return types.SimpleNamespace(content=[types.SimpleNamespace(text=text)])
    fake = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    monkeypatch.setattr(positions.anthropic, "Anthropic", lambda api_key=None: fake)


def _half_close(tmp_db, monkeypatch, yes_price=0.60):
    """Realize half the position at yes_price via the real reevaluate path."""
    pos = _open(tmp_db)
    _fake_claude(monkeypatch, "close_half")
    positions.reevaluate(pos, trigger="news_reeval", yes_price=yes_price)
    return pos


def test_close_half_appears_in_closed_display(tmp_db, monkeypatch):
    _half_close(tmp_db, monkeypatch, yes_price=0.60)  # YES 0.50 -> 0.60 = +20%

    # The position is still open, so the plain closed view is empty...
    assert positions.get_closed_positions() == []
    # ...but the merged display surfaces the partial realization.
    rows = export_status._closed_positions_display(None)
    assert len(rows) == 1
    row = rows[0]
    assert row["partial"] is True
    assert row["exit_reason"] == "½ closed at +20%"
    assert row["market_question"] == "Will X happen?"
    assert row["side"] == "YES"


def test_full_close_after_half_shows_UPD(tmp_db, monkeypatch):
    pos = _half_close(tmp_db, monkeypatch, yes_price=0.60)  # +20%
    # Now fully close the remaining half at the same marked price (+20%).
    positions.close_manual(pos["id"])

    rows = export_status._closed_positions_display(None)
    # Exactly one row for the position — the full-close row, relabeled.
    assert len(rows) == 1
    row = rows[0]
    assert row["partial"] is True
    assert "UPD" in row["exit_reason"]
    assert row["exit_reason"] == "UPD: ½ closed +20% → fully closed +20%"


def test_no_duplication_one_row_per_position(tmp_db, monkeypatch):
    pos = _half_close(tmp_db, monkeypatch, yes_price=0.60)
    positions.close_manual(pos["id"])

    rows = export_status._closed_positions_display(None)
    slugs = [r["slug"] for r in rows]
    # The half-closed-then-fully-closed position is not shown twice.
    assert slugs.count("will-x-happen") == 1


def test_full_close_without_half_is_unaffected(tmp_db, monkeypatch):
    # A position closed outright (no close_half) keeps its plain exit reason and
    # is not flagged partial.
    pos = _open(tmp_db)
    positions.close_manual(pos["id"])

    rows = export_status._closed_positions_display(None)
    assert len(rows) == 1
    row = rows[0]
    assert row["partial"] is False
    assert row["exit_reason"] == "manual"


def test_partial_and_full_close_both_listed(tmp_db, monkeypatch):
    # One position half-closed (still open) and a different one fully closed:
    # both appear, the partial flagged, the full not.
    half_pos = _half_close(tmp_db, monkeypatch, yes_price=0.60)

    trade_id = tmp_db.log_trade(
        market_id="cond2", market_question="Will Y happen?", claude_score=0.7,
        market_price=0.50, edge=0.2, side="YES", amount_usd=25.0,
        status="dry_run", classification="bullish", materiality=0.7,
    )
    other = Market("cond2", "Will Y happen?", "ai", 0.50, 0.50, 5000, "", True, [],
                   slug="will-y-happen")
    positions.open_position(trade_id, other, "YES", 25.0)
    full = positions.get_open_positions()
    full_id = next(p["id"] for p in full if p["slug"] == "will-y-happen")
    positions.close_manual(full_id)

    rows = export_status._closed_positions_display(None)
    by_slug = {r["slug"]: r for r in rows}
    assert by_slug["will-x-happen"]["partial"] is True
    assert by_slug["will-x-happen"]["exit_reason"] == "½ closed at +20%"
    assert by_slug["will-y-happen"]["partial"] is False
