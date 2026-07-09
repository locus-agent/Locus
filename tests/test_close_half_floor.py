"""Anti-dust close_half floor (CLOSE_HALF_MIN_REMAINDER_USD).

Repeated close_half decisions halve the REMAINDER geometrically (position 52:
43 -> 32.5 -> 16.5 -> 8.5 -> 3.5 tokens) until the leftover is below the
exchange minimums and can never be sold. The gate: before executing a
close_half, the post-sale remainder value must clear the floor AND the
exchange minimums, or the decision escalates to a FULL close. The exit prompt
also carries the split history (how many times the position was already
half-closed) so the model can choose a full close itself.
"""
import json
import types

import pytest

from locus import config
from locus.core import positions


@pytest.fixture(autouse=True)
def _floor_defaults(monkeypatch):
    """Pin the floor and exchange minimums to shipped defaults so the tests
    are hermetic against a developer .env."""
    monkeypatch.setattr(config, "CLOSE_HALF_MIN_REMAINDER_USD", 5.00)
    monkeypatch.setattr(config, "MIN_ORDER_SHARES", 5.0)
    monkeypatch.setattr(config, "MIN_ORDER_USD", 1.0)


def _open(amount, tokens, yes=0.5, side="YES"):
    market = types.SimpleNamespace(
        condition_id="c1", question="Will X happen?", slug="", yes_price=yes,
        event_id="", category="crypto", end_date="", volume=5000,
    )
    positions.open_position(1, market, side, amount, token_count=tokens)
    return positions.get_open_positions()[0]


def _fake_claude(monkeypatch, decision="close_half", prompts=None):
    def create(**kwargs):
        if prompts is not None:
            prompts.append(kwargs["messages"][0]["content"])
        text = json.dumps({"decision": decision, "reasoning": "thesis eroding"})
        return types.SimpleNamespace(content=[types.SimpleNamespace(text=text)])

    fake = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    monkeypatch.setattr(positions.anthropic, "Anthropic", lambda api_key=None: fake)


def _last_decision(tmp_db):
    conn = tmp_db._conn()
    row = conn.execute(
        "SELECT decision, reasoning FROM exit_decisions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return dict(row)


# --- resolve_close_half unit behavior -----------------------------------------

def test_remainder_below_floor_escalates(tmp_db):
    # 16 tokens @ 0.50 = $8 held; the remainder after a half would be $4 < $5.
    pos = _open(amount=8.0, tokens=16.0)
    action, why = positions.resolve_close_half(pos, 0.5)
    assert action == "close"
    assert "remainder would be $4.00 < $5.00 floor" in why


def test_remainder_above_floor_proceeds_as_half(tmp_db):
    # 80 tokens @ 0.50 = $40 held; remainder $20 clears the floor.
    pos = _open(amount=40.0, tokens=80.0)
    assert positions.resolve_close_half(pos, 0.5) == ("close_half", None)


def test_current_value_below_floor_escalates(tmp_db):
    # Already worth $3 — a half-close of a sub-floor position is a full close.
    pos = _open(amount=3.0, tokens=6.0)
    action, why = positions.resolve_close_half(pos, 0.5)
    assert action == "close"
    assert "already below" in why


def test_subminimum_remainder_escalates_even_without_floor(tmp_db, monkeypatch):
    # Hard rule: even with the value floor disabled, a remainder that cannot
    # clear the exchange minimums (5 shares AND $1) forces a full close.
    monkeypatch.setattr(config, "CLOSE_HALF_MIN_REMAINDER_USD", 0.0)
    pos = _open(amount=4.0, tokens=8.0)  # remainder: 4 sh < 5 minimum
    action, why = positions.resolve_close_half(pos, 0.5)
    assert action == "close"
    assert "exchange minimums" in why


# --- reevaluate: the escalation end-to-end -------------------------------------

def test_reeval_close_half_escalates_to_full_close(tmp_db, monkeypatch):
    pos = _open(amount=8.0, tokens=16.0)
    _fake_claude(monkeypatch, "close_half")

    result = positions.reevaluate(pos, "take_profit", yes_price=0.5)

    # Claude said close_half, but the executed (and recorded) action is the
    # escalated full close.
    assert result["decision"] == "close"
    assert positions.get_open_positions() == []
    conn = tmp_db._conn()
    row = conn.execute("SELECT status, exit_reason FROM positions").fetchone()
    conn.close()
    assert row["status"] == "closed_tp"
    assert row["exit_reason"] == "tp_decision"
    last = _last_decision(tmp_db)
    assert last["decision"] == "close"
    assert "close_half escalated to full close" in last["reasoning"]
    assert "$4.00 < $5.00 floor" in last["reasoning"]


def test_reeval_healthy_close_half_unchanged(tmp_db, monkeypatch):
    # A remainder comfortably above the floor half-closes exactly as before.
    pos = _open(amount=40.0, tokens=80.0)
    _fake_claude(monkeypatch, "close_half")

    positions.reevaluate(pos, "take_profit", yes_price=0.6)

    p = positions.get_open_positions()[0]
    assert p["token_count"] == pytest.approx(40.0)
    assert p["amount_usd"] == pytest.approx(20.0)
    last = _last_decision(tmp_db)
    assert last["decision"] == "close_half"
    assert "escalated" not in last["reasoning"]


def test_reeval_position_already_below_floor_becomes_full_close(tmp_db, monkeypatch):
    pos = _open(amount=3.0, tokens=6.0)
    _fake_claude(monkeypatch, "close_half")

    positions.reevaluate(pos, "drawdown", yes_price=0.5)

    assert positions.get_open_positions() == []
    last = _last_decision(tmp_db)
    assert last["decision"] == "close"
    assert "already below" in last["reasoning"]


# --- Split history in the exit prompt ------------------------------------------

def test_split_count_appears_in_exit_prompt(tmp_db, monkeypatch):
    pos = _open(amount=40.0, tokens=80.0)
    conn = tmp_db._conn()
    for _ in range(2):
        conn.execute(
            """INSERT INTO exit_decisions
               (position_id, trigger, decision, reasoning, pnl_pct, yes_price)
               VALUES (?, 'news', 'close_half', 'r', 0, 0.5)""",
            (pos["id"],),
        )
    conn.commit()
    conn.close()

    prompts = []
    _fake_claude(monkeypatch, "hold", prompts=prompts)
    positions.reevaluate(pos, "take_profit", yes_price=0.6)

    assert prompts, "Claude was not called"
    assert "already been half-closed 2 times on this developing story" in prompts[0]
    # Guidance travels with the fact — the choice itself stays with the model.
    assert "full close is usually correct" in prompts[0]


def test_no_split_history_section_on_fresh_position(tmp_db, monkeypatch):
    pos = _open(amount=40.0, tokens=80.0)
    prompts = []
    _fake_claude(monkeypatch, "hold", prompts=prompts)
    positions.reevaluate(pos, "take_profit", yes_price=0.6)
    assert "Split history" not in prompts[0]
