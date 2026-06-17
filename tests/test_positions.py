"""Position exits: triggers, cooldown, hard stop-loss, decision execution."""
import json
import types
from datetime import datetime, timedelta, timezone

import pytest

from locus import config
from locus.core import positions
from locus.markets.gamma import Market


@pytest.fixture(autouse=True)
def exit_config(monkeypatch):
    monkeypatch.setattr(config, "TAKE_PROFIT_TRIGGER_PCT", 50.0)
    monkeypatch.setattr(config, "REEVAL_LOSS_PCT", -30.0)
    monkeypatch.setattr(config, "STOP_LOSS_PCT", -50.0)
    monkeypatch.setattr(config, "REEVAL_COOLDOWN_HOURS", 6.0)
    monkeypatch.setattr(config, "NEWS_REEVAL_MATERIALITY", 0.4)
    monkeypatch.setattr(config, "TRUTHSOCIAL_REEVAL_MATERIALITY", 0.6)
    monkeypatch.setattr(config, "NEAR_CERTAIN_THRESHOLD", 0.95)


MKT = Market("cond1", "Will X happen?", "ai", 0.50, 0.50, 5000, "", True, [],
             slug="will-x-happen")


def _open(tmp_db, side="YES", entry=0.50, amount=25.0):
    trade_id = tmp_db.log_trade(
        market_id="cond1", market_question="Will X happen?", claude_score=0.7,
        market_price=entry, edge=0.2, side=side, amount_usd=amount,
        status="dry_run", classification="bullish", materiality=0.7,
    )
    positions.open_position(trade_id, MKT, side, amount,
                            headline="orig headline", reasoning="orig reasoning")
    return positions.get_open_positions()[0]


def _fake_claude(monkeypatch, decision="hold", reasoning="r", calls=None):
    calls = calls if calls is not None else []

    def create(**kwargs):
        calls.append(kwargs)
        text = json.dumps({"decision": decision, "reasoning": reasoning})
        return types.SimpleNamespace(content=[types.SimpleNamespace(text=text)])

    fake = types.SimpleNamespace(
        messages=types.SimpleNamespace(create=create)
    )
    monkeypatch.setattr(positions.anthropic, "Anthropic", lambda api_key=None: fake)
    return calls


# --- trigger math ---

def test_pnl_pct_math():
    assert positions.pnl_pct("YES", 0.50, 0.75) == pytest.approx(50.0)
    assert positions.pnl_pct("YES", 0.50, 0.25) == pytest.approx(-50.0)
    assert positions.pnl_pct("NO", 0.80, 0.90) == pytest.approx(-50.0)  # NO entry 0.20 -> 0.10


def test_no_side_pnl_sign():
    # Entry NO @ 0.45 means entry_yes=0.55; current NO @ 0.43 means now_yes=0.57.
    # NO token fell in value -> loss, must be negative.
    assert positions.pnl_pct("NO", 0.55, 0.57) == pytest.approx(-100 * (1 - 0.43 / 0.45))
    assert positions.pnl_pct("NO", 0.55, 0.57) < 0, "NO position losing value must be negative PnL"
    # NO token rose in value (YES dropped 0.55->0.50, NO 0.45->0.50) -> gain
    assert positions.pnl_pct("NO", 0.55, 0.50) > 0, "NO position gaining value must be positive PnL"


def test_no_entry_price_stored_in_yes_terms(tmp_db):
    # entry_yes_price is ALWAYS the YES price, regardless of side: a NO
    # position on a market at YES=0.55 (NO costs 0.45) stores 0.55.
    mkt = Market("cond1", "Will X happen?", "ai", 0.55, 0.45, 5000, "", True, [])
    trade_id = tmp_db.log_trade(
        market_id="cond1", market_question="Will X happen?", claude_score=0.7,
        market_price=0.55, edge=0.2, side="NO", amount_usd=25.0,
        status="dry_run", classification="bearish", materiality=0.7,
    )
    positions.open_position(trade_id, mkt, "NO", 25.0)
    pos = positions.get_open_positions()[0]
    assert pos["entry_yes_price"] == pytest.approx(0.55)
    # NO gains when YES falls: 0.55 -> 0.50 means NO 0.45 -> 0.50, +11.1%
    assert positions.pnl_pct("NO", pos["entry_yes_price"], 0.50) == pytest.approx(11.111, abs=0.01)


def test_check_trigger_thresholds():
    pos = {"id": 1}
    assert positions.check_trigger(pos, 55.0) == "take_profit"
    assert positions.check_trigger(pos, 50.0) == "take_profit"
    assert positions.check_trigger(pos, 10.0) is None
    assert positions.check_trigger(pos, -29.9) is None
    assert positions.check_trigger(pos, -30.0) == "drawdown"
    assert positions.check_trigger(pos, -50.0) == "stop_loss"
    assert positions.check_trigger(pos, -80.0) == "stop_loss"


# --- cooldown ---

def test_cooldown_blocks_same_trigger_within_window():
    now = datetime(2026, 6, 13, 12, 0, tzinfo=timezone.utc)
    pos = {"last_reeval_at": (now - timedelta(hours=2)).isoformat(), "last_trigger": "take_profit"}
    assert positions.cooldown_allows(pos, "take_profit", now) is False
    # different trigger type bypasses the cooldown
    assert positions.cooldown_allows(pos, "news", now) is True
    # window elapsed
    pos["last_reeval_at"] = (now - timedelta(hours=7)).isoformat()
    assert positions.cooldown_allows(pos, "take_profit", now) is True


def test_cooldown_allows_first_reeval():
    assert positions.cooldown_allows({"last_reeval_at": None}, "take_profit") is True


# --- hard stop-loss ---

def test_hard_stop_loss_closes_without_claude(tmp_db, monkeypatch):
    calls = _fake_claude(monkeypatch, decision="hold", calls=[])
    _open(tmp_db, side="YES", entry=0.50)

    stats = positions.update_and_manage({"cond1": 0.20})  # -60% -> hard SL
    assert stats["stop_losses"] == 1
    assert calls == [], "stop-loss must never wait on a model call"

    closed = positions.get_closed_positions()
    assert len(closed) == 1
    assert closed[0]["status"] == "closed_sl"
    assert closed[0]["exit_reason"] == "sl"
    assert closed[0]["realized_pnl_usd"] == pytest.approx(-15.0)  # 25 * (0.2/0.5 - 1)


# --- decision execution paths ---

def test_take_profit_close_decision(tmp_db, monkeypatch):
    calls = _fake_claude(monkeypatch, decision="close", reasoning="edge captured")
    _open(tmp_db, side="YES", entry=0.50)

    stats = positions.update_and_manage({"cond1": 0.80})  # +60% -> take_profit
    assert stats["reevals"] == 1
    assert len(calls) == 1

    closed = positions.get_closed_positions()
    assert closed[0]["status"] == "closed_tp"
    assert closed[0]["exit_reason"] == "tp_decision"
    assert closed[0]["realized_pnl_usd"] == pytest.approx(15.0)

    decisions = positions.get_recent_exit_decisions()
    assert decisions[0]["decision"] == "close"
    assert decisions[0]["reasoning"] == "edge captured"
    assert decisions[0]["trigger"] == "take_profit"


def test_hold_decision_keeps_position_and_sets_cooldown(tmp_db, monkeypatch):
    _fake_claude(monkeypatch, decision="hold", reasoning="thesis intact")
    _open(tmp_db)
    positions.update_and_manage({"cond1": 0.80})

    open_pos = positions.get_open_positions()
    assert len(open_pos) == 1
    assert open_pos[0]["last_trigger"] == "take_profit"
    assert open_pos[0]["last_reeval_at"] is not None

    # same trigger again inside the window: no second Claude call
    calls = _fake_claude(monkeypatch, decision="close", calls=[])
    stats = positions.update_and_manage({"cond1": 0.85})
    assert stats["reevals"] == 0 and calls == []


def test_close_half_realizes_half_keeps_rest(tmp_db, monkeypatch):
    _fake_claude(monkeypatch, decision="close_half", reasoning="bank some")
    _open(tmp_db, amount=25.0)
    positions.update_and_manage({"cond1": 0.80})

    open_pos = positions.get_open_positions()
    assert len(open_pos) == 1
    assert open_pos[0]["amount_usd"] == pytest.approx(12.5)
    assert open_pos[0]["realized_pnl_usd"] == pytest.approx(7.5)  # half of +15


def test_news_trigger_contradiction_only(tmp_db, monkeypatch):
    calls = _fake_claude(monkeypatch, decision="hold", calls=[])
    _open(tmp_db, side="YES")
    positions.update_and_manage({"cond1": 0.55})  # marks the price, no trigger

    # bullish news on a YES position: agrees, no re-eval
    assert positions.trigger_news_reeval("cond1", "bullish", 0.9, "good news") is False
    # bearish but immaterial: no re-eval
    assert positions.trigger_news_reeval("cond1", "bearish", 0.2, "meh") is False
    assert calls == []
    # bearish and material: re-eval fires
    assert positions.trigger_news_reeval("cond1", "bearish", 0.8, "bad news") is True
    assert len(calls) == 1


def test_truthsocial_contra_forces_reeval_bypassing_cooldown(tmp_db, monkeypatch):
    calls = _fake_claude(monkeypatch, decision="hold", calls=[])
    _open(tmp_db, side="YES")
    positions.update_and_manage({"cond1": 0.55})  # marks the price, no trigger

    # First contra-news re-eval fires and sets the news cooldown.
    assert positions.trigger_news_reeval("cond1", "bearish", 0.8, "n1", "rss") is True
    assert len(calls) == 1
    # A second contra-news from a normal source within the window is blocked.
    assert positions.trigger_news_reeval("cond1", "bearish", 0.8, "n2", "rss") is False
    assert len(calls) == 1
    # But a high-materiality Truth Social contra-post forces an immediate re-eval.
    assert positions.trigger_news_reeval(
        "cond1", "bearish", 0.8, "Trump: the deal is off", "truthsocial") is True
    assert len(calls) == 2


def test_truthsocial_below_force_threshold_respects_cooldown(tmp_db, monkeypatch):
    calls = _fake_claude(monkeypatch, decision="hold", calls=[])
    _open(tmp_db, side="YES")
    positions.update_and_manage({"cond1": 0.55})

    assert positions.trigger_news_reeval("cond1", "bearish", 0.8, "n1", "rss") is True
    assert len(calls) == 1
    # Truth Social, but materiality below the 0.6 force floor -> no bypass, cooldown holds.
    assert positions.trigger_news_reeval(
        "cond1", "bearish", 0.5, "weak ts post", "truthsocial") is False
    assert len(calls) == 1


def test_truthsocial_force_still_requires_contradiction(tmp_db, monkeypatch):
    calls = _fake_claude(monkeypatch, decision="hold", calls=[])
    _open(tmp_db, side="YES")
    positions.update_and_manage({"cond1": 0.55})
    # Agreeing direction (bullish on a YES side) never re-evals, even from Truth Social.
    assert positions.trigger_news_reeval(
        "cond1", "bullish", 0.9, "Trump: I will sign it", "truthsocial") is False
    assert calls == []


def test_resolution_close(tmp_db):
    pos = _open(tmp_db, side="YES", entry=0.50)
    positions.close_on_resolution(pos["trade_id"], 1.0)
    closed = positions.get_closed_positions()
    assert closed[0]["status"] == "resolved"
    assert closed[0]["exit_reason"] == "resolution"
    assert closed[0]["realized_pnl_usd"] == pytest.approx(25.0)


# --- manual close (CLI `close <id>`) ---

def test_manual_close_found(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", True)
    pos = _open(tmp_db, side="YES", entry=0.50)
    positions.update_and_manage({"cond1": 0.60})  # mark +20%, no trigger

    result = positions.close_manual(pos["id"])
    assert result is not None
    assert result["id"] == pos["id"]
    assert result["market_question"] == "Will X happen?"
    assert result["side"] == "YES"
    assert result["price"] == pytest.approx(0.60)
    assert result["pnl_pct"] == pytest.approx(20.0)
    assert result["realized"] == pytest.approx(5.0)  # 25 * (0.6/0.5 - 1)

    closed = positions.get_closed_positions()
    assert len(closed) == 1
    assert closed[0]["status"] == "closed_manual"
    assert closed[0]["exit_reason"] == "manual"

    decisions = positions.get_recent_exit_decisions()
    assert decisions[0]["trigger"] == "manual"
    assert decisions[0]["decision"] == "close"
    assert decisions[0]["reasoning"] == "Manual close requested by user"


def test_manual_close_not_found(tmp_db):
    assert positions.close_manual(99999) is None


def test_manual_close_already_closed(tmp_db):
    pos = _open(tmp_db, side="YES", entry=0.50)
    assert positions.close_manual(pos["id"]) is not None
    # A second close on the now-closed position is rejected (no double-close).
    assert positions.close_manual(pos["id"]) is None
    assert len(positions.get_closed_positions()) == 1


def test_manual_close_unmarked_position_uses_entry(tmp_db):
    # Never marked to a live price -> closes at entry, PnL 0.
    pos = _open(tmp_db, side="YES", entry=0.50)
    result = positions.close_manual(pos["id"])
    assert result["price"] == pytest.approx(0.50)
    assert result["pnl_pct"] == pytest.approx(0.0)


# --- near-certain hard exit ---

_NC_NOW = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def test_near_certain_yes_force_closes():
    # No end_date -> unknown close time, still applies (not within the last hour).
    pos = {"side": "YES", "current_yes_price": 0.96, "end_date": None}
    assert positions.check_hard_exit(pos, 50.0, _NC_NOW) == ("force_close", "near_certain_yes_0.96")


def test_near_certain_yes_below_threshold_holds():
    pos = {"side": "YES", "current_yes_price": 0.94, "end_date": None}
    assert positions.check_hard_exit(pos, 50.0, _NC_NOW) is None


def test_near_certain_no_force_closes():
    pos = {"side": "NO", "current_yes_price": 0.04, "end_date": None}
    assert positions.check_hard_exit(pos, 50.0, _NC_NOW) == ("force_close", "near_certain_no_0.04")


def test_near_certain_no_above_threshold_holds():
    pos = {"side": "NO", "current_yes_price": 0.06, "end_date": None}
    assert positions.check_hard_exit(pos, 50.0, _NC_NOW) is None


def test_near_certain_skipped_within_last_hour():
    # YES @ 0.96 but only 0.5h to close -> the market is naturally converging.
    pos = {"side": "YES", "current_yes_price": 0.96, "end_date": "2026-06-15T12:30:00Z"}
    assert positions.check_hard_exit(pos, 50.0, _NC_NOW) is None
    # Comfortably more than an hour out -> the rule applies.
    pos["end_date"] = "2026-06-15T14:00:00Z"  # 2h out
    assert positions.check_hard_exit(pos, 50.0, _NC_NOW) == ("force_close", "near_certain_yes_0.96")


def test_near_certain_periodic_close_without_news(tmp_db):
    # Purely price-driven, no news/classification — mirrors the periodic
    # update_and_manage sweep the pipeline runs every cycle. Near-certain
    # preempts the take-profit re-eval, so no Claude call is made.
    _open(tmp_db, side="YES", entry=0.50)
    stats = positions.update_and_manage({"cond1": 0.97})

    assert stats["near_certain_exits"] == 1
    assert stats["reevals"] == 0
    closed = positions.get_closed_positions()
    assert len(closed) == 1
    assert closed[0]["status"] == "closed_near_certain"
    assert closed[0]["exit_reason"] == "near_certain_yes_0.97"
