"""Re-entry 2.0: the exit_reason calibration gate (positions.check_reentry_opportunity).

Covers the allow/block lists, the cooldown, the resolution-time floor, the
per-event cap, the materiality floor, and the size-factor reduction."""
import pytest

from locus import config
from locus.core import positions


@pytest.fixture(autouse=True)
def _reentry_defaults(monkeypatch):
    """Pin the Re-entry 2.0 config to its shipped defaults so the suite is
    independent of whatever the developer's .env happens to set."""
    monkeypatch.setattr(config, "REENTRY_ENABLED", True)
    monkeypatch.setattr(config, "REENTRY_ALLOWED_REASONS",
                        ["tp_decision", "manual", "near_certain_yes", "near_certain_no", "resolution"])
    monkeypatch.setattr(config, "REENTRY_BLOCKED_REASONS",
                        ["drawdown_decision", "time_pressure", "news_decision", "already_priced_in", "hard_sl"])
    monkeypatch.setattr(config, "REENTRY_MIN_MATERIALITY", 0.45)
    monkeypatch.setattr(config, "REENTRY_MIN_HOURS", 4)
    monkeypatch.setattr(config, "REENTRY_SIZE_FACTOR", 0.7)
    monkeypatch.setattr(config, "REENTRY_MAX_PER_EVENT", 1)
    monkeypatch.setattr(config, "REENTRY_MIN_HOURS_TO_RESOLUTION", 12)


# Comfortable defaults so each test only has to push the one gate it targets.
PASS = dict(
    materiality=0.6,
    base_size_usd=20.0,
    hours_since_close=10.0,
    hours_to_resolution=48.0,
)


# --- allow / block lists --------------------------------------------------

@pytest.mark.parametrize(
    "exit_reason",
    ["drawdown_decision", "time_pressure", "news_decision", "already_priced_in", "sl"],
)
def test_blocked_reasons_return_none(exit_reason):
    # "sl" normalizes to "hard_sl", which is on the block list.
    assert positions.check_reentry_opportunity(exit_reason, **PASS) is None


@pytest.mark.parametrize(
    "exit_reason",
    ["tp_decision", "manual", "near_certain_yes_0.96", "near_certain_no_0.03", "resolution"],
)
def test_allowed_reasons_pass(exit_reason):
    result = positions.check_reentry_opportunity(exit_reason, **PASS)
    assert result is not None
    assert result["size_usd"] > 0


def test_unknown_reason_blocks():
    # An exit_reason on neither list defaults to blocked.
    assert positions.check_reentry_opportunity("some_new_reason", **PASS) is None


def test_feature_disabled_blocks(monkeypatch):
    monkeypatch.setattr(config, "REENTRY_ENABLED", False)
    assert positions.check_reentry_opportunity("tp_decision", **PASS) is None


# --- size factor ----------------------------------------------------------

def test_size_factor_applied(monkeypatch):
    monkeypatch.setattr(config, "REENTRY_SIZE_FACTOR", 0.7)
    result = positions.check_reentry_opportunity("manual", **{**PASS, "base_size_usd": 20.0})
    assert result["size_usd"] == pytest.approx(14.0)


# --- cooldown (REENTRY_MIN_HOURS) -----------------------------------------

def test_min_hours_cooldown_enforced(monkeypatch):
    monkeypatch.setattr(config, "REENTRY_MIN_HOURS", 4)
    blocked = {**PASS, "hours_since_close": 2.0}
    assert positions.check_reentry_opportunity("manual", **blocked) is None
    # exactly at the threshold is allowed
    ok = {**PASS, "hours_since_close": 4.0}
    assert positions.check_reentry_opportunity("manual", **ok) is not None


# --- resolution-time floor (REENTRY_MIN_HOURS_TO_RESOLUTION) ---------------

def test_min_hours_to_resolution_enforced(monkeypatch):
    monkeypatch.setattr(config, "REENTRY_MIN_HOURS_TO_RESOLUTION", 12)
    blocked = {**PASS, "hours_to_resolution": 6.0}
    assert positions.check_reentry_opportunity("manual", **blocked) is None
    ok = {**PASS, "hours_to_resolution": 12.0}
    assert positions.check_reentry_opportunity("manual", **ok) is not None


def test_unknown_resolution_time_does_not_block():
    # A None close time can't be evaluated, so it doesn't block re-entry.
    args = {**PASS, "hours_to_resolution": None}
    assert positions.check_reentry_opportunity("manual", **args) is not None


# --- materiality floor ----------------------------------------------------

def test_materiality_floor_enforced(monkeypatch):
    monkeypatch.setattr(config, "REENTRY_MIN_MATERIALITY", 0.45)
    blocked = {**PASS, "materiality": 0.4}
    assert positions.check_reentry_opportunity("manual", **blocked) is None
    ok = {**PASS, "materiality": 0.45}
    assert positions.check_reentry_opportunity("manual", **ok) is not None


# --- per-event cap (REENTRY_MAX_PER_EVENT) --------------------------------

def _seed_reentry_trade(conn, event_id):
    conn.execute(
        """INSERT INTO trades (market_id, market_question, claude_score, market_price,
           edge, side, amount_usd, status, edge_type, event_id)
           VALUES ('m1', 'Will e1?', 0.6, 0.4, 0.1, 'YES', 10.0, 'dry_run', 'reentry', ?)""",
        (event_id,),
    )
    conn.commit()


def test_max_per_event_enforced(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "REENTRY_MAX_PER_EVENT", 1)
    conn = tmp_db._conn()
    try:
        # No re-entry yet on this event -> allowed.
        assert positions.check_reentry_opportunity(
            "manual", **PASS, event_id="e1", conn=conn
        ) is not None
        # One re-entry already exists -> capped.
        _seed_reentry_trade(conn, "e1")
        assert positions.check_reentry_opportunity(
            "manual", **PASS, event_id="e1", conn=conn
        ) is None
        # A different event is unaffected.
        assert positions.check_reentry_opportunity(
            "manual", **PASS, event_id="e2", conn=conn
        ) is not None
    finally:
        conn.close()
