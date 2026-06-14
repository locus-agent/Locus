"""Per-category exposure limits: the check, its soft/hard bands, and the
category field flowing from Market through the positions table."""
import pytest

from locus import config
from locus.core import positions
from locus.markets.gamma import Market, _infer_category


@pytest.fixture(autouse=True)
def _pin_limits(monkeypatch):
    """Pin limits so tests don't depend on the developer's .env override."""
    monkeypatch.setattr(config, "MAX_EXPOSURE_PER_CATEGORY", {
        "politics": 75, "crypto": 75, "ai": 50, "technology": 50, "other": 25,
    })
    monkeypatch.setattr(config, "CATEGORY_SOFT_LIMIT_PCT", 0.8)


def _pos(category, amount, question="Q?"):
    return {"category": category, "amount_usd": amount, "market_question": question}


# --- check_category_exposure: bands ---

def test_under_limit_allowed_no_warning():
    # ai limit 50; $25 open -> 50% -> allowed, no warning.
    r = positions.check_category_exposure("ai", [_pos("ai", 25.0)])
    assert r["allowed"] is True
    assert r["warning"] is False
    assert r["current_usd"] == 25.0
    assert r["limit_usd"] == 50.0
    assert r["pct"] == pytest.approx(0.5)


def test_soft_warning_zone():
    # ai limit 50; $42 open -> 84% (>= 80% soft) but <= 100% -> allowed + warning.
    r = positions.check_category_exposure("ai", [_pos("ai", 30.0), _pos("ai", 12.0)])
    assert r["allowed"] is True
    assert r["warning"] is True
    assert r["current_usd"] == 42.0
    assert r["pct"] == pytest.approx(0.84)


def test_exactly_at_limit_is_warning_not_blocked():
    # 100% sits in the soft band: allowed, warned, not blocked.
    r = positions.check_category_exposure("ai", [_pos("ai", 50.0)])
    assert r["allowed"] is True
    assert r["warning"] is True
    assert r["pct"] == pytest.approx(1.0)


def test_hard_limit_blocked():
    # ai limit 50; $55 open -> 110% -> blocked.
    r = positions.check_category_exposure("ai", [_pos("ai", 30.0), _pos("ai", 25.0)])
    assert r["allowed"] is False
    assert r["warning"] is False
    assert r["current_usd"] == 55.0
    assert r["pct"] == pytest.approx(1.1)


def test_only_same_category_counts():
    # crypto limit 75; a big ai position must not count toward crypto.
    book = [_pos("ai", 60.0), _pos("crypto", 25.0)]
    r = positions.check_category_exposure("crypto", book)
    assert r["current_usd"] == 25.0
    assert r["allowed"] is True and r["warning"] is False


def test_unknown_category_uses_other_default():
    # "sports" has no explicit limit -> falls back to 'other' (25).
    r = positions.check_category_exposure("sports", [_pos("sports", 30.0)])
    assert r["limit_usd"] == 25.0
    assert r["allowed"] is False  # 30 > 25


def test_none_category_treated_as_other():
    r = positions.check_category_exposure(None, [])
    assert r["limit_usd"] == 25.0
    assert r["allowed"] is True


def test_legacy_position_without_category_is_inferred():
    # A stored position with no category falls back to question inference.
    book = [{"category": None, "amount_usd": 80.0,
             "market_question": "Will Bitcoin hit $200k in 2026?"}]
    r = positions.check_category_exposure("crypto", book)
    assert r["current_usd"] == 80.0  # inferred crypto
    assert r["allowed"] is False     # 80 > 75


# --- category field flows from Market through the positions table ---

def test_market_category_inferred():
    assert _infer_category("Will Bitcoin hit $200k in 2026?", []) == "crypto"
    assert _infer_category("Will OpenAI release GPT-6?", []) == "ai"
    assert _infer_category("Will it rain in Denver tomorrow?", []) == "other"


def _open(tmp_db, mid, question, category, amount=25.0):
    trade_id = tmp_db.log_trade(
        market_id=mid, market_question=question, claude_score=0.7,
        market_price=0.5, edge=0.2, side="YES", amount_usd=amount,
        status="dry_run", classification="bullish", materiality=0.7,
    )
    mkt = Market(mid, question, category, 0.5, 0.5, 5000, "", True, [])
    positions.open_position(trade_id, mkt, "YES", amount)
    return trade_id


def test_category_persisted_through_open_position(tmp_db):
    q = "Will Bitcoin hit $200k in 2026?"
    _open(tmp_db, "c1", q, _infer_category(q, []))
    op = positions.get_open_positions()
    assert len(op) == 1
    assert op[0]["category"] == "crypto"


def test_check_uses_stored_categories_end_to_end(tmp_db):
    # Two crypto positions ($25 each = $50) vs crypto limit 75 -> allowed.
    _open(tmp_db, "c1", "Will Bitcoin hit $200k?", "crypto")
    _open(tmp_db, "c2", "Will Ethereum flip Bitcoin?", "crypto")
    _open(tmp_db, "a1", "Will OpenAI release GPT-6?", "ai")

    book = positions.get_open_positions()
    crypto = positions.check_category_exposure("crypto", book)
    assert crypto["current_usd"] == 50.0
    assert crypto["allowed"] is True and crypto["warning"] is False

    ai = positions.check_category_exposure("ai", book)
    assert ai["current_usd"] == 25.0
