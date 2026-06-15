"""Dashboard export: dollar PnL on open positions (current_value_usd / pnl_usd)."""
import pytest

from locus.core import export_status


def _row(side, entry, current, amount=25.0):
    return export_status._open_position_row({
        "opened_at": "2026-06-15T00:00:00+00:00",
        "market_question": "Will X happen?",
        "slug": "will-x-happen",
        "side": side,
        "entry_yes_price": entry,
        "current_yes_price": current,
        "unrealized_pnl_pct": 0.0,
        "amount_usd": amount,
        "edge_type": "news",
        "event_id": None,
    })


def test_dollar_pnl_yes_gain():
    # YES 0.50 -> 0.60: value = 0.60 * 25 / 0.50 = 30, pnl = +5.
    row = _row("YES", 0.50, 0.60)
    assert row["current_value_usd"] == pytest.approx(30.0)
    assert row["pnl_usd"] == pytest.approx(5.0)


def test_dollar_pnl_yes_loss():
    # YES 0.50 -> 0.40: value = 0.40 * 25 / 0.50 = 20, pnl = -5.
    row = _row("YES", 0.50, 0.40)
    assert row["current_value_usd"] == pytest.approx(20.0)
    assert row["pnl_usd"] == pytest.approx(-5.0)


def test_dollar_pnl_no_gain():
    # NO with YES 0.50 -> 0.40 (NO 0.50 -> 0.60):
    # value = (1-0.40) * 25 / (1-0.50) = 30, pnl = +5.
    row = _row("NO", 0.50, 0.40)
    assert row["current_value_usd"] == pytest.approx(30.0)
    assert row["pnl_usd"] == pytest.approx(5.0)


def test_dollar_pnl_no_loss():
    # NO with YES 0.50 -> 0.60 (NO 0.50 -> 0.40):
    # value = (1-0.60) * 25 / (1-0.50) = 20, pnl = -5.
    row = _row("NO", 0.50, 0.60)
    assert row["current_value_usd"] == pytest.approx(20.0)
    assert row["pnl_usd"] == pytest.approx(-5.0)


def test_dollar_pnl_unmarked_values_at_entry():
    # No current price yet -> value equals the stake, PnL 0.
    row = _row("YES", 0.50, None)
    assert row["current_value_usd"] == pytest.approx(25.0)
    assert row["pnl_usd"] == pytest.approx(0.0)


def test_open_position_row_keeps_existing_fields():
    row = _row("YES", 0.50, 0.60)
    assert row["pnl_pct"] == 0.0          # existing percent column preserved
    assert row["amount_usd"] == 25.0
    assert row["side"] == "YES"
    assert row["entry_price"] == 0.50
    assert row["current_price"] == 0.60
