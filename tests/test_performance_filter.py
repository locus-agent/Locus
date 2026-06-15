"""PERFORMANCE_START_DATE: a display-only window for the dashboard performance
panel. When set, compute_performance() counts only positions opened on or after
the date; unset, it counts everything. The circuit breaker / calibration /
dynamic-Kelly paths don't go through compute_performance, so they're unaffected."""
import pytest

from locus import config
from locus.core import positions
from locus.core.performance import compute_performance
from locus.markets.gamma import Market


OLD = "2026-06-10 09:00:00"
NEW = "2026-06-14 09:00:00"
CUTOFF = "2026-06-14"

# Live mark for each open position (avoids any Gamma fetch in the test).
PRICES = {"oldO": 0.60, "newO": 0.60}


def _make_position(tmp_db, cond, opened_at, side="YES", entry=0.5, amount=25.0,
                   closed=False, realized=0.0):
    trade_id = tmp_db.log_trade(
        market_id=cond, market_question=f"Q {cond}?", claude_score=0.7,
        market_price=entry, edge=0.2, side=side, amount_usd=amount,
        status="dry_run", classification="bullish", materiality=0.7,
    )
    mkt = Market(cond, f"Q {cond}?", "ai", entry, round(1 - entry, 4),
                 5000, "", True, [])
    positions.open_position(trade_id, mkt, side, amount)
    conn = tmp_db._conn()
    if closed:
        conn.execute(
            "UPDATE positions SET opened_at=?, status='closed_manual', closed_at=?, "
            "exit_yes_price=?, exit_reason='manual', realized_pnl_usd=? WHERE trade_id=?",
            (opened_at, opened_at, entry, realized, trade_id),
        )
    else:
        conn.execute("UPDATE positions SET opened_at=? WHERE trade_id=?",
                     (opened_at, trade_id))
    conn.commit()
    conn.close()


def _setup(tmp_db):
    # Two closed (one old +$10, one new +$5) and two open (old $20, new $30).
    _make_position(tmp_db, "oldC", OLD, amount=25.0, closed=True, realized=10.0)
    _make_position(tmp_db, "newC", NEW, amount=25.0, closed=True, realized=5.0)
    _make_position(tmp_db, "oldO", OLD, amount=20.0)
    _make_position(tmp_db, "newO", NEW, amount=30.0)


def test_performance_filtered_by_date(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "PERFORMANCE_START_DATE", CUTOFF)
    _setup(tmp_db)
    perf = compute_performance(current_prices=PRICES)

    # Only the NEW positions count.
    assert perf["closed_count"] == 1
    assert perf["realized_pnl_usd"] == pytest.approx(5.0)
    assert perf["wins"] == 1 and perf["losses"] == 0
    assert perf["win_rate_pct"] == pytest.approx(100.0)
    assert perf["open_count"] == 1
    assert perf["unrealized_pnl_usd"] == pytest.approx(6.0)   # newO: 30 * (0.6/0.5 - 1)
    assert perf["deployed_usd"] == pytest.approx(55.0)        # newC 25 + newO 30
    assert perf["trades_total"] == 2


def test_performance_unfiltered_counts_all(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "PERFORMANCE_START_DATE", "")
    _setup(tmp_db)
    perf = compute_performance(current_prices=PRICES)

    assert perf["closed_count"] == 2
    assert perf["realized_pnl_usd"] == pytest.approx(15.0)    # 10 + 5
    assert perf["wins"] == 2 and perf["losses"] == 0
    assert perf["open_count"] == 2
    assert perf["unrealized_pnl_usd"] == pytest.approx(10.0)  # oldO 4 + newO 6
    assert perf["deployed_usd"] == pytest.approx(100.0)       # 25+25+20+30
    assert perf["trades_total"] == 4


def test_future_date_excludes_everything(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "PERFORMANCE_START_DATE", "2030-01-01")
    _setup(tmp_db)
    perf = compute_performance(current_prices=PRICES)
    assert perf["closed_count"] == 0
    assert perf["open_count"] == 0
    assert perf["trades_total"] == 0
    assert perf["deployed_usd"] == pytest.approx(0.0)
    assert perf["realized_pnl_usd"] == pytest.approx(0.0)
    assert perf["win_rate_pct"] is None     # no closed positions -> N/A
