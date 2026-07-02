"""Fill-basis PnL accounting (LOGIC_REVIEW.md findings #4/#5/#6).

Every expectation here is hand-computed. The canonical live fixture:
a $25 bet whose BUY filled 40 shares for $24 (fill price 0.60) while the
cached mid recorded at open was entry_yes_price = 0.50. The legacy math
implied 24/0.50 = 48 shares — the real holding is 40.

Conservation property: over any sequence of half-closes, partial sells, and
a final close, the SUM of realized PnL chunks must equal exactly
(total sell proceeds - total cost paid).
"""
import pytest

from locus import config
from locus.core import executor, performance, positions
from locus.core.performance import position_pnl_basis
from locus.markets.gamma import Market


def _open_live(tmp_db, cid="c1", side="YES", mid=0.50, bet=25.0,
               cost=24.0, shares=40.0):
    """A live-style position: real filled cost/shares recorded at open."""
    trade_id = tmp_db.log_trade(
        market_id=cid, market_question=f"Will {cid} happen?", claude_score=0.7,
        market_price=mid, edge=0.2, side=side, amount_usd=bet,
        status="executed", classification="bullish", materiality=0.7,
    )
    mkt = Market(cid, f"Will {cid} happen?", "ai", mid, round(1 - mid, 4),
                 5000, "", True, [])
    positions.open_position(trade_id, mkt, side, bet,
                            actual_cost_usd=cost, token_count=shares)
    return trade_id


def _open_dry(tmp_db, cid="d1", side="YES", entry=0.50, amount=25.0):
    trade_id = tmp_db.log_trade(
        market_id=cid, market_question=f"Will {cid} happen?", claude_score=0.7,
        market_price=entry, edge=0.2, side=side, amount_usd=amount,
        status="dry_run", classification="bullish", materiality=0.7,
    )
    mkt = Market(cid, f"Will {cid} happen?", "ai", entry, round(1 - entry, 4),
                 5000, "", True, [])
    positions.open_position(trade_id, mkt, side, amount)
    return trade_id


def _row(tmp_db, trade_id):
    conn = tmp_db._conn()
    row = conn.execute(
        "SELECT * FROM positions WHERE trade_id=?", (trade_id,)
    ).fetchone()
    conn.close()
    return dict(row)


def _close(tmp_db, pos, yes_price, fraction=1.0, status="closed_manual",
           reason="manual"):
    conn = tmp_db._conn()
    realized = positions._close(conn, pos, yes_price, status, reason, fraction)
    conn.commit()
    conn.close()
    return realized


def _scripted_seller(monkeypatch, results):
    """Patch executor.close_position_live with scripted results, capturing the
    requested share sizes (to assert later sells size from the reduced count)."""
    requested = []
    queue = list(results)

    def fake_close(condition_id, side, shares, max_spread=None, allow_topup=False):
        requested.append(round(shares, 4))
        return queue.pop(0)

    monkeypatch.setattr(executor, "close_position_live", fake_close)
    return requested


# --- FIX 1: marks and realized PnL on the actual fill basis ------------------

def test_unrealized_pnl_uses_real_share_count():
    # 40 real shares costing $24, marked at 0.70:
    # basis PnL = 40*0.70 - 24 = $4.00. Legacy math implied 48 shares
    # (24/0.50) and claimed 48*0.70 - 24 = $9.60 — a 2.4x overstatement.
    pos = {"side": "YES", "entry_yes_price": 0.50, "amount_usd": 24.0,
           "token_count": 40.0}
    assert position_pnl_basis(pos, 0.70) == pytest.approx(4.0)
    assert positions.pnl_pct_basis(pos, 0.70) == pytest.approx(100 * 4.0 / 24.0)


def test_unrealized_pnl_basis_no_side():
    # NO position: 100 shares of NO costing $25 (NO fill 0.25), entry mid
    # yes=0.80. YES falls to 0.70 -> NO side price 0.30:
    # PnL = 100*0.30 - 25 = $5.00.
    pos = {"side": "NO", "entry_yes_price": 0.80, "amount_usd": 25.0,
           "token_count": 100.0}
    assert position_pnl_basis(pos, 0.70) == pytest.approx(5.0)
    assert positions.pnl_pct_basis(pos, 0.70) == pytest.approx(20.0)


def test_legacy_rows_keep_old_math():
    # No token_count -> the legacy derivation (25/0.50 = 50 implied shares).
    pos = {"side": "YES", "entry_yes_price": 0.50, "amount_usd": 25.0,
           "token_count": None}
    assert position_pnl_basis(pos, 0.70) == pytest.approx(10.0)
    assert positions.pnl_pct_basis(pos, 0.70) == pytest.approx(40.0)


def test_dollars_and_percent_agree():
    pos = {"side": "YES", "entry_yes_price": 0.50, "amount_usd": 24.0,
           "token_count": 40.0}
    for mark in (0.35, 0.50, 0.62, 0.90):
        usd = position_pnl_basis(pos, mark)
        pct = positions.pnl_pct_basis(pos, mark)
        assert pct == pytest.approx(usd / 24.0 * 100.0)


def test_realized_close_matches_unrealized_mark(tmp_db):
    # Unrealized -> realized must not jump at close: a dry-run/settled close at
    # the marked price realizes exactly the last unrealized figure.
    tid = _open_live(tmp_db)
    pos = _row(tmp_db, tid)
    unrealized = position_pnl_basis(pos, 0.70)
    realized = _close(tmp_db, pos, 0.70)
    assert realized == pytest.approx(unrealized) == pytest.approx(4.0)
    assert _row(tmp_db, tid)["realized_pnl_usd"] == pytest.approx(4.0)


def test_compute_performance_unrealized_on_basis(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "PERFORMANCE_START_DATE", "")
    tid = _open_live(tmp_db)
    monkeypatch.setattr(performance, "_fetch_current_yes_price", lambda cid: None)
    perf = performance.compute_performance(current_prices={"c1": 0.70})
    assert perf["unrealized_pnl_usd"] == pytest.approx(4.0)  # not the legacy 9.6
    del tid


def test_live_full_close_realizes_at_sell_fill_price(tmp_db, monkeypatch):
    # Live: the SELL fills 40 sh @ 0.68 (not the 0.70 mark) ->
    # realized = 40*0.68 - 24 = $3.20.
    monkeypatch.setattr(config, "DRY_RUN", False)
    _scripted_seller(monkeypatch, [
        {"status": "executed", "order_id": "S1", "price": 0.68, "shares": 40.0,
         "sold_shares": 40.0, "remaining_shares": 0.0},
    ])
    tid = _open_live(tmp_db)
    realized = _close(tmp_db, _row(tmp_db, tid), 0.70)
    assert realized == pytest.approx(40 * 0.68 - 24.0) == pytest.approx(3.2)


# --- FIX 2: close_half shrinks token_count from the fill ---------------------

def test_live_half_close_reduces_token_count_and_basis(tmp_db, monkeypatch):
    # Half of 40 sh sells (20 @ 0.68): realized = 20*0.68 - 24*(20/40) = $1.60;
    # remainder: 20 sh on a $12 basis. The NEXT close must size from 20.
    monkeypatch.setattr(config, "DRY_RUN", False)
    requested = _scripted_seller(monkeypatch, [
        {"status": "executed", "order_id": "S1", "price": 0.68, "shares": 20.0,
         "sold_shares": 20.0, "remaining_shares": 20.0},
        {"status": "executed", "order_id": "S2", "price": 0.72, "shares": 20.0,
         "sold_shares": 20.0, "remaining_shares": 0.0},
    ])
    tid = _open_live(tmp_db)

    realized_half = _close(tmp_db, _row(tmp_db, tid), 0.70, fraction=0.5)
    assert realized_half == pytest.approx(1.6)
    mid_row = _row(tmp_db, tid)
    assert mid_row["token_count"] == pytest.approx(20.0)
    assert mid_row["amount_usd"] == pytest.approx(12.0)
    assert mid_row["status"] == "open"

    realized_final = _close(tmp_db, mid_row, 0.70)
    assert realized_final == pytest.approx(20 * 0.72 - 12.0) == pytest.approx(2.4)
    # The half-close sold 40*0.5=20; the final close sized from the REDUCED
    # count (20), not the stale original 40.
    assert requested == [20.0, 20.0]

    final = _row(tmp_db, tid)
    assert final["status"] == "closed_manual"
    # Conservation: proceeds (13.60 + 14.40) - cost (24) = 4.00.
    assert final["realized_pnl_usd"] == pytest.approx(4.0)


def test_dry_run_half_close_scales_token_count(tmp_db):
    # Dry-run simulates the fill at the marked price: realized =
    # (40*0.70 - 24) * 0.5 = $2.00; token_count halves alongside amount_usd.
    tid = _open_live(tmp_db)
    realized = _close(tmp_db, _row(tmp_db, tid), 0.70, fraction=0.5)
    assert realized == pytest.approx(2.0)
    row = _row(tmp_db, tid)
    assert row["token_count"] == pytest.approx(20.0)
    assert row["amount_usd"] == pytest.approx(12.0)


# --- FIX 3: partial sells realize the sold chunk at ITS fill price -----------

def test_partial_sell_realizes_sold_chunk_then_remainder(tmp_db, monkeypatch):
    # Full close requested, but only 15 of 40 sh match @ 0.68:
    #   sold fraction = 15/40 = 0.375
    #   realized_1 = 15*0.68 - 24*0.375 = 10.20 - 9.00 = $1.20
    #   remainder: 25 sh on a 24*0.625 = $15.00 basis, still open.
    # Later the remaining 25 sh sell @ 0.72:
    #   realized_2 = 25*0.72 - 15.00 = $3.00.
    monkeypatch.setattr(config, "DRY_RUN", False)
    requested = _scripted_seller(monkeypatch, [
        {"status": "partial_close", "order_id": "P1", "price": 0.68,
         "shares": 15.0, "sold_shares": 15.0, "remaining_shares": 25.0},
        {"status": "executed", "order_id": "S2", "price": 0.72, "shares": 25.0,
         "sold_shares": 25.0, "remaining_shares": 0.0},
    ])
    tid = _open_live(tmp_db)

    realized_1 = _close(tmp_db, _row(tmp_db, tid), 0.70)
    assert realized_1 == pytest.approx(1.2)
    mid_row = _row(tmp_db, tid)
    assert mid_row["status"] == "open"                    # NOT closed
    assert mid_row["token_count"] == pytest.approx(25.0)  # on-chain remainder
    assert mid_row["amount_usd"] == pytest.approx(15.0)   # basis less sold 37.5%
    assert mid_row["realized_pnl_usd"] == pytest.approx(1.2)

    realized_2 = _close(tmp_db, mid_row, 0.70)
    assert realized_2 == pytest.approx(3.0)
    assert requested == [40.0, 25.0]  # second sell sized from the remainder

    final = _row(tmp_db, tid)
    assert final["status"] == "closed_manual"
    # Conservation: proceeds (10.20 + 18.00) - cost (24) = 4.20.
    assert final["realized_pnl_usd"] == pytest.approx(4.2)


# --- Conservation property over a mixed multi-step sequence ------------------

def test_conservation_half_close_then_partial_then_final(tmp_db, monkeypatch):
    # Open: 40 sh for $24. Then:
    #   1) close_half: sells 20 @ 0.68  -> realized 20*0.68 - 12    = +1.60
    #      (remainder 20 sh, $12 basis)
    #   2) full close, PARTIAL fill: 12 of 20 sh @ 0.70
    #      sold fraction 0.6 -> realized 12*0.70 - 12*0.6           = +1.20
    #      (remainder 8 sh, $4.80 basis)
    #   3) final close: 8 sh @ 0.75 -> realized 8*0.75 - 4.80       = +1.20
    # Sum of realized chunks: 1.60 + 1.20 + 1.20 = $4.00.
    # Total proceeds: 13.60 + 8.40 + 6.00 = $28.00; total cost $24.00.
    # Conservation: 28.00 - 24.00 = 4.00 == sum of chunks, exactly.
    monkeypatch.setattr(config, "DRY_RUN", False)
    requested = _scripted_seller(monkeypatch, [
        {"status": "executed", "order_id": "S1", "price": 0.68, "shares": 20.0,
         "sold_shares": 20.0, "remaining_shares": 20.0},
        {"status": "partial_close", "order_id": "P1", "price": 0.70,
         "shares": 12.0, "sold_shares": 12.0, "remaining_shares": 8.0},
        {"status": "executed", "order_id": "S2", "price": 0.75, "shares": 8.0,
         "sold_shares": 8.0, "remaining_shares": 0.0},
    ])
    tid = _open_live(tmp_db)

    chunks = [
        _close(tmp_db, _row(tmp_db, tid), 0.70, fraction=0.5),
        _close(tmp_db, _row(tmp_db, tid), 0.70),
        _close(tmp_db, _row(tmp_db, tid), 0.74),
    ]
    assert chunks == [pytest.approx(1.6), pytest.approx(1.2), pytest.approx(1.2)]

    total_proceeds = 20 * 0.68 + 12 * 0.70 + 8 * 0.75   # 28.00
    total_cost = 24.0
    final = _row(tmp_db, tid)
    assert sum(chunks) == pytest.approx(total_proceeds - total_cost)
    assert final["realized_pnl_usd"] == pytest.approx(total_proceeds - total_cost)
    assert final["status"] == "closed_manual"
    # Each sell was sized from the then-current reduced holding.
    assert requested == [20.0, 20.0, 8.0]


def test_conservation_topup_and_sell(tmp_db, monkeypatch):
    # Top-up-and-sell (dust position, position 54's shape): 3.25 sh costing
    # $0.16 — below the exchange minimums, unsellable on its own. The close
    # tops up 17 sh for $1.02, then sells 20.2 of the combined 20.25 @ 0.05
    # (the 0.05 sh remainder is dust, written off with the close):
    #   total cost = 0.16 + 1.02              = $1.18
    #   proceeds   = 20.2 * 0.05              = $1.01
    #   realized   = 1.01 - 1.18              = -$0.17
    # Conservation: realized == proceeds - total cost INCLUDING the top-up.
    monkeypatch.setattr(config, "DRY_RUN", False)
    requested = _scripted_seller(monkeypatch, [
        {"status": "executed", "order_id": "T1", "price": 0.05, "shares": 20.2,
         "sold_shares": 20.2, "remaining_shares": 0.05,
         "topup_cost_usd": 1.02, "topup_shares": 17.0},
    ])
    tid = _open_live(tmp_db, mid=0.05, bet=2.0, cost=0.16, shares=3.25)

    realized = _close(tmp_db, _row(tmp_db, tid), 0.05)

    total_proceeds = 20.2 * 0.05                       # 1.01
    total_cost = 0.16 + 1.02                           # 1.18
    assert realized == pytest.approx(total_proceeds - total_cost)
    assert realized == pytest.approx(-0.17)
    final = _row(tmp_db, tid)
    assert final["status"] == "closed_manual"
    assert final["realized_pnl_usd"] == pytest.approx(-0.17)
    assert final["amount_usd"] == pytest.approx(1.18)  # basis includes top-up
    # actual_cost_usd tracks the same total real cost (0.16 open + 1.02 top-up)
    assert final["actual_cost_usd"] == pytest.approx(1.18)
    # the SELL request was sized from the dust holding (top-up happens inside)
    assert requested == [3.25]


def test_conservation_topup_survives_failed_sell(tmp_db, monkeypatch):
    # The top-up BUY fills but the follow-up SELL fails: real dollars were
    # spent and real tokens received, so the still-open position must absorb
    # them (basis $0.16 -> $1.18, holding 3.25 -> 20.25 sh) and realize
    # nothing. The retry then sells the topped-up holding:
    #   chunk 1 = $0
    #   chunk 2 = 20.25 * 0.06 - 1.18 = 1.215 - 1.18 = +$0.035
    # Conservation across the sequence: chunks sum to proceeds - total cost.
    monkeypatch.setattr(config, "DRY_RUN", False)
    requested = _scripted_seller(monkeypatch, [
        {"status": "close_failed", "order_id": None, "price": None,
         "shares": None, "error": "SELL not confirmed",
         "topup_cost_usd": 1.02, "topup_shares": 17.0},
        {"status": "executed", "order_id": "S2", "price": 0.06, "shares": 20.25,
         "sold_shares": 20.25, "remaining_shares": 0.0},
    ])
    tid = _open_live(tmp_db, mid=0.05, bet=2.0, cost=0.16, shares=3.25)

    chunk1 = _close(tmp_db, _row(tmp_db, tid), 0.05)
    assert chunk1 == 0.0
    mid_row = _row(tmp_db, tid)
    assert mid_row["status"] == "open"                       # sell failed
    assert mid_row["token_count"] == pytest.approx(20.25)    # tokens absorbed
    assert mid_row["amount_usd"] == pytest.approx(1.18)      # cost absorbed
    assert mid_row["actual_cost_usd"] == pytest.approx(1.18) # in lockstep

    chunk2 = _close(tmp_db, mid_row, 0.06)

    total_proceeds = 20.25 * 0.06                            # 1.215
    total_cost = 0.16 + 1.02                                 # 1.18
    assert chunk2 == pytest.approx(0.035)
    assert chunk1 + chunk2 == pytest.approx(total_proceeds - total_cost)
    final = _row(tmp_db, tid)
    assert final["realized_pnl_usd"] == pytest.approx(total_proceeds - total_cost)
    assert final["status"] == "closed_manual"
    # the retry sized its SELL from the topped-up holding, not the stale 3.25
    assert requested == [3.25, 20.25]


def test_actual_cost_stays_consistent_with_amount(tmp_db, monkeypatch):
    # actual_cost_usd must move in lockstep with amount_usd through every basis
    # change — both describe the real cost of the REMAINING holding: scaled by
    # the sold fraction on a partial sell, increased by a top-up. Position 54's
    # anomaly: actual_cost_usd stayed frozen at the original $7.65 open fill
    # while three partial closes scaled amount_usd to $0.23 and a top-up lifted
    # it to $1.88, so return-on-cost displays divided by a holding that no
    # longer existed.
    monkeypatch.setattr(config, "DRY_RUN", False)
    _scripted_seller(monkeypatch, [
        {"status": "partial_close", "order_id": "P1", "price": 0.68,
         "shares": 15.0, "sold_shares": 15.0, "remaining_shares": 25.0},
        {"status": "close_failed", "order_id": None, "price": None,
         "shares": None, "error": "SELL not confirmed",
         "topup_cost_usd": 1.02, "topup_shares": 17.0},
        {"status": "executed", "order_id": "S2", "price": 0.50, "shares": 42.0,
         "sold_shares": 42.0, "remaining_shares": 0.0},
    ])
    tid = _open_live(tmp_db)  # 40 sh for $24 (amount == actual_cost == 24)

    r1 = _close(tmp_db, _row(tmp_db, tid), 0.70)           # partial: 15 of 40
    row = _row(tmp_db, tid)
    assert row["amount_usd"] == pytest.approx(15.0)         # 24 * 25/40
    assert row["actual_cost_usd"] == pytest.approx(15.0)    # scaled in lockstep

    r2 = _close(tmp_db, row, 0.70)                          # top-up, sell fails
    row = _row(tmp_db, tid)
    assert row["amount_usd"] == pytest.approx(16.02)        # 15 + 1.02
    assert row["actual_cost_usd"] == pytest.approx(16.02)   # in lockstep
    assert row["token_count"] == pytest.approx(42.0)        # 25 + 17

    r3 = _close(tmp_db, row, 0.50)                          # final: 42 @ 0.50
    # conservation incl. the top-up:
    # proceeds (15*0.68 + 42*0.50 = 31.20) - cost (24 + 1.02 = 25.02) = 6.18
    assert r1 + r2 + r3 == pytest.approx((15 * 0.68 + 42 * 0.50) - (24.0 + 1.02))
    assert _row(tmp_db, tid)["status"] == "closed_manual"


def test_conservation_dry_run_parity(tmp_db):
    # Dry-run: $25 at 0.50 -> 50 simulated shares. Half closes at 0.80
    # (realized 25 sh * 0.80 - 12.50 = 7.50), remainder closes at 0.60
    # (realized 25 sh * 0.60 - 12.50 = 2.50). Sum 10.00 == simulated proceeds
    # (20.00 + 15.00) - cost (25.00).
    tid = _open_dry(tmp_db)
    r1 = _close(tmp_db, _row(tmp_db, tid), 0.80, fraction=0.5)
    r2 = _close(tmp_db, _row(tmp_db, tid), 0.60)
    assert r1 == pytest.approx(7.5)
    assert r2 == pytest.approx(2.5)
    assert _row(tmp_db, tid)["realized_pnl_usd"] == pytest.approx(10.0)
    assert r1 + r2 == pytest.approx((25 * 0.80 + 25 * 0.60) - 25.0)
