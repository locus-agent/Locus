"""Passive limit-order entry (core/passive.py, PASSIVE_LIMIT_ENABLED).

The contract under test, in order of importance:
1. Flag OFF (the default): byte-identical behavior — aggressive routing only,
   no pending_orders row is ever written.
2. Routing: flag on routes long-horizon markets passive; short-horizon and
   unknown-end-date markets keep the aggressive path; dry-run never routes
   passive (its simulated instant fill is unchanged).
3. Placement: a GTC BUY inside the spread (bid + improve ticks, never
   crossing), persisted as a pending row — and NEVER passed through
   reconcile_order, whose resting-order CANCEL is the phantom-position
   defense the passive path must avoid.
4. Lifecycle: fill -> position with aggressive-identical accounting; timeout
   -> cancel + release; partial at timeout -> MIN_FILL_USD split; chase-away;
   vanished-order resolution by on-chain balance (both directions of the
   crash-restart reconcile).
"""
import asyncio
import sys
import types
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from locus import config
from locus.core import executor, passive, positions
from locus.core.edge import Signal
from locus.markets.gamma import Market
from locus.memory import logger as db


FAR = "2030-01-01T00:00:00Z"    # ~4y out: clears any passive horizon
SOON = (datetime.now(timezone.utc) + timedelta(hours=10)).strftime(
    "%Y-%m-%dT%H:%M:%SZ")       # 10h out: under the 72h default


def _mkt(end_date=FAR, cid="c1", price=0.42):
    return Market(cid, "Will X happen?", "politics", price,
                  round(1 - price, 4), 5000, end_date, True, [],
                  slug="will-x", event_id="e1")


def _signal(market=None, side="YES", bet=10.0, headline="h1"):
    market = market or _mkt()
    return Signal(market=market, claude_score=0.6, market_price=market.yes_price,
                  edge=0.2, side=side, bet_amount=bet, reasoning="r",
                  headlines=headline, news_source="rss",
                  classification="bullish", materiality=0.6, confidence=0.6)


# --- fake CLOB plumbing for the placement path --------------------------------

def _install_fake_v2(monkeypatch, client_cls):
    class OrderArgs:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class PartialCreateOrderOptions:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    mod = types.ModuleType("py_clob_client_v2")
    mod.ClobClient = client_cls
    mod.OrderArgs = OrderArgs
    mod.PartialCreateOrderOptions = PartialCreateOrderOptions
    mod.OrderType = SimpleNamespace(GTC="GTC", FOK="FOK", FAK="FAK", GTD="GTD")
    mod.Side = SimpleNamespace(BUY="BUY", SELL="SELL")
    mod.BalanceAllowanceParams = lambda **kw: ("params", kw)
    mod.AssetType = SimpleNamespace(COLLATERAL="COLLATERAL", CONDITIONAL="CONDITIONAL")
    mod.OrderPayload = lambda **kw: ("payload", kw)
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", mod)


def _level(p, s):
    return SimpleNamespace(price=str(p), size=str(s))


def _install_placement_clob(monkeypatch, captured, book):
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    monkeypatch.setattr(config, "POLYMARKET_FUNDER_ADDRESS", "")

    class FakeClient:
        def __init__(self, **kw):
            captured["init"] = kw

        def create_or_derive_api_key(self):
            return "creds"

        def set_api_creds(self, creds):
            pass

        def get_market(self, condition_id):
            return {"tokens": [{"token_id": "tok-yes", "outcome": "YES"},
                               {"token_id": "tok-no", "outcome": "NO"}]}

        def get_order_book(self, token_id):
            captured["book_token"] = token_id
            return book

        def get_tick_size(self, token_id):
            return "0.01"

        def get_neg_risk(self, token_id):
            return False

        def create_order(self, order_args, options=None):
            captured.setdefault("orders", []).append(order_args)
            captured["options"] = options
            return "signed"

        def post_order(self, signed, order_type):
            captured["order_type"] = order_type
            return {"orderID": "PASSIVE-1"}

        def get_order(self, order_id):
            # placement must NEVER reconcile (reconcile cancels resting orders)
            captured.setdefault("reconciled", []).append(order_id)
            return {"status": "LIVE", "size_matched": "0"}

    _install_fake_v2(monkeypatch, FakeClient)


def _pending_rows(status=None):
    conn = db._conn()
    sql = "SELECT * FROM pending_orders"
    if status:
        sql += f" WHERE status='{status}'"
    rows = [dict(r) for r in conn.execute(sql).fetchall()]
    conn.close()
    return rows


def _trade_status(trade_id):
    conn = db._conn()
    row = conn.execute("SELECT status FROM trades WHERE id=?", (trade_id,)).fetchone()
    conn.close()
    return row["status"] if row else None


# =============================================================================
# 1. Flag OFF: byte-identical behavior
# =============================================================================

def test_flag_defaults_off():
    # The runtime attribute reflects the developer's .env (and is pinned False
    # by the autouse _pin_passive_defaults fixture anyway), so asserting it
    # proves nothing about the shipped default. Verify the default literal in
    # config.py itself: an environment without the var must parse to off.
    import re
    from pathlib import Path

    src = Path(config.__file__).read_text()
    m = re.search(
        r'PASSIVE_LIMIT_ENABLED = os\.getenv\("PASSIVE_LIMIT_ENABLED", "(\w+)"\)', src
    )
    assert m is not None, "PASSIVE_LIMIT_ENABLED default not found in config.py"
    assert m.group(1) == "false"


def test_flag_off_routes_aggressive_and_writes_no_pending_rows(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "PASSIVE_LIMIT_ENABLED", False)
    monkeypatch.setattr(config, "DRY_RUN", False)
    monkeypatch.setattr(executor, "_execute_live",
                        lambda s: {"status": "aggressive-sentinel"})

    def never(signal):
        raise AssertionError("passive placement must not run with the flag off")

    monkeypatch.setattr(passive, "place_passive_order", never)
    result = executor.execute_trade(_signal(_mkt(end_date=FAR)))
    assert result["status"] == "aggressive-sentinel"
    assert _pending_rows() == []


def test_flag_off_lifecycle_is_a_single_noop_select(tmp_db, monkeypatch):
    # With no pending rows the lifecycle never builds a CLOB client.
    def boom():
        raise AssertionError("no client should be built with nothing pending")

    monkeypatch.setattr(executor, "create_clob_client", boom)
    summary = passive.check_pending_orders()
    assert summary["checked"] == 0
    assert summary["released_headlines"] == []


# =============================================================================
# 2. Routing
# =============================================================================

def test_flag_on_long_horizon_routes_passive(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "PASSIVE_LIMIT_ENABLED", True)
    monkeypatch.setattr(config, "DRY_RUN", False)
    monkeypatch.setattr(passive, "place_passive_order",
                        lambda s: {"status": "passive-sentinel"})
    monkeypatch.setattr(executor, "_execute_live",
                        lambda s: {"status": "aggressive-sentinel"})
    assert executor.execute_trade(_signal(_mkt(end_date=FAR)))["status"] == \
        "passive-sentinel"


@pytest.mark.parametrize("end_date", [SOON, "", None])
def test_flag_on_short_or_unknown_horizon_routes_aggressive(tmp_db, monkeypatch,
                                                            end_date):
    monkeypatch.setattr(config, "PASSIVE_LIMIT_ENABLED", True)
    monkeypatch.setattr(config, "DRY_RUN", False)

    def never(signal):
        raise AssertionError("short/unknown horizon must stay aggressive")

    monkeypatch.setattr(passive, "place_passive_order", never)
    monkeypatch.setattr(executor, "_execute_live",
                        lambda s: {"status": "aggressive-sentinel"})
    m = _mkt(end_date=end_date or "")
    assert executor.execute_trade(_signal(m))["status"] == "aggressive-sentinel"


def test_flag_on_dry_run_is_unchanged(tmp_db, monkeypatch):
    # DRY_RUN=True via conftest: the simulated instant fill at the cached
    # price stands in for the limit fill — no pending rows, no CLOB.
    monkeypatch.setattr(config, "PASSIVE_LIMIT_ENABLED", True)
    result = executor.execute_trade(_signal(_mkt(end_date=FAR)))
    assert result["status"] == "dry_run"
    assert _pending_rows() == []


# =============================================================================
# 3. Placement
# =============================================================================

def _place(monkeypatch, book, bet=10.0, side="YES"):
    monkeypatch.setattr(config, "DRY_RUN", False)
    monkeypatch.setattr(config, "PASSIVE_LIMIT_ENABLED", True)
    captured = {}
    _install_placement_clob(monkeypatch, captured, book)
    result = passive.place_passive_order(_signal(_mkt(end_date=FAR), side=side,
                                                 bet=bet))
    return result, captured


def test_placement_rests_inside_spread_and_persists_pending(tmp_db, monkeypatch):
    # Book bid 0.40 / ask 0.45, tick 0.01, improve 1 -> limit 0.41.
    # $10 / 0.41 = 24.39 sh, snapped down to the CLOB step (1.0 sh) -> 24.
    book = SimpleNamespace(bids=[_level(0.40, 500)], asks=[_level(0.45, 500)])
    result, captured = _place(monkeypatch, book)

    assert result["status"] == "passive_pending"
    assert result["order_id"] == "PASSIVE-1"
    assert result["limit_price"] == pytest.approx(0.41)
    assert result["shares"] == pytest.approx(24.0)

    [order] = captured["orders"]
    assert order.side == "BUY"
    assert order.price == pytest.approx(0.41)
    assert order.size == pytest.approx(24.0)
    assert captured["order_type"] == "GTC"
    # CRITICAL: no reconcile pass — reconcile_order cancels resting orders.
    assert "reconciled" not in captured

    [row] = _pending_rows("pending")
    assert row["order_id"] == "PASSIVE-1"
    assert row["limit_price"] == pytest.approx(0.41)
    assert row["shares"] == pytest.approx(24.0)
    assert row["bet_amount"] == pytest.approx(10.0)
    assert row["entry_yes_price"] == pytest.approx(0.41)  # YES side: limit itself
    assert row["headline"] == "h1"
    assert row["expires_at"] > row["placed_at"]
    assert _trade_status(row["trade_id"]) == "passive_pending"
    # a resting passive order commits capital against the daily spend limit
    assert db.get_daily_pnl() == pytest.approx(-10.0)
    # and no position exists yet
    assert positions.get_open_positions() == []


def test_placement_joins_bid_on_one_tick_spread(tmp_db, monkeypatch):
    # bid 0.40 / ask 0.41: bid+1 tick would BE the ask (crossing = taking);
    # the passive order joins the bid instead.
    book = SimpleNamespace(bids=[_level(0.40, 500)], asks=[_level(0.41, 500)])
    result, captured = _place(monkeypatch, book)
    assert result["status"] == "passive_pending"
    assert result["limit_price"] == pytest.approx(0.40)


def test_placement_below_minimums_skips_without_pending_row(tmp_db, monkeypatch):
    book = SimpleNamespace(bids=[_level(0.40, 500)], asks=[_level(0.45, 500)])
    result, _ = _place(monkeypatch, book, bet=0.50)  # ~1 share, < both minimums
    assert result["status"] == "skipped_thin_book"
    assert _pending_rows() == []


def test_placement_no_bid_falls_back_to_aggressive(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", False)
    monkeypatch.setattr(config, "PASSIVE_LIMIT_ENABLED", True)
    captured = {}
    book = SimpleNamespace(bids=[], asks=[_level(0.45, 500)])
    _install_placement_clob(monkeypatch, captured, book)
    monkeypatch.setattr(executor, "_execute_live",
                        lambda s: {"status": "aggressive-sentinel"})
    result = passive.place_passive_order(_signal(_mkt(end_date=FAR)))
    assert result["status"] == "aggressive-sentinel"
    assert _pending_rows() == []


# =============================================================================
# 4. Lifecycle
# =============================================================================

class LifecycleClient:
    """Fake CLOB client for check_pending_orders: scripted get_order result
    and book; get_market resolves YES/NO tokens."""

    def __init__(self, order="missing", book=None):
        self.order = order
        self.book = book or SimpleNamespace(bids=[_level(0.40, 500)],
                                            asks=[_level(0.45, 500)])

    def get_order(self, order_id):
        return None if self.order == "missing" else self.order

    def get_market(self, condition_id):
        return {"tokens": [{"token_id": "tok-yes", "outcome": "YES"},
                           {"token_id": "tok-no", "outcome": "NO"}]}

    def get_order_book(self, token_id):
        return self.book


def _seed_pending(tmp_db, limit=0.41, shares=24.0, bet=10.0, headline="h1",
                  hours_to_expiry=6.0, side="YES", cid="c1"):
    trade_id = tmp_db.log_trade(
        market_id=cid, market_question="Will X happen?", claude_score=0.6,
        market_price=0.42, edge=0.2, side=side, amount_usd=bet,
        status="passive_pending", classification="bullish", materiality=0.6,
    )
    expires = (datetime.now(timezone.utc)
               + timedelta(hours=hours_to_expiry)).isoformat()
    pending_id = db.insert_pending_order(
        order_id="PASSIVE-1", trade_id=trade_id, condition_id=cid,
        market_question="Will X happen?", slug="will-x", side=side,
        limit_price=limit, shares=shares, bet_amount=bet,
        entry_yes_price=limit if side == "YES" else round(1 - limit, 4),
        headline=headline, reasoning="r", news_source="rss", event_id="e1",
        category="politics", end_date=FAR, expires_at=expires,
    )
    return pending_id, trade_id


def _stub_cancel(monkeypatch):
    cancels = []
    monkeypatch.setattr(executor, "cancel_order_safe",
                        lambda client, oid: cancels.append(oid) or True)
    return cancels


def test_fill_opens_position_with_aggressive_identical_accounting(tmp_db, monkeypatch):
    pending_id, trade_id = _seed_pending(tmp_db)
    cancels = _stub_cancel(monkeypatch)
    client = LifecycleClient(order={"status": "MATCHED", "size_matched": "24"})

    summary = passive.check_pending_orders(client=client)

    assert summary["filled"] == 1
    assert summary["released_headlines"] == []   # fill COMMITS the reservation
    assert cancels == []
    [pos] = positions.get_open_positions()
    assert pos["token_count"] == pytest.approx(24.0)
    assert pos["actual_cost_usd"] == pytest.approx(24 * 0.41)  # real fill basis
    assert pos["amount_usd"] == pytest.approx(24 * 0.41)       # notional = cost
    assert pos["entry_yes_price"] == pytest.approx(0.41)
    assert pos["headline"] == "h1"
    assert _trade_status(trade_id) == "executed"
    [row] = _pending_rows("filled")
    assert row["filled_shares"] == pytest.approx(24.0)
    assert row["filled_cost_usd"] == pytest.approx(9.84)
    # terminal row: a second pass must be a no-op (no double open)
    assert passive.check_pending_orders(client=client)["checked"] == 0
    assert len(positions.get_open_positions()) == 1


def test_timeout_unfilled_cancels_and_releases(tmp_db, monkeypatch):
    pending_id, trade_id = _seed_pending(tmp_db, hours_to_expiry=-0.1)  # expired
    cancels = _stub_cancel(monkeypatch)
    client = LifecycleClient(order={"status": "LIVE", "size_matched": "0"})

    summary = passive.check_pending_orders(client=client)

    assert summary["expired"] == 1
    assert summary["released_headlines"] == ["h1"]   # reservation handed back
    assert cancels == ["PASSIVE-1"]
    assert positions.get_open_positions() == []
    assert _trade_status(trade_id) == "passive_expired"
    assert _pending_rows("expired")


def test_timeout_partial_above_min_fill_opens_the_filled_part(tmp_db, monkeypatch):
    # 10 of 24 sh really filled by expiry (confirmed by on-chain balance):
    # 10 * 0.41 = $4.10 >= MIN_FILL_USD -> cancel the remainder, open the
    # position with the real partial fill.
    monkeypatch.setattr(config, "MIN_FILL_USD", 1.0)
    _seed_pending(tmp_db, hours_to_expiry=-0.1)
    cancels = _stub_cancel(monkeypatch)
    monkeypatch.setattr(executor, "held_token_shares", lambda c, t: 10.0)
    client = LifecycleClient(order={"status": "LIVE", "size_matched": "10"})

    summary = passive.check_pending_orders(client=client)

    assert summary["filled"] == 1
    assert summary["released_headlines"] == []
    assert cancels == ["PASSIVE-1"]
    [pos] = positions.get_open_positions()
    assert pos["token_count"] == pytest.approx(10.0)
    assert pos["actual_cost_usd"] == pytest.approx(4.1)


def test_timeout_partial_matched_report_accepted_without_balance(tmp_db, monkeypatch):
    # Balance unverifiable but the order reports MATCHED (a real match report,
    # not a LIVE echo): the reported size is accepted.
    monkeypatch.setattr(config, "MIN_FILL_USD", 1.0)
    _seed_pending(tmp_db, hours_to_expiry=-0.1)
    _stub_cancel(monkeypatch)
    monkeypatch.setattr(executor, "held_token_shares", lambda c, t: None)
    client = LifecycleClient(order={"status": "MATCHED", "size_matched": "10"})

    summary = passive.check_pending_orders(client=client)

    assert summary["filled"] == 1
    [pos] = positions.get_open_positions()
    assert pos["token_count"] == pytest.approx(10.0)


def test_live_echo_fill_is_never_trusted(tmp_db, monkeypatch):
    # Phantom defense: a LIVE order echoing a "fill" of the full order size,
    # with an unverifiable balance, must NOT open a position or release — the
    # row stays pending until the balance can tell the truth.
    _seed_pending(tmp_db, hours_to_expiry=-0.1)
    cancels = _stub_cancel(monkeypatch)
    monkeypatch.setattr(executor, "held_token_shares", lambda c, t: None)
    client = LifecycleClient(order={"status": "LIVE", "size_matched": "24"})

    summary = passive.check_pending_orders(client=client)

    assert summary["filled"] == 0
    assert summary["expired"] == 0
    assert summary["released_headlines"] == []
    assert cancels == ["PASSIVE-1"]           # the order WAS cancelled
    assert positions.get_open_positions() == []  # but nothing was booked
    assert _pending_rows("pending")           # retried next cycle


def test_timeout_partial_below_min_fill_is_dust(tmp_db, monkeypatch):
    # 2 of 24 sh really filled (balance-confirmed): $0.82 < MIN_FILL_USD ->
    # dust-guard: sell back best-effort, no position, reservation released.
    monkeypatch.setattr(config, "MIN_FILL_USD", 1.0)
    _, trade_id = _seed_pending(tmp_db, hours_to_expiry=-0.1)
    cancels = _stub_cancel(monkeypatch)
    monkeypatch.setattr(executor, "held_token_shares", lambda c, t: 2.0)
    sellbacks = []
    monkeypatch.setattr(
        executor, "close_position_live",
        lambda cid, side, shares, **kw: sellbacks.append((cid, side, shares))
        or {"status": "executed"},
    )
    client = LifecycleClient(order={"status": "LIVE", "size_matched": "2"})

    summary = passive.check_pending_orders(client=client)

    assert summary["dust"] == 1
    assert summary["released_headlines"] == ["h1"]
    assert cancels == ["PASSIVE-1"]
    assert sellbacks == [("c1", "YES", 2.0)]
    assert positions.get_open_positions() == []
    assert _trade_status(trade_id) == "dust_fill"
    assert _pending_rows("dust")


def test_chase_away_cancels_when_ask_runs(tmp_db, monkeypatch):
    # Limit 0.41, ask now 0.50 (+22% > the 10% chase threshold) -> cancel.
    monkeypatch.setattr(config, "PASSIVE_CHASE_AWAY_PCT", 10.0)
    _, trade_id = _seed_pending(tmp_db, hours_to_expiry=6.0)  # NOT expired
    cancels = _stub_cancel(monkeypatch)
    client = LifecycleClient(
        order={"status": "LIVE", "size_matched": "0"},
        book=SimpleNamespace(bids=[_level(0.44, 500)], asks=[_level(0.50, 500)]),
    )

    summary = passive.check_pending_orders(client=client)

    assert summary["chased_away"] == 1
    assert summary["released_headlines"] == ["h1"]
    assert cancels == ["PASSIVE-1"]
    assert _trade_status(trade_id) == "passive_chased_away"
    assert _pending_rows("chased_away")


def test_no_chase_when_ask_within_threshold(tmp_db, monkeypatch):
    # Ask 0.43 is +4.9% over the 0.41 limit — inside the 10% band: keep resting.
    monkeypatch.setattr(config, "PASSIVE_CHASE_AWAY_PCT", 10.0)
    _seed_pending(tmp_db, hours_to_expiry=6.0)
    cancels = _stub_cancel(monkeypatch)
    client = LifecycleClient(
        order={"status": "LIVE", "size_matched": "0"},
        book=SimpleNamespace(bids=[_level(0.40, 500)], asks=[_level(0.43, 500)]),
    )

    summary = passive.check_pending_orders(client=client)

    assert summary == {"checked": 1, "filled": 0, "expired": 0,
                       "chased_away": 0, "dust": 0, "released_headlines": []}
    assert cancels == []
    assert _pending_rows("pending")   # still resting


# --- Telegram wiring -----------------------------------------------------------

def _capture_telegram(monkeypatch):
    """Capture passive-lifecycle Telegram notifications instead of sending."""
    from locus.core import telegram_bot
    calls = {"filled": [], "expired": []}
    monkeypatch.setattr(telegram_bot, "notify_passive_filled",
                        lambda position: calls["filled"].append(position) or True)
    monkeypatch.setattr(
        telegram_bot, "notify_passive_expired",
        lambda q, tokens, reason="expired":
        calls["expired"].append((q, tokens, reason)) or True,
    )
    return calls


def test_fill_sends_passive_fill_notification(tmp_db, monkeypatch):
    _seed_pending(tmp_db)
    calls = _capture_telegram(monkeypatch)
    client = LifecycleClient(order={"status": "MATCHED", "size_matched": "24"})

    passive.check_pending_orders(client=client)

    [note] = calls["filled"]
    assert note["market_question"] == "Will X happen?"
    assert note["side"] == "YES"
    assert note["price"] == pytest.approx(0.41)
    assert note["token_count"] == pytest.approx(24.0)
    assert note["actual_cost_usd"] == pytest.approx(9.84)
    assert calls["expired"] == []


def test_expiry_sends_passive_expired_notification(tmp_db, monkeypatch):
    _seed_pending(tmp_db, hours_to_expiry=-0.1)
    _stub_cancel(monkeypatch)
    calls = _capture_telegram(monkeypatch)
    client = LifecycleClient(order={"status": "LIVE", "size_matched": "0"})

    passive.check_pending_orders(client=client)

    assert calls["expired"] == [("Will X happen?", 0.0, "expired")]
    assert calls["filled"] == []


def test_chase_away_sends_passive_expired_notification(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "PASSIVE_CHASE_AWAY_PCT", 10.0)
    _seed_pending(tmp_db, hours_to_expiry=6.0)
    _stub_cancel(monkeypatch)
    calls = _capture_telegram(monkeypatch)
    client = LifecycleClient(
        order={"status": "LIVE", "size_matched": "0"},
        book=SimpleNamespace(bids=[_level(0.44, 500)], asks=[_level(0.50, 500)]),
    )

    passive.check_pending_orders(client=client)

    assert calls["expired"] == [("Will X happen?", 0.0, "chased_away")]
    assert calls["filled"] == []


# --- crash-restart reconciliation (both directions) ---------------------------

def test_startup_reconcile_opens_fill_that_happened_while_down(tmp_db, monkeypatch):
    # The order vanished from the CLOB and the tokens are held on-chain: it
    # filled while we were down. Open the position late with real fill data.
    _, trade_id = _seed_pending(tmp_db)
    monkeypatch.setattr(executor, "held_token_shares", lambda c, t: 24.0)
    client = LifecycleClient(order="missing")

    summary = passive.reconcile_on_startup(client=client)

    assert summary["filled"] == 1
    assert summary["reserved_headlines"] == []   # nothing left pending
    [pos] = positions.get_open_positions()
    assert pos["token_count"] == pytest.approx(24.0)
    assert pos["actual_cost_usd"] == pytest.approx(9.84)
    assert _trade_status(trade_id) == "executed"


def test_startup_reconcile_releases_vanished_unfilled_order(tmp_db, monkeypatch):
    _, trade_id = _seed_pending(tmp_db)
    monkeypatch.setattr(executor, "held_token_shares", lambda c, t: 0.0)
    client = LifecycleClient(order="missing")

    summary = passive.reconcile_on_startup(client=client)

    assert summary["expired"] == 1
    assert summary["released_headlines"] == ["h1"]
    assert summary["reserved_headlines"] == []
    assert positions.get_open_positions() == []
    assert _trade_status(trade_id) == "passive_expired"


def test_startup_reconcile_keeps_unverifiable_row_pending(tmp_db, monkeypatch):
    # Balance unverifiable: never guess — stay pending, re-reserve the headline.
    _seed_pending(tmp_db)
    monkeypatch.setattr(executor, "held_token_shares", lambda c, t: None)
    client = LifecycleClient(order="missing")

    summary = passive.reconcile_on_startup(client=client)

    assert summary["filled"] == summary["expired"] == 0
    assert summary["reserved_headlines"] == ["h1"]   # headline re-reserved
    assert _pending_rows("pending")


def test_startup_reconcile_rereserves_still_live_orders(tmp_db, monkeypatch):
    # The order still rests on the CLOB inside its window: keep pending, and
    # hand its headline back for re-reservation into the fresh in-memory set.
    _seed_pending(tmp_db, hours_to_expiry=6.0)
    monkeypatch.setattr(config, "PASSIVE_CHASE_AWAY_PCT", 0.0)  # isolate
    client = LifecycleClient(order={"status": "LIVE", "size_matched": "0"})

    summary = passive.reconcile_on_startup(client=client)

    assert summary["reserved_headlines"] == ["h1"]
    assert summary["released_headlines"] == []
    assert _pending_rows("pending")
