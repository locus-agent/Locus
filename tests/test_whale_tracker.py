"""Whale tracking: missed-opportunity detection, cooldown, graceful disable,
and the closing-soon / size / matching guards."""
from datetime import datetime, timedelta, timezone

import pytest

from locus import config
from locus.core import whale_tracker
from locus.markets.gamma import Market

NOW = datetime(2026, 6, 14, 12, 0, 0, tzinfo=timezone.utc)


def mkt(cid, end_hours=48, price=0.3, tokens=None):
    end = (NOW + timedelta(hours=end_hours)).isoformat()
    return Market(cid, f"Will {cid} happen?", "politics", price, round(1 - price, 4),
                  5000, end, True, tokens or [])


def trade(cid="m1", size_usd=5000.0, outcome="Yes", token_id="", ts=None):
    return {
        "wallet": "0xabc", "market_token_id": token_id, "condition_id": cid,
        "side": "BUY", "outcome": outcome, "size_usd": size_usd,
        "timestamp": (ts or NOW).timestamp(), "title": "Some market",
    }


def _seed_classification(db, condition_id, direction, action, ago_hours=0.5):
    created_at = (NOW - timedelta(hours=ago_hours)).strftime("%Y-%m-%d %H:%M:%S")
    conn = db._conn()
    conn.execute(
        """INSERT INTO classifications
           (market_question, headline, news_source, direction, materiality, edge,
            action, condition_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("Will m1 happen?", "h", "rss", direction, 0.5, None, action, condition_id, created_at),
    )
    conn.commit()
    conn.close()


def _find(db, trades, markets):
    conn = db._conn()
    try:
        return whale_tracker.find_missed_opportunities(trades, markets, conn, now=NOW)
    finally:
        conn.close()


# --- graceful disable ----------------------------------------------------

def test_empty_wallet_list_disables_fetch_without_network():
    # No wallets -> returns [] immediately, never touches the network.
    assert whale_tracker.fetch_recent_whale_trades(wallets=[]) == []


def test_empty_wallet_list_via_config(monkeypatch):
    monkeypatch.setattr(config, "WHALE_WALLETS", [])
    assert whale_tracker.fetch_recent_whale_trades() == []


# --- missed opportunity detection ----------------------------------------

def test_missed_when_no_classification(tmp_db):
    missed = _find(tmp_db, [trade("m1")], [mkt("m1")])
    assert len(missed) == 1
    assert missed[0]["market"].condition_id == "m1"


def test_not_missed_when_actionable_classification(tmp_db):
    _seed_classification(tmp_db, "m1", direction="bullish", action="signal")
    assert _find(tmp_db, [trade("m1")], [mkt("m1")]) == []


def test_missed_when_only_neutral_or_skip(tmp_db):
    # Neutral, and a directional-but-skipped read: neither counts as acting on it.
    _seed_classification(tmp_db, "m1", direction="neutral", action="skip")
    _seed_classification(tmp_db, "m1", direction="bullish", action="skip")
    assert len(_find(tmp_db, [trade("m1")], [mkt("m1")])) == 1


def test_old_classification_outside_window_is_still_missed(tmp_db):
    # Actionable, but 5h ago — outside the 2h lookback, so still a miss.
    _seed_classification(tmp_db, "m1", direction="bullish", action="signal", ago_hours=5)
    assert len(_find(tmp_db, [trade("m1")], [mkt("m1")])) == 1


def test_untracked_market_is_ignored(tmp_db):
    assert _find(tmp_db, [trade("ghost")], [mkt("m1")]) == []


def test_match_by_token_id_when_condition_missing(tmp_db):
    market = mkt("m1", tokens=[{"token_id": "tok-1", "outcome": "Yes"}])
    tr = trade(cid="", token_id="tok-1")
    missed = _find(tmp_db, [tr], [market])
    assert len(missed) == 1 and missed[0]["market"].condition_id == "m1"


# --- cooldown ------------------------------------------------------------

def test_cooldown_skips_recent_whale_triggered(tmp_db):
    _seed_classification(tmp_db, "m1", direction="bullish", action="whale_triggered", ago_hours=1)
    assert _find(tmp_db, [trade("m1")], [mkt("m1")]) == []


def test_cooldown_expired_allows_again(tmp_db):
    # whale_triggered 7h ago is past the 6h cooldown.
    _seed_classification(tmp_db, "m1", direction="bullish", action="whale_triggered", ago_hours=7)
    assert len(_find(tmp_db, [trade("m1")], [mkt("m1")])) == 1


# --- closing soon / size / dedup -----------------------------------------

def test_market_closing_soon_is_skipped(tmp_db):
    assert _find(tmp_db, [trade("m1")], [mkt("m1", end_hours=1)]) == []


def test_market_closing_after_window_is_kept(tmp_db):
    assert len(_find(tmp_db, [trade("m1")], [mkt("m1", end_hours=3)])) == 1


def test_min_trade_usd_filters_small_trades(tmp_db):
    conn = tmp_db._conn()
    try:
        missed = whale_tracker.find_missed_opportunities(
            [trade("m1", size_usd=500.0)], [mkt("m1")], conn, now=NOW, min_trade_usd=1000.0
        )
    finally:
        conn.close()
    assert missed == []


def test_one_opportunity_per_market_largest_trade_wins(tmp_db):
    trades = [trade("m1", size_usd=2000.0), trade("m1", size_usd=9000.0)]
    missed = _find(tmp_db, trades, [mkt("m1")])
    assert len(missed) == 1
    assert missed[0]["size_usd"] == 9000.0


# --- helpers -------------------------------------------------------------

def test_whale_headline_includes_market_and_size():
    h = whale_tracker.whale_headline({**trade("m1", size_usd=12345.0), "market": mkt("m1")})
    assert "12,345" in h and "Yes" in h
