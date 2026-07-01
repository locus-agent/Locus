"""end_date extraction and backfill.

Some Gamma markets — e.g. the per-candidate outcomes of election "Winner"
events — carry NO market-level endDate at all (only startDate/startDateIso);
the parent event does carry endDate (election day). Positions opened on such
markets stored end_date NULL, which silently disabled the time_pressure /
near-certain exit logic (hours_to_close returns None). Two-layer fix:
_parse_market / fetch_markets_by_condition_ids fall back to the event endDate,
and the management cycle backfills NULL end_dates on open positions.
"""
import pytest

from locus.core import positions
from locus.markets import gamma
from locus.markets.gamma import Market


# A raw Gamma market shaped like the real responses for positions 51/52:
# no market-level endDate key at all, event endDate present.
def _raw_market(cid="0xabc", **overrides):
    m = {
        "conditionId": cid,
        "question": "Will the Republicans win the Texas Senate race in 2026?",
        "outcomePrices": '["0.555", "0.445"]',
        "clobTokenIds": '["111", "222"]',
        "volume": "306482.0",
        "active": True,
        "closed": False,
        "startDate": "2025-10-13T21:28:56.505Z",
        "startDateIso": "2025-10-13",
        "events": [{"id": "57672", "slug": "texas-senate-election-winner",
                    "endDate": "2026-11-03T00:00:00Z"}],
    }
    m.update(overrides)
    return m


# --- parser-level fallback ----------------------------------------------------

def test_end_date_falls_back_to_event_end_date():
    mkt = gamma._parse_market(_raw_market())
    assert mkt is not None
    assert mkt.end_date == "2026-11-03T00:00:00Z"
    assert mkt.question.startswith("Will the Republicans")


def test_market_level_end_date_wins_over_event():
    mkt = gamma._parse_market(_raw_market(endDate="2026-06-30T12:00:00Z"))
    assert mkt.end_date == "2026-06-30T12:00:00Z"


def test_explicit_null_end_date_also_falls_back():
    # An endDate key present but null must fall through to the event too.
    mkt = gamma._parse_market(_raw_market(endDate=None))
    assert mkt.end_date == "2026-11-03T00:00:00Z"


def test_no_end_date_anywhere_is_empty():
    mkt = gamma._parse_market(_raw_market(events=[{"id": "1", "slug": "s"}]))
    assert mkt.end_date == ""


def test_fetch_by_condition_ids_includes_end_date(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return [_raw_market("0xabc"),
                    _raw_market("0xdef", events=[{"id": "1", "slug": "s"}])]

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            return FakeResp()

    monkeypatch.setattr(gamma.httpx, "Client", FakeClient)
    out = gamma.fetch_markets_by_condition_ids(["0xabc", "0xdef"])
    assert out["0xabc"]["end_date"] == "2026-11-03T00:00:00Z"  # event fallback
    assert out["0xdef"]["end_date"] is None                    # nothing anywhere


# --- position backfill ----------------------------------------------------------

def _open_position(tmp_db, cid, end_date=""):
    trade_id = tmp_db.log_trade(
        market_id=cid, market_question=f"Will {cid}?", claude_score=0.7,
        market_price=0.5, edge=0.2, side="YES", amount_usd=25.0,
        status="dry_run", classification="bullish", materiality=0.7,
    )
    mkt = Market(cid, f"Will {cid}?", "politics", 0.5, 0.5, 5000,
                 end_date, True, [])
    positions.open_position(trade_id, mkt, "YES", 25.0)
    return trade_id


def _end_dates(tmp_db):
    conn = tmp_db._conn()
    rows = conn.execute(
        "SELECT condition_id, end_date FROM positions ORDER BY id"
    ).fetchall()
    conn.close()
    return {r["condition_id"]: r["end_date"] for r in rows}


def test_backfill_fills_null_end_dates_only(tmp_db, monkeypatch):
    # c1/c2 opened with no end date (stored NULL); c3 already has one.
    _open_position(tmp_db, "c1")
    _open_position(tmp_db, "c2")
    _open_position(tmp_db, "c3", end_date="2026-08-01T00:00:00Z")
    assert _end_dates(tmp_db)["c1"] is None  # stored NULL at open

    asked = []

    def fake_fetch(ids):
        asked.append(list(ids))
        return {
            "c1": {"condition_id": "c1", "end_date": "2026-11-03T00:00:00Z"},
            "c2": {"condition_id": "c2", "end_date": None},  # Gamma still has none
        }

    monkeypatch.setattr(positions.gamma, "fetch_markets_by_condition_ids", fake_fetch)
    updated = positions.backfill_missing_end_dates()

    assert updated == 1
    # Only the NULL rows were fetched — c3 was never asked about.
    assert asked == [["c1", "c2"]]
    dates = _end_dates(tmp_db)
    assert dates["c1"] == "2026-11-03T00:00:00Z"       # filled
    assert dates["c2"] is None                          # still unknown, retried later
    assert dates["c3"] == "2026-08-01T00:00:00Z"        # untouched


def test_backfill_noop_without_null_rows(tmp_db, monkeypatch):
    _open_position(tmp_db, "c1", end_date="2026-08-01T00:00:00Z")

    def boom(ids):
        raise AssertionError("no fetch should happen when nothing is NULL")

    monkeypatch.setattr(positions.gamma, "fetch_markets_by_condition_ids", boom)
    assert positions.backfill_missing_end_dates() == 0


def test_backfill_skips_closed_positions(tmp_db, monkeypatch):
    tid = _open_position(tmp_db, "c1")
    conn = tmp_db._conn()
    conn.execute("UPDATE positions SET status='closed_manual', "
                 "closed_at='2026-06-30T00:00:00+00:00' WHERE trade_id=?", (tid,))
    conn.commit()
    conn.close()

    def boom(ids):
        raise AssertionError("closed positions must not be re-fetched")

    monkeypatch.setattr(positions.gamma, "fetch_markets_by_condition_ids", boom)
    assert positions.backfill_missing_end_dates() == 0


def test_management_cycle_runs_backfill_and_restores_time_exit_input(tmp_db, monkeypatch):
    # update_and_manage triggers the (throttled) backfill, so the same cycle's
    # hard-exit checks see the date: hours_to_close goes None -> a real number.
    _open_position(tmp_db, "c1")
    monkeypatch.setattr(
        positions.gamma, "fetch_markets_by_condition_ids",
        lambda ids: {"c1": {"condition_id": "c1",
                            "end_date": "2026-11-03T00:00:00Z"}},
    )
    monkeypatch.setattr(positions, "_last_end_date_backfill", 0.0)

    assert positions.hours_to_close(_end_dates(tmp_db)["c1"]) is None  # unprotected
    positions.update_and_manage(prices={"c1": 0.5})
    filled = _end_dates(tmp_db)["c1"]
    assert filled == "2026-11-03T00:00:00Z"
    assert positions.hours_to_close(filled) is not None  # time exits live again


def test_backfill_is_throttled_between_cycles(tmp_db, monkeypatch):
    _open_position(tmp_db, "c1")
    calls = []
    monkeypatch.setattr(
        positions.gamma, "fetch_markets_by_condition_ids",
        lambda ids: calls.append(list(ids)) or {},   # never returns a date
    )
    monkeypatch.setattr(positions, "_last_end_date_backfill", 0.0)

    positions.update_and_manage(prices={})
    positions.update_and_manage(prices={})  # within the throttle window
    assert len(calls) == 1
