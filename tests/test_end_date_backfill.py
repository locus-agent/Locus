"""end_date extraction, backfill, and stale-date refresh.

Some Gamma markets — e.g. the per-candidate outcomes of election "Winner"
events — carry NO market-level endDate at all (only startDate/startDateIso);
the parent event does carry endDate (election day). Positions opened on such
markets stored end_date NULL, which silently disabled the time_pressure /
near-certain exit logic (hours_to_close returns None). Two-layer fix:
_parse_market / fetch_markets_by_condition_ids fall back to the event endDate,
and the management cycle backfills NULL end_dates on open positions.

The same hourly pass also REFRESHES stale-but-populated dates: Polymarket uses
placeholder end dates and moves them (position 54 carried a past end_date while
Gamma reported the market open), and a stale past date fires a wrong
time_pressure trigger every cycle. A past end_date is replaced only when Gamma
now reports a LATER one; when Gamma agrees it's past, the date stands and
time_pressure keeps applying.
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
    # c1/c2 opened with no end date (stored NULL); c3 already has a future one.
    _open_position(tmp_db, "c1")
    _open_position(tmp_db, "c2")
    _open_position(tmp_db, "c3", end_date="2030-08-01T00:00:00Z")
    assert _end_dates(tmp_db)["c1"] is None  # stored NULL at open

    asked = []

    def fake_fetch(ids):
        asked.append(list(ids))
        return {
            "c1": {"condition_id": "c1", "end_date": "2026-11-03T00:00:00Z"},
            "c2": {"condition_id": "c2", "end_date": None},  # Gamma still has none
        }

    monkeypatch.setattr(positions.gamma, "fetch_markets_by_condition_ids", fake_fetch)
    updated = positions.refresh_end_dates()

    assert updated == 1
    # Only the NULL rows were fetched — c3 (future date) was never asked about.
    assert asked == [["c1", "c2"]]
    dates = _end_dates(tmp_db)
    assert dates["c1"] == "2026-11-03T00:00:00Z"       # filled
    assert dates["c2"] is None                          # still unknown, retried later
    assert dates["c3"] == "2030-08-01T00:00:00Z"        # untouched


def test_refresh_noop_without_null_or_stale_rows(tmp_db, monkeypatch):
    _open_position(tmp_db, "c1", end_date="2030-08-01T00:00:00Z")

    def boom(ids):
        raise AssertionError("no fetch should happen when nothing is NULL or stale")

    monkeypatch.setattr(positions.gamma, "fetch_markets_by_condition_ids", boom)
    assert positions.refresh_end_dates() == 0


def test_refresh_skips_closed_positions(tmp_db, monkeypatch):
    tid = _open_position(tmp_db, "c1")
    conn = tmp_db._conn()
    conn.execute("UPDATE positions SET status='closed_manual', "
                 "closed_at='2026-06-30T00:00:00+00:00' WHERE trade_id=?", (tid,))
    conn.commit()
    conn.close()

    def boom(ids):
        raise AssertionError("closed positions must not be re-fetched")

    monkeypatch.setattr(positions.gamma, "fetch_markets_by_condition_ids", boom)
    assert positions.refresh_end_dates() == 0


# --- stale (past) end_date refresh ---------------------------------------------

def test_stale_past_date_refreshed_when_gamma_reports_later(tmp_db, monkeypatch):
    # Position 54's shape: stored end_date is in the past, but the market is
    # alive on Gamma with a moved (later) placeholder date — take Gamma's.
    _open_position(tmp_db, "c1", end_date="2026-06-30T12:00:00Z")  # past
    monkeypatch.setattr(
        positions.gamma, "fetch_markets_by_condition_ids",
        lambda ids: {"c1": {"condition_id": "c1", "closed": False,
                            "end_date": "2030-09-30T12:00:00Z"}},
    )
    assert positions.refresh_end_dates() == 1
    assert _end_dates(tmp_db)["c1"] == "2030-09-30T12:00:00Z"


def test_stale_past_date_left_when_gamma_agrees(tmp_db, monkeypatch, caplog):
    # Gamma returns the SAME past date with closed=False: the oddity is noted
    # at debug level, the date stands, and time_pressure keeps applying — this
    # pass only corrects placeholders Gamma itself has moved.
    _open_position(tmp_db, "c1", end_date="2026-06-30T12:00:00Z")  # past
    monkeypatch.setattr(
        positions.gamma, "fetch_markets_by_condition_ids",
        lambda ids: {"c1": {"condition_id": "c1", "closed": False,
                            "end_date": "2026-06-30T12:00:00Z"}},
    )
    with caplog.at_level("DEBUG", logger="locus.core.positions"):
        assert positions.refresh_end_dates() == 0
    assert _end_dates(tmp_db)["c1"] == "2026-06-30T12:00:00Z"  # unchanged
    assert "past but Gamma reports the market still open" in caplog.text


def test_stale_past_date_left_when_market_absent_from_gamma(tmp_db, monkeypatch):
    # A market Gamma doesn't return is resolved/unknown (reconcile territory):
    # the stale date is left alone, silently.
    _open_position(tmp_db, "c1", end_date="2026-06-30T12:00:00Z")  # past
    monkeypatch.setattr(
        positions.gamma, "fetch_markets_by_condition_ids", lambda ids: {},
    )
    assert positions.refresh_end_dates() == 0
    assert _end_dates(tmp_db)["c1"] == "2026-06-30T12:00:00Z"


def test_stale_and_null_rows_refreshed_in_one_batch(tmp_db, monkeypatch):
    _open_position(tmp_db, "c1")                                   # NULL
    _open_position(tmp_db, "c2", end_date="2026-06-30T12:00:00Z")  # past
    asked = []

    def fake_fetch(ids):
        asked.append(sorted(ids))
        return {
            "c1": {"condition_id": "c1", "end_date": "2030-11-03T00:00:00Z"},
            "c2": {"condition_id": "c2", "closed": False,
                   "end_date": "2030-09-30T12:00:00Z"},
        }

    monkeypatch.setattr(positions.gamma, "fetch_markets_by_condition_ids", fake_fetch)
    assert positions.refresh_end_dates() == 2
    assert asked == [["c1", "c2"]]  # one batched fetch covers both classes
    dates = _end_dates(tmp_db)
    assert dates["c1"] == "2030-11-03T00:00:00Z"
    assert dates["c2"] == "2030-09-30T12:00:00Z"


def test_management_cycle_runs_backfill_and_restores_time_exit_input(tmp_db, monkeypatch):
    # update_and_manage triggers the (throttled) backfill, so the same cycle's
    # hard-exit checks see the date: hours_to_close goes None -> a real number.
    _open_position(tmp_db, "c1")
    monkeypatch.setattr(
        positions.gamma, "fetch_markets_by_condition_ids",
        lambda ids: {"c1": {"condition_id": "c1",
                            "end_date": "2026-11-03T00:00:00Z"}},
    )
    monkeypatch.setattr(positions, "_last_end_date_refresh", 0.0)

    assert positions.hours_to_close(_end_dates(tmp_db)["c1"]) is None  # unprotected
    positions.update_and_manage(prices={"c1": 0.5})
    filled = _end_dates(tmp_db)["c1"]
    assert filled == "2026-11-03T00:00:00Z"
    assert positions.hours_to_close(filled) is not None  # time exits live again


def test_refresh_is_throttled_between_cycles(tmp_db, monkeypatch):
    _open_position(tmp_db, "c1")
    calls = []
    monkeypatch.setattr(
        positions.gamma, "fetch_markets_by_condition_ids",
        lambda ids: calls.append(list(ids)) or {},   # never returns a date
    )
    monkeypatch.setattr(positions, "_last_end_date_refresh", 0.0)

    positions.update_and_manage(prices={})
    positions.update_and_manage(prices={})  # within the throttle window
    assert len(calls) == 1
