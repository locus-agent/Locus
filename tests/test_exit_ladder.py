"""Exit ladder: FAK sells walked down the book under a slippage cap, retries,
partial-fill continuation, stop-loss ignores the spread guard."""
import types

import pytest

from locus import config
from locus.core import executor
from tests.test_live_close import _install_fake_clob, _level


@pytest.fixture(autouse=True)
def ladder(monkeypatch):
    monkeypatch.setattr(config, "EXIT_LADDER_STEPS", 3)
    monkeypatch.setattr(config, "EXIT_MAX_SLIPPAGE_PCT", 9.0)
    monkeypatch.setattr(config, "STOP_LOSS_MAX_SLIPPAGE_PCT", 30.0)
    monkeypatch.setattr(config, "EXIT_RETRY_DELAY_SECONDS", 0.0)


def _scripted(monkeypatch, results):
    calls = []

    def fake(cid, side, shares, max_spread=None, allow_topup=False, max_slippage_pct=None):
        calls.append({"shares": shares, "slip": max_slippage_pct,
                      "max_spread": max_spread, "topup": allow_topup})
        return results.pop(0)

    monkeypatch.setattr(executor, "close_position_live", fake)
    return calls


def test_ladder_widens_slippage_until_filled(monkeypatch):
    calls = _scripted(monkeypatch, [
        {"status": "close_failed", "order_id": None, "price": None, "shares": None},
        {"status": "executed", "order_id": "B", "price": 0.55, "shares": 40.0,
         "sold_shares": 40.0, "remaining_shares": 0.0},
    ])
    res = executor.close_position_ladder("c", "YES", 40.0, allow_topup=True)
    assert [c["slip"] for c in calls] == [pytest.approx(3.0), pytest.approx(6.0)]
    assert calls[0]["topup"] is True and calls[1]["topup"] is False
    assert res["status"] == "executed" and res["sold_shares"] == 40.0
    assert res["price"] == pytest.approx(0.55)


def test_ladder_continues_partial_with_remainder(monkeypatch):
    calls = _scripted(monkeypatch, [
        {"status": "partial_close", "order_id": "A", "price": 0.60, "shares": 10.0,
         "sold_shares": 10.0, "remaining_shares": 30.0},
        {"status": "executed", "order_id": "B", "price": 0.50, "shares": 30.0,
         "sold_shares": 30.0, "remaining_shares": 0.0},
    ])
    res = executor.close_position_ladder("c", "YES", 40.0)
    assert calls[1]["shares"] == 30.0
    assert res["status"] == "executed"
    assert res["sold_shares"] == 40.0
    assert res["price"] == pytest.approx((10 * 0.60 + 30 * 0.50) / 40)


def test_ladder_reports_partial_when_book_exhausted(monkeypatch):
    _scripted(monkeypatch, [
        {"status": "partial_close", "order_id": "A", "price": 0.6, "shares": 10.0,
         "sold_shares": 10.0, "remaining_shares": 30.0},
        {"status": "skipped_thin_book", "order_id": None, "price": None, "shares": None},
        {"status": "skipped_thin_book", "order_id": None, "price": None, "shares": None},
    ])
    res = executor.close_position_ladder("c", "YES", 40.0)
    assert res["status"] == "partial_close"
    assert res["remaining_shares"] == 30.0 and res["sold_shares"] == 10.0


def test_stop_loss_uses_wider_cap_and_ignores_spread(monkeypatch):
    calls = _scripted(monkeypatch, [{"status": "close_failed"}] * 3)
    res = executor.close_position_ladder("c", "YES", 40.0, stop_loss=True)
    assert [c["slip"] for c in calls] == [pytest.approx(10), pytest.approx(20), pytest.approx(30)]
    assert all(c["max_spread"] == 1.0 for c in calls)
    assert res["status"] == "close_failed"


def test_error_stops_ladder(monkeypatch):
    calls = _scripted(monkeypatch, [{"status": "error_no_clob_client"}])
    assert executor.close_position_ladder("c", "YES", 5.0)["status"] == "error_no_clob_client"
    assert len(calls) == 1


def test_fak_order_priced_at_slippage_floor(monkeypatch):
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    book = types.SimpleNamespace(
        bids=[_level(0.60, 10), _level(0.57, 20), _level(0.40, 1000)],
        asks=[_level(0.62, 1000)])
    captured = {}
    _install_fake_clob(monkeypatch, captured, book)
    res = executor.close_position_live("cond1", "YES", 40.0, max_spread=0.05,
                                       max_slippage_pct=10.0)
    oa = captured["order_args"]
    assert captured["posted"][1] == "FAK"           # never rests on the book
    assert oa.price == pytest.approx(0.54)          # ceil(0.60 * 0.90) to tick
    assert oa.size == pytest.approx(30.0)           # depth inside the band only
    assert res["status"] == "partial_close"         # 30 of 40 sold
    # realized price estimated by walking the book, not the floor
    assert res["price"] == pytest.approx((10 * 0.60 + 20 * 0.57) / 30, abs=1e-4)


def test_legacy_close_still_gtc(monkeypatch):
    monkeypatch.setattr(config, "POLYMARKET_PRIVATE_KEY", "0xkey")
    book = types.SimpleNamespace(bids=[_level(0.60, 1000)], asks=[_level(0.62, 1000)])
    captured = {}
    _install_fake_clob(monkeypatch, captured, book)
    executor.close_position_live("cond1", "YES", 40.0, max_spread=0.05)
    assert captured["posted"][1] == "GTC"
