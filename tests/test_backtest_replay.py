"""Replay backtest harness: cost model, entry modes, walk-forward calibration
(no lookahead), exits, metrics. No network."""
import pytest

from locus import config
from locus.backtest import replay as R
from locus.backtest.costs import BookModel, buy_cost, sell_proceeds

H = 3600
T0 = 1_780_000_000 - (1_780_000_000 % H)


def _hist(prices, start=T0, step=H):
    return [{"t": start + i * step, "p": p} for i, p in enumerate(prices)]


def _markets(q="Will X happen?", closed=True, yes_wins=True):
    return {q: {"question": q, "clob_token_ids": '["tY", "tN"]', "outcomes": '["Yes", "No"]',
                "closed": closed, "outcome_prices": '["1", "0"]' if yes_wins else '["0", "1"]'}}


def _row(ts, direction="bullish", conf=0.8, mat=0.6, q="Will X happen?", headline="h"):
    from datetime import datetime, timezone
    return {"time": datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "market_question": q, "headline": headline, "direction": direction,
            "materiality": mat, "confidence": conf}


def test_book_model_and_costs():
    bm = BookModel(half_spread=0.01, tick=0.01, depth_usd=10.0, levels=5)
    book = bm.book(0.50)
    assert book["asks"][0][0] == pytest.approx(0.51)
    assert book["bids"][0][0] == pytest.approx(0.49)
    shares, vwap, cash = buy_cost(book, 15.0, fee_rate=0.04)
    assert vwap > 0.51  # walked into the second level
    assert cash == pytest.approx(15.0 + 0.04 * vwap * (1 - vwap) * shares)
    sold, cash_in = sell_proceeds(book, 10.0, fee_rate=0.0)
    assert cash_in == pytest.approx(10.0 * 0.49)
    no = bm.side_book(0.30, "NO")
    assert no["asks"][0][0] == pytest.approx(0.71)


def test_entry_modes():
    hist = {"tY": _hist([0.40, 0.60, 0.60])}
    ts = T0 + 30 * 60  # call half-way through the first hour, price jumps after
    calls_after, _ = R.build_calls([_row(ts)], _markets(), hist, entry="after")
    calls_before, _ = R.build_calls([_row(ts)], _markets(), hist, entry="before")
    assert calls_before[0].yes_mid == 0.40   # optimistic: stale pre-move price
    assert calls_after[0].yes_mid == 0.60    # conservative: next print
    minute = {"tY": _hist([0.40, 0.45, 0.50], start=T0 + 30 * 60, step=60)}
    calls_lat, _ = R.build_calls([_row(ts)], _markets(), hist, entry="latency",
                                 minute=minute, latency_s=60)
    assert calls_lat[0].yes_mid == 0.45


def test_resolution_and_unmatched_stats():
    rows = [_row(T0), _row(T0, q="Unknown market?")]
    calls, stats = R.build_calls(rows, _markets(), {"tY": _hist([0.4, 0.5])}, entry="before")
    assert stats["usable"] == 1 and stats["no_market_match"] == 1
    assert calls[0].resolution == 1.0


def test_walk_forward_calibrator_uses_only_past_labels(monkeypatch):
    seen = []
    orig = R.pc.ProbCalibrator

    class Spy(orig):
        def __init__(self, rows, *a, **k):
            seen.append(len(rows))
            super().__init__(rows, *a, **k)

    monkeypatch.setattr(R.pc, "ProbCalibrator", Spy)
    hist = {"tY": _hist([0.5] * 200)}
    rows = [_row(T0 + i * 7 * H, headline=f"h{i}") for i in range(6)]
    calls, _ = R.build_calls(rows, _markets(), hist, entry="before")
    R.run(calls, "calibrated", BookModel())
    # first rebuild happens at the first call: nothing is labelled yet
    assert seen[1] == 0
    # labels arrive 24h after each call -> by call 4 (t=28h) only call 0 is labelled
    assert max(seen) <= 3


def test_run_trades_and_metrics(monkeypatch):
    monkeypatch.setattr(config, "MIN_MATERIALITY_DEFAULT", 0.0)
    monkeypatch.setattr(config, "MIN_MATERIALITY_BULLISH", 0.0)
    hist = {"tY": _hist([0.40] + [0.42] * 5 + [0.97])}
    calls, _ = R.build_calls([_row(T0)], _markets(), hist, entry="before")
    res = R.run(calls, "legacy", BookModel())
    m = R.metrics(res)
    assert m["trades"] == 1
    t = res.trades[0]
    assert t.exit_kind == "near_certain" and t.pnl > 0
    assert m["total_pnl_usd"] == pytest.approx(round(t.pnl, 2))


def test_stop_loss_exit(monkeypatch):
    monkeypatch.setattr(config, "MIN_MATERIALITY_DEFAULT", 0.0)
    monkeypatch.setattr(config, "MIN_MATERIALITY_BULLISH", 0.0)
    hist = {"tY": _hist([0.40, 0.35, 0.15, 0.10])}
    calls, _ = R.build_calls([_row(T0)], _markets(yes_wins=False), hist, entry="before")
    res = R.run(calls, "legacy", BookModel())
    assert res.trades[0].exit_kind == "stop_loss" and res.trades[0].pnl < 0


def test_calibration_quality_runs():
    hist = {"tY": _hist([0.5] * 200)}
    rows = [_row(T0 + i * 30 * H, headline=f"h{i}") for i in range(5)]
    calls, _ = R.build_calls(rows, _markets(), hist, entry="before")
    q = R.calibration_quality(calls)
    assert q["n_calls"] == 5
    assert q["mse"]["market_price"] == pytest.approx(0.0)
