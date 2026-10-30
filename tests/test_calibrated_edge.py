"""Calibrated edge model: edge = p_cal - effective ask - fee, depth-walked
slippage, Kelly on p_cal, and the probability calibrator itself."""
from datetime import datetime, timezone

import pytest

from locus import config
from locus.core import edge
from locus.core.classifier import Classification
from locus.markets.gamma import Market
from locus.memory import prob_calibration as pc
from locus.sources.news_stream import NewsEvent


@pytest.fixture(autouse=True)
def calibrated(monkeypatch):
    monkeypatch.setattr(config, "EDGE_MODEL", "calibrated")
    monkeypatch.setattr(config, "KELLY_BANKROLL_USD", 100.0)
    monkeypatch.setattr(config, "MAX_BET_USD", 25.0)
    monkeypatch.setattr(config, "KELLY_MIN_BET_USD", 2.0)
    monkeypatch.setattr(config, "CAL_EDGE_THRESHOLD", 0.04)
    monkeypatch.setattr(config, "CAL_KELLY_FRACTION", 0.5)
    monkeypatch.setattr(config, "DEFAULT_SLIPPAGE", 0.01)
    monkeypatch.setattr(config, "BULLISH_MIN_PRICE", 0.12)
    monkeypatch.setattr(config, "BULLISH_MAX_PRICE", 0.82)
    monkeypatch.setattr(config, "BEARISH_MIN_PRICE", 0.18)
    monkeypatch.setattr(config, "BEARISH_MAX_PRICE", 0.88)


EVENT = NewsEvent(headline="h", source="rss", url="",
                  received_at=datetime.now(timezone.utc),
                  published_at=datetime.now(timezone.utc), latency_ms=0)


def _cls(direction, confidence=0.8, materiality=0.5):
    return Classification(direction=direction, materiality=materiality,
                          confidence=confidence, reasoning="", latency_ms=1, model="t")


def _mkt(price, fee_rate=0.0, category="politics"):
    return Market("c1", "Will X?", category, price, round(1 - price, 4), 5000, "",
                  True, [], fee_rate=fee_rate)


def fixed(p):
    return lambda conf, cat, side_price: (p, 100, "test")


# --- fee + Kelly primitives ---------------------------------------------------

def test_fee_matches_polymarket_formula():
    # docs.polymarket.com/trading/fees: fee = C * feeRate * p * (1 - p)
    assert edge.polymarket_fee_per_share(0.07, 0.5) == pytest.approx(0.0175)
    assert edge.polymarket_fee_per_share(0.04, 0.3) == pytest.approx(0.04 * 0.3 * 0.7)
    assert edge.polymarket_fee_per_share(0.0, 0.5) == 0.0


def test_kelly_fraction_binary():
    assert edge.calibrated_kelly_fraction(0.6, 0.5) == pytest.approx(0.2)
    assert edge.calibrated_kelly_fraction(0.5, 0.5) == 0.0
    assert edge.calibrated_kelly_fraction(0.4, 0.5) < 0


# --- detect_edge_calibrated ---------------------------------------------------

def test_edge_is_pcal_minus_effective_ask_minus_fee():
    book = {"bids": [(0.48, 1000)], "asks": [(0.50, 1000)]}
    m = edge.detect_edge_calibrated(_mkt(0.49, fee_rate=0.04), _cls("bullish"), EVENT,
                                    book=book, win_prob=fixed(0.65))
    fee = 0.04 * 0.5 * 0.5
    assert m.net_edge == pytest.approx(0.65 - 0.50 - fee)
    assert m.effective_price == pytest.approx(0.50)
    assert m.slippage == pytest.approx(0.0)
    assert m.signal.side == "YES"
    assert m.signal.p_cal == 0.65
    # half Kelly on p_cal at all-in cost: f = (p - c) / (1 - c)
    c = 0.5 + fee
    expected = min(100 * (0.65 - c) / (1 - c) * 0.5, 25.0)
    assert m.signal.bet_amount == pytest.approx(expected, abs=0.02)


def test_no_side_uses_mirrored_yes_bids():
    # YES bids at 0.40 -> NO asks at 0.60
    book = {"bids": [(0.40, 1000)], "asks": [(0.42, 1000)]}
    m = edge.detect_edge_calibrated(_mkt(0.41), _cls("bearish"), EVENT,
                                    book=book, win_prob=fixed(0.75))
    assert m.signal.side == "NO"
    assert m.effective_price == pytest.approx(0.60)
    assert m.net_edge == pytest.approx(0.15)


def test_slippage_walks_depth_and_can_kill_edge():
    thin = {"bids": [(0.48, 1000)], "asks": [(0.50, 4.0), (0.58, 10.0), (0.70, 1000)]}
    deep = {"bids": [(0.48, 1000)], "asks": [(0.50, 1000)]}
    thin_m = edge.detect_edge_calibrated(_mkt(0.49), _cls("bullish"), EVENT,
                                         book=thin, win_prob=fixed(0.60))
    deep_m = edge.detect_edge_calibrated(_mkt(0.49), _cls("bullish"), EVENT,
                                         book=deep, win_prob=fixed(0.60))
    assert deep_m is not None and deep_m.signal is not None
    # same p_cal, but walking the thin book costs more than the edge
    assert thin_m is None or thin_m.signal is None or thin_m.effective_price > 0.5


def test_no_edge_when_pcal_equals_market():
    book = {"bids": [(0.48, 1000)], "asks": [(0.50, 1000)]}
    assert edge.detect_edge_calibrated(_mkt(0.49), _cls("bullish"), EVENT,
                                       book=book, win_prob=fixed(0.50)) is None


def test_empty_side_book_no_signal():
    book = {"bids": [], "asks": [(0.5, 10)]}
    # bearish needs YES bids (mirrored NO asks): none -> no trade
    assert edge.detect_edge_calibrated(_mkt(0.5), _cls("bearish"), EVENT,
                                       book=book, win_prob=fixed(0.9)) is None


def test_neutral_and_price_guards():
    assert edge.detect_edge_calibrated(_mkt(0.5), _cls("neutral"), EVENT,
                                       win_prob=fixed(0.9)) is None
    assert edge.detect_edge_calibrated(_mkt(0.05), _cls("bullish"), EVENT,
                                       win_prob=fixed(0.9)) is None


def test_dispatch_uses_calibrated_by_config(monkeypatch):
    monkeypatch.setattr(pc, "calibrated_win_prob", lambda c, cat, p: (p + 0.2, 99, "cell"))
    m = edge.detect_edge_v2(_mkt(0.40), _cls("bullish"), EVENT)  # no book -> default slippage
    assert m.p_cal == pytest.approx(0.60)
    assert m.effective_price == pytest.approx(0.41)
    monkeypatch.setattr(config, "EDGE_MODEL", "legacy")
    legacy = edge.detect_edge_v2(_mkt(0.40), _cls("bullish"), EVENT)
    assert legacy is None or legacy.p_cal is None


def test_no_calibration_data_means_no_trade(monkeypatch, tmp_db):
    pc.reset_cache()
    m = edge.detect_edge_v2(_mkt(0.40), _cls("bullish", confidence=0.95), EVENT)
    assert m is None  # p_cal falls back to the market price -> zero edge
    pc.reset_cache()


# --- calibrator ---------------------------------------------------------------

def test_isotonic_is_monotone_and_pools_violators():
    kx, ky = pc.isotonic_fit([1, 2, 3, 4], [0.1, 0.3, 0.2, 0.4])
    assert ky == sorted(ky)
    assert pc.isotonic_predict(kx, ky, 2.5) == pytest.approx(0.25)


def test_calibrator_prior_when_sparse():
    cal = pc.ProbCalibrator([pc.CalRow(0.8, "ai", 0.4, 1.0)] * 5, min_cell=30)
    p, n, level = cal.predict(0.8, "ai", 0.4)
    assert (p, n, level) == (0.4, 0, "prior")


def test_calibrator_learns_excess_and_shrinks():
    rows = ([pc.CalRow(0.9, "ai", 0.4, 0.6)] * 50 + [pc.CalRow(0.6, "ai", 0.4, 0.4)] * 50)
    cal = pc.ProbCalibrator(rows, min_cell=30, prior_strength=100)
    p_hi, n, level = cal.predict(0.9, "ai", 0.4)
    p_lo, _, _ = cal.predict(0.6, "ai", 0.4)
    assert level == "cell" and n == 100
    assert p_hi == pytest.approx(0.4 + 0.5 * 0.2)  # shrink 100/(100+100)
    assert p_lo == pytest.approx(0.4)
    assert p_hi > p_lo


def test_calibrator_falls_back_category_then_global():
    rows = [pc.CalRow(0.8, "ai", 0.2, 0.3)] * 40  # bucket 1 only
    cal = pc.ProbCalibrator(rows, min_cell=30, prior_strength=0)
    _, _, level = cal.predict(0.8, "ai", 0.6)   # other bucket -> category level
    assert level == "category"
    _, _, level = cal.predict(0.8, "crypto", 0.6)  # other category+bucket -> global
    assert level == "global"


def test_row_from_grade_side_conventions():
    r = pc.row_from_grade("bearish", 0.7, 0.3, 0.1, "ai")
    assert r.side_price == pytest.approx(0.7) and r.y == pytest.approx(0.9)
    assert pc.row_from_grade("neutral", 0.7, 0.3, 0.1, "ai") is None
    assert pc.row_from_grade("bullish", None, 0.3, 0.1, "ai") is None


def test_load_rows_from_db(tmp_db):
    conn = tmp_db._conn()
    conn.execute("""INSERT INTO classifications (market_question, direction, materiality,
                    action, confidence, yes_price) VALUES ('Will Bitcoin hit 1M?', 'bullish',
                    0.5, 'signal', 0.8, 0.3)""")
    cid = conn.execute("SELECT id FROM classifications").fetchone()[0]
    conn.execute("""INSERT INTO classification_grades (classification_id, direction,
                    entry_price, price_after, correct) VALUES (?, 'bullish', 0.3, 0.45, 1)""",
                 (cid,))
    conn.commit()
    conn.close()
    rows = pc.load_rows_from_db()
    assert len(rows) == 1
    assert rows[0].side_price == pytest.approx(0.3) and rows[0].y == pytest.approx(0.45)
