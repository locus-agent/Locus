"""Empirical win-probability calibration for Kelly sizing.

Claude's ``confidence`` is not a probability: live data showed directional
calls at 0.6-0.8 confidence grading only 7-42% "correct" depending on the price
bucket. This module maps (confidence, category, price paid) to a calibrated
probability ``p_cal`` that the bought outcome pays out, learned from the
agent's own graded classifications.

Model
-----
Target is the *excess over the market*: for each graded directional call

    side_price = yes_price (bullish) | 1 - yes_price (bearish)
    y          = value of the bought side after the grading horizon
               = price_after (bullish) | 1 - price_after (bearish)
    r          = y - side_price

``price_after`` (CALIBRATION_HORIZON_HOURS later, from the CLOB history) is the
market's updated probability — a martingale estimate of the payout — so ``y``
is a low-variance soft label for "did the call carry real information". For
resolved markets it equals the 0/1 payout. ``r`` is fit as an isotonic
(monotone non-decreasing) function of confidence with pool-adjacent-violators.

Hierarchy and shrinkage (sparse data):
    (category, price bucket) -> category -> price bucket -> global
The first level with >= CALIBRATION_MIN_CELL rows is used, then shrunk toward
the market prior (r = 0, i.e. p_cal = side_price):

    p_cal = side_price + n / (n + k) * r_fit(confidence),  k = PRIOR_STRENGTH

With no data at all p_cal == side_price -> zero edge -> no trade. That is
deliberate: until the agent has evidence its calls beat the price, it should
not bet on them.
"""
from __future__ import annotations

import bisect
import logging
import time
from dataclasses import dataclass, field

from locus import config

log = logging.getLogger(__name__)

PRICE_BUCKETS = [(0.0, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 0.85), (0.85, 1.0001)]


def price_bucket(p: float) -> int:
    for i, (lo, hi) in enumerate(PRICE_BUCKETS):
        if lo <= p < hi:
            return i
    return len(PRICE_BUCKETS) - 1


def isotonic_fit(xs: list[float], ys: list[float], ws: list[float] | None = None
                 ) -> tuple[list[float], list[float]]:
    """Weighted pool-adjacent-violators. Returns (knots_x, knots_y): block
    mean-x and block-fitted y, non-decreasing in y."""
    if not xs:
        return [], []
    ws = ws or [1.0] * len(xs)
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    blocks: list[list[float]] = []  # [sum_w, sum_wy, sum_wx]
    for i in order:
        blocks.append([ws[i], ws[i] * ys[i], ws[i] * xs[i]])
        while len(blocks) > 1 and blocks[-2][1] / blocks[-2][0] > blocks[-1][1] / blocks[-1][0]:
            w, wy, wx = blocks.pop()
            blocks[-1][0] += w
            blocks[-1][1] += wy
            blocks[-1][2] += wx
    kx = [b[2] / b[0] for b in blocks]
    ky = [b[1] / b[0] for b in blocks]
    return kx, ky


def isotonic_predict(kx: list[float], ky: list[float], x: float) -> float:
    """Linear interpolation between knots, flat beyond the ends."""
    if not kx:
        return 0.0
    if x <= kx[0]:
        return ky[0]
    if x >= kx[-1]:
        return ky[-1]
    j = bisect.bisect_right(kx, x)
    x0, x1, y0, y1 = kx[j - 1], kx[j], ky[j - 1], ky[j]
    return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@dataclass
class CalRow:
    confidence: float
    category: str
    side_price: float
    y: float  # realized value of the bought side (soft or 0/1 label)

    @property
    def r(self) -> float:
        return self.y - self.side_price


@dataclass
class ProbCalibrator:
    rows: list[CalRow] = field(default_factory=list)
    min_cell: int = 30
    prior_strength: float = 50.0
    _fits: dict = field(default_factory=dict, repr=False)

    def _rows_for(self, key) -> list[CalRow]:
        level, cat, b = key
        if level == "cell":
            return [r for r in self.rows if r.category == cat and price_bucket(r.side_price) == b]
        if level == "category":
            return [r for r in self.rows if r.category == cat]
        if level == "bucket":
            return [r for r in self.rows if price_bucket(r.side_price) == b]
        return self.rows

    def _fit(self, key):
        if key not in self._fits:
            rows = self._rows_for(key)
            kx, ky = isotonic_fit([r.confidence for r in rows], [r.r for r in rows])
            self._fits[key] = (len(rows), kx, ky)
        return self._fits[key]

    def predict(self, confidence: float, category: str, side_price: float
                ) -> tuple[float, int, str]:
        """(p_cal, support n, level used). Level 'prior' => no usable data."""
        b = price_bucket(side_price)
        for key in (("cell", category, b), ("category", category, None),
                    ("bucket", None, b), ("global", None, None)):
            n, kx, ky = self._fit(key)
            if n >= self.min_cell:
                r_hat = isotonic_predict(kx, ky, confidence)
                shrink = n / (n + self.prior_strength)
                p = side_price + shrink * r_hat
                return min(max(p, 0.001), 0.999), n, key[0]
        return side_price, 0, "prior"

    def brier(self, rows: list[CalRow] | None = None) -> float | None:
        rows = rows if rows is not None else self.rows
        if not rows:
            return None
        return sum((self.predict(r.confidence, r.category, r.side_price)[0] - r.y) ** 2
                   for r in rows) / len(rows)


def row_from_grade(direction: str, confidence: float | None, yes_price: float | None,
                   price_after: float | None, category: str) -> CalRow | None:
    if direction not in ("bullish", "bearish") or confidence is None:
        return None
    if yes_price is None or price_after is None or not (0 < yes_price < 1):
        return None
    side_price = yes_price if direction == "bullish" else 1.0 - yes_price
    y = price_after if direction == "bullish" else 1.0 - price_after
    return CalRow(float(confidence), category or "other", side_price, min(max(y, 0.0), 1.0))


def load_rows_from_db() -> list[CalRow]:
    """Graded directional classifications from trades.db."""
    from locus.memory import logger
    from locus.markets.gamma import _infer_category

    conn = logger._conn()
    try:
        cur = conn.execute(
            """SELECT c.direction, c.confidence, c.yes_price, c.market_question,
                      g.price_after
                 FROM classification_grades g
                 JOIN classifications c ON c.id = g.classification_id
                WHERE g.price_after IS NOT NULL AND c.confidence IS NOT NULL
                  AND c.direction IN ('bullish', 'bearish')"""
        )
        rows = []
        for d in cur.fetchall():
            row = row_from_grade(d["direction"], d["confidence"], d["yes_price"],
                                 d["price_after"], _infer_category(d["market_question"] or "", []))
            if row:
                rows.append(row)
        return rows
    finally:
        conn.close()


_cache: dict = {"at": 0.0, "cal": None}


def get_calibrator() -> ProbCalibrator:
    now = time.monotonic()
    if _cache["cal"] is None or now - _cache["at"] > config.CALIBRATION_CACHE_TTL_SECONDS:
        try:
            rows = load_rows_from_db()
        except Exception as e:  # missing DB/table -> empty calibrator (prior only)
            log.warning(f"[prob_calibration] load failed, using market prior: {e}")
            rows = []
        _cache["cal"] = ProbCalibrator(rows, config.CALIBRATION_MIN_CELL,
                                       config.CALIBRATION_PRIOR_STRENGTH)
        _cache["at"] = now
    return _cache["cal"]


def reset_cache() -> None:
    _cache["cal"] = None
    _cache["at"] = 0.0


def calibrated_win_prob(confidence: float, category: str, side_price: float
                        ) -> tuple[float, int, str]:
    return get_calibrator().predict(confidence, category, side_price)
