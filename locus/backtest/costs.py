"""Execution-cost model shared by the backtests.

Public Polymarket APIs expose price HISTORY but not historical order books, so
past depth/spread cannot be replayed. Fills are therefore priced against a
parametric synthetic book centred on the historical mid:

    asks: mid + half_spread + k * tick,  k = 0..levels-1, each `depth_usd` deep
    bids: mid - half_spread - k * tick   (mirror)

Entry walks the asks for the stake (VWAP), exits walk the bids, and the
Polymarket taker fee (feeRate * p * (1 - p) per share) is charged on both
taker legs. Resolution payouts carry no fee. The defaults are deliberately
conservative for niche (<$500K volume) markets; run with several settings and
report the sensitivity rather than trusting one number.
"""
from __future__ import annotations

from dataclasses import dataclass

from locus.core.edge import polymarket_fee_per_share
from locus.core.orderbook import walk_asks_for_usd, walk_bids_for_shares


@dataclass(frozen=True)
class BookModel:
    half_spread: float = 0.01
    tick: float = 0.01
    depth_usd: float = 40.0
    levels: int = 30

    def book(self, mid: float) -> dict:
        asks, bids = [], []
        for k in range(self.levels):
            a = round(mid + self.half_spread + k * self.tick, 4)
            b = round(mid - self.half_spread - k * self.tick, 4)
            if a < 1.0:
                asks.append((a, self.depth_usd / a))
            if b > 0.0:
                bids.append((b, self.depth_usd / b))
        return {"asks": asks, "bids": bids}

    def side_book(self, yes_mid: float, side: str) -> dict:
        return self.book(yes_mid if side == "YES" else 1.0 - yes_mid)


def buy_cost(book: dict, usd: float, fee_rate: float) -> tuple[float, float, float]:
    """(shares, vwap, total_cash_out incl. fee) for buying `usd` of notional."""
    shares, vwap, spent = walk_asks_for_usd(book["asks"], usd)
    fee = polymarket_fee_per_share(fee_rate, vwap) * shares if shares else 0.0
    return shares, vwap, spent + fee


def sell_proceeds(book: dict, shares: float, fee_rate: float) -> tuple[float, float]:
    """(shares_sold, cash_in net of fee) walking the bids with no floor.
    Shares the synthetic book can't absorb are valued at the worst level."""
    filled, vwap = walk_bids_for_shares(book["bids"], shares)
    cash = filled * vwap - polymarket_fee_per_share(fee_rate, vwap) * filled
    rest = shares - filled
    if rest > 1e-9 and book["bids"]:
        worst = book["bids"][-1][0]
        cash += rest * worst - polymarket_fee_per_share(fee_rate, worst) * rest
    return shares, cash
