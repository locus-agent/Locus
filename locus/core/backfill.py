"""Backfill realized PnL for closed positions from Polymarket's public Data API.

Positions closed outside the agent (e.g. a manual flatten in the Polymarket UI
while the pipeline was down, later reconciled as closed_reconciled with
realized $0) leave realized_pnl_usd missing or stale. This module recomputes
each closed position's true realized PnL from the wallet's on-chain activity
feed (`{POLYMARKET_DATA_HOST}/activity`) and writes only the DELTA against what
the DB already realized — so prior partial realizations are never double-
counted and a repeated --apply is a no-op.

Activity types folded in (symmetric cost/proceeds on the matched outcome
token):
  - TRADE (side BUY/SELL): the ordinary fills.
  - REDEEM: a resolution payout for shares held through market close — pure
    proceeds, no cost leg. Confirmed present in this wallet's feed (position
    48 redeemed $0.75 of post-close dust that a drawdown exit never sold,
    money the old TRADE-only accounting silently missed).
  - MERGE / SPLIT: not observed in this wallet's feed, but handled the same
    way if they ever appear — MERGE (burning a YES+NO pair back to USDC) reads
    as proceeds on the matched outcome; SPLIT ($1 -> one YES + one NO) reads
    as cost, both at usdcSize/size for the matched outcome's leg.

Matching: conditionId + outcomeIndex (YES=0 / NO=1, the same positional
convention as gamma._parse_market and executor._resolve_token_id), windowed to
this position's life and clipped at any sibling position that re-entered the
same market+side — so a trade/redeem is never double-attributed across a
close-then-reenter pair (observed for positions 44/49, 34/52, 45/46, 30/55).

Known Data API gap: the activity feed can MISS buy fills entirely (observed:
position 59 shows 31.75 of 36.75 bought shares; position 57's whole ~150-share
BUY is absent). Rather than guess at the missing cost, each row reports a
completeness check — bought_shares - sold_shares - redeemed_shares should be
~0 for a fully-exited position — and flags `complete=False` with the signed
`unexplained_shares` gap when it doesn't. Only complete rows are trustworthy
enough to ever write; an incomplete row's realized_true is shown for
reference but is NOT a reliable number.
"""
from __future__ import annotations

import collections
import logging
from datetime import datetime, timezone

import httpx

from locus import config
from locus.memory import logger

log = logging.getLogger(__name__)

ACTIVITY_SOURCE = "data-api.polymarket.com/activity"

# TRADE rows land on-chain seconds around the DB's opened_at/closed_at stamps
# (the row is written after the order reconcile) — widen the life window by
# this much for matching BUY/SELL fills.
MATCH_WINDOW_SLACK_SECONDS = 600.0

# REDEEM/MERGE/SPLIT can lag the position's closed_at by hours or longer (dust
# left over from a partial close, claimed whenever someone bothers to): no
# upper bound beyond the next sibling position's open (see _sibling_bounds).

# Deltas below this are float noise, not missing accounting — never written.
MIN_ADJUSTMENT_USD = 0.01

# Share-count tolerance for the bought/sold/redeemed completeness check.
COMPLETENESS_TOLERANCE_SHARES = 0.5

# Our positional YES/NO convention -> Data API outcomeIndex.
_OUTCOME_INDEX = {"YES": 0, "NO": 1}

TRADE_TYPE = "TRADE"
# Pure-proceeds activity on the matched outcome (no cost leg of their own).
PROCEEDS_TYPES = frozenset({"REDEEM", "MERGE"})
# Pure-cost activity on the matched outcome.
COST_TYPES = frozenset({"SPLIT"})


def fetch_wallet_activity(wallet: str, page_size: int = 500,
                          max_pages: int = 40) -> list[dict]:
    """All activity rows for `wallet` from the public Data API, paginated by
    offset until a short page. TLS verification stays on (httpx default)."""
    rows: list[dict] = []
    offset = 0
    with httpx.Client(timeout=30.0) as client:
        for _ in range(max_pages):
            resp = client.get(
                f"{config.POLYMARKET_DATA_HOST}/activity",
                params={"user": wallet, "limit": page_size, "offset": offset},
            )
            resp.raise_for_status()
            page = resp.json() or []
            rows.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
    return rows


def activity_type_counts(activity: list[dict]) -> dict[str, int]:
    """How many rows of each `type` appear in a fetched activity list — for
    surfacing what kinds of events this wallet's feed actually contains."""
    return dict(collections.Counter(r.get("type") for r in activity))


def _epoch(stamp: str | None) -> float | None:
    """A positions timestamp (SQLite 'YYYY-MM-DD HH:MM:SS' or ISO with offset)
    as a UTC epoch, or None when missing/unparseable."""
    if not stamp:
        return None
    try:
        dt = datetime.fromisoformat(str(stamp).replace(" ", "T").replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _sibling_bounds(position: dict, all_positions: list[dict]) -> tuple[float | None, float | None]:
    """(prev_closed_epoch, next_opened_epoch) among OTHER positions sharing
    this position's condition_id + side, so a re-entered market's activity
    rows are never attributed to the wrong occupant of that market+side.
    Either side is None when there is no such sibling."""
    opened = _epoch(position.get("opened_at"))
    siblings = [
        p for p in all_positions
        if p["id"] != position["id"]
        and p["condition_id"] == position["condition_id"]
        and (p.get("side") or "").upper() == (position.get("side") or "").upper()
    ]
    prev_closed = max(
        (_epoch(p.get("closed_at")) for p in siblings
         if _epoch(p.get("opened_at")) is not None and opened is not None
         and _epoch(p.get("opened_at")) < opened and _epoch(p.get("closed_at")) is not None),
        default=None,
    )
    next_opened = min(
        (_epoch(p.get("opened_at")) for p in siblings
         if _epoch(p.get("opened_at")) is not None and opened is not None
         and _epoch(p.get("opened_at")) > opened),
        default=None,
    )
    return prev_closed, next_opened


def match_position_activity(position: dict, activity: list[dict],
                            all_positions: list[dict] | None = None) -> dict[str, list[dict]]:
    """Activity rows belonging to this position, split by role: {"trade":
    [...], "proceeds_extra": [...REDEEM/MERGE...], "cost_extra": [...SPLIT...]}.

    Matched by conditionId + outcomeIndex, within the position's life window
    — TRADE rows get a tight slack around opened_at/closed_at;
    REDEEM/MERGE/SPLIT get no upper bound beyond closed_at (they can lag
    arbitrarily), but all rows are clipped at any sibling re-entry on the same
    market+side (see _sibling_bounds), so a later re-entry's own fills/redeems
    are never misattributed to this (earlier, closed) occupant, or vice versa.
    """
    outcome_index = _OUTCOME_INDEX.get((position.get("side") or "").upper())
    if outcome_index is None:
        return {"trade": [], "proceeds_extra": [], "cost_extra": []}

    opened = _epoch(position.get("opened_at"))
    closed = _epoch(position.get("closed_at"))
    prev_closed, next_opened = _sibling_bounds(position, all_positions or [position])

    lo = (opened - MATCH_WINDOW_SLACK_SECONDS) if opened is not None else None
    if prev_closed is not None:
        lo = prev_closed if lo is None else max(lo, prev_closed)

    hi_trade = (closed + MATCH_WINDOW_SLACK_SECONDS) if closed is not None else None
    if next_opened is not None:
        hi_trade = next_opened if hi_trade is None else min(hi_trade, next_opened)

    hi_wide = next_opened  # None (unbounded) unless a later re-entry caps it

    def _in(row, hi):
        ts = row.get("timestamp") or 0
        if lo is not None and ts < lo:
            return False
        if hi is not None and ts > hi:
            return False
        return True

    same_market = [
        row for row in activity
        if row.get("conditionId") == position["condition_id"]
        and row.get("outcomeIndex") == outcome_index
    ]
    result = {
        "trade": sorted((r for r in same_market
                         if r.get("type") == TRADE_TYPE and _in(r, hi_trade)),
                        key=lambda r: r.get("timestamp") or 0),
        "proceeds_extra": sorted((r for r in same_market
                                  if r.get("type") in PROCEEDS_TYPES and _in(r, hi_wide)),
                                 key=lambda r: r.get("timestamp") or 0),
        "cost_extra": sorted((r for r in same_market
                              if r.get("type") in COST_TYPES and _in(r, hi_wide)),
                             key=lambda r: r.get("timestamp") or 0),
    }
    return result


def compute_adjustment(position: dict, matched: dict[str, list[dict]]) -> dict:
    """True realized PnL for a CLOSED position from its matched activity, and
    the delta against what the DB already realized.

    realized_true = (SELL + REDEEM/MERGE usdcSize) - (BUY + SPLIT usdcSize) —
    a strict reading of the ledger, no estimation. Completeness is checked
    separately: bought_shares - sold_shares - redeemed_shares should be ~0 for
    a position holding nothing (COMPLETENESS_TOLERANCE_SHARES); when it isn't,
    `complete` is False and `unexplained_shares` carries the signed gap
    (positive = shares vanished with no recorded SELL/REDEEM, i.e. realized_true
    UNDERSTATES true proceeds; negative = more was sold/redeemed than bought,
    i.e. a missing BUY, so realized_true OVERSTATES true profit)."""
    trade = matched["trade"]
    proceeds_extra = matched["proceeds_extra"]
    cost_extra = matched["cost_extra"]

    buys_usd = sum(r.get("usdcSize") or 0.0 for r in trade if r.get("side") == "BUY")
    sells_usd = sum(r.get("usdcSize") or 0.0 for r in trade if r.get("side") == "SELL")
    bought_shares = sum(r.get("size") or 0.0 for r in trade if r.get("side") == "BUY")
    sold_shares = sum(r.get("size") or 0.0 for r in trade if r.get("side") == "SELL")

    redeemed_usd = sum(r.get("usdcSize") or 0.0 for r in proceeds_extra)
    redeemed_shares = sum(r.get("size") or 0.0 for r in proceeds_extra)

    split_usd = sum(r.get("usdcSize") or 0.0 for r in cost_extra)
    split_shares = sum(r.get("size") or 0.0 for r in cost_extra)

    total_buys_usd = buys_usd + split_usd
    total_bought_shares = bought_shares + split_shares
    total_proceeds_usd = sells_usd + redeemed_usd

    unexplained_shares = total_bought_shares - sold_shares - redeemed_shares
    complete = abs(unexplained_shares) <= COMPLETENESS_TOLERANCE_SHARES

    realized_true = total_proceeds_usd - total_buys_usd
    already = position.get("realized_pnl_usd") or 0.0
    return {
        "position_id": position["id"],
        "status": position.get("status"),
        "side": position.get("side"),
        "market_question": position.get("market_question") or "",
        "trades_matched": len(trade),
        "redeems_matched": len(proceeds_extra),
        "splits_matched": len(cost_extra),
        "buys_usd": round(total_buys_usd, 4),
        "sells_usd": round(sells_usd, 4),
        "redeemed_usd": round(redeemed_usd, 4),
        "bought_shares": round(total_bought_shares, 4),
        "sold_shares": round(sold_shares, 4),
        "redeemed_shares": round(redeemed_shares, 4),
        "unexplained_shares": round(unexplained_shares, 4),
        "complete": complete,
        "realized_true": round(realized_true, 4),
        "already_realized": round(already, 4),
        "delta": round(realized_true - already, 4),
    }


def _audit_reasoning(row: dict, wallet: str, position: dict) -> str:
    """One self-contained audit line for exit_decisions: where the correction
    came from and how it was computed."""
    text = (
        f"Realized PnL backfilled from {ACTIVITY_SOURCE} (wallet {wallet}): "
        f"window {position.get('opened_at')}..{position.get('closed_at') or 'now'}, "
        f"{row['trades_matched']} TRADE + {row['redeems_matched']} REDEEM/MERGE rows "
        f"matched — BUY ${row['buys_usd']:.4f} ({row['bought_shares']:g} sh), "
        f"SELL ${row['sells_usd']:.4f} + REDEEM ${row['redeemed_usd']:.4f} "
        f"({row['sold_shares']:g} + {row['redeemed_shares']:g} sh); "
        f"completeness check: {row['bought_shares']:g} - {row['sold_shares']:g} - "
        f"{row['redeemed_shares']:g} = {row['unexplained_shares']:+g} sh unexplained "
        f"({'within' if row['complete'] else 'OUTSIDE'} the "
        f"{COMPLETENESS_TOLERANCE_SHARES:g}-token tolerance); "
        f"realized_true ${row['realized_true']:+.4f} "
        f"minus already recorded ${row['already_realized']:+.4f} "
        f"= delta ${row['delta']:+.4f}"
    )
    return text


def _fetch_positions(conn, include_open: bool = False) -> list[dict]:
    sql = "SELECT * FROM positions" if include_open else \
        "SELECT * FROM positions WHERE status != 'open'"
    return [dict(r) for r in conn.execute(sql + " ORDER BY id").fetchall()]


def backfill_realized(apply: bool = False, ids: list[int] | None = None,
                      wallet: str | None = None,
                      fetcher=fetch_wallet_activity) -> dict:
    """Compute (and with apply=True, write) realized-PnL corrections for closed
    positions from the wallet's Data API activity (TRADE + REDEEM/MERGE/SPLIT).

    Open positions are never touched — their basis/realized bookkeeping belongs
    to the live close paths. `ids` restricts which positions may be WRITTEN
    (the report always covers every closed position with matched activity).
    Writes require BOTH a non-trivial delta AND `complete=True` — an
    incomplete ledger's realized_true is not trustworthy enough to apply, even
    if the position id is in `ids`. One exit_decisions audit row
    (trigger='backfill', decision='pnl_backfill') is written per adjusted
    position. Returns {"rows", "applied", "skipped_ids", "wallet",
    "activity_types"}."""
    wallet = wallet or config.POLYMARKET_FUNDER_ADDRESS
    if not wallet:
        raise RuntimeError(
            "No wallet to audit: POLYMARKET_FUNDER_ADDRESS is not set"
        )
    activity = fetcher(wallet)
    activity_types = activity_type_counts(activity)

    conn = logger._conn()
    try:
        closed_positions = _fetch_positions(conn, include_open=False)
        all_positions = _fetch_positions(conn, include_open=True)

        rows: list[dict] = []
        for position in closed_positions:
            matched = match_position_activity(position, activity, all_positions)
            if not any(matched.values()):
                continue
            row = compute_adjustment(position, matched)
            row["writable"] = (
                row["complete"]
                and abs(row["delta"]) >= MIN_ADJUSTMENT_USD
                and (ids is None or position["id"] in ids)
            )
            rows.append(row)
            if not apply or not row["writable"]:
                continue
            cur = conn.execute(
                """UPDATE positions
                   SET realized_pnl_usd = COALESCE(realized_pnl_usd, 0) + ?
                   WHERE id = ? AND status != 'open'""",
                (row["delta"], position["id"]),
            )
            if cur.rowcount == 0:
                row["writable"] = False
                continue
            conn.execute(
                """INSERT INTO exit_decisions
                   (position_id, trigger, decision, reasoning, pnl_pct, yes_price)
                   VALUES (?, 'backfill', 'pnl_backfill', ?, NULL, NULL)""",
                (position["id"], _audit_reasoning(row, wallet, position)),
            )
            log.info(
                "[backfill] position %s realized_pnl_usd %+.4f -> %+.4f "
                "(delta %+.4f, source %s)",
                position["id"], row["already_realized"], row["realized_true"],
                row["delta"], ACTIVITY_SOURCE,
            )
        if apply:
            conn.commit()
    finally:
        conn.close()

    applied = [r["position_id"] for r in rows if apply and r["writable"]]
    skipped = [r["position_id"] for r in rows
               if abs(r["delta"]) >= MIN_ADJUSTMENT_USD and not r["writable"]]
    return {"rows": rows, "applied": applied, "skipped_ids": skipped,
            "wallet": wallet, "activity_types": activity_types}


def preview_reconcile_with_ledger(position_id: int, wallet: str | None = None,
                                  fetcher=fetch_wallet_activity) -> dict | None:
    """Preview-only: what closing `position_id` as closed_reconciled WOULD
    look like if its realized_pnl_usd were set from the Data API ledger,
    instead of the normal reconcile fix (which realizes $0 on top of whatever
    was already realized). Returns the same shape as one `backfill_realized`
    row (plus "preview_total_realized"), or None if the position doesn't
    exist. Never writes anything — purely informational, for a position still
    status='open' (reconcile_positions --fix is the actual mechanism, and it
    never touches a held>0 mismatch; this just shows what the number would be
    if a human chose to override that)."""
    wallet = wallet or config.POLYMARKET_FUNDER_ADDRESS
    if not wallet:
        raise RuntimeError(
            "No wallet to audit: POLYMARKET_FUNDER_ADDRESS is not set"
        )
    activity = fetcher(wallet)

    conn = logger._conn()
    try:
        row = conn.execute(
            "SELECT * FROM positions WHERE id=?", (position_id,)
        ).fetchone()
        if row is None:
            return None
        position = dict(row)
        all_positions = _fetch_positions(conn, include_open=True)
    finally:
        conn.close()

    matched = match_position_activity(position, activity, all_positions)
    result = compute_adjustment(position, matched)
    result["writable"] = False  # preview only — this function never writes
    result["preview_total_realized"] = round(
        result["already_realized"] + result["delta"], 4
    )
    return result
