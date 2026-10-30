"""Backfill realized PnL for closed positions from Polymarket's public Data API.

Positions closed outside the agent (e.g. a manual flatten in the Polymarket UI
while the pipeline was down, later reconciled as closed_reconciled with
realized $0) leave realized_pnl_usd missing or stale. This module recomputes
each closed position's true realized PnL from the wallet's on-chain activity
feed (`{POLYMARKET_DATA_HOST}/activity`, TRADE rows) and writes only the DELTA
against what the DB already realized — so prior partial realizations are never
double-counted and a repeated --apply is a no-op.

Matching: conditionId + outcomeIndex (YES=0 / NO=1, the same positional
convention as gamma._parse_market and executor._resolve_token_id) within the
position's life window [opened_at - slack, closed_at + slack].

Known Data API gap: the activity feed can MISS buy fills (observed: position 59
shows 31.75 of 36.75 bought shares; position 57's whole BUY is absent), which
would inflate sells-minus-buys. When the matched BUY shares don't cover the
matched SELL shares, the missing shares' cost is estimated at the position's
entry price for its side and the row is flagged `incomplete_buy` — proceeds
come from the feed, the gap in cost from the DB's own entry.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from locus import config
from locus.memory import logger

log = logging.getLogger(__name__)

ACTIVITY_SOURCE = "data-api.polymarket.com/activity"

# Fills land on-chain seconds around the DB's opened_at/closed_at stamps (the
# row is written after the order reconcile); widen the life window by this much.
MATCH_WINDOW_SLACK_SECONDS = 600.0

# Deltas below this are float noise, not missing accounting — never written.
MIN_ADJUSTMENT_USD = 0.01

# Share-count slack when deciding the BUY side of the feed is incomplete.
_SHARE_EPSILON = 0.05

# Our positional YES/NO convention -> Data API outcomeIndex.
_OUTCOME_INDEX = {"YES": 0, "NO": 1}


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


def match_position_activity(position: dict, activity: list[dict]) -> list[dict]:
    """TRADE rows belonging to this position: same conditionId, same outcome
    token (by outcomeIndex), timestamped within the position's life window."""
    outcome_index = _OUTCOME_INDEX.get((position.get("side") or "").upper())
    if outcome_index is None:
        return []
    opened = _epoch(position.get("opened_at"))
    closed = _epoch(position.get("closed_at"))
    lo = (opened - MATCH_WINDOW_SLACK_SECONDS) if opened is not None else None
    hi = (closed + MATCH_WINDOW_SLACK_SECONDS) if closed is not None else None
    matched = [
        row for row in activity
        if row.get("type") == "TRADE"
        and row.get("conditionId") == position["condition_id"]
        and row.get("outcomeIndex") == outcome_index
        and (lo is None or (row.get("timestamp") or 0) >= lo)
        and (hi is None or (row.get("timestamp") or 0) <= hi)
    ]
    return sorted(matched, key=lambda row: row.get("timestamp") or 0)


def compute_adjustment(position: dict, matched: list[dict]) -> dict:
    """True realized PnL for a CLOSED position from its matched activity, and
    the delta against what the DB already realized.

    realized_true = SELL usdcSize - BUY usdcSize - estimated cost of any buy
    fills the feed is missing (missing shares x the side's entry price; see the
    module docstring). delta = realized_true - realized_pnl_usd(DB)."""
    buys_usd = sum(r.get("usdcSize") or 0.0 for r in matched if r.get("side") == "BUY")
    sells_usd = sum(r.get("usdcSize") or 0.0 for r in matched if r.get("side") == "SELL")
    bought_shares = sum(r.get("size") or 0.0 for r in matched if r.get("side") == "BUY")
    sold_shares = sum(r.get("size") or 0.0 for r in matched if r.get("side") == "SELL")

    # A closed position holds nothing, so every sold share was bought; feed
    # BUYs covering fewer shares than the SELLs mean the feed lost buy fills.
    missing_buy_shares = max(sold_shares - bought_shares, 0.0)
    incomplete_buy = missing_buy_shares > _SHARE_EPSILON
    if incomplete_buy:
        entry_yes = position.get("entry_yes_price") or 0.0
        side_entry = entry_yes if (position.get("side") or "").upper() == "YES" \
            else 1.0 - entry_yes
        est_missing_cost = missing_buy_shares * side_entry
    else:
        missing_buy_shares = 0.0
        est_missing_cost = 0.0

    realized_true = sells_usd - buys_usd - est_missing_cost
    already = position.get("realized_pnl_usd") or 0.0
    return {
        "position_id": position["id"],
        "status": position.get("status"),
        "side": position.get("side"),
        "market_question": position.get("market_question") or "",
        "trades_matched": len(matched),
        "buys_usd": round(buys_usd, 4),
        "sells_usd": round(sells_usd, 4),
        "bought_shares": round(bought_shares, 4),
        "sold_shares": round(sold_shares, 4),
        "missing_buy_shares": round(missing_buy_shares, 4),
        "est_missing_cost_usd": round(est_missing_cost, 4),
        "incomplete_buy": incomplete_buy,
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
        f"{row['trades_matched']} TRADE rows matched — "
        f"BUY ${row['buys_usd']:.4f} ({row['bought_shares']:g} sh), "
        f"SELL ${row['sells_usd']:.4f} ({row['sold_shares']:g} sh); "
        f"realized_true ${row['realized_true']:+.4f} "
        f"minus already recorded ${row['already_realized']:+.4f} "
        f"= delta ${row['delta']:+.4f}"
    )
    if row["incomplete_buy"]:
        text += (
            f". BUY rows incomplete in the feed: {row['missing_buy_shares']:g} sh "
            f"missing, costed at the entry price "
            f"(est ${row['est_missing_cost_usd']:.4f})"
        )
    return text


def backfill_realized(apply: bool = False, ids: list[int] | None = None,
                      wallet: str | None = None,
                      fetcher=fetch_wallet_activity) -> dict:
    """Compute (and with apply=True, write) realized-PnL corrections for closed
    positions from the wallet's Data API activity.

    Open positions are never touched — their basis/realized bookkeeping belongs
    to the live close paths. `ids` restricts which positions may be WRITTEN
    (the report always covers every closed position with matched activity).
    Writes are the delta only, guarded by status != 'open', with one
    exit_decisions audit row (trigger='backfill', decision='pnl_backfill') per
    adjusted position. Returns {"rows", "applied", "skipped_ids", "wallet"}."""
    wallet = wallet or config.POLYMARKET_FUNDER_ADDRESS
    if not wallet:
        raise RuntimeError(
            "No wallet to audit: POLYMARKET_FUNDER_ADDRESS is not set"
        )
    activity = [r for r in fetcher(wallet) if r.get("type") == "TRADE"]

    conn = logger._conn()
    try:
        positions = [dict(r) for r in conn.execute(
            "SELECT * FROM positions WHERE status != 'open' ORDER BY id"
        ).fetchall()]

        rows: list[dict] = []
        for position in positions:
            matched = match_position_activity(position, activity)
            if not matched:
                continue
            row = compute_adjustment(position, matched)
            row["writable"] = (
                abs(row["delta"]) >= MIN_ADJUSTMENT_USD
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
            "wallet": wallet}
