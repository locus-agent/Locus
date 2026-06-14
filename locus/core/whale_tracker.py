"""
Whale tracking — shadow top-performing wallets and investigate the markets they
move that we missed.

A small set of consistently profitable wallets (`WHALE_WALLETS`) is polled from
Polymarket's public trades feed. When one of them *opens* a position (a BUY) on
a niche market we track but had no actionable read on, that's a "missed
opportunity": the pipeline asks Claude whether it's worth investigating and, if
so, runs the market through the normal classify -> gates -> trade path with
edge_type='whale'.

This module is pure data + two Claude-free helpers plus one Claude call
(`decide_investigation`); the async scheduling lives in the pipeline.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import anthropic
import httpx

from locus import config

log = logging.getLogger(__name__)


def fetch_recent_whale_trades(
    since_minutes: float = 30,
    wallets: list[str] | None = None,
    now: datetime | None = None,
) -> list[dict]:
    """Recent trades by the watched wallets, newest first.

    Polls the public trades feed once per wallet (the feed supports a `user`
    filter), keeps BUYs within the last `since_minutes` (whales *opening*
    positions), and returns dicts of
    {wallet, market_token_id, condition_id, side, outcome, size_usd,
     timestamp, title}.

    Empty WHALE_WALLETS -> [] (whale tracking disabled). Network/parse errors
    on one wallet are logged and skipped, never raised.
    """
    wallets = wallets if wallets is not None else config.WHALE_WALLETS
    if not wallets:
        return []

    now = now or datetime.now(timezone.utc)
    cutoff_ts = now.timestamp() - since_minutes * 60.0
    out: list[dict] = []

    with httpx.Client(timeout=15) as client:
        for wallet in wallets:
            try:
                resp = client.get(
                    f"{config.POLYMARKET_DATA_HOST}/trades",
                    params={"user": wallet, "limit": 100},
                )
                resp.raise_for_status()
                trades = resp.json()
            except (httpx.HTTPError, ValueError) as e:
                log.warning(f"[whale] trade fetch failed for {wallet[:10]}: {e}")
                continue

            for t in trades if isinstance(trades, list) else []:
                try:
                    ts = float(t.get("timestamp", 0) or 0)
                except (TypeError, ValueError):
                    continue
                if ts < cutoff_ts:
                    continue
                if str(t.get("side", "")).upper() != "BUY":
                    continue  # opening a position, not closing
                size = float(t.get("size", 0) or 0)
                price = float(t.get("price", 0) or 0)
                out.append({
                    "wallet": str(t.get("proxyWallet", "") or "").lower(),
                    "market_token_id": str(t.get("asset", "") or ""),
                    "condition_id": str(t.get("conditionId", "") or ""),
                    "side": str(t.get("side", "") or "").upper(),
                    "outcome": str(t.get("outcome", "") or ""),
                    "size_usd": round(size * price, 2),
                    "timestamp": ts,
                    "title": str(t.get("title", "") or ""),
                })

    out.sort(key=lambda x: x["timestamp"], reverse=True)
    return out


def _hours_to_close(end_date: str, now: datetime) -> float | None:
    """Hours from `now` until the market closes, or None if unparseable."""
    try:
        end = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
    except (ValueError, AttributeError, TypeError):
        return None
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return (end - now).total_seconds() / 3600.0


def _has_actionable_classification(db_conn, condition_id: str, since: str) -> bool:
    """True if we already made a directional, non-skipped call on this market
    since `since` — i.e. we did NOT miss it."""
    row = db_conn.execute(
        """SELECT 1 FROM classifications
           WHERE condition_id = ?
             AND created_at >= ?
             AND direction IN ('bullish', 'bearish')
             AND action != 'skip'
           LIMIT 1""",
        (condition_id, since),
    ).fetchone()
    return row is not None


def _on_cooldown(db_conn, condition_id: str, since: str) -> bool:
    """True if this market was whale-triggered within the cooldown window."""
    row = db_conn.execute(
        """SELECT 1 FROM classifications
           WHERE condition_id = ?
             AND action = 'whale_triggered'
             AND created_at >= ?
           LIMIT 1""",
        (condition_id, since),
    ).fetchone()
    return row is not None


def find_missed_opportunities(
    whale_trades: list[dict],
    tracked_markets: list,
    db_conn,
    *,
    now: datetime | None = None,
    min_trade_usd: float = 0.0,
    classification_lookback_hours: float | None = None,
    min_hours_to_close: float | None = None,
    cooldown_hours: float | None = None,
) -> list[dict]:
    """Whale trades on tracked markets we plausibly missed.

    A trade is a missed opportunity when, for a market we track:
      - we had NO classification on it in the last
        `classification_lookback_hours`, OR every such classification was
        neutral/skip (no actionable read), AND
      - the market is not closing within `min_hours_to_close` (too late), AND
      - the market is not already on whale cooldown.

    At most one opportunity per market per call (largest trade wins). Each
    result is the trade dict with the resolved `market` attached.
    """
    now = now or datetime.now(timezone.utc)
    if classification_lookback_hours is None:
        classification_lookback_hours = config.WHALE_CLASSIFICATION_LOOKBACK_HOURS
    if min_hours_to_close is None:
        min_hours_to_close = config.WHALE_MIN_HOURS_TO_CLOSE
    if cooldown_hours is None:
        cooldown_hours = config.WHALE_COOLDOWN_HOURS

    class_since = (
        now - _timedelta_hours(classification_lookback_hours)
    ).strftime("%Y-%m-%d %H:%M:%S")
    cooldown_since = (
        now - _timedelta_hours(cooldown_hours)
    ).strftime("%Y-%m-%d %H:%M:%S")

    by_condition = {m.condition_id: m for m in tracked_markets}
    by_token: dict[str, object] = {}
    for m in tracked_markets:
        for tok in (getattr(m, "tokens", None) or []):
            tid = tok.get("token_id")
            if tid:
                by_token[str(tid)] = m

    # Largest trade per market first, so the one opportunity we keep is the biggest.
    ordered = sorted(whale_trades, key=lambda t: t.get("size_usd", 0), reverse=True)

    missed: list[dict] = []
    seen: set[str] = set()
    for tr in ordered:
        if tr.get("size_usd", 0) < min_trade_usd:
            continue
        market = by_condition.get(tr.get("condition_id")) or by_token.get(tr.get("market_token_id"))
        if market is None:
            continue  # not a market we track
        cid = market.condition_id
        if cid in seen:
            continue

        ttl = _hours_to_close(getattr(market, "end_date", ""), now)
        if ttl is not None and ttl < min_hours_to_close:
            continue  # too late to act
        if _on_cooldown(db_conn, cid, cooldown_since):
            continue  # already investigated recently
        if _has_actionable_classification(db_conn, cid, class_since):
            continue  # we didn't miss it

        seen.add(cid)
        missed.append({**tr, "market": market})

    return missed


def _timedelta_hours(hours: float):
    from datetime import timedelta
    return timedelta(hours=hours)


# --- Claude: should we investigate this whale trade? ------------------------

INVESTIGATE_PROMPT = """You are Locus, an autonomous agent trading niche Polymarket \
prediction markets. You shadow a few consistently profitable "whale" wallets. One of them \
just opened a position on a market you had not acted on. Decide whether it is worth \
investigating now.

## Market
Question: {question}
Current YES price: {yes_price:.3f} (implied probability: {yes_price:.1%})
Time until market close: {time_remaining}

## Whale activity
Wallet: {wallet}
Action: bought {outcome} (${size_usd:,.0f} notional)

## Task
Decide "investigate" (run a full analysis and possibly trade) or "hold" (skip — the whale's \
move looks unremarkable, too late, already priced in, or not your kind of edge). Investigate \
when a sharp wallet taking real size on a market you missed plausibly reflects information \
you should price in while there is still room and time to act.

Respond with ONLY valid JSON:
{{"decision": "investigate" | "hold", "reasoning": "<1 sentence>"}}"""


def whale_headline(opp: dict) -> str:
    """Synthetic headline describing the whale activity, fed to classify()."""
    return (
        f"Tracked whale wallet {opp.get('wallet', '')[:10]} bought "
        f"{opp.get('outcome', '')} (${opp.get('size_usd', 0):,.0f}) on "
        f"\"{opp.get('title') or (opp.get('market').question if opp.get('market') else '')}\""
    )


def decide_investigation(opp: dict, now: datetime | None = None) -> str:
    """One Claude call: 'investigate' or 'hold' for a missed whale opportunity.
    Fails closed to 'hold' on any error (never trade off a broken call)."""
    market = opp["market"]
    now = now or datetime.now(timezone.utc)
    # Local import avoids a circular import at module load (classifier -> memory).
    from locus.core.classifier import _format_time_remaining

    prompt = INVESTIGATE_PROMPT.format(
        question=market.question,
        yes_price=market.yes_price,
        time_remaining=_format_time_remaining(getattr(market, "end_date", ""), now),
        wallet=opp.get("wallet", ""),
        outcome=opp.get("outcome", ""),
        size_usd=opp.get("size_usd", 0),
    )
    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=config.SCORING_MODEL, max_tokens=150, temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1].removeprefix("json").strip()
        decision = json.loads(text).get("decision", "hold")
        return "investigate" if decision == "investigate" else "hold"
    except Exception as e:
        log.warning(f"[whale] investigation decision failed: {e}")
        return "hold"
