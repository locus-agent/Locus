"""
Textual TUI dashboard — live, read-only view over trades.db and docs/status.json.

Runs alongside `python cli.py watch` (or on its own): it only SELECTs from the
SQLite database (WAL mode allows a concurrent writer) and never scans markets,
classifies, or trades. Colors match the web dashboard (docs/index.html).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Static

from locus import config
from locus.memory import logger
from locus import memory

STATUS_PATH = config.PROJECT_ROOT / "docs" / "status.json"

PURPLE = "#534AB7"
GREEN = "#0f6e56"
RED = "#a32d2d"

DIRECTION_STYLES = {
    "bullish": f"bold {GREEN}",
    "bearish": f"bold {RED}",
    "neutral": "dim",
}

ACTION_STYLES = {
    "SIGNAL": f"bold {GREEN}",
    "STALE": "dim yellow",
    "CAPPED": "dim cyan",
    "ERROR": f"bold {RED}",
}

MATCH_BADGES = {
    "keyword": ("kw", "dim"),
    "embedding": ("sem", PURPLE),
    "both": ("both", GREEN),
}


def _read_status_json() -> dict:
    try:
        return json.loads(STATUS_PATH.read_text())
    except Exception:
        return {}


def _news_counts_by_source(since: str) -> dict[str, int]:
    conn = logger._conn()
    rows = conn.execute(
        "SELECT source, COUNT(*) AS c FROM news_events WHERE created_at >= ? GROUP BY source",
        (since,),
    ).fetchall()
    conn.close()
    return {r["source"]: r["c"] for r in rows}


def _action_counts(since: str) -> dict[str, int]:
    """Classification counts by action (signal/skip/stale/capped) since `since`."""
    conn = logger._conn()
    rows = conn.execute(
        "SELECT action, COUNT(*) AS c FROM classifications WHERE created_at >= ? GROUP BY action",
        (since,),
    ).fetchall()
    conn.close()
    return {r["action"]: r["c"] for r in rows}


def _fmt_time(created_at: str) -> str:
    return (created_at or "")[11:19] or "--:--:--"


class LocusTUI(App):
    """Bloomberg-for-one-bot: header, stats, live feed, track record, open positions."""

    TITLE = "Locus"
    BINDINGS = [("q", "quit", "Quit")]

    CSS = f"""
    Screen {{
        background: #1a1a1a;
        color: #e2e2e2;
        layout: vertical;
    }}
    #header {{
        height: 1;
        background: {PURPLE};
        color: #fafafa;
        text-style: bold;
        padding: 0 1;
    }}
    #body {{
        height: 1fr;
    }}
    #stats, #record {{
        width: 32;
        border: round {PURPLE};
        padding: 0 1;
    }}
    #feed {{
        width: 1fr;
        border: round {PURPLE};
        background: #141414;
    }}
    #feed > DataTable {{
        background: #141414;
        height: 1fr;
    }}
    DataTable > .datatable--header {{
        background: #1a1a1a;
        color: {PURPLE};
        text-style: bold;
    }}
    #footer {{
        height: 7;
        border: round {GREEN};
        padding: 0 1;
    }}
    .panel-title {{
        color: {PURPLE};
        text-style: bold;
    }}
    """

    def __init__(self):
        super().__init__()
        self._started = time.monotonic()
        self._feed_last_id = -1

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        with Horizontal(id="body"):
            yield Static(id="stats")
            with Vertical(id="feed"):
                yield DataTable(cursor_type="none", zebra_stripes=False)
            yield Static(id="record")
        yield Static(id="footer")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_column("TIME", width=8)
        table.add_column("MARKET", width=32)
        table.add_column("HEADLINE", width=34)
        table.add_column("MATCH", width=5)
        table.add_column("DIR", width=8)
        table.add_column("MAT", width=4)
        table.add_column("ACTION", width=6)
        self.refresh_header()
        self.refresh_data()
        self.set_interval(2.0, self.refresh_data)
        self.set_interval(1.0, self.refresh_header)

    # --- refreshers -------------------------------------------------------

    def refresh_header(self) -> None:
        status = _read_status_json()
        dry = status.get("dry_run", config.DRY_RUN)
        mode = "DRY RUN" if dry else "LIVE"
        up = int(time.monotonic() - self._started)
        uptime = f"{up // 3600:02d}:{up % 3600 // 60:02d}:{up % 60:02d}"
        header = Text(" LOCUS ", style=f"bold #fafafa on {PURPLE}")
        header.append("  ● ", style="bold" if dry else f"bold {RED}")
        header.append(mode, style="bold")
        header.append(f"   up {uptime}", style="#e2e2e2")
        header.append("   q to quit", style="dim")
        self.query_one("#header", Static).update(header)

    def refresh_data(self) -> None:
        self._refresh_stats()
        self._refresh_feed()
        self._refresh_record()
        self._refresh_footer()

    def _refresh_stats(self) -> None:
        now = datetime.now(timezone.utc)
        since_24h = (now - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        by_source = _news_counts_by_source(since_24h)
        status = _read_status_json()
        trade_stats = logger.get_trade_stats()

        text = Text()
        text.append("STATS (24h)\n\n", style=f"bold {PURPLE}")
        text.append("news\n", style="bold")
        for source in ("twitter", "telegram", "rss", "newsapi"):
            text.append(f"  {source:<10}", style="dim")
            text.append(f"{by_source.get(source, 0):>6}\n")
        for source, count in sorted(by_source.items()):
            if source not in ("twitter", "telegram", "rss", "newsapi"):
                text.append(f"  {source:<10}", style="dim")
                text.append(f"{count:>6}\n")
        text.append("\n")
        acts = _action_counts(since_24h)
        classified = sum(acts.values())
        signals = acts.get("signal", 0)
        selectivity = f"{signals / classified * 100:.1f}%" if classified else "—"
        rows = [
            ("matched", logger.get_matched_headline_count_since(since_24h)),
            ("classified", classified),
            ("signals", signals),
            ("selectivity", selectivity),
            ("trades", trade_stats.get("total", 0)),
            ("markets", status.get("markets_tracked", "—")),
        ]
        for label, value in rows:
            text.append(f"{label:<12}", style="bold")
            text.append(f"{value:>6}\n")

        text.append("\ngates (24h)\n", style=f"bold {PURPLE}")
        text.append("  stale     ", style="dim")
        text.append(f"{acts.get('stale', 0):>6}\n", style="yellow")
        text.append("  capped    ", style="dim")
        text.append(f"{acts.get('capped', 0):>6}\n", style="cyan")
        self.query_one("#stats", Static).update(text)

    def _refresh_feed(self) -> None:
        rows = logger.get_recent_classifications(limit=50)
        if not rows:
            return
        newest_id = rows[0]["id"]
        if newest_id == self._feed_last_id:
            return
        self._feed_last_id = newest_id

        table = self.query_one(DataTable)
        table.clear()
        for c in rows:
            direction = (c["direction"] or "?").lower()
            dir_style = DIRECTION_STYLES.get(direction, "")
            action = (c["action"] or "").upper()
            mat = c["materiality"]
            badge, badge_style = MATCH_BADGES.get(c["match_source"], ("—", "dim"))
            table.add_row(
                Text(_fmt_time(c["created_at"]), style="dim"),
                (c["market_question"] or "")[:32],
                Text((c["headline"] or "")[:34], style="#e2e2e2"),
                Text(badge, style=badge_style),
                Text(direction, style=dir_style),
                f"{mat:.2f}" if mat is not None else "—",
                Text(action, style=ACTION_STYLES.get(action, "dim")),
            )

    def _refresh_record(self) -> None:
        record = memory.get_track_record()
        lessons = logger.get_recent_lessons(limit=5)

        text = Text()
        text.append("TRACK RECORD\n\n", style=f"bold {PURPLE}")
        text.append(f"resolved  {record['total']:>5}\n", style="bold")
        acc = record["accuracy"]
        acc_style = GREEN if acc >= 50 else RED
        text.append("accuracy  ", style="bold")
        text.append(f"{acc:>4.1f}%\n\n", style=f"bold {acc_style}" if record["total"] else "dim")
        for cat, pct in sorted(record["by_category"].items()):
            text.append(f"  {cat:<11}", style="dim")
            text.append(f"{pct:>5.1f}%\n", style=GREEN if pct >= 50 else RED)

        text.append("\nLESSONS\n", style=f"bold {PURPLE}")
        if not lessons:
            text.append("\n(none yet)", style="dim")
        for l in lessons:
            text.append(f"\n• ", style=PURPLE)
            text.append(l["lesson"].strip(), style="#e2e2e2")
            text.append("\n")
        self.query_one("#record", Static).update(text)

    def _refresh_footer(self) -> None:
        status = _read_status_json()
        mode = "DRY RUN" if status.get("dry_run", config.DRY_RUN) else "LIVE"
        trades = logger.get_recent_trades(limit=4)

        text = Text()
        text.append(f"OPEN POSITIONS ({mode})", style=f"bold {GREEN}")
        if not trades:
            text.append("\nnone yet — waiting for a signal to clear the gates", style="dim")
        for t in trades:
            side = (t["side"] or "?").upper()
            side_style = GREEN if side == "YES" else RED
            edge = t["edge"]
            text.append(f"\n{_fmt_time(t['created_at'])}  ", style="dim")
            text.append(f"{side:<3}", style=f"bold {side_style}")
            text.append(f" ${t['amount_usd']:>6.2f}", style="bold")
            text.append(f" @{t['market_price']:.2f}")
            text.append(f"  edge {edge:.0%}" if edge is not None else "  edge —", style="bold")
            text.append(f"  [{t['status']}]", style="dim")
            text.append(f"  {(t['market_question'] or '')[:52]}")
        self.query_one("#footer", Static).update(text)


def run_tui():
    LocusTUI().run()


if __name__ == "__main__":
    run_tui()
