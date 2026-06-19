#!/usr/bin/env python3
"""
Polymarket Pipeline — CLI Interface

Usage:
    python cli.py watch                # Event-driven pipeline (real-time news → classify → trade)
    python cli.py watch --live         # With live trading
    python cli.py dashboard            # Launch live terminal dashboard (TUI)
    python cli.py calibrate            # Show classification accuracy report
    python cli.py niche                # Browse niche markets (< $500K volume)
    python cli.py verify               # Check all API keys and connections
    python cli.py scrape               # Test news scraper only
    python cli.py markets              # Browse all active markets
    python cli.py trades               # View trade log
    python cli.py close <position_id>  # Manually close an open position
    python cli.py stats                # Performance statistics
    python cli.py evolve               # Manually evolve the classification prompt
    python cli.py suggestion list      # List pending threshold-adjustment suggestions
    python cli.py suggestion review <id>  # Mark a suggestion as reviewed
"""

import argparse
import sys

from rich.console import Console
from rich.table import Table

console = Console()


def cmd_watch(args):
    """V2: Event-driven pipeline — real-time news → classify → trade."""
    from datetime import datetime, timezone
    from locus import config
    from locus.core.pipeline import run_pipeline_v2

    # Stamp the launch time so export_status can publish uptime on the dashboard.
    watch_start_time = datetime.now(timezone.utc)
    config.WATCH_START_TIME = watch_start_time

    if args.live:
        config.DRY_RUN = False
        console.print("[red bold]LIVE TRADING ENABLED[/red bold]\n")
    else:
        console.print("[yellow]Dry-run mode (use --live to trade for real)[/yellow]\n")

    if args.threshold:
        config.MATERIALITY_THRESHOLD_BULLISH = args.threshold
        config.MATERIALITY_THRESHOLD_BEARISH = args.threshold

    # Real-time Telegram notifications + interactive /portfolio bot (no-op when
    # TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are unset). Runs in a daemon thread.
    from locus.core import telegram_bot
    if telegram_bot.start_bot_polling() is not None:
        console.print("[dim]Telegram interactive bot polling started[/dim]")

    run_pipeline_v2()


def cmd_calibrate(args):
    """Show classification accuracy report."""
    from locus.memory.calibrator import check_resolutions, get_report, grade_classifications
    from locus import memory
    from rich.panel import Panel

    console.print("[bold]Checking for resolved markets...[/bold]")
    resolved = check_resolutions()
    if resolved:
        console.print(f"  Updated {resolved} trade resolutions")

    console.print("[bold]Grading non-traded classifications vs price moves...[/bold]")
    graded = grade_classifications()
    if graded:
        console.print(f"  Graded {graded} classifications")
    record = memory.get_track_record()
    console.print(
        f"  Combined track record: {record['total']} graded calls, "
        f"{record['accuracy']:.1f}% accurate"
    )

    report = get_report()

    console.print(Panel(f"[bold]CALIBRATION REPORT[/bold]", style="bright_cyan"))
    console.print(f"  Total resolved: {report.total}")
    console.print(f"  Accuracy: {report.accuracy:.1f}%")

    if report.by_source:
        console.print(f"\n  [bold]By Source:[/bold]")
        for source, acc in report.by_source.items():
            color = "bright_green" if acc >= 55 else ("yellow" if acc >= 45 else "red")
            console.print(f"    {source}: [{color}]{acc:.1f}%[/{color}]")

    if report.by_classification:
        console.print(f"\n  [bold]By Classification:[/bold]")
        for cls, acc in report.by_classification.items():
            color = "bright_green" if acc >= 55 else ("yellow" if acc >= 45 else "red")
            console.print(f"    {cls}: [{color}]{acc:.1f}%[/{color}]")

    console.print(f"\n  [dim]{report.recommendation}[/dim]")


def cmd_niche(args):
    """Browse niche markets only (volume-filtered)."""
    from locus import config
    from locus.markets.gamma import fetch_active_markets, filter_by_categories

    all_markets = fetch_active_markets(limit=200)
    categorized = filter_by_categories(all_markets)
    niche = [
        m for m in categorized
        if config.MIN_VOLUME_USD <= m.volume <= config.MAX_VOLUME_USD
    ]

    console.print(f"\n[bold]{len(niche)} niche markets[/bold] (${config.MIN_VOLUME_USD:,.0f} - ${config.MAX_VOLUME_USD:,.0f} volume)\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Category", width=12)
    table.add_column("Question", max_width=50)
    table.add_column("YES", justify="right")
    table.add_column("NO", justify="right")
    table.add_column("Volume", justify="right")

    for m in niche[:30]:
        table.add_row(
            m.category,
            m.question[:50],
            f"{m.yes_price:.2f}",
            f"{m.no_price:.2f}",
            f"${m.volume:,.0f}",
        )

    console.print(table)


def cmd_dashboard(args):
    try:
        from locus.ui.tui import run_tui
    except ImportError:
        console.print("[red]Textual not installed — run: pip install textual[/red]")
        return
    run_tui()


def cmd_verify(args):
    """Check all API keys and connections work."""
    from rich.panel import Panel

    console.print(Panel("[bold]POLYMARKET PIPELINE V2 — VERIFICATION[/bold]", style="bright_green"))
    all_good = True

    # 1. Python version
    v = sys.version_info
    py_ok = v.major == 3 and v.minor >= 9
    status = "[bright_green]PASS[/bright_green]" if py_ok else "[red]FAIL[/red]"
    console.print(f"  {status}  Python {v.major}.{v.minor}.{v.micro}")
    if not py_ok:
        all_good = False

    # 2. Dependencies
    deps_ok = True
    for mod in ["anthropic", "feedparser", "httpx", "rich", "dotenv", "websockets"]:
        try:
            __import__(mod)
        except ImportError:
            console.print(f"  [red]FAIL[/red]  Missing module: {mod}")
            deps_ok = False
            all_good = False
    if deps_ok:
        console.print(f"  [bright_green]PASS[/bright_green]  All dependencies installed")

    # 3. .env exists
    import os
    env_exists = os.path.exists(os.path.join(os.path.dirname(__file__), ".env"))
    status = "[bright_green]PASS[/bright_green]" if env_exists else "[red]FAIL[/red] — run: cp .env.example .env"
    console.print(f"  {status}  .env file")
    if not env_exists:
        all_good = False

    # 4. Anthropic API key
    from locus import config
    has_key = bool(config.ANTHROPIC_API_KEY) and config.ANTHROPIC_API_KEY != "sk-ant-..."
    if has_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=10,
                messages=[{"role": "user", "content": "Say OK"}],
            )
            console.print(f"  [bright_green]PASS[/bright_green]  Anthropic API key (verified)")
        except Exception as e:
            console.print(f"  [red]FAIL[/red]  Anthropic API key — {type(e).__name__}: {e}")
            all_good = False
    else:
        console.print(f"  [red]FAIL[/red]  Anthropic API key not set")
        all_good = False

    # 5. News scraper (RSS)
    try:
        from locus.sources.scraper import scrape_rss
        items = scrape_rss(config.RSS_FEEDS[0], 12)
        console.print(f"  [bright_green]PASS[/bright_green]  RSS scraper ({len(items)} headlines)")
    except Exception as e:
        console.print(f"  [yellow]WARN[/yellow]  RSS scraper — {e}")

    # 6. Twitter API (optional)
    has_twitter = bool(config.TWITTER_BEARER_TOKEN)
    if has_twitter:
        console.print(f"  [bright_green]PASS[/bright_green]  Twitter bearer token set")
    else:
        console.print(f"  [dim]SKIP[/dim]  Twitter API (optional — enables real-time news stream)")

    # 7. Telegram (optional)
    has_telegram = bool(config.TELEGRAM_BOT_TOKEN)
    if has_telegram:
        console.print(f"  [bright_green]PASS[/bright_green]  Telegram bot token set")
    else:
        console.print(f"  [dim]SKIP[/dim]  Telegram bot (optional — enables channel monitoring)")

    # 8. NewsAPI (optional)
    has_newsapi = bool(config.NEWSAPI_KEY)
    if has_newsapi:
        try:
            import httpx
            resp = httpx.get(
                "https://newsapi.org/v2/top-headlines",
                params={"country": "us", "pageSize": 1, "apiKey": config.NEWSAPI_KEY},
                timeout=15,
            )
            resp.raise_for_status()
            total = resp.json().get("totalResults", 0)
            console.print(f"  [bright_green]PASS[/bright_green]  NewsAPI key (verified, {total} headlines available)")
        except Exception as e:
            console.print(f"  [red]FAIL[/red]  NewsAPI key — {type(e).__name__}: {e}")
            all_good = False
    else:
        console.print(f"  [dim]SKIP[/dim]  NewsAPI (optional — broader news coverage)")

    # 9. Polymarket API
    try:
        from locus.markets.gamma import fetch_active_markets
        mkts = fetch_active_markets(limit=5)
        console.print(f"  [bright_green]PASS[/bright_green]  Polymarket API ({len(mkts)} markets)")
    except Exception as e:
        console.print(f"  [yellow]WARN[/yellow]  Polymarket API — {e}")

    # 10. Niche market filter
    try:
        from locus.markets.gamma import fetch_active_markets, filter_by_categories
        all_m = fetch_active_markets(limit=100)
        cat = filter_by_categories(all_m)
        niche = [m for m in cat if config.MIN_VOLUME_USD <= m.volume <= config.MAX_VOLUME_USD]
        console.print(f"  [bright_green]PASS[/bright_green]  Niche filter ({len(niche)} markets in range)")
    except Exception as e:
        console.print(f"  [yellow]WARN[/yellow]  Niche filter — {e}")

    # 11. Polymarket trading credentials (optional)
    has_poly = bool(config.POLYMARKET_API_KEY)
    if has_poly:
        console.print(f"  [bright_green]PASS[/bright_green]  Polymarket trading credentials set")
    else:
        console.print(f"  [dim]SKIP[/dim]  Polymarket trading credentials (optional — needed for --live)")

    # 12. SQLite
    try:
        from locus.memory import logger as _
        console.print(f"  [bright_green]PASS[/bright_green]  SQLite database (V2 schema)")
    except Exception as e:
        console.print(f"  [red]FAIL[/red]  SQLite — {e}")
        all_good = False

    # Summary
    console.print()
    if all_good:
        console.print(Panel(
            "[bright_green bold]ALL CHECKS PASSED[/bright_green bold]\n\n"
            "You're ready to go. Run:\n"
            "  python cli.py watch             # Event-driven pipeline\n"
            "  python cli.py dashboard          # Live terminal dashboard (TUI)\n"
            "  python cli.py watch --live       # Real trading (careful!)",
            style="bright_green",
        ))
    else:
        console.print(Panel(
            "[yellow bold]SOME CHECKS FAILED[/yellow bold]\n\n"
            "Fix the issues above, then run: python cli.py verify",
            style="yellow",
        ))


def cmd_scrape(args):
    from locus.sources.scraper import scrape_all

    news = scrape_all(args.hours)
    console.print(f"\n[bold]Scraped {len(news)} headlines[/bold] (last {args.hours}h)\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Age", justify="right", width=6)
    table.add_column("Source", max_width=20)
    table.add_column("Headline", max_width=80)

    for item in news[:30]:
        table.add_row(f"{item.age_hours():.1f}h", item.source[:20], item.headline[:80])

    console.print(table)


def cmd_markets(args):
    from locus.markets.gamma import fetch_active_markets, filter_by_categories

    all_markets = fetch_active_markets(limit=args.max)
    markets = filter_by_categories(all_markets)

    console.print(f"\n[bold]{len(markets)} markets in target categories[/bold] (of {len(all_markets)} fetched)\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Category", width=12)
    table.add_column("Question", max_width=60)
    table.add_column("YES", justify="right")
    table.add_column("NO", justify="right")
    table.add_column("Volume", justify="right")

    for m in markets:
        table.add_row(m.category, m.question[:60], f"{m.yes_price:.2f}", f"{m.no_price:.2f}", f"${m.volume:,.0f}")

    console.print(table)


def cmd_trades(args):
    from locus.memory import logger

    trades = logger.get_recent_trades(limit=args.limit)
    if not trades:
        console.print("[yellow]No trades logged yet.[/yellow]")
        return

    console.print(f"\n[bold]Last {len(trades)} trades[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("ID", justify="right", width=4)
    table.add_column("Market", max_width=35)
    table.add_column("Signal", width=8)
    table.add_column("Mat.", justify="right", width=5)
    table.add_column("Side", width=4)
    table.add_column("Edge", justify="right", width=6)
    table.add_column("Bet", justify="right", width=7)
    table.add_column("Src", width=6)
    table.add_column("Lat.", justify="right", width=6)
    table.add_column("Status", width=8)

    for t in trades:
        cls = t.get("classification") or "—"
        mat = f"{t.get('materiality', 0) or 0:.2f}"
        src = (t.get("news_source") or "—")[:6]
        lat = f"{t.get('total_latency_ms') or 0}ms"
        table.add_row(
            str(t["id"]),
            t["market_question"][:35],
            cls[:8],
            mat,
            t["side"],
            f"{t['edge']:.1%}",
            f"${t['amount_usd']:.2f}",
            src,
            lat,
            t["status"][:8],
        )

    console.print(table)


def cmd_close(args):
    """Manually close an open position at its last-marked price."""
    from locus import config
    from locus.core import positions

    result = positions.close_manual(args.position_id)
    if result is None:
        console.print(
            f"[red]Position #{args.position_id} not found or already closed.[/red]"
        )
        sys.exit(1)

    if config.DRY_RUN:
        console.print("[yellow]Dry-run mode — simulated close.[/yellow]")
    else:
        console.print(
            "[red bold]LIVE mode[/red bold] — CLOB sell order not yet wired up; "
            "recorded the close at the marked price."
        )

    console.print(
        f"Closed position #{result['id']}: {result['market_question']} "
        f"at {result['price']:.3f} (PnL: {result['pnl_pct']:+.1f}%)"
    )


def cmd_stats(args):
    from locus.memory import logger

    stats = logger.get_trade_stats()
    daily = logger.get_daily_pnl()
    latency = logger.get_latency_stats()
    cal = logger.get_calibration_stats()

    console.print(f"\n[bold]Pipeline Statistics[/bold]\n")
    console.print(f"  Total signals: {stats['total_trades']}")
    console.print(f"  Daily exposure: ${abs(daily):.2f}")
    console.print(f"  By status:")
    for status, count in stats["by_status"].items():
        console.print(f"    {status}: {count}")

    if latency["count"] > 0:
        console.print(f"\n  [bold]Latency:[/bold]")
        console.print(f"    Avg total: {latency['avg_total_ms']}ms")
        console.print(f"    Avg news: {latency['avg_news_ms']}ms")
        console.print(f"    Avg classification: {latency['avg_class_ms']}ms")

    if cal["total"] > 0:
        console.print(f"\n  [bold]Calibration:[/bold]")
        console.print(f"    Accuracy: {cal['accuracy']:.1f}% ({cal['total']} resolved)")


def cmd_evolve(args):
    """Manually trigger a meta-prompt evolution (normally weekly, after journal)."""
    import asyncio
    from locus.memory import meta_evolver

    console.print("[cyan]Evolving classification prompt from lessons + accuracy...[/cyan]")
    result = asyncio.run(meta_evolver.evolve_prompt())
    if result:
        console.print(
            f"[green]Evolved to v{result['version']}[/green] "
            f"({result['lessons_count']} lessons, accuracy {result['accuracy_at_creation']}%, "
            f"{result['prev_chars']} → {result['chars']} chars)"
        )
        console.print(f"  Saved to [dim]{result['path']}[/dim]")
    else:
        console.print(
            "[yellow]No new prompt saved[/yellow] — the model's output failed "
            "validation or generation errored (see logs)."
        )


def cmd_suggestion(args):
    """List or review the calibrator's missed-opportunity threshold suggestions."""
    from locus.memory import logger

    if args.action == "list":
        suggestions = logger.get_pending_suggestions()
        if not suggestions:
            console.print("[yellow]No pending suggestions.[/yellow]")
            return

        console.print(f"\n[bold]{len(suggestions)} pending suggestion(s)[/bold]\n")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("ID", justify="right", width=4)
        table.add_column("Type", width=18)
        table.add_column("Category", width=12)
        table.add_column("Avg Move", justify="right", width=8)
        table.add_column("Misses", justify="right", width=6)
        table.add_column("Suggestion", max_width=60)

        for s in suggestions:
            avg = f"+{s['avg_pct_move']:.0f}%" if s.get("avg_pct_move") is not None else "—"
            misses = str(s["miss_count"]) if s.get("miss_count") is not None else "—"
            table.add_row(
                str(s["id"]), s["suggestion_type"], s.get("category") or "—",
                avg, misses, s["suggestion_text"],
            )

        console.print(table)
        console.print(
            "\n[dim]Mark one reviewed with: python3 cli.py suggestion review <id>[/dim]"
        )

    elif args.action == "review":
        pending_ids = {s["id"] for s in logger.get_pending_suggestions()}
        if args.id not in pending_ids:
            console.print(
                f"[red]Suggestion #{args.id} not found among pending suggestions.[/red]"
            )
            sys.exit(1)
        logger.mark_suggestion_reviewed(args.id)
        console.print(f"[green]Marked suggestion #{args.id} as reviewed.[/green]")


def main():
    parser = argparse.ArgumentParser(description="Polymarket Pipeline V2")
    sub = parser.add_subparsers(dest="command")

    # watch
    p_watch = sub.add_parser("watch", help="Event-driven pipeline (real-time)")
    p_watch.add_argument("--live", action="store_true", help="Enable live trading")
    p_watch.add_argument("--threshold", type=float, default=None, help="Materiality threshold override")
    p_watch.set_defaults(func=cmd_watch)

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Launch live terminal dashboard (TUI)")
    p_dash.set_defaults(func=cmd_dashboard)

    # calibrate
    p_cal = sub.add_parser("calibrate", help="Show classification accuracy report")
    p_cal.set_defaults(func=cmd_calibrate)

    # niche
    p_niche = sub.add_parser("niche", help="Browse niche markets (volume-filtered)")
    p_niche.set_defaults(func=cmd_niche)

    # verify
    p_verify = sub.add_parser("verify", help="Check API keys and connections")
    p_verify.set_defaults(func=cmd_verify)

    # scrape
    p_scrape = sub.add_parser("scrape", help="Test the news scraper")
    p_scrape.add_argument("--hours", type=int, default=6, help="Lookback hours")
    p_scrape.set_defaults(func=cmd_scrape)

    # markets
    p_markets = sub.add_parser("markets", help="View all available markets")
    p_markets.add_argument("--max", type=int, default=50, help="Max markets to fetch")
    p_markets.set_defaults(func=cmd_markets)

    # trades
    p_trades = sub.add_parser("trades", help="View trade log")
    p_trades.add_argument("--limit", type=int, default=20, help="Number of trades to show")
    p_trades.set_defaults(func=cmd_trades)

    # close
    p_close = sub.add_parser("close", help="Manually close an open position by id")
    p_close.add_argument("position_id", type=int, help="Position id to close")
    p_close.set_defaults(func=cmd_close)

    # stats
    p_stats = sub.add_parser("stats", help="Performance statistics")
    p_stats.set_defaults(func=cmd_stats)

    # evolve
    p_evolve = sub.add_parser("evolve", help="Manually evolve the classification prompt")
    p_evolve.set_defaults(func=cmd_evolve)

    # suggestion (list / review missed-opportunity threshold suggestions)
    p_sugg = sub.add_parser("suggestion", help="View / review adjustment suggestions")
    sugg_sub = p_sugg.add_subparsers(dest="action", required=True)
    sugg_sub.add_parser("list", help="Show all pending suggestions")
    p_sugg_review = sugg_sub.add_parser("review", help="Mark a suggestion as reviewed")
    p_sugg_review.add_argument("id", type=int, help="Suggestion id to mark reviewed")
    p_sugg.set_defaults(func=cmd_suggestion)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
