# Locus

Locus is an autonomous agent that reads breaking news, classifies it with Claude, and trades niche Polymarket markets.

```
Breaking News (Twitter / Telegram / RSS)
        ↓ (< 5 seconds)
Match to niche markets (< $500K volume)
        ↓
Claude Classification: bullish / bearish / neutral + materiality
        ↓
Edge detection + quarter-Kelly sizing
        ↓
Instant execution → SQLite log → calibration tracking
```

## What Changed From V1

V1 scraped RSS feeds (5-60 min delay), asked Claude "what's the probability?" (wrong question for LLMs), and competed on high-volume markets (where every bot already operates).

V2 inverts all three:
- **Speed**: Real-time Twitter/Telegram streams instead of stale RSS
- **Classification**: Claude classifies "bullish or bearish?" instead of estimating probability — a task LLMs are actually good at
- **Niche markets**: Only trades markets under $500K volume where the crowd is small and slow

---

## Setup (2 minutes)

### One-Command Setup

```bash
git clone https://github.com/locus-agent/Locus.git
cd Locus
bash setup.sh
```

### Manual Setup

```bash
git clone https://github.com/locus-agent/Locus.git
cd Locus
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your keys to `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...         # Required
TWITTER_BEARER_TOKEN=...             # Optional — real-time news stream
TELEGRAM_BOT_TOKEN=...               # Optional — channel monitoring
POLYMARKET_API_KEY=...               # Optional — live trading only
```

### Verify

```bash
python cli.py verify
```

---

## How to Use

### V2: Event-Driven Pipeline (Recommended)

```bash
# Start the real-time pipeline — monitors news streams, classifies, trades
python cli.py watch

# Enable live trading
python cli.py watch --live
```

The `watch` command runs indefinitely. It connects to your configured news sources (Twitter, Telegram, RSS fallback), matches breaking headlines to niche Polymarket markets, classifies each with Claude, and executes trades when it finds edge.

### V1: Synchronous Pipeline

```bash
# Single scan — scrape RSS, score markets, log signals
python cli.py run

python cli.py run --max 15 --hours 12
```

### Live Dashboard

```bash
python cli.py dashboard
```

### Backtest

```bash
# Validate the V2 strategy against resolved markets
python cli.py backtest

python cli.py backtest --limit 50 --category ai
```

### All Commands

| Command | What it does |
|---|---|
| `python cli.py watch` | V2: Real-time event-driven pipeline |
| `python cli.py run` | V1: Synchronous RSS-based pipeline |
| `python cli.py dashboard` | Live terminal dashboard |
| `python cli.py backtest` | Backtest against resolved markets |
| `python cli.py calibrate` | Classification accuracy report |
| `python cli.py niche` | Browse niche markets (volume-filtered) |
| `python cli.py verify` | Check all API keys and connections |
| `python cli.py scrape` | Test news scraper |
| `python cli.py markets` | Browse all active markets |
| `python cli.py trades` | View trade log |
| `python cli.py stats` | Performance + latency + calibration stats |

---

## Architecture

```
                cli.py   (entry point: watch, run, verify, …)
                   │
                   ▼
              pipeline.py   (asyncio event loop)
                   │
       ┌───────────┴──────────────┐
       ▼                          ▼
 news_stream.py             market_watcher.py
 Twitter · RSS ·            niche markets ($1k–$500k volume),
 NewsAPI · Telegram         Gamma API refresh + WebSocket prices
 → deduped event queue            │
       │                          │
       └───────────┬──────────────┘
                   ▼
              matcher.py          headline → candidate markets
                   ▼
             classifier.py        Claude: bullish / bearish / neutral
                   │  ▲           + materiality (0–1)
                   │  └─────────────────────────────┐
                   ▼                                │ track record
                edge.py           signal filter +   │ + lessons
                   │              quarter-Kelly     │
                   ▼              sizing            │
             executor.py          dry-run or live   │
                   │              CLOB order,       │
                   ▼              daily loss limit  │
          logger.py (trades.db)                     │
                   │                                │
                   ├──► calibrator.py ──► memory.py ┘
                   │    resolved markets   accuracy stats
                   ▼    vs predictions     + lessons
           export_status.py
                   ▼
        docs/status.json → GitHub Pages dashboard
```

The loop at the right is the feedback cycle: as markets resolve, `calibrator.py` grades each
classification, `memory.py` turns the results into a track record and one-line lessons, and
`classifier.py` injects both into every future prompt — so Claude sees its own accuracy
history before classifying the next headline.

### Module Map

| File | What it does |
|---|---|
| `cli.py` | Entry point — `watch`, `run`, `dashboard`, `backtest`, `verify`, and friends |
| `pipeline.py` | Orchestrators: V2 asyncio event loop (`watch`) and V1 synchronous loop (`run`) |
| `news_stream.py` | Aggregates Twitter stream, RSS, NewsAPI, and Telegram into one deduped queue |
| `market_watcher.py` | Tracks niche markets — 5-min Gamma refresh, live WebSocket prices, momentum |
| `markets.py` | Polymarket Gamma API client, `Market` model, category inference |
| `matcher.py` | Matches headlines to tracked markets by keyword overlap |
| `classifier.py` | Asks Claude for direction + materiality, with track record injected |
| `edge.py` | Turns classifications into signals; quarter-Kelly position sizing |
| `executor.py` | Executes trades — dry-run log or live CLOB order; enforces daily loss limit |
| `logger.py` | SQLite store (`trades.db`): trades, outcomes, news events, calibration, lessons |
| `calibrator.py` | Grades classifications once markets resolve |
| `memory.py` | Track record + lessons — the classifier's feedback loop |
| `export_status.py` | Writes `docs/status.json` for the public GitHub Pages dashboard |
| `dashboard.py` | Terminal dashboard (runs the V1 scan loop) |
| `scraper.py` | V1 news scraper (RSS + NewsAPI) |
| `scorer.py` | V1 probability scoring with Claude |
| `backtest.py` | Replays resolved markets through the V2 classifier |
| `backtest_real.py` | Backtest pilot on real data: Gamma markets, GDELT/NewsAPI headlines, CLOB prices |
| `config.py` | All settings — `.env` keys, thresholds, market categories |

### How a Headline Becomes a Trade Decision

1. **A headline arrives** — say, from the Twitter stream. `news_stream.py` dedupes it against recent events and stamps how long it took to arrive.
2. **Matching** — `matcher.py` compares its words against every tracked niche market's question and keeps the few markets it might be about. No API call, just keyword overlap.
3. **Classification** — for each candidate market, `classifier.py` asks Claude one question: does this news make the market *more likely YES*, *more likely NO*, or is it *not relevant*? Claude also scores materiality — how much this news should move the price — and sees its own historical accuracy and recent mistakes before answering.
4. **Edge check** — `edge.py` drops the signal unless the direction is non-neutral, materiality clears the threshold, and the price actually has room to move (no buying YES at 0.95).
5. **Sizing** — surviving signals are sized with quarter-Kelly, capped at `MAX_BET_USD`.
6. **Execution** — `executor.py` checks the daily loss limit, then either logs a dry-run trade (default) or places a real CLOB order.
7. **Logging** — the trade, the headline, and the news-to-decision latency land in `trades.db`, and the public dashboard updates.
8. **Learning** — when the market eventually resolves, `calibrator.py` grades the call. Wrong calls become one-line lessons that Claude reads next time, closing the loop.

---

## Configuration

| Setting | Default | What it does |
|---|---|---|
| `DRY_RUN` | `true` | Set to `false` for live trading |
| `MAX_BET_USD` | `25` | Maximum single bet |
| `DAILY_LOSS_LIMIT_USD` | `100` | Pipeline halts if breached |
| `EDGE_THRESHOLD` | `0.10` | Minimum edge to trigger trade |
| `MAX_VOLUME_USD` | `500000` | Only trade markets below this volume |
| `MIN_VOLUME_USD` | `1000` | Skip dead markets |
| `MATERIALITY_THRESHOLD` | `0.6` | Minimum materiality to act on |
| `SPEED_TARGET_SECONDS` | `5` | Target news-to-trade latency |

---

## Safety

- Dry-run mode ON by default
- $25 max single bet, $100 daily limit
- Quarter-Kelly position sizing
- Niche market filter prevents competing against sophisticated bots
- Calibration tracking — auto-detects if strategy accuracy drops
- All API keys in `.env`, never committed

---

Built by [@locus_agent](https://x.com/locus_agent)

---

## Disclaimer

This project is for **entertainment and educational purposes only**. It is not financial advice. The authors are not responsible for any financial losses incurred through the use of this software. Prediction market trading carries significant risk — you can lose money. Never trade with funds you cannot afford to lose. Past performance of any strategy does not guarantee future results. Use at your own risk.
