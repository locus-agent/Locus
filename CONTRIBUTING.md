# Contributing to Locus

First off — thank you for being here. 🎉

**Locus** is an autonomous trading agent that reads breaking news, classifies its
directional impact with Claude, and trades niche Polymarket prediction markets
(dry-run by default). It detects news, sizes positions with confidence-based
half-Kelly, runs a stack of risk gates, grades its own calibration over time, and
even rewrites its own classification prompt from the lessons it accumulates.

It's a fun, self-contained codebase to hack on: real-time data, an LLM in the
decision loop, a tidy SQLite memory, and a live public dashboard. Whether you
want to fix a typo, add a risk gate, or propose a whole new trading strategy,
there's a place for you here — and you don't need to be a quant or an ML expert
to make a meaningful contribution.

This guide explains how to get set up and how we work together. If anything is
unclear, please open an issue and ask — improving this doc is itself a great
first contribution.

---

## Table of Contents

1. [Ways to Contribute](#ways-to-contribute)
2. [Development Setup](#development-setup)
3. [Code Standards](#code-standards)
4. [Pull Request Process](#pull-request-process)
5. [Architecture Overview](#architecture-overview)
6. [Community Strategies (Future)](#community-strategies-future)
7. [Code of Conduct](#code-of-conduct)

---

## Ways to Contribute

There's no contribution too small. Here are the main ways to get involved:

### 🐛 Bug Reports

Found something broken or surprising? Open a **GitHub Issue**. A good report makes
a fix far more likely. Please include:

- **What happened** vs. **what you expected**
- **Steps to reproduce** (the exact command, e.g. `python cli.py watch`)
- **Environment**: OS, Python version (`python --version`), and whether you're in
  dry-run or live mode
- **Logs / traceback**, with any secrets redacted (never paste your `.env` or API keys)

> **Suggested issue template**
> ```
> **Describe the bug**
> A clear, concise description of what went wrong.
>
> **To reproduce**
> 1. Run `...`
> 2. See error
>
> **Expected behavior**
> What you expected to happen.
>
> **Environment**
> - OS:
> - Python version:
> - Mode: dry-run / live
>
> **Logs**
> ```
> (paste relevant output, secrets removed)
> ```
> ```

### ✨ New Features

Have an idea for a new risk gate, news source, edge type, or dashboard panel?

1. **Open an issue first** to discuss the idea — it saves everyone time and helps
   shape the design before you write code.
2. Once there's rough agreement, implement it (see the
   [Pull Request Process](#pull-request-process)).
3. **Every new feature needs tests.** See [Code Standards](#code-standards).

### 📊 Community Strategies (Future)

We plan to support drop-in trading strategies under a `community_strategies/`
folder so people can share their own approaches without touching the core engine.
This is still being designed — see
[Community Strategies (Future)](#community-strategies-future) for the proposed
format and how to get involved early.

### 📚 Documentation Improvements

Docs contributions are hugely valued and a fantastic first PR. That includes the
`README.md`, this file, code comments, docstrings, and the public dashboard copy.
If something confused you, it probably confused someone else — fixing it helps
the whole community.

---

## Development Setup

Locus targets **Python 3.9+** and runs entirely in dry-run mode by default, so
you can develop safely without ever risking real money.

```bash
# 1. Clone the repo (or your fork — see the PR process below)
git clone https://github.com/locus-agent/Locus.git
cd Locus

# 2. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure your environment
cp .env.example .env
#    Then open .env and add your Anthropic API key:
#    ANTHROPIC_API_KEY=sk-ant-...   (required)
#    Everything else is optional — see .env.example for what each key unlocks.

# 4. Verify your setup (checks deps, API keys, DB, and connectivity)
python cli.py verify

# 5. Run the pipeline in dry-run mode (no real trades — safe to leave running)
python cli.py watch
```

`python cli.py watch` is the event-driven pipeline. It runs forever,
streaming news → classifying → sizing → logging simulated trades. Add `--live` only
when you genuinely intend to trade real money — and never with someone else's keys.

Run `python cli.py` with no arguments to see every available command (`verify`,
`watch`, `dashboard`, `calibrate`, `niche`, `markets`, `trades`,
`stats`, `evolve`, ...).

---

## Code Standards

We keep the bar friendly but firm — the goal is a codebase that stays easy to
trust and easy to change.

- **All new features (and bug fixes) need tests.** We use `pytest`. Tests live in
  `tests/` and need **no network access or API keys** — the database is a temp
  fixture and external calls are faked, so the suite runs fast and offline.
- **Run the full suite before opening a PR:**
  ```bash
  python -m pytest tests/
  ```
  Please make sure everything passes.
- **Follow the existing module structure.** Put code where similar code already
  lives (`locus/core/`, `locus/memory/`, `locus/markets/`, `locus/sources/`, ...).
  See [Architecture Overview](#architecture-overview).
- **Match the surrounding style.** Mirror the naming, comment density, and idioms
  of the file you're editing. Most modules also include a small `__main__` smoke
  test — keep that pattern where it helps.
- **Read config as `config.X` at call time** (not `from locus.config import X`) so
  CLI/env overrides are respected — this is a load-bearing convention in the
  codebase.
- **Never commit secrets.** `.env` is git-ignored; keep it that way. Redact keys
  and wallet details from logs, tests, and screenshots.

---

## Pull Request Process

1. **Fork** the repository to your own GitHub account.
2. **Create a branch** off `main` with a descriptive name
   (e.g. `fix/stale-gate-timezone` or `feat/momentum-edge`). Please don't commit
   directly to `main`.
3. **Make your change**, with tests, keeping commits focused and readable.
4. **Run the full test suite** (`python -m pytest tests/`) and make sure it's green.
5. **Open a Pull Request to `main`** and in the description tell us:
   - **What** changed
   - **Why** it changed (link the related issue if there is one)
   - **How it's tested** — note the new/updated tests and that the suite passes
6. A maintainer will review, maybe suggest tweaks, and merge. Be patient and kind;
   reviews are a conversation, not a verdict. 💛

Small, well-scoped PRs are much easier to review and merge than large ones — when
in doubt, split it up.

---

## Architecture Overview

Locus is an asyncio pipeline with a single entry point (`cli.py`) and everything
else under `locus/`:

```
locus/
  config.py     .env loading, thresholds, PROJECT_ROOT (anchors trades.db, docs/, chroma_db/)
  core/         pipeline, classifier, multi_classifier, matcher, edge, executor,
                event_context, reentry, whale_tracker, performance, export_status, journal
  sources/      news_stream (Twitter/Telegram/RSS/NewsAPI), scraper (RSS/NewsAPI fetch helpers)
  markets/      gamma (Polymarket API client + Market model), market_watcher
  memory/       __init__ (track record + lessons), logger (SQLite), calibrator, meta_evolver
  backtest/     real (real-data replay, parked)
  ui/           tui (Textual dashboard)
```

The high-level flow:

```
Breaking news → match to niche markets → Claude classifies (bullish/bearish/neutral
+ materiality + confidence) → edge detection → risk gates → half-Kelly sizing →
execute (dry-run by default) → SQLite log → calibration & self-improvement
```

This is intentionally brief. **For the full architecture, module map, data flow,
and the "how a headline becomes a trade" walkthrough, see the
[README](README.md#architecture)** and the in-repo [`CLAUDE.md`](CLAUDE.md).

---

## Community Strategies (Future)

> ⚠️ **Status: proposed / not yet implemented.** The interface below may change.
> If you're interested, open an issue to help shape it — early input is welcome.

The idea is to let contributors share complete trading strategies as self-contained
plugins, without modifying the core engine. We're planning a `community_strategies/`
folder so the community can experiment, compare, and learn from each other.

### How to propose a new strategy

1. **Start with an issue** describing your strategy: the thesis (what edge it
   captures), which markets/news it targets, and how it decides direction and size.
2. **Discuss** the design with maintainers and the community.
3. Once there's agreement, submit it as a PR following the format below.

### Proposed format for `community_strategies/`

Each strategy lives in its own folder:

```
community_strategies/
  your_strategy_name/
    README.md          # thesis, assumptions, parameters, and backtest results
    strategy.py        # the strategy implementation (a documented entry point)
    tests/             # pytest tests covering the strategy's logic
    config.example     # any strategy-specific settings, with safe defaults
```

Guidelines:

- **Dry-run first.** Strategies must run safely in simulation; live trading is opt-in.
- **Be transparent.** Document the assumptions and known failure modes in the
  strategy's `README.md`.
- **Bring evidence.** Include backtest or calibration results where you can.
- **Test it.** Same standard as the rest of the project — strategies need tests.

If you'd like to help design this system itself (the plugin interface, the shared
risk gates, the scoring), that's one of the most valuable contributions you can make
right now. Open an issue and let's build it together.

---

## Code of Conduct

Be kind, be patient, and assume good intent. We want Locus to be a welcoming place
for people of all backgrounds and experience levels. Harassment or disrespect of any
kind isn't tolerated. If you see a problem, please reach out to the maintainers.

---

> **A note on responsible use:** Locus is a research and educational project. It runs
> in dry-run mode by default. If you choose to enable live trading, you do so at your
> own risk — markets are unforgiving, and nothing here is financial advice. Please
> trade responsibly and never risk more than you can afford to lose.

Thanks again for contributing. We can't wait to see what you build. 🚀
