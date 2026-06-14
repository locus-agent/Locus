import os
from pathlib import Path

from dotenv import load_dotenv

# Repository root — runtime artifacts (trades.db, chroma_db/, docs/, .env)
# live here regardless of where a module sits in the package tree.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# --- Anthropic ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- Polymarket CLOB ---
POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY", "")
POLYMARKET_API_SECRET = os.getenv("POLYMARKET_API_SECRET", "")
POLYMARKET_API_PASSPHRASE = os.getenv("POLYMARKET_API_PASSPHRASE", "")
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
# Proxy-wallet (Polymarket UI account) trading only: the proxy ADDRESS and
# its signature type (1 = email/Magic, 2 = browser wallet proxy). Leave
# unset for a plain EOA wallet signing with POLYMARKET_PRIVATE_KEY.
POLYMARKET_FUNDER_ADDRESS = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
POLYMARKET_SIGNATURE_TYPE = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "1"))
# Live orders: skip when the book's spread exceeds this (the apparent edge
# is mostly spread on thin niche books).
LIVE_MAX_SPREAD = float(os.getenv("LIVE_MAX_SPREAD", "0.05"))
POLYMARKET_HOST = "https://clob.polymarket.com"
POLYMARKET_WS_HOST = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
# Public trades feed (no auth). The CLOB /data/trades endpoint requires API
# keys; this read-only data API exposes the same recent-trade stream.
POLYMARKET_DATA_HOST = os.getenv("POLYMARKET_DATA_HOST", "https://data-api.polymarket.com")

# --- Twitter API v2 ---
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

# --- Telegram ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHANNEL_IDS = [
    c.strip() for c in os.getenv("TELEGRAM_CHANNEL_IDS", "").split(",") if c.strip()
]

# --- NewsAPI (optional, broader news coverage) ---
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
# Free "Developer" plan allows 100 requests/day across all endpoints.
NEWSAPI_DAILY_LIMIT = int(os.getenv("NEWSAPI_DAILY_LIMIT", "100"))

# /v2/everything search queries, one per market category (NewsAPI source)
NEWSAPI_CATEGORY_QUERIES = {
    "ai": '"artificial intelligence" OR OpenAI OR Anthropic OR "Claude AI" OR "Google AI" OR Gemini OR GPT-5',
    "technology": 'Apple OR NVIDIA OR Microsoft OR "big tech"',
    "crypto": "Bitcoin OR Ethereum OR Solana OR cryptocurrency OR blockchain OR Coinbase OR stablecoin",
    "politics": 'Congress OR "White House" OR Senate OR election OR tariff OR "Fed rate"',
}

# /v2/top-headlines categories (country=us) to cover the markets above.
# NewsAPI has no 'crypto' top-headline category — crypto coverage comes from
# the NEWSAPI_CATEGORY_QUERIES /v2/everything search above.
NEWSAPI_TOP_HEADLINE_CATEGORIES = ["general", "technology", "business"]

# --- RSS Feeds (fallback) ---
RSS_FEEDS = [
    "https://news.google.com/rss/search?q=AI+artificial+intelligence&hl=en-US&gl=US&ceid=US:en",
    "https://feeds.feedburner.com/TechCrunch",
    "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "https://www.theverge.com/rss/index.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
]

# --- Pipeline Settings ---
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
MAX_BET_USD = float(os.getenv("MAX_BET_USD", "25"))
# Notional bankroll for quarter-Kelly sizing (size = bankroll * kelly / 4,
# capped at MAX_BET_USD). With the defaults, materiality 1.0 hits the cap.
KELLY_BANKROLL_USD = float(os.getenv("KELLY_BANKROLL_USD", "100"))
DAILY_LOSS_LIMIT_USD = float(os.getenv("DAILY_LOSS_LIMIT_USD", "100"))
EDGE_THRESHOLD = float(os.getenv("EDGE_THRESHOLD", "0.10"))
NEWS_LOOKBACK_HOURS = 6

# --- V2 Settings ---
MAX_VOLUME_USD = float(os.getenv("MAX_VOLUME_USD", "500000"))
MIN_VOLUME_USD = float(os.getenv("MIN_VOLUME_USD", "1000"))
SPEED_TARGET_SECONDS = float(os.getenv("SPEED_TARGET_SECONDS", "5"))

# Direction-specific materiality floors (applied in pipeline.gate_trade).
# Calibration on 700 graded calls showed bearish calls far less accurate
# (11.1%, worse than random) than bullish (18.1%), so bearish signals need a
# higher materiality bar; low-materiality bullish signals still pay off.
MATERIALITY_THRESHOLD_BULLISH = float(os.getenv("MATERIALITY_THRESHOLD_BULLISH", "0.3"))
MATERIALITY_THRESHOLD_BEARISH = float(os.getenv("MATERIALITY_THRESHOLD_BEARISH", "0.4"))

# High-materiality confirmation gate: the most "obvious" news (materiality >=
# HIGH_MATERIALITY_THRESHOLD) graded *worst* (12.0%), likely because it is
# already priced in. Require the same directional read from at least
# MIN_CONFIRMING_SOURCES distinct news sources on the same market within
# CONFIRMATION_WINDOW_HOURS before trading; otherwise hold for confirmation.
HIGH_MATERIALITY_THRESHOLD = float(os.getenv("HIGH_MATERIALITY_THRESHOLD", "0.5"))
CONFIRMATION_WINDOW_HOURS = float(os.getenv("CONFIRMATION_WINDOW_HOURS", "2"))
MIN_CONFIRMING_SOURCES = int(os.getenv("MIN_CONFIRMING_SOURCES", "2"))

# Event context awareness: markets sharing a Gamma event_id are sibling
# outcomes of one event. Cap how many positions we hold across one event so a
# single event can't quietly dominate the book (mutually-exclusive outcomes are
# highly correlated). Default 1 = one position per event.
MAX_POSITIONS_PER_EVENT = int(os.getenv("MAX_POSITIONS_PER_EVENT", "1"))


def materiality_threshold(direction: str) -> float:
    """Direction-specific materiality floor for a would-be signal."""
    return (
        MATERIALITY_THRESHOLD_BEARISH if direction == "bearish"
        else MATERIALITY_THRESHOLD_BULLISH
    )

# Trade only on fresh news: suppress signals (classification still runs and
# is logged) when the headline was published more than this long before we
# received it. Stale RSS/NewsAPI articles must not trigger trades.
MAX_NEWS_AGE_SECONDS = float(os.getenv("MAX_NEWS_AGE_SECONDS", "900"))

# --- Semantic matching (V2) ---
# Cosine distance ceiling for headline -> market embedding matches.
EMBED_DISTANCE_THRESHOLD = float(os.getenv("EMBED_DISTANCE_THRESHOLD", "0.6"))
EMBED_TOP_K = int(os.getenv("EMBED_TOP_K", "8"))
# Embedding-only matches added on top of keyword matches per headline.
EMBED_MAX_EXTRA_MATCHES = int(os.getenv("EMBED_MAX_EXTRA_MATCHES", "3"))
CLASSIFICATION_MODEL = "claude-haiku-4-5-20251001"
SCORING_MODEL = "claude-sonnet-4-6"

# --- Claude-call efficiency ---
# Prefilter: skip classification for keyword-only matches whose overlap
# score is below this AND whose headline topic mismatches the market category.
PREFILTER_KEYWORD_SCORE = float(os.getenv("PREFILTER_KEYWORD_SCORE", "0.25"))
# Classification cache: reuse a stored result for the same (headline, market)
# within this window if the market price hasn't moved beyond the tolerance.
CLASSIFY_CACHE_HOURS = float(os.getenv("CLASSIFY_CACHE_HOURS", "24"))
CLASSIFY_CACHE_PRICE_TOLERANCE = float(os.getenv("CLASSIFY_CACHE_PRICE_TOLERANCE", "0.02"))

# --- Position exits ---
# Rules trigger a Claude re-evaluation; the hard stop never waits for one.
TAKE_PROFIT_TRIGGER_PCT = float(os.getenv("TAKE_PROFIT_TRIGGER_PCT", "50"))
REEVAL_LOSS_PCT = float(os.getenv("REEVAL_LOSS_PCT", "-30"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "-50"))
REEVAL_COOLDOWN_HOURS = float(os.getenv("REEVAL_COOLDOWN_HOURS", "6"))
NEWS_REEVAL_MATERIALITY = float(os.getenv("NEWS_REEVAL_MATERIALITY", "0.4"))

# --- Classification grading (non-traded calls vs later price moves) ---
CALIBRATION_HORIZON_HOURS = float(os.getenv("CALIBRATION_HORIZON_HOURS", "24"))
# Automatic calibration cadence inside the pipeline (first run ~5 min after start).
CALIBRATION_INTERVAL_HOURS = float(os.getenv("CALIBRATION_INTERVAL_HOURS", "4"))
CALIBRATION_MOVE_THRESHOLD = float(os.getenv("CALIBRATION_MOVE_THRESHOLD", "0.02"))

# --- Journal ---
# Daily first-person journal entry (one Sonnet call/day, written at the
# first pipeline cycle after 21:00 UTC; see locus/core/journal.py).
JOURNAL_ENABLED = os.getenv("JOURNAL_ENABLED", "true").lower() == "true"

# --- Dashboard ---
# Auto-commit and push docs/status.json after each pipeline cycle.
AUTO_PUSH_STATUS = os.getenv("AUTO_PUSH_STATUS", "true").lower() == "true"
# Minimum time between auto-pushes, regardless of how often cycles run.
AUTO_PUSH_MIN_INTERVAL_SECONDS = float(os.getenv("AUTO_PUSH_MIN_INTERVAL_SECONDS", "300"))

# --- Whale tracking ---
# Top-performing wallets to shadow (comma-separated addresses in .env). When a
# whale opens a position on a niche market we hadn't acted on, the pipeline
# investigates it. Empty list disables whale tracking entirely.
WHALE_WALLETS = [
    w.strip().lower() for w in os.getenv("WHALE_WALLETS", "").split(",") if w.strip()
]
# How often the whale-check task runs, and how far back each poll looks.
WHALE_CHECK_INTERVAL_MINUTES = float(os.getenv("WHALE_CHECK_INTERVAL_MINUTES", "15"))
WHALE_LOOKBACK_MINUTES = float(os.getenv("WHALE_LOOKBACK_MINUTES", "30"))
# After a whale-triggered investigation on a market, skip it for this long.
WHALE_COOLDOWN_HOURS = float(os.getenv("WHALE_COOLDOWN_HOURS", "6"))
# Ignore whale trades smaller than this (USD notional).
WHALE_MIN_TRADE_USD = float(os.getenv("WHALE_MIN_TRADE_USD", "1000"))
# A whale trade is a "missed opportunity" only if we have no actionable
# classification on that market within this window, and the market is not
# closing sooner than the minimum time-to-close (too late to act).
WHALE_CLASSIFICATION_LOOKBACK_HOURS = float(os.getenv("WHALE_CLASSIFICATION_LOOKBACK_HOURS", "2"))
WHALE_MIN_HOURS_TO_CLOSE = float(os.getenv("WHALE_MIN_HOURS_TO_CLOSE", "2"))

# --- Categories to track ---
MARKET_CATEGORIES = [
    "ai",
    "technology",
    "crypto",
    "politics",
]

# --- Twitter filter keywords (for filtered stream rules) ---
TWITTER_KEYWORDS = [
    "OpenAI", "GPT-5", "Anthropic", "Claude", "Google AI", "Gemini",
    "Bitcoin", "Ethereum", "Solana", "crypto",
    "Fed rate", "tariff", "Congress", "White House",
    "SpaceX", "Starship", "NASA",
    "Apple", "NVIDIA", "Microsoft", "Google",
]
