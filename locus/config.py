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
POLYMARKET_HOST = "https://clob.polymarket.com"
POLYMARKET_WS_HOST = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

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
    "crypto": "Bitcoin OR Ethereum OR Solana OR cryptocurrency OR blockchain",
    "politics": 'Congress OR "White House" OR Senate OR election OR tariff OR "Fed rate"',
    "science": 'NASA OR SpaceX OR Starship OR "scientific research"',
}

# /v2/top-headlines categories (country=us) to cover the markets above
NEWSAPI_TOP_HEADLINE_CATEGORIES = ["general", "technology", "science", "business"]

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
MATERIALITY_THRESHOLD = float(os.getenv("MATERIALITY_THRESHOLD", "0.6"))
SPEED_TARGET_SECONDS = float(os.getenv("SPEED_TARGET_SECONDS", "5"))

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

# --- Dashboard ---
# Auto-commit and push docs/status.json after each pipeline cycle.
AUTO_PUSH_STATUS = os.getenv("AUTO_PUSH_STATUS", "true").lower() == "true"
# Minimum time between auto-pushes, regardless of how often cycles run.
AUTO_PUSH_MIN_INTERVAL_SECONDS = float(os.getenv("AUTO_PUSH_MIN_INTERVAL_SECONDS", "300"))

# --- Categories to track ---
MARKET_CATEGORIES = [
    "ai",
    "technology",
    "crypto",
    "politics",
    "science",
]

# --- Twitter filter keywords (for filtered stream rules) ---
TWITTER_KEYWORDS = [
    "OpenAI", "GPT-5", "Anthropic", "Claude", "Google AI", "Gemini",
    "Bitcoin", "Ethereum", "Solana", "crypto",
    "Fed rate", "tariff", "Congress", "White House",
    "SpaceX", "Starship", "NASA",
    "Apple", "NVIDIA", "Microsoft", "Google",
]
