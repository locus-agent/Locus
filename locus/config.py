import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Repository root — runtime artifacts (trades.db, chroma_db/, docs/, .env)
# live here regardless of where a module sits in the package tree.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# --- Anthropic ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- Grok (xAI) — PARKED (kept only for future multi-LLM experiments) ---
# OpenAI-compatible API config for the parked Grok ensemble path. Not used by
# the default pipeline (tiered Haiku->Sonnet). See multi_classifier.py.
GROK_API_KEY = os.getenv("GROK_API_KEY", "")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4")

# --- Multi-LLM consensus classification — DISABLED by default ---
# Parked experiment: when ENSEMBLE_ENABLED is true AND tiered classification is
# off, headlines are classified by Claude + Grok and blended with a
# consensus_score. The default pipeline uses tiered Haiku->Sonnet, so this never
# runs unless explicitly turned on. Off by default — Grok is parked.
ENSEMBLE_ENABLED = os.getenv("ENSEMBLE_ENABLED", "false").lower() == "true"
ENSEMBLE_MIN_CONSENSUS = float(os.getenv("ENSEMBLE_MIN_CONSENSUS", "0.5"))

# --- Polymarket CLOB ---
POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY", "")
POLYMARKET_API_SECRET = os.getenv("POLYMARKET_API_SECRET", "")
POLYMARKET_API_PASSPHRASE = os.getenv("POLYMARKET_API_PASSPHRASE", "")
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
# Funded-wallet trading: the deposit wallet ADDRESS that holds the USDC and its
# signature type (1 = email/Magic proxy, 2 = browser wallet proxy, 3 = POLY_1271
# deposit wallet). Defaults to 3 (deposit wallet). Leave the address unset for a
# plain EOA wallet signing with POLYMARKET_PRIVATE_KEY.
POLYMARKET_FUNDER_ADDRESS = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
POLYMARKET_SIGNATURE_TYPE = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "3"))
# Live orders: skip when the book's spread exceeds this (the apparent edge
# is mostly spread on thin niche books).
LIVE_MAX_SPREAD = float(os.getenv("LIVE_MAX_SPREAD", "0.05"))
# After posting a GTC order, wait this long before re-querying the exchange to
# reconcile the real fill status (MATCHED/filled -> executed, LIVE/resting ->
# not filled yet, missing -> error). A local position is only opened on a
# confirmed fill.
ORDER_RECONCILE_WAIT_SECONDS = float(os.getenv("ORDER_RECONCILE_WAIT_SECONDS", "2"))
# Polymarket CLOB order minimums (server-enforced). An order must be worth at
# least MIN_ORDER_USD in notional AND at least MIN_ORDER_SHARES outcome tokens,
# or the exchange rejects it (e.g. "Size (1.08) lower than the minimum: 5").
MIN_ORDER_USD = float(os.getenv("MIN_ORDER_USD", "1.0"))
MIN_ORDER_SHARES = float(os.getenv("MIN_ORDER_SHARES", "5.0"))
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
# Chat the trading-notification bot posts to (and accepts /portfolio from). When
# empty, locus/core/telegram_bot.py is fully disabled (all notifications no-op).
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
# Send a drawdown alert once an open position's unrealized loss exceeds this
# fraction (0.25 = -25%). See positions.update_and_manage.
TELEGRAM_DRAWDOWN_ALERT_PCT = float(os.getenv("TELEGRAM_DRAWDOWN_ALERT_PCT", "0.25"))

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
    # --- Breaking-news feeds (faster political/crypto coverage) ---
    "http://feeds.bbci.co.uk/news/rss.xml",                              # BBC Breaking
    "https://www.coindesk.com/arc/outboundfeeds/rss/",                   # CoinDesk
    "https://www.theblock.co/rss.xml",                                   # The Block
    # Reuters/AP/Politico killed their direct RSS (dead hosts / Cloudflare),
    # so we pull their breaking headlines via Google News site-search.
    "https://news.google.com/rss/search?q=when:24h+site:reuters.com&hl=en-US&gl=US&ceid=US:en",   # Reuters
    "https://news.google.com/rss/search?q=when:24h+site:apnews.com&hl=en-US&gl=US&ceid=US:en",    # AP News
    "https://news.google.com/rss/search?q=when:24h+site:politico.com&hl=en-US&gl=US&ceid=US:en",  # Politico
    # Trump's Truth Social posts — often break policy decisions ahead of the wires.
    "https://www.trumpstruth.org/feed",                                  # truthsocial
]

# --- Pipeline Settings ---
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
MAX_BET_USD = float(os.getenv("MAX_BET_USD", "25"))
# Notional bankroll for quarter-Kelly sizing (size = bankroll * kelly / 4,
# capped at MAX_BET_USD). With the defaults, materiality 1.0 hits the cap.
KELLY_BANKROLL_USD = float(os.getenv("KELLY_BANKROLL_USD", "100"))
# Caps total notional deployed per day, not realized losses
DAILY_SPEND_LIMIT_USD = float(os.getenv("DAILY_SPEND_LIMIT_USD", "100"))
EDGE_THRESHOLD = float(os.getenv("EDGE_THRESHOLD", "0.10"))
NEWS_LOOKBACK_HOURS = 6

# --- Time-horizon materiality penalty (classifier) ---
# Claude tags each signal with when the market is likely to resolve given the
# news. Long-horizon resolutions are discounted: the penalty is added to the raw
# materiality to produce adjusted_materiality (floored at 0), which the gates
# check against. A soft penalty — strong long-term signals still clear the bar,
# weak ones drop below it. Override with a TIME_HORIZON_PENALTY JSON string.
TIME_HORIZON_PENALTY = {"immediate": 0.0, "medium": -0.03, "long_term": -0.10}
_time_horizon_penalty_override = os.getenv("TIME_HORIZON_PENALTY", "")
if _time_horizon_penalty_override:
    try:
        TIME_HORIZON_PENALTY = json.loads(_time_horizon_penalty_override)
    except (ValueError, TypeError):
        pass  # malformed override -> keep the defaults above

# --- Momentum hybrid edge (edge.detect_edge_v2) ---
# When MOMENTUM_ENABLED, a signal whose direction agrees with the market's
# recent price drift (over MOMENTUM_LOOKBACK_MINUTES, via the Gamma/CLOB price
# history API) gets a small additive edge boost. Fails open: if price history is
# unavailable the boost is simply skipped.
MOMENTUM_ENABLED = os.getenv("MOMENTUM_ENABLED", "true").lower() == "true"
MOMENTUM_LOOKBACK_MINUTES = int(os.getenv("MOMENTUM_LOOKBACK_MINUTES", "60"))

# --- Multi-source confirmation for high-materiality signals (pipeline) ---
# A high-materiality signal (adjusted_materiality >= MULTI_SOURCE_CONFIRM_THRESHOLD)
# is sized down by MULTI_SOURCE_SIZE_REDUCTION unless a second, independent news
# source produced a matching directional call (materiality >= 0.35) on the same
# market in the last few hours. Obvious news is least accurate, so an unconfirmed
# big signal bets smaller rather than being blocked outright.
MULTI_SOURCE_CONFIRM_THRESHOLD = float(os.getenv("MULTI_SOURCE_CONFIRM_THRESHOLD", "0.65"))
MULTI_SOURCE_SIZE_REDUCTION = float(os.getenv("MULTI_SOURCE_SIZE_REDUCTION", "0.30"))
MULTI_SOURCE_CONFIRM_WINDOW_HOURS = float(os.getenv("MULTI_SOURCE_CONFIRM_WINDOW_HOURS", "3"))
MULTI_SOURCE_CONFIRM_MIN_MATERIALITY = float(os.getenv("MULTI_SOURCE_CONFIRM_MIN_MATERIALITY", "0.35"))

# --- Polymarket trading fees (per-category) ---
# Per-share fee modeled as feeRate * p * (1 - p) (see edge.detect_edge_v2).
# The fee is subtracted from raw edge before the EDGE_THRESHOLD check, so a
# market whose fee eats the edge never signals. Geopolitics/world markets are
# fee-free. Override the whole mapping with a FEE_RATES JSON string in .env,
# e.g. FEE_RATES='{"crypto": 0.07, "other": 0.05}'. 'other' is the fallback
# for any category without an explicit entry (see gamma._fee_rate_for_category).
FEE_RATES = {
    "geopolitics": 0.0,
    "world": 0.0,
    "politics": 0.04,
    "mentions": 0.04,
    "technology": 0.04,
    "ai": 0.04,
    "crypto": 0.07,
    "sports": 0.03,
    "finance": 0.04,
    "other": 0.05,
}
_fee_rates_override = os.getenv("FEE_RATES", "")
if _fee_rates_override:
    try:
        FEE_RATES = json.loads(_fee_rates_override)
    except (ValueError, TypeError):
        pass  # malformed override -> keep the defaults above

# --- Price-room guards (edge.detect_edge_v2) ---
# Skip a signal when the market is priced outside the band where its direction
# has historically paid. Bullish (YES): min raised from 0.05 to 0.12 because
# calibration shows markets under 0.15 grade only 5.8% accurate (longshot
# lottery tickets), and the max tightened slightly from 0.85 to 0.82 (little
# room left to profit above). Bearish (NO): min 0.18 because NO bets on very
# low-priced markets are risky, max kept wide at 0.88 since good NO edge exists
# in the 0.87-0.93 range.
BULLISH_MIN_PRICE = float(os.getenv("BULLISH_MIN_PRICE", "0.12"))
BULLISH_MAX_PRICE = float(os.getenv("BULLISH_MAX_PRICE", "0.82"))
BEARISH_MIN_PRICE = float(os.getenv("BEARISH_MIN_PRICE", "0.18"))
BEARISH_MAX_PRICE = float(os.getenv("BEARISH_MAX_PRICE", "0.88"))

# --- Dynamic Kelly sizing ---
# Scale half-Kelly bets by recent realized win rate: a cold streak shrinks
# size, a hot streak restores it. Win rate is the fraction of the last
# KELLY_WINRATE_LOOKBACK closed positions that were profitable (0.5 until at
# least KELLY_WINRATE_MIN_SAMPLES closes), cached for KELLY_WINRATE_CACHE_TTL
# seconds. The factor maps win rate linearly: 0.25 -> MIN_FACTOR, 0.75 ->
# MAX_FACTOR (clamped to that band). The final bet is floored at
# KELLY_MIN_BET_USD so a real signal is never sized down to dust.
KELLY_WINRATE_LOOKBACK = int(os.getenv("KELLY_WINRATE_LOOKBACK", "20"))
KELLY_WINRATE_MIN_SAMPLES = int(os.getenv("KELLY_WINRATE_MIN_SAMPLES", "5"))
KELLY_DYNAMIC_MIN_FACTOR = float(os.getenv("KELLY_DYNAMIC_MIN_FACTOR", "0.25"))
KELLY_DYNAMIC_MAX_FACTOR = float(os.getenv("KELLY_DYNAMIC_MAX_FACTOR", "1.0"))
KELLY_MIN_BET_USD = float(os.getenv("KELLY_MIN_BET_USD", "2.0"))
KELLY_WINRATE_CACHE_TTL = float(os.getenv("KELLY_WINRATE_CACHE_TTL", "1800"))

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

# --- Market-structure filters (applied in pipeline.gate_trade) ---
# Never open a position in a market resolving within this many hours: the
# thesis has too little time to play out (action 'too_close_to_resolution').
MIN_HOURS_TO_RESOLUTION = float(os.getenv("MIN_HOURS_TO_RESOLUTION", "4"))
# Skip "price-target" markets (e.g. "Will Bitcoin reach $100k") — they resolve
# on a price threshold that news rarely moves cleanly in our favor (action
# 'price_target_market'). Matched case-insensitively against the keyword list
# below; override the list with a PRICE_TARGET_KEYWORDS JSON array in .env.
EXCLUDE_PRICE_TARGET_MARKETS = os.getenv("EXCLUDE_PRICE_TARGET_MARKETS", "true").lower() == "true"
PRICE_TARGET_KEYWORDS = [
    "reach $", "hit $", "above $", "below $", "exceed $", "surpass $",
    "new all-time high", "new ath", "all time high",
    "will bitcoin reach", "will ethereum hit",
]
_price_target_keywords_override = os.getenv("PRICE_TARGET_KEYWORDS", "")
if _price_target_keywords_override:
    try:
        PRICE_TARGET_KEYWORDS = json.loads(_price_target_keywords_override)
    except (ValueError, TypeError):
        pass  # malformed override -> keep the defaults above


# --- Sports markets (extra-strict; see classifier SPORTS block + pipeline gates) ---
# Sports headlines are noisy (rumors, lineup speculation, pre-game punditry), so
# sports markets get a higher materiality bar, a tighter resolution-time floor,
# and a per-event headline cap. SPORTS_ENABLED=false skips them entirely
# (action 'sports_disabled') and drops 'sports' from the tracked categories.
SPORTS_ENABLED = os.getenv("SPORTS_ENABLED", "true").lower() == "true"
SPORTS_MATERIALITY_THRESHOLD = float(os.getenv("SPORTS_MATERIALITY_THRESHOLD", "0.48"))
SPORTS_MAX_EXPOSURE = float(os.getenv("SPORTS_MAX_EXPOSURE", "50"))
SPORTS_MIN_HOURS_TO_RESOLUTION = float(os.getenv("SPORTS_MIN_HOURS_TO_RESOLUTION", "4"))
MAX_HEADLINES_PER_SPORTS_EVENT = int(os.getenv("MAX_HEADLINES_PER_SPORTS_EVENT", "2"))


def materiality_threshold(direction: str) -> float:
    """Direction-specific materiality floor for a would-be signal."""
    return (
        MATERIALITY_THRESHOLD_BEARISH if direction == "bearish"
        else MATERIALITY_THRESHOLD_BULLISH
    )

# Trade only on fresh news: suppress signals (classification still runs and
# is logged) when the headline was published more than this long before we
# received it. Stale RSS/NewsAPI articles must not trigger trades.
#
# Limits are source-specific: real-time feeds (Twitter) get a tight window,
# while slower aggregators (RSS/NewsAPI) get more slack. MAX_NEWS_AGE_SECONDS
# is the fallback for unknown/unlisted sources. Use get_max_age_seconds().
MAX_NEWS_AGE_SECONDS = float(os.getenv("MAX_NEWS_AGE_SECONDS", "900"))
MAX_NEWS_AGE_SECONDS_TWITTER = float(os.getenv("MAX_NEWS_AGE_SECONDS_TWITTER", "900"))
MAX_NEWS_AGE_SECONDS_RSS = float(os.getenv("MAX_NEWS_AGE_SECONDS_RSS", "7200"))
MAX_NEWS_AGE_SECONDS_NEWSAPI = float(os.getenv("MAX_NEWS_AGE_SECONDS_NEWSAPI", "14400"))
MAX_NEWS_AGE_SECONDS_TELEGRAM = float(os.getenv("MAX_NEWS_AGE_SECONDS_TELEGRAM", "1800"))
# Truth Social posts are breaking news straight from the source — treat them
# as nearly as time-sensitive as Twitter (30 min), not slow RSS (2h).
MAX_NEWS_AGE_SECONDS_TRUTHSOCIAL = float(os.getenv("MAX_NEWS_AGE_SECONDS_TRUTHSOCIAL", "1800"))

# Geopolitical markets move slowly: long-horizon diplomatic/military/political
# developments (Iran deals, Ukraine, Taiwan, sanctions, ...) stay relevant for
# days, so headlines on those markets get a much wider freshness window (12h)
# than the source default — but only when the market resolves > 7 days out (see
# pipeline.is_geopolitical + gate_trade).
MAX_NEWS_AGE_SECONDS_GEOPOLITICAL = float(os.getenv("MAX_NEWS_AGE_SECONDS_GEOPOLITICAL", "43200"))
GEOPOLITICAL_KEYWORDS = [
    "iran", "tehran", "ayatollah", "nuclear deal", "sanctions", "ceasefire",
    "ukraine", "russia", "putin", "zelensky", "taiwan", "china invasion",
    "nato", "g7", "g20", "diplomatic", "treaty", "signing ceremony",
    "peace deal", "trade war", "tariff", "congress", "senate", "election",
    "coup", "regime", "withdrawal", "troops", "middle east",
]


def get_max_age_seconds(news_source: str) -> float:
    """Freshness limit (seconds) for a given news source, falling back to
    MAX_NEWS_AGE_SECONDS for unknown/unlisted sources. Read these as
    config.X at call time so CLI/env overrides are honored."""
    return {
        "twitter": MAX_NEWS_AGE_SECONDS_TWITTER,
        "rss": MAX_NEWS_AGE_SECONDS_RSS,
        "newsapi": MAX_NEWS_AGE_SECONDS_NEWSAPI,
        "telegram": MAX_NEWS_AGE_SECONDS_TELEGRAM,
        "truthsocial": MAX_NEWS_AGE_SECONDS_TRUTHSOCIAL,
    }.get((news_source or "").lower(), MAX_NEWS_AGE_SECONDS)

# --- Semantic matching (V2) ---
# Cosine distance ceiling for headline -> market embedding matches.
EMBED_DISTANCE_THRESHOLD = float(os.getenv("EMBED_DISTANCE_THRESHOLD", "0.6"))
EMBED_TOP_K = int(os.getenv("EMBED_TOP_K", "8"))
# Embedding-only matches added on top of keyword matches per headline.
EMBED_MAX_EXTRA_MATCHES = int(os.getenv("EMBED_MAX_EXTRA_MATCHES", "3"))
CLASSIFICATION_MODEL = "claude-haiku-4-5-20251001"
SCORING_MODEL = "claude-sonnet-4-6"

# --- Tiered classification (Haiku prefilter -> Sonnet deep analysis) ---
# A cheap Haiku call triages each matched headline (relevant? roughly material?)
# before the full Sonnet (SCORING_MODEL) classification runs, so only headlines
# worth a deep read pay for the expensive call.
# Expected cost reduction: ~55% (Haiku ~4x cheaper + filters ~60-70% of low-value headlines)
HAIKU_MODEL = os.getenv("HAIKU_MODEL", "claude-haiku-4-5-20251001")  # fallback: "claude-3-5-haiku-20241022"
HAIKU_MATERIALITY_THRESHOLD = float(os.getenv("HAIKU_MATERIALITY_THRESHOLD", "0.25"))
TIERED_CLASSIFICATION_ENABLED = os.getenv("TIERED_CLASSIFICATION_ENABLED", "true").lower() == "true"

# --- Chain-of-Verification (CoV) novelty check ---
# The most obvious, high-materiality news is often already reflected in the
# price (no edge left). Before trading a high-materiality signal at a
# non-extreme price, a cheap Haiku call asks whether the news is genuinely NEW
# information not yet priced in; if it is confidently already priced, the
# signal is suppressed (action 'already_priced_in'). Read as config.X at call
# time so env overrides are honored.
COV_ENABLED = os.getenv("COV_ENABLED", "true").lower() == "true"
COV_MATERIALITY_THRESHOLD = float(os.getenv("COV_MATERIALITY_THRESHOLD", "0.65"))
COV_CONFIDENCE_THRESHOLD = float(os.getenv("COV_CONFIDENCE_THRESHOLD", "0.75"))
COV_MODEL = os.getenv("COV_MODEL", "claude-haiku-4-5-20251001")  # fallback: "claude-3-5-haiku-20241022"

# --- Throughput: parallel candidate processing + concurrency limits ---
# Each news event can match several markets; they're classified concurrently in
# batches of PARALLEL_BATCH_SIZE. Separate semaphores cap in-flight model calls:
# Haiku is cheap/fast (allow more), Sonnet is expensive (allow fewer).
PARALLEL_BATCH_SIZE = int(os.getenv("PARALLEL_BATCH_SIZE", "8"))
HAIKU_SEMAPHORE_SIZE = int(os.getenv("HAIKU_SEMAPHORE_SIZE", "16"))
SONNET_SEMAPHORE_SIZE = int(os.getenv("SONNET_SEMAPHORE_SIZE", "8"))

# --- Dedup + embedding cache ---
# Near-duplicate headlines are dropped when their MiniLM cosine similarity to a
# recent headline exceeds DEDUP_COSINE_THRESHOLD. EMBEDDING_CACHE_SIZE bounds the
# in-memory LRU of market embeddings that fronts the Chroma disk lookup.
DEDUP_COSINE_THRESHOLD = float(os.getenv("DEDUP_COSINE_THRESHOLD", "0.92"))
EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "1000"))

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
# A position sitting on a big unrealized gain deserves a fresh look even inside
# the cooldown window — at/above this PnL%, re-eval bypasses the cooldown.
REEVAL_FORCE_PCT = float(os.getenv("REEVAL_FORCE_PCT", "150.0"))
NEWS_REEVAL_MATERIALITY = float(os.getenv("NEWS_REEVAL_MATERIALITY", "0.4"))
# A Truth Social post (direct from Trump) that contradicts a held side at this
# materiality or higher forces an immediate re-evaluation, bypassing the
# per-position re-eval cooldown (don't wait for the -30% drawdown trigger).
TRUTHSOCIAL_REEVAL_MATERIALITY = float(os.getenv("TRUTHSOCIAL_REEVAL_MATERIALITY", "0.6"))

# Hard, model-free time-pressure exit: a deep loser running into market close
# has little time left to recover, so force it out without a Claude call.
TIME_PRESSURE_HOURS = float(os.getenv("TIME_PRESSURE_HOURS", "4"))
TIME_PRESSURE_LOSS_PCT = float(os.getenv("TIME_PRESSURE_LOSS_PCT", "-20"))

# Hard, model-free near-certain exit: when the held side is all but resolved
# (YES price >= NEAR_CERTAIN_THRESHOLD, or NO price <= 1 - threshold), lock it in
# rather than tie up capital until resolution. Skipped within the last hour of a
# market, where prices naturally drift to certainty anyway.
NEAR_CERTAIN_THRESHOLD = float(os.getenv("NEAR_CERTAIN_THRESHOLD", "0.95"))

# --- Classification grading (non-traded calls vs later price moves) ---
CALIBRATION_HORIZON_HOURS = float(os.getenv("CALIBRATION_HORIZON_HOURS", "24"))
# Automatic calibration cadence inside the pipeline (first run ~5 min after start).
CALIBRATION_INTERVAL_HOURS = float(os.getenv("CALIBRATION_INTERVAL_HOURS", "4"))
CALIBRATION_MOVE_THRESHOLD = float(os.getenv("CALIBRATION_MOVE_THRESHOLD", "0.02"))

# --- Journal ---
# Daily first-person journal entry (one Sonnet call/day, written at the
# first pipeline cycle after 21:00 UTC; see locus/core/journal.py).
JOURNAL_ENABLED = os.getenv("JOURNAL_ENABLED", "true").lower() == "true"

# --- Missed-opportunity tracking ---
# Once a day the calibrator looks back at directional classifications we did NOT
# trade (skip/low_materiality/stale/prefiltered_haiku) and checks whether the
# market moved our way anyway. A relative move of MISSED_OPPORTUNITY_THRESHOLD
# (12%) in the predicted direction logs a lesson suggesting we lower the
# materiality bar for that category. Candidates are restricted to a tradeable
# entry price (>= MISSED_MIN_ENTRY_PRICE) so dust longshots don't dominate.
# Set MISSED_OPPORTUNITY_ENABLED=false to skip the daily check.
MISSED_OPPORTUNITY_ENABLED = os.getenv("MISSED_OPPORTUNITY_ENABLED", "true").lower() == "true"
MISSED_OPPORTUNITY_THRESHOLD = float(os.getenv("MISSED_OPPORTUNITY_THRESHOLD", "0.12"))
MISSED_MIN_ENTRY_PRICE = float(os.getenv("MISSED_MIN_ENTRY_PRICE", "0.08"))

# --- Missed-opportunity-driven threshold suggestions (conservative mode) ---
# When the calibrator detects a *recurring* pattern of misses, it stores a
# human-readable suggestion (e.g. "loosen the geopolitical freshness window")
# rather than silently changing a threshold. AUTO_ADJUST stays off: these are
# suggestions for a human to review on the dashboard, never auto-applied.
MISSED_OPPORTUNITY_AUTO_ADJUST_ENABLED = os.getenv("MISSED_OPPORTUNITY_AUTO_ADJUST_ENABLED", "false").lower() == "true"
# Recurrence bars: N qualifying misses within the window before we suggest.
MISSED_STALE_GEOPOLITICAL_THRESHOLD = int(os.getenv("MISSED_STALE_GEOPOLITICAL_THRESHOLD", "3"))   # in 14 days
MISSED_MATERIALITY_THRESHOLD = int(os.getenv("MISSED_MATERIALITY_THRESHOLD", "2"))                  # per category, 7 days
# Conservative magnitude the suggestion proposes (a freshness-window widening).
STALE_WINDOW_INCREASE_PCT = float(os.getenv("STALE_WINDOW_INCREASE_PCT", "0.25"))
# Pending suggestions older than this are treated as expired (counters reset).
MISSED_ADJUSTMENT_DECAY_DAYS = int(os.getenv("MISSED_ADJUSTMENT_DECAY_DAYS", "30"))

# --- Meta-prompt evolution ---
# Once every PROMPT_EVOLUTION_INTERVAL_DAYS, after the daily journal, Claude
# (Sonnet) rewrites its own classification prompt from accumulated lessons and
# accuracy stats (see locus/memory/meta_evolver.py). The first evolution waits
# until there are that many days of lessons. Set PROMPT_EVOLUTION_ENABLED=false
# to keep the hardcoded prompt.
PROMPT_EVOLUTION_ENABLED = os.getenv("PROMPT_EVOLUTION_ENABLED", "true").lower() == "true"
PROMPT_EVOLUTION_INTERVAL_DAYS = float(os.getenv("PROMPT_EVOLUTION_INTERVAL_DAYS", "7"))

# --- Dashboard ---
# Pipeline start time (UTC), set by `cli.py watch` at launch so export_status
# can publish uptime. None when nothing set it (standalone export, dashboard).
WATCH_START_TIME = None
# Auto-commit and push docs/status.json after each pipeline cycle.
AUTO_PUSH_STATUS = os.getenv("AUTO_PUSH_STATUS", "true").lower() == "true"
# Minimum time between auto-pushes, regardless of how often cycles run.
AUTO_PUSH_MIN_INTERVAL_SECONDS = float(os.getenv("AUTO_PUSH_MIN_INTERVAL_SECONDS", "300"))
# Display-only filter: when set to an ISO date (e.g. "2026-06-14"), the
# dashboard's open/closed position lists only show positions opened on or after
# that date (hides old test positions). Empty = show all. Positions are still in
# the DB and still count toward calibration, the circuit breaker, and performance.
DASHBOARD_POSITIONS_START_DATE = os.getenv("DASHBOARD_POSITIONS_START_DATE", "")
# Display-only filter for the dashboard performance panel: when set to an ISO
# date, compute_performance() only counts positions opened on or after it
# (closed_count, win rate, realized/unrealized PnL, deployed capital). Empty =
# count all. The circuit breaker, calibration, and dynamic-Kelly win rate keep
# using full history (they have their own date filters).
PERFORMANCE_START_DATE = os.getenv("PERFORMANCE_START_DATE", "")

# --- Whale tracking ---
# Master switch for opening whale-triggered trades. Investigations still run and
# are logged either way; when false, a whale signal is never queued for a trade.
WHALE_TRADING_ENABLED = os.getenv("WHALE_TRADING_ENABLED", "true").lower() == "true"  # Set to false if you want extra safety
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

# --- Re-entry logic ---
# After a non-resolution close, keep watching the market for REENTRY_WATCH_HOURS
# and re-enter at most MAX_REENTRY_PER_MARKET times if a fresh classification
# clears the Re-entry 2.0 gate (positions.check_reentry_opportunity, configured
# below). These two just bound the watch window and per-market re-entry budget.
REENTRY_WATCH_HOURS = float(os.getenv("REENTRY_WATCH_HOURS", "72"))
MAX_REENTRY_PER_MARKET = int(os.getenv("MAX_REENTRY_PER_MARKET", "1"))

# --- Re-entry 2.0 (exit_reason calibration) ---
# A calibration-driven re-entry gate keyed on the *granular* exit_reason rather
# than the bucketed close reason above. Some exits leave a clean thesis to
# re-enter on (a take-profit decision, a manual close, a near-certain lock-in, a
# resolution); others mean the market beat us and re-entering loses money (a
# drawdown or time-pressure exit, a hard stop, a news reversal, an
# already-priced-in skip). Reasons are matched after normalizing the stored
# exit_reason (e.g. "near_certain_yes_0.96" -> "near_certain_yes", the hard stop
# "sl" -> "hard_sl"); see positions.check_reentry_opportunity. REENTRY_ENABLED
# is the master switch. Override the lists with JSON arrays in .env.
REENTRY_ENABLED = os.getenv("REENTRY_ENABLED", "true").lower() == "true"
REENTRY_ALLOWED_REASONS = [
    "tp_decision", "manual", "near_certain_yes", "near_certain_no", "resolution",
]
REENTRY_BLOCKED_REASONS = [
    "drawdown_decision", "time_pressure", "news_decision", "already_priced_in", "hard_sl",
]
_reentry_allowed_override = os.getenv("REENTRY_ALLOWED_REASONS", "")
if _reentry_allowed_override:
    try:
        REENTRY_ALLOWED_REASONS = json.loads(_reentry_allowed_override)
    except (ValueError, TypeError):
        pass  # malformed override -> keep the defaults above
_reentry_blocked_override = os.getenv("REENTRY_BLOCKED_REASONS", "")
if _reentry_blocked_override:
    try:
        REENTRY_BLOCKED_REASONS = json.loads(_reentry_blocked_override)
    except (ValueError, TypeError):
        pass  # malformed override -> keep the defaults above
REENTRY_MIN_MATERIALITY = float(os.getenv("REENTRY_MIN_MATERIALITY", "0.45"))
REENTRY_MIN_HOURS = float(os.getenv("REENTRY_MIN_HOURS", "4"))
# Reduce position size on a re-entry (0.7 = 70% of the size we'd otherwise open).
REENTRY_SIZE_FACTOR = float(os.getenv("REENTRY_SIZE_FACTOR", "0.7"))
# At most this many re-entries across one Gamma event_id.
REENTRY_MAX_PER_EVENT = int(os.getenv("REENTRY_MAX_PER_EVENT", "1"))
# Don't re-enter a market that resolves within this many hours (too little time
# for the re-entered thesis to play out).
REENTRY_MIN_HOURS_TO_RESOLUTION = float(os.getenv("REENTRY_MIN_HOURS_TO_RESOLUTION", "12"))

# --- Circuit breaker ---
# Auto-pause trading when recent realized performance deteriorates. Evaluated
# at the start of each signal-processing cycle: a tripped breaker holds every
# would-be trade (logged with action 'circuit_breaker') instead of executing.
# Trips when the 7-day realized-PnL drawdown exceeds CIRCUIT_BREAKER_DD (a
# fraction of the running peak) OR the 7-day daily Sharpe falls below
# CIRCUIT_BREAKER_SHARPE. CIRCUIT_BREAKER_ENABLED=false disables it entirely
# (compute_circuit_breaker still reports the metrics, just never trips). Read
# these as config.X at call time so env/CLI overrides are honored.
CIRCUIT_BREAKER_ENABLED = os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
CIRCUIT_BREAKER_DD = float(os.getenv("CIRCUIT_BREAKER_DD", "0.20"))
CIRCUIT_BREAKER_SHARPE = float(os.getenv("CIRCUIT_BREAKER_SHARPE", "-1.0"))
# Only count positions closed at/after this date in the drawdown/Sharpe
# calculation. Empty = count all closes (within the rolling 7-day window).
# Set to an ISO date (e.g. "2026-06-14") to ignore legacy closes from before a
# strategy change, so old losses don't keep the breaker tripped. Read as
# config.X at call time so env overrides are honored.
CIRCUIT_BREAKER_START_DATE = os.getenv("CIRCUIT_BREAKER_START_DATE", "")

# --- Categories to track ---
MARKET_CATEGORIES = [
    "ai",
    "technology",
    "crypto",
    "politics",
]
# Only track sports markets when the sports feature is on (otherwise the
# market_watcher never tracks them and the sports gates never fire).
if SPORTS_ENABLED:
    MARKET_CATEGORIES.append("sports")

# --- Per-category exposure limits ---
# Hard cap (USD) on combined open-position exposure per inferred market
# category. A would-be trade is blocked (action 'category_limit') once a
# category's existing exposure is over its hard limit; between
# CATEGORY_SOFT_LIMIT_PCT and 100% of the limit it is allowed but warned.
# 'other' is the fallback for any category without an explicit entry.
# Override the whole mapping with a CATEGORY_EXPOSURE_LIMITS JSON string, e.g.
# CATEGORY_EXPOSURE_LIMITS='{"politics": 100, "crypto": 100, "other": 25}'.
MAX_EXPOSURE_PER_CATEGORY = {
    "politics": 75,
    "crypto": 75,
    "ai": 50,
    "technology": 50,
    "sports": SPORTS_MAX_EXPOSURE,
    "other": 25,
}
_category_exposure_override = os.getenv("CATEGORY_EXPOSURE_LIMITS", "")
if _category_exposure_override:
    try:
        MAX_EXPOSURE_PER_CATEGORY = json.loads(_category_exposure_override)
    except (ValueError, TypeError):
        pass  # malformed override -> keep the defaults above
# Warn (but still allow) once a category reaches this fraction of its hard limit.
CATEGORY_SOFT_LIMIT_PCT = float(os.getenv("CATEGORY_SOFT_LIMIT_PCT", "0.8"))

# --- Twitter filter keywords (for filtered stream rules) ---
TWITTER_KEYWORDS = [
    "OpenAI", "GPT-5", "Anthropic", "Claude", "Google AI", "Gemini",
    "Bitcoin", "Ethereum", "Solana", "crypto",
    "Fed rate", "tariff", "Congress", "White House",
    "SpaceX", "Starship", "NASA",
    "Apple", "NVIDIA", "Microsoft", "Google",
    # Expanded breaking political/crypto/AI coverage. (Stream matching is
    # case-insensitive, so spacex/openai/anthropic/bitcoin/ethereum/congress/
    # white house from the requested set are already covered above.)
    "Trump", "Iran", "Fed", "Federal Reserve", "Polymarket",
    "election", "Senate", "breaking", "just in", "developing",
    # Catch chatter about Trump's Truth Social posts.
    "truth social", "truthsocial",
]
