from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

import feedparser
import httpx
from dateutil import parser as date_parser

from locus import config


def _parse_rss_date(entry) -> datetime | None:
    """Best-effort parse of an RSS item's raw <pubDate> when feedparser's own
    struct_time parse failed (non-standard date formats). Tries dateutil across
    the common raw date fields with multiple format fallbacks; returns a
    tz-aware UTC datetime, or None when nothing parses."""
    for field_name in ("published", "pubDate", "updated", "created", "date"):
        raw = entry.get(field_name) if hasattr(entry, "get") else None
        if not raw or not isinstance(raw, str):
            continue
        try:
            dt = date_parser.parse(raw)
        except (ValueError, OverflowError, TypeError):
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return None


# Sports-specific feeds, scraped only when config.SPORTS_ENABLED is set (see
# scrape_all). Kept separate from config.RSS_FEEDS so toggling the sports
# feature also toggles its news sources.
SPORTS_RSS_FEEDS = [
    "http://feeds.bbci.co.uk/sport/rss.xml",        # BBC Sport
    "https://www.espn.com/espn/rss/news",           # ESPN
    "https://www.goal.com/feeds/en/news",           # Goal.com
]


@dataclass
class NewsItem:
    headline: str
    source: str
    url: str
    published_at: datetime
    summary: str = ""
    date_known: bool = True  # False when the feed/article had no parseable date

    def age_hours(self) -> float:
        delta = datetime.now(timezone.utc) - self.published_at
        return delta.total_seconds() / 3600


def scrape_rss(feed_url: str, lookback_hours: int) -> list[NewsItem]:
    """Parse a single RSS feed and return recent items."""
    items = []
    try:
        # Fetch via httpx (uses certifi's CA bundle) instead of feedparser's
        # built-in urllib fetch, which fails with CERTIFICATE_VERIFY_FAILED
        # on systems where the stdlib ssl module has no default CA bundle.
        resp = httpx.get(feed_url, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except Exception:
        return items

    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    # Truth Social is a first-class fast source (direct from Trump), tagged by
    # URL so the pipeline can apply its tighter freshness window + materiality
    # boost. Everything else keeps the feed's own title.
    if "trumpstruth.org" in feed_url:
        source_name = "truthsocial"
    else:
        source_name = feed.feed.get("title", feed_url)

    for entry in feed.entries:
        date_known = True
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
            published = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
        else:
            # feedparser couldn't parse the date — fall back to a dateutil parse
            # of the raw <pubDate> string before giving up and treating the item
            # as undated (received_at is used as the basis downstream).
            published = _parse_rss_date(entry)
            if published is None:
                published = datetime.now(timezone.utc)
                date_known = False

        if published < cutoff:
            continue

        items.append(NewsItem(
            headline=entry.get("title", "").strip(),
            source=source_name,
            url=entry.get("link", ""),
            published_at=published,
            summary=entry.get("summary", "")[:500],
            date_known=date_known,
        ))

    return items


# Tracks NewsAPI requests against the free-tier daily limit (resets at UTC midnight).
_newsapi_day = None
_newsapi_request_count = 0


def _newsapi_budget_ok() -> bool:
    global _newsapi_day, _newsapi_request_count
    today = datetime.now(timezone.utc).date()
    if today != _newsapi_day:
        _newsapi_day = today
        _newsapi_request_count = 0
    if _newsapi_request_count >= config.NEWSAPI_DAILY_LIMIT:
        return False
    _newsapi_request_count += 1
    return True


def scrape_newsapi(query: str, lookback_hours: int) -> list[NewsItem]:
    """Pull from NewsAPI.org /v2/everything if key is configured and within budget."""
    if not config.NEWSAPI_KEY or not _newsapi_budget_ok():
        return []

    items = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    from_dt = cutoff.strftime("%Y-%m-%dT%H:%M:%S")

    try:
        resp = httpx.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "from": from_dt,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": 50,
            },
            headers={"X-Api-Key": config.NEWSAPI_KEY},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return items

    for article in data.get("articles", []):
        pub_str = article.get("publishedAt", "")
        date_known = True
        try:
            published = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            published = datetime.now(timezone.utc)
            date_known = False

        items.append(NewsItem(
            headline=article.get("title", "").strip(),
            source=article.get("source", {}).get("name", "NewsAPI"),
            url=article.get("url", ""),
            published_at=published,
            summary=(article.get("description") or "")[:500],
            date_known=date_known,
        ))

    return items


def scrape_newsapi_top_headlines(category: str | None = None, country: str = "us") -> list[NewsItem]:
    """Pull from NewsAPI.org /v2/top-headlines if key is configured and within budget."""
    if not config.NEWSAPI_KEY or not _newsapi_budget_ok():
        return []

    items = []
    params = {
        "country": country,
        "pageSize": 50,
    }
    if category:
        params["category"] = category

    try:
        resp = httpx.get(
            "https://newsapi.org/v2/top-headlines",
            params=params,
            headers={"X-Api-Key": config.NEWSAPI_KEY},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return items

    for article in data.get("articles", []):
        pub_str = article.get("publishedAt", "")
        date_known = True
        try:
            published = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            published = datetime.now(timezone.utc)
            date_known = False

        items.append(NewsItem(
            headline=article.get("title", "").strip(),
            source=article.get("source", {}).get("name", "NewsAPI"),
            url=article.get("url", ""),
            published_at=published,
            summary=(article.get("description") or "")[:500],
            date_known=date_known,
        ))

    return items


def deduplicate(items: list[NewsItem]) -> list[NewsItem]:
    """Remove near-duplicate headlines by normalized prefix matching."""
    seen = set()
    unique = []
    for item in items:
        key = item.headline.lower()[:80]
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def scrape_all(lookback_hours: int | None = None) -> list[NewsItem]:
    """Run all scrapers and return deduplicated, sorted results."""
    hours = lookback_hours or config.NEWS_LOOKBACK_HOURS
    all_items = []

    feeds = list(config.RSS_FEEDS)
    if config.SPORTS_ENABLED:
        feeds += SPORTS_RSS_FEEDS
    for feed_url in feeds:
        all_items.extend(scrape_rss(feed_url, hours))
        time.sleep(0.5)  # polite crawling

    unique = deduplicate(all_items)
    unique.sort(key=lambda x: x.published_at, reverse=True)
    return unique


if __name__ == "__main__":
    items = scrape_all()
    print(f"\n--- Scraped {len(items)} unique headlines ---\n")
    for item in items[:20]:
        age = item.age_hours()
        print(f"  [{age:.1f}h ago] [{item.source}] {item.headline}")
