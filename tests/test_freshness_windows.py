"""Source-aware freshness windows: config.get_max_age_seconds(source, category,
hours_to_resolution) and the dateutil <pubDate> fallback in the RSS scraper."""
from datetime import datetime, timezone

import pytest

from locus import config
from locus.sources import scraper


@pytest.fixture(autouse=True)
def _pin_windows(monkeypatch):
    """Pin every freshness window to its default so the tests don't depend on a
    developer's .env overrides."""
    monkeypatch.setattr(config, "MAX_NEWS_AGE_SECONDS_DEFAULT", 14400)
    monkeypatch.setattr(config, "MAX_NEWS_AGE_SECONDS_TWITTER", 10800)
    monkeypatch.setattr(config, "MAX_NEWS_AGE_SECONDS_TRUTHSOCIAL", 7200)
    monkeypatch.setattr(config, "MAX_NEWS_AGE_SECONDS_RSS", 21600)
    monkeypatch.setattr(config, "MAX_NEWS_AGE_SECONDS_NEWSAPI", 18000)
    monkeypatch.setattr(config, "MAX_NEWS_AGE_SECONDS_GEOPOLITICAL", 86400)


# --- Source windows ----------------------------------------------------------

def test_twitter_window():
    assert config.get_max_age_seconds("twitter") == 10800


def test_truthsocial_window():
    assert config.get_max_age_seconds("truthsocial") == 7200


def test_rss_window():
    assert config.get_max_age_seconds("rss") == 21600


def test_newsapi_window():
    assert config.get_max_age_seconds("newsapi") == 18000


def test_unknown_source_falls_back_to_default():
    assert config.get_max_age_seconds("mystery-wire") == 14400
    assert config.get_max_age_seconds("") == 14400
    assert config.get_max_age_seconds(None) == 14400


def test_case_insensitive_source():
    assert config.get_max_age_seconds("RSS") == 21600


# --- Geopolitical override ---------------------------------------------------

def test_geopolitical_category_overrides_source_window():
    # Even a fast source gets the wide geopolitical window when flagged.
    assert config.get_max_age_seconds("twitter", "geopolitical") == 86400
    assert config.get_max_age_seconds("rss", "geopolitical") == 86400


def test_non_geopolitical_category_uses_source_window():
    assert config.get_max_age_seconds("rss", "politics") == 21600


def test_far_resolution_overrides_to_geopolitical():
    # Resolution far beyond the 7-day threshold (passed in seconds) also widens.
    assert config.get_max_age_seconds("rss", "ai", 8 * 24 * 3600) == 86400


def test_near_resolution_does_not_override():
    assert config.get_max_age_seconds("rss", "ai", 24 * 3600) == 21600
    assert config.get_max_age_seconds("rss", "ai", None) == 21600


# --- RSS <pubDate> dateutil parsing -----------------------------------------

class _Entry(dict):
    """feedparser entries support both attribute and .get() access; here we only
    need .get(), which dict already provides."""


def test_parse_rfc822_pubdate():
    dt = scraper._parse_rss_date(_Entry(published="Tue, 23 Jun 2026 14:30:00 GMT"))
    assert dt == datetime(2026, 6, 23, 14, 30, tzinfo=timezone.utc)


def test_parse_iso8601_pubdate():
    dt = scraper._parse_rss_date(_Entry(pubDate="2026-06-23T14:30:00+00:00"))
    assert dt == datetime(2026, 6, 23, 14, 30, tzinfo=timezone.utc)


def test_parse_naive_date_assumed_utc():
    dt = scraper._parse_rss_date(_Entry(updated="2026-06-23 14:30:00"))
    assert dt == datetime(2026, 6, 23, 14, 30, tzinfo=timezone.utc)


def test_parse_offset_date_normalized_to_utc():
    # +02:00 -> 12:30 UTC.
    dt = scraper._parse_rss_date(_Entry(published="2026-06-23T14:30:00+02:00"))
    assert dt == datetime(2026, 6, 23, 12, 30, tzinfo=timezone.utc)


def test_parse_rfc822_with_zone_name():
    dt = scraper._parse_rss_date(_Entry(published="Tue, 23 Jun 2026 14:30:00 UTC"))
    assert dt == datetime(2026, 6, 23, 14, 30, tzinfo=timezone.utc)


def test_parse_iso_zulu_pubdate():
    dt = scraper._parse_rss_date(_Entry(pubDate="2026-06-23T14:30:00Z"))
    assert dt == datetime(2026, 6, 23, 14, 30, tzinfo=timezone.utc)


def test_parse_date_only_pubdate():
    dt = scraper._parse_rss_date(_Entry(published="2026-06-23"))
    assert dt == datetime(2026, 6, 23, 0, 0, tzinfo=timezone.utc)


def test_parse_no_weekday_pubdate():
    dt = scraper._parse_rss_date(_Entry(published="23 Jun 2026 14:30:00"))
    assert dt == datetime(2026, 6, 23, 14, 30, tzinfo=timezone.utc)


@pytest.mark.parametrize("raw,expected", [
    ("Tue, 23 Jun 2026 14:30:00 GMT", datetime(2026, 6, 23, 14, 30, tzinfo=timezone.utc)),
    ("2026-06-23T14:30:00+00:00", datetime(2026, 6, 23, 14, 30, tzinfo=timezone.utc)),
    ("2026-06-23T16:30:00+02:00", datetime(2026, 6, 23, 14, 30, tzinfo=timezone.utc)),
    ("2026-06-23 14:30:00", datetime(2026, 6, 23, 14, 30, tzinfo=timezone.utc)),
])
def test_parse_one_date_multiple_formats(raw, expected):
    assert scraper._parse_one_date(raw) == expected


def test_parse_one_date_unparseable_returns_none():
    assert scraper._parse_one_date("definitely not a date") is None


def test_unparseable_pubdate_returns_none():
    assert scraper._parse_rss_date(_Entry(published="not a date at all !!!")) is None


def test_missing_pubdate_returns_none():
    assert scraper._parse_rss_date(_Entry()) is None
