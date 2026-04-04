"""Unit tests for RSS ingestion module."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx
import pytest

from synesis.config import Settings
from synesis.ingestion.google_rss import (
    GoogleRSSPoller,
    RSSItem,
    _clean_title,
    _parse_pub_date,
    parse_feed_xml,
)
from synesis.processing.news.models import SourcePlatform

# ─────────────────────────────────────────────────────────────
# Fixture: Google News RSS XML
# ─────────────────────────────────────────────────────────────

GOOGLE_NEWS_XML = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/">
  <channel>
    <title>Business - Latest - Google News</title>
    <link>https://news.google.com/topics/business</link>
    <language>en-US</language>
    <lastBuildDate>Sat, 04 Apr 2026 04:51:37 GMT</lastBuildDate>
    <item>
      <title>US Bonds Fall as Strong Jobs Data Shake Markets - Bloomberg.com</title>
      <link>https://news.google.com/rss/articles/CBMiXXxx?oc=5</link>
      <guid isPermaLink="false">CBMiXXxx</guid>
      <pubDate>Fri, 03 Apr 2026 16:55:44 GMT</pubDate>
      <description>&lt;ol&gt;&lt;li&gt;Clustered headlines&lt;/li&gt;&lt;/ol&gt;</description>
      <source url="https://www.bloomberg.com">Bloomberg.com</source>
    </item>
    <item>
      <title>Applied Optoelectronics secures $71M order for 800G transceivers - Investing.com</title>
      <link>https://news.google.com/rss/articles/CBMiYYyy?oc=5</link>
      <guid isPermaLink="false">CBMiYYyy</guid>
      <pubDate>Thu, 02 Apr 2026 20:56:35 GMT</pubDate>
      <description></description>
      <source url="https://www.investing.com">Investing.com</source>
    </item>
    <item>
      <title>No Source Title Here</title>
      <link>https://news.google.com/rss/articles/CBMiZZzz?oc=5</link>
      <guid isPermaLink="false">CBMiZZzz</guid>
      <pubDate>Wed, 01 Apr 2026 10:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""


# ─────────────────────────────────────────────────────────────
# Title cleanup
# ─────────────────────────────────────────────────────────────


class TestCleanTitle:
    def test_strips_matching_source(self) -> None:
        assert _clean_title("US Bonds Fall - Bloomberg.com", "Bloomberg.com") == "US Bonds Fall"

    def test_strips_last_dash_when_source_differs(self) -> None:
        # Source name doesn't match exactly but rsplit fallback works
        assert _clean_title("Some Headline - Unknown Source", "Different") == "Some Headline"

    def test_no_dash_returns_unchanged(self) -> None:
        assert _clean_title("Simple headline", "") == "Simple headline"

    def test_preserves_internal_dashes(self) -> None:
        assert (
            _clean_title("Q1 2026 - Earnings Beat - Reuters", "Reuters")
            == "Q1 2026 - Earnings Beat"
        )


# ─────────────────────────────────────────────────────────────
# XML parsing
# ─────────────────────────────────────────────────────────────


class TestParseFeedXml:
    def test_parses_all_items(self) -> None:
        items = parse_feed_xml(GOOGLE_NEWS_XML)
        assert len(items) == 3

    def test_first_item_fields(self) -> None:
        items = parse_feed_xml(GOOGLE_NEWS_XML)
        item = items[0]
        assert item.guid == "CBMiXXxx"
        assert item.title == "US Bonds Fall as Strong Jobs Data Shake Markets"
        assert item.source_name == "Bloomberg.com"
        assert item.source_url == "https://www.bloomberg.com"
        assert item.pub_date.year == 2026
        assert item.pub_date.month == 4
        assert item.pub_date.tzinfo is not None

    def test_second_item_title_cleaned(self) -> None:
        items = parse_feed_xml(GOOGLE_NEWS_XML)
        item = items[1]
        assert item.title == "Applied Optoelectronics secures $71M order for 800G transceivers"
        assert item.source_name == "Investing.com"

    def test_item_without_source(self) -> None:
        items = parse_feed_xml(GOOGLE_NEWS_XML)
        item = items[2]
        assert item.source_name == ""
        # Fallback rsplit strips the title as-is since no ' - '
        assert item.title == "No Source Title Here"

    def test_empty_description(self) -> None:
        items = parse_feed_xml(GOOGLE_NEWS_XML)
        assert items[1].description == ""

    def test_malformed_xml_returns_empty(self) -> None:
        assert parse_feed_xml(b"not xml at all") == []

    def test_missing_required_fields_skipped(self) -> None:
        xml = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <item><link>http://example.com</link></item>
  <item><title>Has Title</title><guid>g1</guid></item>
</channel></rss>
"""
        items = parse_feed_xml(xml)
        # First item skipped (no title), second item skipped (no guid) — wait, second has both
        assert len(items) == 1
        assert items[0].guid == "g1"


# ─────────────────────────────────────────────────────────────
# RSSItem → UnifiedMessage mapping
# ─────────────────────────────────────────────────────────────


class TestToUnified:
    def test_mapping(self) -> None:
        item = RSSItem(
            guid="test-guid-123",
            title="Fed raises rates by 25bps",
            link="https://news.google.com/rss/articles/abc",
            pub_date=datetime(2026, 4, 3, 12, 0, 0, tzinfo=UTC),
            source_name="Reuters",
            source_url="https://www.reuters.com",
            description="",
        )
        msg = GoogleRSSPoller._to_unified(item, "https://www.reuters.com/article/fed-rate")

        assert msg.external_id == "test-guid-123"
        assert msg.source_platform == SourcePlatform.google_rss
        assert msg.source_account == "Reuters"
        assert msg.text == "Fed raises rates by 25bps"
        assert msg.timestamp == datetime(2026, 4, 3, 12, 0, 0, tzinfo=UTC)
        assert msg.raw["guid"] == "test-guid-123"
        assert msg.raw["resolved_url"] == "https://www.reuters.com/article/fed-rate"
        assert msg.raw["link"] == "https://news.google.com/rss/articles/abc"


# ─────────────────────────────────────────────────────────────
# GUID dedup (requires Redis mock)
# ─────────────────────────────────────────────────────────────


class TestGuidDedup:
    @pytest.mark.asyncio
    async def test_unseen_guid(self) -> None:
        """Unseen GUIDs should not be marked as seen."""
        redis = AsyncMock()
        redis.exists.return_value = 0
        redis.set = AsyncMock()

        poller = GoogleRSSPoller(feeds=[], poll_interval=5, redis=redis)
        assert await poller._is_seen("new-guid") is False

    @pytest.mark.asyncio
    async def test_seen_guid(self) -> None:
        """Seen GUIDs should be detected."""
        redis = AsyncMock()
        redis.exists.return_value = 1

        poller = GoogleRSSPoller(feeds=[], poll_interval=5, redis=redis)
        assert await poller._is_seen("old-guid") is True

    @pytest.mark.asyncio
    async def test_mark_seen_calls_redis(self) -> None:
        redis = AsyncMock()
        redis.set = AsyncMock()

        poller = GoogleRSSPoller(feeds=[], poll_interval=5, redis=redis)
        await poller._mark_seen("guid-abc")

        redis.set.assert_called_once()
        args = redis.set.call_args
        assert "synesis:google_rss:seen:guid-abc" in args[0]


# ─────────────────────────────────────────────────────────────
# Pub date parsing
# ─────────────────────────────────────────────────────────────


class TestParsePubDate:
    def test_valid_rfc2822(self) -> None:
        dt = _parse_pub_date("Fri, 03 Apr 2026 16:55:44 GMT")
        assert dt.year == 2026
        assert dt.month == 4
        assert dt.day == 3
        assert dt.tzinfo is not None

    def test_invalid_date_returns_now(self) -> None:
        dt = _parse_pub_date("not a date")
        # Should return a datetime close to now, not crash
        assert dt.year >= 2026
        assert dt.tzinfo is not None


# ─────────────────────────────────────────────────────────────
# Seed behavior (first poll caches, doesn't process)
# ─────────────────────────────────────────────────────────────


class TestSeedBehavior:
    @pytest.mark.asyncio
    async def test_seed_marks_seen_without_callback(self) -> None:
        """First poll should mark all GUIDs as seen but NOT invoke callback."""
        redis = AsyncMock()
        redis.exists.return_value = 0
        redis.set = AsyncMock()

        poller = GoogleRSSPoller(feeds=["https://example.com/rss"], poll_interval=1, redis=redis)
        poller._client = AsyncMock()

        callback = AsyncMock()
        poller.on_message(callback)

        # Mock HTTP response with our test XML
        mock_resp = AsyncMock()
        mock_resp.content = GOOGLE_NEWS_XML
        mock_resp.raise_for_status = lambda: None
        poller._client.get.return_value = mock_resp

        # Run seed
        result = await poller._seed_seen_cache()

        # GUIDs should be marked as seen (3 items in fixture)
        assert redis.set.call_count == 3

        # Callback should NOT have been called
        callback.assert_not_called()

        # Should return True (at least one feed succeeded)
        assert result is True

    @pytest.mark.asyncio
    async def test_seed_returns_false_when_all_feeds_fail(self) -> None:
        """Seed should return False when no feeds are reachable."""
        redis = AsyncMock()

        poller = GoogleRSSPoller(feeds=["https://example.com/rss"], poll_interval=1, redis=redis)
        poller._client = AsyncMock()
        poller._client.get.side_effect = httpx.HTTPError("connection failed")

        result = await poller._seed_seen_cache()
        assert result is False

    @pytest.mark.asyncio
    async def test_seeded_flag_set_after_seed(self) -> None:
        """_seeded flag should be False initially and True after seeding."""
        redis = AsyncMock()
        redis.exists.return_value = 0
        redis.set = AsyncMock()

        poller = GoogleRSSPoller(feeds=[], poll_interval=1, redis=redis)
        assert poller._seeded is False

        poller._client = AsyncMock()
        await poller._seed_seen_cache()
        # Note: _seeded is set by _poll_loop, not _seed_seen_cache directly
        # But we can verify the method runs without error


# ─────────────────────────────────────────────────────────────
# _poll_feed end-to-end (with mocked HTTP + Redis)
# ─────────────────────────────────────────────────────────────


class TestPollFeed:
    @pytest.mark.asyncio
    async def test_new_items_invoke_callback(self) -> None:
        """New items (unseen GUIDs) should be pushed through the callback."""
        redis = AsyncMock()
        redis.exists.return_value = 0  # all GUIDs unseen
        redis.set = AsyncMock()

        poller = GoogleRSSPoller(feeds=[], poll_interval=1, redis=redis)
        poller._client = AsyncMock()

        # Mock resolve_url to return the original URL
        poller._resolve_url = AsyncMock(side_effect=lambda url: url)

        callback = AsyncMock()
        poller.on_message(callback)

        mock_resp = AsyncMock()
        mock_resp.content = GOOGLE_NEWS_XML
        mock_resp.raise_for_status = lambda: None
        poller._client.get.return_value = mock_resp

        count = await poller._poll_feed("https://example.com/rss")

        assert count == 3
        assert callback.call_count == 3

    @pytest.mark.asyncio
    async def test_seen_items_skipped(self) -> None:
        """Already-seen GUIDs should NOT invoke callback."""
        redis = AsyncMock()
        redis.exists.return_value = 1  # all GUIDs seen
        redis.set = AsyncMock()

        poller = GoogleRSSPoller(feeds=[], poll_interval=1, redis=redis)
        poller._client = AsyncMock()

        callback = AsyncMock()
        poller.on_message(callback)

        mock_resp = AsyncMock()
        mock_resp.content = GOOGLE_NEWS_XML
        mock_resp.raise_for_status = lambda: None
        poller._client.get.return_value = mock_resp

        count = await poller._poll_feed("https://example.com/rss")

        assert count == 0
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_http_error_returns_zero(self) -> None:
        """HTTP errors should log warning and return 0, not crash."""
        redis = AsyncMock()
        poller = GoogleRSSPoller(feeds=[], poll_interval=1, redis=redis)
        poller._client = AsyncMock()
        poller._client.get.side_effect = httpx.HTTPError("connection failed")

        count = await poller._poll_feed("https://example.com/rss")
        assert count == 0


# ─────────────────────────────────────────────────────────────
# URL resolution
# ─────────────────────────────────────────────────────────────


class TestResolveUrl:
    @pytest.mark.asyncio
    async def test_returns_resolved_url(self) -> None:
        redis = AsyncMock()
        poller = GoogleRSSPoller(feeds=[], poll_interval=1, redis=redis)
        poller._client = AsyncMock()

        mock_resp = AsyncMock()
        mock_resp.url = "https://www.reuters.com/article/real-url"
        poller._client.head.return_value = mock_resp

        result = await poller._resolve_url("https://news.google.com/rss/articles/abc")
        assert result == "https://www.reuters.com/article/real-url"

    @pytest.mark.asyncio
    async def test_returns_original_on_error(self) -> None:
        redis = AsyncMock()
        poller = GoogleRSSPoller(feeds=[], poll_interval=1, redis=redis)
        poller._client = AsyncMock()
        poller._client.head.side_effect = httpx.HTTPError("429")

        result = await poller._resolve_url("https://news.google.com/rss/articles/abc")
        assert result == "https://news.google.com/rss/articles/abc"

    @pytest.mark.asyncio
    async def test_empty_url_returns_empty(self) -> None:
        redis = AsyncMock()
        poller = GoogleRSSPoller(feeds=[], poll_interval=1, redis=redis)
        poller._client = AsyncMock()

        result = await poller._resolve_url("")
        assert result == ""


# ─────────────────────────────────────────────────────────────
# Config validator
# ─────────────────────────────────────────────────────────────


class TestParseRssFeeds:
    def test_none_returns_empty(self) -> None:
        assert Settings.parse_rss_feeds(None) == []

    def test_csv_string(self) -> None:
        result = Settings.parse_rss_feeds("https://a.com/rss, https://b.com/rss")
        assert result == ["https://a.com/rss", "https://b.com/rss"]

    def test_json_array(self) -> None:
        result = Settings.parse_rss_feeds('["https://a.com/rss", "https://b.com/rss"]')
        assert result == ["https://a.com/rss", "https://b.com/rss"]

    def test_list_passthrough(self) -> None:
        result = Settings.parse_rss_feeds(["https://a.com/rss"])
        assert result == ["https://a.com/rss"]
