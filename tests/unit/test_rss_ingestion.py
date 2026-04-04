"""Unit tests for RSS ingestion module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
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
from synesis.processing.news.deduplication import DeduplicationResult
from synesis.processing.news.models import SourcePlatform


def _mock_deduplicator(*, is_duplicate: bool = False) -> AsyncMock:
    """Create a mock MessageDeduplicator that returns the given duplicate status."""
    dedup = AsyncMock()
    dedup.process_message.return_value = DeduplicationResult(is_duplicate=is_duplicate)
    return dedup


def _make_fresh_xml(pub_date: datetime | None = None) -> bytes:
    """Build RSS XML with a configurable pubDate for freshness testing."""
    if pub_date is None:
        pub_date = datetime.now(UTC)
    # Format as RFC 2822
    date_str = pub_date.strftime("%a, %d %b %Y %H:%M:%S GMT")
    return f"""\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Fresh Headline - Reuters</title>
      <link>https://news.google.com/rss/articles/abc</link>
      <guid isPermaLink="false">guid-fresh</guid>
      <pubDate>{date_str}</pubDate>
      <source url="https://www.reuters.com">Reuters</source>
    </item>
  </channel>
</rss>
""".encode()


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
# Pub date parsing
# ─────────────────────────────────────────────────────────────


class TestParsePubDate:
    def test_valid_rfc2822(self) -> None:
        dt = _parse_pub_date("Fri, 03 Apr 2026 16:55:44 GMT")
        assert dt.year == 2026
        assert dt.month == 4
        assert dt.day == 3
        assert dt.tzinfo is not None

    def test_invalid_date_returns_epoch(self) -> None:
        dt = _parse_pub_date("not a date")
        # Should return epoch (stale) so freshness filter rejects it
        assert dt == datetime.min.replace(tzinfo=UTC)
        assert dt.tzinfo is not None

    def test_timezone_aware_utc(self) -> None:
        """Parsed dates must be UTC-aware for freshness comparison."""
        dt = _parse_pub_date("Fri, 03 Apr 2026 16:55:44 GMT")
        assert dt.tzinfo is not None
        assert dt.utcoffset() == timedelta(0)

    def test_non_gmt_timezone_converted(self) -> None:
        """Non-GMT timezones should be converted to UTC."""
        dt = _parse_pub_date("Fri, 03 Apr 2026 12:55:44 -0400")
        assert dt.tzinfo is not None
        # -0400 means 12:55 local = 16:55 UTC
        assert dt.hour == 16
        assert dt.minute == 55


# ─────────────────────────────────────────────────────────────
# _poll_feed: freshness filter + semantic dedup
# ─────────────────────────────────────────────────────────────


class TestPollFeed:
    @pytest.mark.asyncio
    async def test_fresh_items_invoke_callback(self) -> None:
        """Fresh items that pass semantic dedup should be pushed through the callback."""
        dedup = _mock_deduplicator()
        poller = GoogleRSSPoller(feeds=[], poll_interval=1, deduplicator=dedup)
        poller._client = AsyncMock()
        poller._resolve_url = AsyncMock(side_effect=lambda url: url)

        callback = AsyncMock()
        poller.on_message(callback)

        # Use fresh XML (pubDate = now)
        mock_resp = AsyncMock()
        mock_resp.content = _make_fresh_xml()
        mock_resp.raise_for_status = lambda: None
        poller._client.get.return_value = mock_resp

        count = await poller._poll_feed("https://example.com/rss")

        assert count == 1
        assert callback.call_count == 1
        assert dedup.process_message.call_count == 1

    @pytest.mark.asyncio
    async def test_stale_items_skipped(self) -> None:
        """Articles older than 15 minutes should be skipped entirely."""
        dedup = _mock_deduplicator()
        poller = GoogleRSSPoller(feeds=[], poll_interval=1, deduplicator=dedup)
        poller._client = AsyncMock()

        callback = AsyncMock()
        poller.on_message(callback)

        # Use stale XML (pubDate = 30 min ago)
        stale_time = datetime.now(UTC) - timedelta(minutes=30)
        mock_resp = AsyncMock()
        mock_resp.content = _make_fresh_xml(stale_time)
        mock_resp.raise_for_status = lambda: None
        poller._client.get.return_value = mock_resp

        count = await poller._poll_feed("https://example.com/rss")

        assert count == 0
        callback.assert_not_called()
        # Dedup should NOT be called for stale items
        dedup.process_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_semantic_dedup_drops_duplicate(self) -> None:
        """Fresh items flagged as duplicate by semantic dedup should be dropped."""
        dedup = _mock_deduplicator(is_duplicate=True)
        poller = GoogleRSSPoller(feeds=[], poll_interval=1, deduplicator=dedup)
        poller._client = AsyncMock()
        poller._resolve_url = AsyncMock(side_effect=lambda url: url)

        callback = AsyncMock()
        poller.on_message(callback)

        mock_resp = AsyncMock()
        mock_resp.content = _make_fresh_xml()
        mock_resp.raise_for_status = lambda: None
        poller._client.get.return_value = mock_resp

        count = await poller._poll_feed("https://example.com/rss")

        assert count == 0
        callback.assert_not_called()
        assert dedup.process_message.call_count == 1

    @pytest.mark.asyncio
    async def test_item_at_14min_passes_freshness(self) -> None:
        """Article published 14 min ago (inside 15-min window) should pass."""
        dedup = _mock_deduplicator()
        poller = GoogleRSSPoller(feeds=[], poll_interval=1, deduplicator=dedup)
        poller._client = AsyncMock()
        poller._resolve_url = AsyncMock(side_effect=lambda url: url)

        callback = AsyncMock()
        poller.on_message(callback)

        pub_time = datetime.now(UTC) - timedelta(minutes=14)
        mock_resp = AsyncMock()
        mock_resp.content = _make_fresh_xml(pub_time)
        mock_resp.raise_for_status = lambda: None
        poller._client.get.return_value = mock_resp

        count = await poller._poll_feed("https://example.com/rss")

        assert count == 1
        assert callback.call_count == 1

    @pytest.mark.asyncio
    async def test_item_at_16min_skipped(self) -> None:
        """Article published 16 min ago (outside 15-min window) should be skipped."""
        dedup = _mock_deduplicator()
        poller = GoogleRSSPoller(feeds=[], poll_interval=1, deduplicator=dedup)
        poller._client = AsyncMock()

        callback = AsyncMock()
        poller.on_message(callback)

        pub_time = datetime.now(UTC) - timedelta(minutes=16)
        mock_resp = AsyncMock()
        mock_resp.content = _make_fresh_xml(pub_time)
        mock_resp.raise_for_status = lambda: None
        poller._client.get.return_value = mock_resp

        count = await poller._poll_feed("https://example.com/rss")

        assert count == 0
        callback.assert_not_called()
        dedup.process_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_fixture_items_skipped_when_stale(self) -> None:
        """The GOOGLE_NEWS_XML fixture has old dates — all items should be skipped."""
        dedup = _mock_deduplicator()
        poller = GoogleRSSPoller(feeds=[], poll_interval=1, deduplicator=dedup)
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
        dedup = _mock_deduplicator()
        poller = GoogleRSSPoller(feeds=[], poll_interval=1, deduplicator=dedup)
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
        poller = GoogleRSSPoller(feeds=[], poll_interval=1, deduplicator=_mock_deduplicator())
        poller._client = AsyncMock()

        mock_resp = AsyncMock()
        mock_resp.url = "https://www.reuters.com/article/real-url"
        poller._client.head.return_value = mock_resp

        result = await poller._resolve_url("https://news.google.com/rss/articles/abc")
        assert result == "https://www.reuters.com/article/real-url"

    @pytest.mark.asyncio
    async def test_returns_original_on_error(self) -> None:
        poller = GoogleRSSPoller(feeds=[], poll_interval=1, deduplicator=_mock_deduplicator())
        poller._client = AsyncMock()
        poller._client.head.side_effect = httpx.HTTPError("429")

        result = await poller._resolve_url("https://news.google.com/rss/articles/abc")
        assert result == "https://news.google.com/rss/articles/abc"

    @pytest.mark.asyncio
    async def test_empty_url_returns_empty(self) -> None:
        poller = GoogleRSSPoller(feeds=[], poll_interval=1, deduplicator=_mock_deduplicator())
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
