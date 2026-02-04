"""Unit tests for Reddit RSS ingestion."""

import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from synesis.ingestion.reddit import (
    RedditPost,
    RedditRSSClient,
    _clean_html,
    _extract_selftext,
    _parse_rss_timestamp,
)


class TestParseRssTimestamp:
    """Tests for _parse_rss_timestamp."""

    def test_none_returns_now(self) -> None:
        result = _parse_rss_timestamp(None)
        assert isinstance(result, datetime)
        assert result.tzinfo is not None

    def test_valid_struct_time(self) -> None:
        ts = time.strptime("2024-01-15 10:30:00", "%Y-%m-%d %H:%M:%S")
        result = _parse_rss_timestamp(ts)
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_overflow_returns_now(self) -> None:
        # Create a struct_time that causes OverflowError in mktime
        ts = time.strptime("9999-12-31 23:59:59", "%Y-%m-%d %H:%M:%S")
        result = _parse_rss_timestamp(ts)
        assert isinstance(result, datetime)

    def test_non_struct_time_returns_now(self) -> None:
        result = _parse_rss_timestamp("not a time struct")
        assert isinstance(result, datetime)


class TestCleanHtml:
    """Tests for _clean_html."""

    def test_strips_tags(self) -> None:
        result = _clean_html("<b>bold</b> and <i>italic</i>")
        assert result == "bold and italic"

    def test_converts_links(self) -> None:
        result = _clean_html('<a href="https://example.com">click here</a>')
        assert result == "click here (https://example.com)"

    def test_unescapes_entities(self) -> None:
        result = _clean_html("&amp; test &quot;hello&quot;")
        assert "&" in result
        assert '"hello"' in result

    def test_collapses_newlines(self) -> None:
        result = _clean_html("<p>first</p><p>second</p><p>third</p>")
        # Should not have more than 2 consecutive newlines
        assert "\n\n\n" not in result

    def test_br_to_newline(self) -> None:
        result = _clean_html("line1<br/>line2<br>line3")
        assert "line1\nline2\nline3" == result

    def test_empty_string(self) -> None:
        assert _clean_html("") == ""


class TestExtractSelftext:
    """Tests for _extract_selftext."""

    def test_content_list(self) -> None:
        entry = {"content": [{"value": "<p>Hello world</p>"}]}
        result = _extract_selftext(entry)
        assert "Hello world" in result

    def test_content_dict(self) -> None:
        entry = {"content": {"value": "<b>test</b>"}}
        result = _extract_selftext(entry)
        assert result == "test"

    def test_falls_back_to_summary(self) -> None:
        entry = {"summary": "<p>Summary text</p>"}
        result = _extract_selftext(entry)
        assert "Summary text" in result

    def test_returns_empty_if_nothing(self) -> None:
        entry: dict[str, str] = {}
        result = _extract_selftext(entry)
        assert result == ""

    def test_empty_content_list_falls_back(self) -> None:
        entry = {"content": [], "summary": "fallback"}
        result = _extract_selftext(entry)
        assert result == "fallback"


class TestRedditPostFullText:
    """Tests for RedditPost.full_text property."""

    def test_title_and_content(self) -> None:
        post = RedditPost(
            post_id="abc",
            subreddit="wsb",
            author="user1",
            title="Big News",
            content="Details here",
            url="https://reddit.com/r/wsb/abc",
            permalink="https://reddit.com/r/wsb/abc",
            timestamp=datetime.now(timezone.utc),
            raw={},
        )
        assert post.full_text == "Big News\n\nDetails here"

    def test_title_only(self) -> None:
        post = RedditPost(
            post_id="abc",
            subreddit="wsb",
            author="user1",
            title="Just a link",
            content="",
            url="https://reddit.com/r/wsb/abc",
            permalink="https://reddit.com/r/wsb/abc",
            timestamp=datetime.now(timezone.utc),
            raw={},
        )
        assert post.full_text == "Just a link"


class TestRedditRSSClientParseEntry:
    """Tests for RedditRSSClient._parse_entry."""

    def _make_client(self) -> RedditRSSClient:
        return RedditRSSClient(subreddits=["test"])

    def test_extracts_post_id_from_url(self) -> None:
        client = self._make_client()
        entry = {
            "id": "https://www.reddit.com/r/wsb/comments/abc123/some_title/",
            "title": "Test",
            "link": "https://www.reddit.com/r/wsb/comments/abc123/",
        }
        post = client._parse_entry(entry, "wsb")
        assert post.post_id == "abc123"

    def test_handles_t3_prefix(self) -> None:
        client = self._make_client()
        entry = {
            "id": "t3_xyz789",
            "title": "Test",
            "link": "https://reddit.com/r/wsb/xyz789/",
        }
        post = client._parse_entry(entry, "wsb")
        assert post.post_id == "xyz789"

    def test_parses_author_with_u_prefix(self) -> None:
        client = self._make_client()
        entry = {
            "id": "t3_abc",
            "title": "Test",
            "link": "",
            "author": "/u/testuser",
        }
        post = client._parse_entry(entry, "wsb")
        assert post.author == "testuser"

    def test_parses_author_detail(self) -> None:
        client = self._make_client()
        entry = {
            "id": "t3_abc",
            "title": "Test",
            "link": "",
            "author_detail": {"name": "detailuser"},
        }
        post = client._parse_entry(entry, "wsb")
        assert post.author == "detailuser"

    def test_no_author(self) -> None:
        client = self._make_client()
        entry = {
            "id": "t3_abc",
            "title": "Test",
            "link": "",
        }
        post = client._parse_entry(entry, "wsb")
        assert post.author is None


class TestPollAllSubreddits:
    """Tests for RedditRSSClient.poll_all_subreddits deduplication."""

    @pytest.mark.asyncio
    async def test_deduplication_via_seen_ids(self) -> None:
        client = RedditRSSClient(subreddits=["wsb"])

        post1 = RedditPost(
            post_id="aaa",
            subreddit="wsb",
            author="u1",
            title="Post 1",
            content="",
            url="",
            permalink="",
            timestamp=datetime.now(timezone.utc),
            raw={},
        )
        post2 = RedditPost(
            post_id="bbb",
            subreddit="wsb",
            author="u2",
            title="Post 2",
            content="",
            url="",
            permalink="",
            timestamp=datetime.now(timezone.utc),
            raw={},
        )

        # First poll returns both posts
        with patch.object(client, "fetch_subreddit", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = [post1, post2]
            new_posts = await client.poll_all_subreddits()
            assert len(new_posts) == 2

        # Second poll returns same posts - should be filtered out
        with patch.object(client, "fetch_subreddit", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = [post1, post2]
            new_posts = await client.poll_all_subreddits()
            assert len(new_posts) == 0

    @pytest.mark.asyncio
    async def test_seen_ids_size_limit(self) -> None:
        client = RedditRSSClient(subreddits=["wsb"])

        # Pre-fill seen_ids to just under the limit
        for i in range(10000):
            client._seen_ids[f"old_{i}"] = None

        # Add one more via polling
        post = RedditPost(
            post_id="new_one",
            subreddit="wsb",
            author="u1",
            title="New",
            content="",
            url="",
            permalink="",
            timestamp=datetime.now(timezone.utc),
            raw={},
        )

        with patch.object(client, "fetch_subreddit", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = [post]
            await client.poll_all_subreddits()

        # Should have pruned old entries, keeping <= 10000
        assert len(client._seen_ids) <= 10001  # 10000 + 1 new
