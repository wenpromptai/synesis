"""Tests for Twitter API client."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synesis.ingestion.twitterapi import (
    Tweet,
    TwitterClient,
    TwitterStreamClient,
    _extract_full_text,
    parse_twitter_timestamp,
)


class TestParseTwitterTimestamp:
    """Tests for timestamp parsing."""

    def test_iso_format_with_z(self) -> None:
        """Test parsing ISO format with Z suffix."""
        result = parse_twitter_timestamp("2025-01-15T10:30:00Z")
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30

    def test_iso_format_with_offset(self) -> None:
        """Test parsing ISO format with timezone offset."""
        result = parse_twitter_timestamp("2025-01-15T10:30:00+00:00")
        assert result.year == 2025
        assert result.tzinfo is not None

    def test_twitter_standard_format(self) -> None:
        """Test parsing Twitter's standard timestamp format."""
        result = parse_twitter_timestamp("Sun Jan 18 13:14:48 +0000 2026")
        assert result.year == 2026
        assert result.month == 1
        assert result.day == 18

    def test_none_returns_now(self) -> None:
        """Test that None returns current time."""
        before = datetime.now(timezone.utc)
        result = parse_twitter_timestamp(None)
        after = datetime.now(timezone.utc)

        assert before <= result <= after

    def test_invalid_format_returns_now(self) -> None:
        """Test that invalid format returns current time."""
        before = datetime.now(timezone.utc)
        result = parse_twitter_timestamp("invalid timestamp")
        after = datetime.now(timezone.utc)

        assert before <= result <= after


class TestExtractFullText:
    """Tests for _extract_full_text helper."""

    def test_standard_text(self) -> None:
        """Test extracting standard tweet text."""
        data = {"text": "This is a normal tweet"}
        result = _extract_full_text(data)
        assert result == "This is a normal tweet"

    def test_full_text_field(self) -> None:
        """Test extracting from full_text field."""
        data = {"text": "Truncated...", "full_text": "This is the full text of the tweet"}
        result = _extract_full_text(data)
        assert result == "This is the full text of the tweet"

    def test_extended_tweet(self) -> None:
        """Test extracting from extended_tweet."""
        data = {
            "text": "Truncated...",
            "extended_tweet": {"full_text": "This is the extended tweet text"},
        }
        result = _extract_full_text(data)
        assert result == "This is the extended tweet text"

    def test_note_tweet(self) -> None:
        """Test extracting from note_tweet (long-form)."""
        data = {
            "text": "Truncated...",
            "note_tweet": {"text": "This is a very long Twitter Note with 4000+ characters..."},
        }
        result = _extract_full_text(data)
        assert "Twitter Note" in result

    def test_priority_note_over_extended(self) -> None:
        """Test that note_tweet takes priority."""
        data = {
            "text": "Standard",
            "full_text": "Full",
            "extended_tweet": {"full_text": "Extended"},
            "note_tweet": {"text": "Note"},
        }
        result = _extract_full_text(data)
        assert result == "Note"

    def test_empty_data(self) -> None:
        """Test extracting from empty data."""
        result = _extract_full_text({})
        assert result == ""


class TestTweet:
    """Tests for Tweet dataclass."""

    def test_create_tweet(self) -> None:
        """Test creating a Tweet."""
        tweet = Tweet(
            tweet_id="123456",
            user_id="user_789",
            username="testuser",
            text="Hello world",
            timestamp=datetime.now(timezone.utc),
            raw={"id": "123456"},
        )

        assert tweet.tweet_id == "123456"
        assert tweet.user_id == "user_789"
        assert tweet.username == "testuser"
        assert tweet.text == "Hello world"


class TestTwitterClient:
    """Tests for TwitterClient class."""

    def test_init(self) -> None:
        """Test client initialization."""
        client = TwitterClient(
            api_key="test-key",
            accounts=["DeItaone", "zerohedge"],
        )

        assert client.api_key == "test-key"
        assert client.accounts == ["DeItaone", "zerohedge"]
        assert client.base_url == "https://api.twitterapi.io"
        assert client._client is None
        assert client._running is False

    def test_get_client_creates_httpx_client(self) -> None:
        """Test that _get_client creates HTTP client."""
        client = TwitterClient(api_key="test-key")

        http_client = client._get_client()

        assert http_client is not None
        assert client._client is http_client

    def test_parse_tweet(self) -> None:
        """Test parsing tweet from API response."""
        client = TwitterClient(api_key="test-key")

        data = {
            "id": "123456",
            "createdAt": "2025-01-15T10:30:00Z",
            "text": "Test tweet text",
            "author": {
                "id": "user_789",
                "userName": "testuser",
            },
        }

        tweet = client._parse_tweet(data)

        assert tweet.tweet_id == "123456"
        assert tweet.username == "testuser"
        assert tweet.text == "Test tweet text"

    @pytest.mark.anyio
    async def test_get_user_tweets(self) -> None:
        """Test fetching user tweets."""
        client = TwitterClient(api_key="test-key")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "tweets": [
                {
                    "id": "123",
                    "text": "Tweet 1",
                    "createdAt": "2025-01-15T10:00:00Z",
                    "author": {"id": "u1", "userName": "user1"},
                },
            ],
            "next_cursor": "cursor_abc",
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        client._client = mock_http

        tweets, cursor = await client.get_user_tweets("testuser")

        assert len(tweets) == 1
        assert tweets[0].tweet_id == "123"
        assert cursor == "cursor_abc"

    def test_on_tweet_callback(self) -> None:
        """Test registering tweet callback."""
        client = TwitterClient(api_key="test-key")

        async def callback(tweet: Tweet) -> None:
            pass

        client.on_tweet(callback)

        assert len(client._callbacks) == 1
        assert client._callbacks[0] == callback

    @pytest.mark.anyio
    async def test_start_stop(self) -> None:
        """Test starting and stopping client."""
        client = TwitterClient(api_key="test-key", accounts=["test"])

        # Mock to avoid actual polling
        with patch.object(client, "_poll_loop", new_callable=AsyncMock):
            await client.start()
            assert client._running is True
            assert client._poll_task is not None

            await client.stop()
            assert client._running is False

    @pytest.mark.anyio
    async def test_context_manager(self) -> None:
        """Test async context manager."""
        async with TwitterClient(api_key="test-key") as client:
            assert isinstance(client, TwitterClient)


class TestTwitterStreamClient:
    """Tests for TwitterStreamClient class."""

    def test_init(self) -> None:
        """Test stream client initialization."""
        client = TwitterStreamClient(api_key="test-key")

        assert client.api_key == "test-key"
        assert "wss://" in client.ws_url
        assert client._running is False
        assert client._ws is None

    def test_parse_tweet(self) -> None:
        """Test parsing tweet from WebSocket data."""
        client = TwitterStreamClient(api_key="test-key")

        data = {
            "id": "ws_123",
            "createdAt": "2025-01-15T10:30:00Z",
            "text": "WebSocket tweet",
            "author": {"id": "u1", "userName": "wsuser"},
        }

        tweet = client._parse_tweet(data)

        assert tweet.tweet_id == "ws_123"
        assert tweet.username == "wsuser"

    def test_on_tweet_callback(self) -> None:
        """Test registering callback."""
        client = TwitterStreamClient(api_key="test-key")

        async def callback(tweet: Tweet) -> None:
            pass

        client.on_tweet(callback)

        assert len(client._callbacks) == 1

    @pytest.mark.anyio
    async def test_handle_message_connected(self) -> None:
        """Test handling connected event."""
        client = TwitterStreamClient(api_key="test-key")

        # Should not raise
        await client._handle_message('{"event_type": "connected"}')

    @pytest.mark.anyio
    async def test_handle_message_ping(self) -> None:
        """Test handling ping event."""
        client = TwitterStreamClient(api_key="test-key")

        # Should not raise
        await client._handle_message('{"event_type": "ping", "timestamp": 123}')

    @pytest.mark.anyio
    async def test_handle_message_tweet(self) -> None:
        """Test handling tweet event."""
        client = TwitterStreamClient(api_key="test-key")

        received_tweets: list[Tweet] = []

        async def callback(tweet: Tweet) -> None:
            received_tweets.append(tweet)

        client.on_tweet(callback)

        message = """{"event_type": "tweet", "rule_tag": "news", "tweets": [
            {"id": "123", "text": "Test", "author": {"id": "u1", "userName": "test"}}
        ]}"""

        await client._handle_message(message)

        assert len(received_tweets) == 1
        assert received_tweets[0].tweet_id == "123"

    @pytest.mark.anyio
    async def test_handle_message_invalid_json(self) -> None:
        """Test handling invalid JSON."""
        client = TwitterStreamClient(api_key="test-key")

        # Should not raise, just log warning
        await client._handle_message("not valid json")

    @pytest.mark.anyio
    async def test_stop_closes_websocket(self) -> None:
        """Test that stop closes WebSocket."""
        client = TwitterStreamClient(api_key="test-key")

        mock_ws = AsyncMock()
        client._ws = mock_ws
        client._running = True

        await client.stop()

        mock_ws.close.assert_called_once()
        assert client._running is False
        assert client._ws is None
