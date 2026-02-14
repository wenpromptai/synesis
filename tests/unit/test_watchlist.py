"""Unit tests for WatchlistManager."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from synesis.processing.common.watchlist import (
    WATCHLIST_KEY,
    TickerMetadata,
    WatchlistManager,
)


@pytest.fixture
def mock_redis() -> AsyncMock:
    """Create a mock Redis client."""
    redis = AsyncMock()
    # Default behaviors
    redis.sismember = AsyncMock(return_value=False)
    redis.sadd = AsyncMock()
    redis.set = AsyncMock()
    redis.hset = AsyncMock()
    redis.expire = AsyncMock()
    redis.srem = AsyncMock()
    redis.delete = AsyncMock()
    redis.smembers = AsyncMock(return_value=set())
    redis.hgetall = AsyncMock(return_value={})
    redis.hincrby = AsyncMock()
    return redis


@pytest.fixture
def manager(mock_redis: AsyncMock) -> WatchlistManager:
    """Create WatchlistManager with mock Redis."""
    return WatchlistManager(redis=mock_redis, ttl_days=7)


class TestTickerMetadata:
    """Tests for TickerMetadata dataclass."""

    def test_to_dict_roundtrip(self) -> None:
        now = datetime.now(UTC)
        meta = TickerMetadata(
            ticker="AAPL",
            source="twitter",
            subreddit="wsb",
            added_at=now,
            last_seen_at=now,
            mention_count=5,
        )
        d = meta.to_dict()
        restored = TickerMetadata.from_dict(d)
        assert restored.ticker == "AAPL"
        assert restored.source == "twitter"
        assert restored.subreddit == "wsb"
        assert restored.mention_count == 5

    def test_none_subreddit_handling(self) -> None:
        meta = TickerMetadata(ticker="TSLA", source="telegram", subreddit=None)
        d = meta.to_dict()
        assert d["subreddit"] == ""
        restored = TickerMetadata.from_dict(d)
        assert restored.subreddit is None

    def test_to_dict_values_are_strings(self) -> None:
        meta = TickerMetadata(ticker="GME", source="reddit", mention_count=3)
        d = meta.to_dict()
        for v in d.values():
            assert isinstance(v, str)


class TestWatchlistAddTicker:
    """Tests for WatchlistManager.add_ticker."""

    @pytest.mark.asyncio
    async def test_new_ticker_returns_true(
        self, manager: WatchlistManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.sismember.return_value = False
        result = await manager.add_ticker("aapl", "twitter")
        assert result is True
        mock_redis.sadd.assert_called_once_with(WATCHLIST_KEY, "AAPL")

    @pytest.mark.asyncio
    async def test_existing_ticker_returns_false(
        self, manager: WatchlistManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.sismember.return_value = True
        result = await manager.add_ticker("AAPL", "twitter")
        assert result is False

    @pytest.mark.asyncio
    async def test_uppercases_ticker(
        self, manager: WatchlistManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.sismember.return_value = False
        await manager.add_ticker("tsla", "reddit")
        mock_redis.sadd.assert_called_once_with(WATCHLIST_KEY, "TSLA")

    @pytest.mark.asyncio
    async def test_refreshes_ttl_on_existing(
        self, manager: WatchlistManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.sismember.return_value = True
        await manager.add_ticker("AAPL", "twitter")
        # Should still set TTL key even for existing
        mock_redis.set.assert_called_once()
        mock_redis.hincrby.assert_called_once()


class TestWatchlistRemoveTicker:
    """Tests for WatchlistManager.remove_ticker."""

    @pytest.mark.asyncio
    async def test_remove_existing_returns_true(
        self, manager: WatchlistManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.sismember.return_value = True
        result = await manager.remove_ticker("AAPL")
        assert result is True
        mock_redis.srem.assert_called_once_with(WATCHLIST_KEY, "AAPL")

    @pytest.mark.asyncio
    async def test_remove_nonexistent_returns_false(
        self, manager: WatchlistManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.sismember.return_value = False
        result = await manager.remove_ticker("AAPL")
        assert result is False
        mock_redis.srem.assert_not_called()


class TestWatchlistGetAll:
    """Tests for WatchlistManager.get_all."""

    @pytest.mark.asyncio
    async def test_returns_sorted_decoded(
        self, manager: WatchlistManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.smembers.return_value = {b"TSLA", b"AAPL", b"NVDA"}
        result = await manager.get_all()
        assert result == ["AAPL", "NVDA", "TSLA"]

    @pytest.mark.asyncio
    async def test_handles_string_members(
        self, manager: WatchlistManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.smembers.return_value = {"TSLA", "AAPL"}
        result = await manager.get_all()
        assert sorted(result) == ["AAPL", "TSLA"]

    @pytest.mark.asyncio
    async def test_empty_set(self, manager: WatchlistManager, mock_redis: AsyncMock) -> None:
        mock_redis.smembers.return_value = set()
        result = await manager.get_all()
        assert result == []


class TestWatchlistGetMetadata:
    """Tests for WatchlistManager.get_metadata."""

    @pytest.mark.asyncio
    async def test_returns_metadata(self, manager: WatchlistManager, mock_redis: AsyncMock) -> None:
        now = datetime.now(UTC)
        mock_redis.hgetall.return_value = {
            "ticker": "AAPL",
            "source": "twitter",
            "subreddit": "",
            "added_at": now.isoformat(),
            "last_seen_at": now.isoformat(),
            "mention_count": "3",
        }
        result = await manager.get_metadata("aapl")
        assert result is not None
        assert result.ticker == "AAPL"
        assert result.mention_count == 3

    @pytest.mark.asyncio
    async def test_returns_none_when_missing(
        self, manager: WatchlistManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.hgetall.return_value = {}
        result = await manager.get_metadata("AAPL")
        assert result is None


class TestWatchlistContains:
    """Tests for WatchlistManager.contains."""

    @pytest.mark.asyncio
    async def test_contains_true(self, manager: WatchlistManager, mock_redis: AsyncMock) -> None:
        mock_redis.sismember.return_value = True
        assert await manager.contains("aapl") is True
        mock_redis.sismember.assert_called_with(WATCHLIST_KEY, "AAPL")

    @pytest.mark.asyncio
    async def test_contains_false(self, manager: WatchlistManager, mock_redis: AsyncMock) -> None:
        mock_redis.sismember.return_value = False
        assert await manager.contains("XYZ") is False


class TestWatchlistCleanupExpired:
    """Tests for WatchlistManager.cleanup_expired."""

    @pytest.mark.asyncio
    async def test_removes_expired_tickers(
        self, manager: WatchlistManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.smembers.return_value = {b"AAPL", b"TSLA"}

        # Mock Lua script: AAPL expired (returns 1), TSLA still active (returns 0)
        mock_script = AsyncMock(side_effect=[1, 0])
        mock_redis.register_script = MagicMock(return_value=mock_script)

        removed = await manager.cleanup_expired()
        assert "AAPL" in removed
        assert "TSLA" not in removed


class TestWatchlistBulkAdd:
    """Tests for WatchlistManager.bulk_add."""

    @pytest.mark.asyncio
    async def test_returns_new_and_refreshed(
        self, manager: WatchlistManager, mock_redis: AsyncMock
    ) -> None:
        # First ticker is new, second already exists
        mock_redis.sismember.side_effect = [False, True]
        newly_added, refreshed = await manager.bulk_add(
            ["AAPL", "TSLA"], source="reddit", subreddit="wsb"
        )
        assert "AAPL" in newly_added
        assert "TSLA" in refreshed
