"""Tests for Redis client module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synesis.storage.redis import (
    close_redis,
    get_redis,
    init_redis,
)


class TestGetRedis:
    """Tests for get_redis function."""

    def test_not_initialized(self) -> None:
        """Test get_redis raises if not initialized."""
        import synesis.storage.redis as redis_module

        redis_module._redis = None

        with pytest.raises(RuntimeError, match="not initialized"):
            get_redis()

    def test_returns_instance(self) -> None:
        """Test get_redis returns the instance."""
        import synesis.storage.redis as redis_module

        mock_redis = MagicMock()
        redis_module._redis = mock_redis

        result = get_redis()

        assert result is mock_redis

        # Cleanup
        redis_module._redis = None


class TestInitRedis:
    """Tests for init_redis function."""

    @pytest.mark.anyio
    async def test_init_creates_client(self) -> None:
        """Test init_redis creates Redis client."""
        import synesis.storage.redis as redis_module

        mock_redis = MagicMock()
        # ping() returns a coroutine that we need to mock
        mock_ping = AsyncMock(return_value=True)
        mock_redis.ping = MagicMock(return_value=mock_ping())

        with patch("synesis.storage.redis.Redis") as mock_redis_cls:
            mock_redis_cls.from_url.return_value = mock_redis

            result = await init_redis("redis://localhost:6379")

        assert result is mock_redis
        assert redis_module._redis is mock_redis
        mock_redis_cls.from_url.assert_called_once_with(
            "redis://localhost:6379",
            decode_responses=False,
        )

        # Cleanup
        redis_module._redis = None

    @pytest.mark.anyio
    async def test_init_tests_connection(self) -> None:
        """Test init_redis pings to verify connection."""
        import synesis.storage.redis as redis_module

        mock_redis = MagicMock()
        mock_ping = AsyncMock(return_value=True)
        mock_redis.ping = MagicMock(return_value=mock_ping())

        with patch("synesis.storage.redis.Redis") as mock_redis_cls:
            mock_redis_cls.from_url.return_value = mock_redis

            await init_redis("redis://localhost:6379")

        # Should have called ping
        mock_redis.ping.assert_called_once()

        # Cleanup
        redis_module._redis = None


class TestCloseRedis:
    """Tests for close_redis function."""

    @pytest.mark.anyio
    async def test_close_when_initialized(self) -> None:
        """Test close_redis closes the client."""
        import synesis.storage.redis as redis_module

        mock_redis = MagicMock()
        mock_redis.aclose = AsyncMock()
        redis_module._redis = mock_redis

        await close_redis()

        mock_redis.aclose.assert_called_once()
        assert redis_module._redis is None

    @pytest.mark.anyio
    async def test_close_when_not_initialized(self) -> None:
        """Test close_redis handles None gracefully."""
        import synesis.storage.redis as redis_module

        redis_module._redis = None

        # Should not raise
        await close_redis()

        assert redis_module._redis is None


class TestRedisIntegration:
    """Integration-style tests for Redis workflow."""

    @pytest.mark.anyio
    async def test_init_get_close_lifecycle(self) -> None:
        """Test full Redis lifecycle: init -> get -> close."""
        import synesis.storage.redis as redis_module

        mock_redis = MagicMock()
        mock_ping = AsyncMock(return_value=True)
        mock_redis.ping = MagicMock(return_value=mock_ping())
        mock_redis.aclose = AsyncMock()

        with patch("synesis.storage.redis.Redis") as mock_redis_cls:
            mock_redis_cls.from_url.return_value = mock_redis

            # Init
            await init_redis("redis://localhost:6379")

            # Get should return same instance
            client = get_redis()
            assert client is mock_redis

            # Close
            await close_redis()
            assert redis_module._redis is None

            # Get should now raise
            with pytest.raises(RuntimeError):
                get_redis()
