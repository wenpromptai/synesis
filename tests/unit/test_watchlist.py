"""Unit tests for WatchlistManager (DB-only)."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from synesis.processing.common.watchlist import WatchlistManager


@pytest.fixture
def mock_db() -> AsyncMock:
    """Create a mock Database."""
    db = AsyncMock()
    db.upsert_watchlist_ticker = AsyncMock(return_value=True)
    db.remove_watchlist_ticker = AsyncMock(return_value=True)
    db.get_active_watchlist = AsyncMock(return_value=[])
    db.get_active_watchlist_with_metadata = AsyncMock(return_value=[])
    db.get_watchlist_metadata = AsyncMock(return_value=None)
    db.get_watchlist_stats = AsyncMock(return_value={"total_tickers": 0, "sources": {}})
    db.watchlist_contains = AsyncMock(return_value=False)
    db.deactivate_expired_watchlist = AsyncMock(return_value=[])
    return db


@pytest.fixture
def manager(mock_db: AsyncMock) -> WatchlistManager:
    """Create WatchlistManager with mock Database."""
    return WatchlistManager(db=mock_db, ttl_days=7)


class TestWatchlistAddTicker:
    """Tests for WatchlistManager.add_ticker."""

    @pytest.mark.asyncio
    async def test_new_ticker_returns_true(
        self, manager: WatchlistManager, mock_db: AsyncMock
    ) -> None:
        mock_db.upsert_watchlist_ticker.return_value = True
        result = await manager.add_ticker("aapl", "telegram")
        assert result is True
        mock_db.upsert_watchlist_ticker.assert_called_once()
        call_kwargs = mock_db.upsert_watchlist_ticker.call_args
        assert call_kwargs.kwargs["ticker"] == "AAPL"
        assert call_kwargs.kwargs["added_by"] == "telegram"

    @pytest.mark.asyncio
    async def test_existing_ticker_returns_false(
        self, manager: WatchlistManager, mock_db: AsyncMock
    ) -> None:
        mock_db.upsert_watchlist_ticker.return_value = False
        result = await manager.add_ticker("AAPL", "telegram")
        assert result is False

    @pytest.mark.asyncio
    async def test_uppercases_ticker(self, manager: WatchlistManager, mock_db: AsyncMock) -> None:
        mock_db.upsert_watchlist_ticker.return_value = True
        await manager.add_ticker("tsla", "twitter")
        call_kwargs = mock_db.upsert_watchlist_ticker.call_args
        assert call_kwargs.kwargs["ticker"] == "TSLA"

    @pytest.mark.asyncio
    async def test_sets_correct_expiry(self, manager: WatchlistManager, mock_db: AsyncMock) -> None:
        before = datetime.now(UTC)
        await manager.add_ticker("AAPL", "telegram")
        after = datetime.now(UTC)

        call_kwargs = mock_db.upsert_watchlist_ticker.call_args
        expires_at = call_kwargs.kwargs["expires_at"]
        assert before + timedelta(days=7) <= expires_at <= after + timedelta(days=7)


class TestWatchlistRemoveTicker:
    """Tests for WatchlistManager.remove_ticker."""

    @pytest.mark.asyncio
    async def test_remove_existing_returns_true(
        self, manager: WatchlistManager, mock_db: AsyncMock
    ) -> None:
        mock_db.remove_watchlist_ticker.return_value = True
        result = await manager.remove_ticker("AAPL")
        assert result is True
        mock_db.remove_watchlist_ticker.assert_called_once_with("AAPL")

    @pytest.mark.asyncio
    async def test_remove_nonexistent_returns_false(
        self, manager: WatchlistManager, mock_db: AsyncMock
    ) -> None:
        mock_db.remove_watchlist_ticker.return_value = False
        result = await manager.remove_ticker("AAPL")
        assert result is False


class TestWatchlistGetAll:
    """Tests for WatchlistManager.get_all."""

    @pytest.mark.asyncio
    async def test_delegates_to_db(self, manager: WatchlistManager, mock_db: AsyncMock) -> None:
        mock_db.get_active_watchlist.return_value = ["AAPL", "NVDA", "TSLA"]
        result = await manager.get_all()
        assert result == ["AAPL", "NVDA", "TSLA"]
        mock_db.get_active_watchlist.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_list(self, manager: WatchlistManager, mock_db: AsyncMock) -> None:
        mock_db.get_active_watchlist.return_value = []
        result = await manager.get_all()
        assert result == []


class TestWatchlistGetMetadata:
    """Tests for WatchlistManager.get_metadata."""

    @pytest.mark.asyncio
    async def test_returns_metadata(self, manager: WatchlistManager, mock_db: AsyncMock) -> None:
        now = datetime.now(UTC)
        mock_db.get_watchlist_metadata.return_value = {
            "ticker": "AAPL",
            "added_by": "telegram",
            "added_reason": "Signal from telegram",
            "added_at": now,
            "expires_at": now + timedelta(days=7),
        }
        result = await manager.get_metadata("aapl")
        assert result is not None
        assert result["ticker"] == "AAPL"
        mock_db.get_watchlist_metadata.assert_called_once_with("AAPL")

    @pytest.mark.asyncio
    async def test_returns_none_when_missing(
        self, manager: WatchlistManager, mock_db: AsyncMock
    ) -> None:
        mock_db.get_watchlist_metadata.return_value = None
        result = await manager.get_metadata("AAPL")
        assert result is None


class TestWatchlistContains:
    """Tests for WatchlistManager.contains."""

    @pytest.mark.asyncio
    async def test_contains_true(self, manager: WatchlistManager, mock_db: AsyncMock) -> None:
        mock_db.watchlist_contains.return_value = True
        assert await manager.contains("aapl") is True
        mock_db.watchlist_contains.assert_called_with("AAPL")

    @pytest.mark.asyncio
    async def test_contains_false(self, manager: WatchlistManager, mock_db: AsyncMock) -> None:
        mock_db.watchlist_contains.return_value = False
        assert await manager.contains("XYZ") is False


class TestWatchlistCleanupExpired:
    """Tests for WatchlistManager.cleanup_expired."""

    @pytest.mark.asyncio
    async def test_delegates_to_db(self, manager: WatchlistManager, mock_db: AsyncMock) -> None:
        mock_db.deactivate_expired_watchlist.return_value = ["AAPL", "GME"]
        removed = await manager.cleanup_expired()
        assert removed == ["AAPL", "GME"]
        mock_db.deactivate_expired_watchlist.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_cleanup(self, manager: WatchlistManager, mock_db: AsyncMock) -> None:
        mock_db.deactivate_expired_watchlist.return_value = []
        removed = await manager.cleanup_expired()
        assert removed == []


class TestWatchlistGetStats:
    """Tests for WatchlistManager.get_stats."""

    @pytest.mark.asyncio
    async def test_delegates_to_db(self, manager: WatchlistManager, mock_db: AsyncMock) -> None:
        mock_db.get_watchlist_stats.return_value = {
            "total_tickers": 5,
            "sources": {"telegram": 3, "api": 2},
        }
        result = await manager.get_stats()
        assert result["total_tickers"] == 5
        assert result["sources"]["telegram"] == 3
        mock_db.get_watchlist_stats.assert_called_once()


class TestWatchlistGetAllWithMetadata:
    """Tests for WatchlistManager.get_all_with_metadata."""

    @pytest.mark.asyncio
    async def test_converts_records_to_dicts(
        self, manager: WatchlistManager, mock_db: AsyncMock
    ) -> None:
        now = datetime.now(UTC)
        # Simulate asyncpg.Record-like objects (MagicMock with dict conversion)
        mock_record = {"ticker": "AAPL", "added_by": "telegram", "added_at": now, "expires_at": now}
        mock_db.get_active_watchlist_with_metadata.return_value = [mock_record]
        result = await manager.get_all_with_metadata()
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert result[0]["ticker"] == "AAPL"

    @pytest.mark.asyncio
    async def test_empty(self, manager: WatchlistManager, mock_db: AsyncMock) -> None:
        mock_db.get_active_watchlist_with_metadata.return_value = []
        result = await manager.get_all_with_metadata()
        assert result == []


class TestWatchlistRemoveUppercases:
    """Tests for WatchlistManager.remove_ticker uppercasing."""

    @pytest.mark.asyncio
    async def test_remove_uppercases(self, manager: WatchlistManager, mock_db: AsyncMock) -> None:
        mock_db.remove_watchlist_ticker.return_value = True
        await manager.remove_ticker("aapl")
        mock_db.remove_watchlist_ticker.assert_called_once_with("AAPL")


class TestWatchlistCustomTtl:
    """Tests for WatchlistManager with custom ttl_days."""

    @pytest.mark.asyncio
    async def test_custom_ttl_days(self, mock_db: AsyncMock) -> None:
        manager = WatchlistManager(db=mock_db, ttl_days=14)
        before = datetime.now(UTC)
        await manager.add_ticker("AAPL", "api")
        after = datetime.now(UTC)
        call_kwargs = mock_db.upsert_watchlist_ticker.call_args
        expires_at = call_kwargs.kwargs["expires_at"]
        assert before + timedelta(days=14) <= expires_at <= after + timedelta(days=14)


class TestWatchlistAddReasonFormat:
    """Tests for WatchlistManager.add_ticker added_reason formatting."""

    @pytest.mark.asyncio
    async def test_added_reason_contains_source(
        self, manager: WatchlistManager, mock_db: AsyncMock
    ) -> None:
        await manager.add_ticker("AAPL", "twitter")
        call_kwargs = mock_db.upsert_watchlist_ticker.call_args
        assert call_kwargs.kwargs["added_reason"] == "Signal from twitter"


class TestWatchlistGetMetadataConvertsToDict:
    """Tests for WatchlistManager.get_metadata dict conversion."""

    @pytest.mark.asyncio
    async def test_converts_record_to_dict(
        self, manager: WatchlistManager, mock_db: AsyncMock
    ) -> None:
        """Ensure get_metadata converts asyncpg.Record-like objects to plain dicts."""
        now = datetime.now(UTC)
        # asyncpg.Record supports dict() conversion; mock that
        mock_record = {"ticker": "AAPL", "added_by": "api", "added_at": now, "expires_at": now}
        mock_db.get_watchlist_metadata.return_value = mock_record
        result = await manager.get_metadata("aapl")
        assert isinstance(result, dict)
        assert result["ticker"] == "AAPL"


class TestWatchlistBulkAdd:
    """Tests for WatchlistManager.bulk_add."""

    @pytest.mark.asyncio
    async def test_returns_new_and_refreshed(
        self, manager: WatchlistManager, mock_db: AsyncMock
    ) -> None:
        # First ticker is new, second already exists
        mock_db.upsert_watchlist_ticker.side_effect = [True, False]
        newly_added, refreshed = await manager.bulk_add(["AAPL", "TSLA"], source="telegram")
        assert "AAPL" in newly_added
        assert "TSLA" in refreshed

    @pytest.mark.asyncio
    async def test_empty_list(self, manager: WatchlistManager, mock_db: AsyncMock) -> None:
        newly_added, refreshed = await manager.bulk_add([], source="api")
        assert newly_added == []
        assert refreshed == []
        mock_db.upsert_watchlist_ticker.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_new(self, manager: WatchlistManager, mock_db: AsyncMock) -> None:
        mock_db.upsert_watchlist_ticker.side_effect = [True, True, True]
        newly_added, refreshed = await manager.bulk_add(["AAPL", "TSLA", "NVDA"], source="telegram")
        assert len(newly_added) == 3
        assert len(refreshed) == 0

    @pytest.mark.asyncio
    async def test_all_existing(self, manager: WatchlistManager, mock_db: AsyncMock) -> None:
        mock_db.upsert_watchlist_ticker.side_effect = [False, False]
        newly_added, refreshed = await manager.bulk_add(["AAPL", "TSLA"], source="telegram")
        assert len(newly_added) == 0
        assert len(refreshed) == 2
