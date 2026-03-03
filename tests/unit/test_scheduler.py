"""Tests for scheduler cleanup job."""

from unittest.mock import AsyncMock

import pytest

from synesis.agent.scheduler import watchlist_cleanup_job


class TestWatchlistCleanupJob:
    """Tests for watchlist_cleanup_job."""

    @pytest.mark.asyncio
    async def test_deactivates_expired_tickers(self) -> None:
        db = AsyncMock()
        db.deactivate_expired_watchlist = AsyncMock(return_value=["AAPL", "GME"])
        expired = await watchlist_cleanup_job(db)
        db.deactivate_expired_watchlist.assert_called_once()
        # Job doesn't return anything (logs instead), just verify no exception
        assert expired is None

    @pytest.mark.asyncio
    async def test_no_expired_tickers(self) -> None:
        db = AsyncMock()
        db.deactivate_expired_watchlist = AsyncMock(return_value=[])
        await watchlist_cleanup_job(db)
        db.deactivate_expired_watchlist.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_db_exception(self) -> None:
        db = AsyncMock()
        db.deactivate_expired_watchlist = AsyncMock(side_effect=RuntimeError("db down"))
        # Should not raise — exception is caught and logged
        await watchlist_cleanup_job(db)
        db.deactivate_expired_watchlist.assert_called_once()
