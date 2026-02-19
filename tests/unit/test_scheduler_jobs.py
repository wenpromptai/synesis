"""Tests for centralized scheduler jobs (agent/scheduler.py)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synesis.agent.scheduler import (
    create_scheduler,
    sentiment_signal_job,
    watchlist_cleanup_job,
    watchlist_intel_job,
)


class TestCreateScheduler:
    """Tests for create_scheduler."""

    def test_creates_scheduler(self) -> None:
        scheduler = create_scheduler()
        assert scheduler is not None
        assert str(scheduler.timezone) == "UTC"


class TestSentimentSignalJob:
    """Tests for sentiment_signal_job."""

    @pytest.mark.asyncio
    async def test_generates_and_publishes_signal(self) -> None:
        mock_processor = AsyncMock()
        mock_signal = MagicMock()
        mock_signal.model_dump_json.return_value = '{"test": true}'
        mock_signal.watchlist = ["AAPL"]
        mock_signal.total_posts_analyzed = 10
        mock_signal.overall_sentiment = "bullish"
        mock_processor.generate_signal.return_value = mock_signal

        mock_redis = AsyncMock()

        with (
            patch(
                "synesis.agent.scheduler.format_sentiment_signal",
                return_value="formatted",
            ),
            patch(
                "synesis.agent.scheduler.send_long_telegram",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_send,
        ):
            await sentiment_signal_job(mock_processor, mock_redis)

        mock_processor.generate_signal.assert_awaited_once()
        mock_redis.publish.assert_awaited_once_with("synesis:sentiment:signals", '{"test": true}')
        mock_send.assert_awaited_once_with("formatted")

    @pytest.mark.asyncio
    async def test_logs_error_on_telegram_failure(self) -> None:
        mock_processor = AsyncMock()
        mock_signal = MagicMock()
        mock_signal.model_dump_json.return_value = "{}"
        mock_signal.watchlist = []
        mock_signal.total_posts_analyzed = 0
        mock_signal.overall_sentiment = "neutral"
        mock_processor.generate_signal.return_value = mock_signal

        mock_redis = AsyncMock()

        with (
            patch(
                "synesis.agent.scheduler.format_sentiment_signal",
                return_value="msg",
            ),
            patch(
                "synesis.agent.scheduler.send_long_telegram",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            # Should not raise
            await sentiment_signal_job(mock_processor, mock_redis)


class TestWatchlistIntelJob:
    """Tests for watchlist_intel_job."""

    @pytest.mark.asyncio
    async def test_runs_analysis_and_publishes(self) -> None:
        mock_processor = AsyncMock()
        mock_signal = MagicMock()
        mock_signal.model_dump_json.return_value = '{"tickers": 5}'
        mock_signal.tickers_analyzed = 5
        mock_signal.alerts = []
        mock_processor.run_analysis.return_value = mock_signal

        mock_redis = AsyncMock()

        with (
            patch(
                "synesis.agent.scheduler.format_watchlist_signal",
                return_value="formatted",
            ),
            patch(
                "synesis.agent.scheduler.send_long_telegram",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_send,
        ):
            await watchlist_intel_job(mock_processor, mock_redis)

        mock_processor.run_analysis.assert_awaited_once()
        mock_redis.publish.assert_awaited_once_with(
            "synesis:watchlist_intel:signals", '{"tickers": 5}'
        )
        mock_send.assert_awaited_once_with("formatted")


class TestWatchlistCleanupJob:
    """Tests for watchlist_cleanup_job."""

    @pytest.mark.asyncio
    async def test_cleanup_runs_db_and_redis(self) -> None:
        mock_db = AsyncMock()
        mock_db.deactivate_expired_watchlist.return_value = ["OLDTICKER"]

        mock_watchlist = AsyncMock()
        mock_watchlist.cleanup_expired.return_value = ["OLDTICKER"]

        await watchlist_cleanup_job(mock_db, mock_watchlist)

        mock_db.deactivate_expired_watchlist.assert_awaited_once()
        mock_watchlist.cleanup_expired.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cleanup_without_watchlist(self) -> None:
        mock_db = AsyncMock()
        mock_db.deactivate_expired_watchlist.return_value = []

        await watchlist_cleanup_job(mock_db, None)

        mock_db.deactivate_expired_watchlist.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cleanup_no_expired(self) -> None:
        mock_db = AsyncMock()
        mock_db.deactivate_expired_watchlist.return_value = []

        mock_watchlist = AsyncMock()
        mock_watchlist.cleanup_expired.return_value = []

        await watchlist_cleanup_job(mock_db, mock_watchlist)

        mock_db.deactivate_expired_watchlist.assert_awaited_once()
        mock_watchlist.cleanup_expired.assert_awaited_once()
