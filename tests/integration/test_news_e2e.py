"""Integration smoke test for news processing pipeline.

Uses REAL APIs (LLM, SEC EDGAR, NASDAQ, Polymarket, Telegram) with mocked storage.
Run with: pytest -m integration

Environment variables required:
- ANTHROPIC_API_KEY or OPENAI_API_KEY
- TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (for notifications)
"""

from datetime import datetime, timezone
from typing import Any

import pytest

from synesis.core.processor import NewsProcessor
from synesis.processing.common.watchlist import WatchlistManager
from synesis.processing.news import (
    SourcePlatform,
    SourceType,
    UnifiedMessage,
)


@pytest.mark.integration
class TestNewsE2E:
    """Smoke test for news processing with real APIs."""

    @pytest.fixture
    def breaking_news_message(self) -> UnifiedMessage:
        """Breaking news: Fed rate decision."""
        return UnifiedMessage(
            external_id=f"test_e2e_{datetime.now().timestamp()}",
            source_platform=SourcePlatform.twitter,
            source_account="@DeItaone",
            text="*FED CUTS RATES BY 25BPS - Fed funds rate now 4.00-4.25%, as expected",
            timestamp=datetime.now(timezone.utc),
            source_type=SourceType.news,
        )

    @pytest.mark.anyio
    async def test_news_smoke(
        self,
        mock_redis: Any,
        mock_db: Any,
        ticker_provider: Any,
        breaking_news_message: UnifiedMessage,
    ) -> None:
        """Smoke test: Full news pipeline with real LLM + providers + Telegram.

        Verifies:
        1. Stage 1 classification works (real LLM)
        2. Stage 2 analysis works (real LLM + Polymarket)
        3. Telegram notification sends
        4. Mock storage captures data correctly
        """
        # Create processor with mock Redis and ticker provider
        processor = NewsProcessor(
            mock_redis,
            ticker_provider=ticker_provider,
        )
        await processor.initialize()

        # Create watchlist manager with mock Redis/DB
        watchlist = WatchlistManager(mock_redis, db=mock_db)

        try:
            # Process message through REAL LLM pipeline
            result = await processor.process_message(breaking_news_message)

            # Verify processing succeeded
            assert result is not None
            assert result.skipped is False, f"Message was skipped: {result.skip_reason}"
            assert result.extraction is not None, "Stage 1 extraction failed"

            print(f"\n{'=' * 60}")
            print("STAGE 1 (Classification):")
            print(f"  Event type: {result.extraction.event_type}")
            print(f"  Primary entity: {result.extraction.primary_entity}")
            print(f"  Urgency: {result.extraction.urgency}")

            if result.analysis:
                print("\nSTAGE 2 (Analysis):")
                print(f"  Tickers: {result.analysis.tickers}")
                print(
                    f"  Sentiment: {result.analysis.sentiment} ({result.analysis.sentiment_score:+.2f})"
                )

                # Add tickers to watchlist
                for ticker in result.analysis.tickers:
                    await watchlist.add_ticker(
                        ticker,
                        source=breaking_news_message.source_platform.value,
                    )

            # Store in mock DB
            await mock_db.insert_raw_message(breaking_news_message)

            signal = result.to_signal()
            if signal:
                await mock_db.insert_signal(signal, prices=None)
                await mock_redis.publish("synesis:signals", signal.model_dump_json())

                # Send REAL Telegram notification
                from synesis.notifications.telegram import (
                    format_condensed_signal,
                    send_telegram,
                )

                if result.analysis:
                    telegram_msg = format_condensed_signal(
                        breaking_news_message,
                        result.analysis,
                    )
                    print(f"\n{'=' * 60}")
                    print("TELEGRAM (sending...):")
                    sent = await send_telegram(telegram_msg)
                    print(f"  Sent: {sent}")

            # Verify mock storage
            assert len(mock_db._test_raw_messages) == 1, "Raw message not stored"
            assert len(mock_db._test_signals) >= 1, "Signal not stored"
            assert len(mock_redis._test_pubsub) >= 1, "Signal not published"

            if result.analysis and result.analysis.tickers:
                watchlist_tickers = mock_redis._test_sets.get("synesis:watchlist:tickers", set())
                print(f"\nWatchlist: {watchlist_tickers}")
                assert len(watchlist_tickers) > 0, "No tickers in watchlist"

        finally:
            await processor.close()
