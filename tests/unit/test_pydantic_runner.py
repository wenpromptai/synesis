"""Tests for PydanticAI agent runner."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synesis.agent.pydantic_runner import (
    INCOMING_QUEUE,
    SIGNAL_CHANNEL,
    emit_combined_telegram,
    emit_prediction_to_db,
    emit_raw_message_to_db,
    emit_signal,
    emit_signal_to_db,
    emit_stage1_telegram,
    enqueue_test_message,
    store_signal,
)
from synesis.processing.news import (
    Direction,
    LightClassification,
    MarketEvaluation,
    NewsCategory,
    NewsSignal,
    PrimaryTopic,
    SmartAnalysis,
    SourcePlatform,
    UnifiedMessage,
    UrgencyLevel,
)


class TestConstants:
    """Tests for module constants."""

    def test_queue_keys(self) -> None:
        """Test Redis queue keys are defined."""
        assert INCOMING_QUEUE == "synesis:queue:incoming"
        assert SIGNAL_CHANNEL == "synesis:signals"


class TestStoreSignal:
    """Tests for store_signal function."""

    @pytest.fixture
    def sample_signal(self) -> NewsSignal:
        """Create a sample signal for testing."""
        extraction = LightClassification(
            primary_topics=[PrimaryTopic.monetary_policy],
            summary="Fed cuts rates",
            confidence=0.9,
            primary_entity="Federal Reserve",
        )
        return NewsSignal(
            timestamp=datetime.now(timezone.utc),
            source_platform=SourcePlatform.telegram,
            source_account="@test",
            raw_text="Fed cuts rates",
            external_id="test_123",
            extraction=extraction,
        )

    @pytest.mark.anyio
    async def test_store_signal_publishes_to_redis(self, sample_signal: NewsSignal) -> None:
        """Test that store_signal publishes to Redis."""
        mock_redis = AsyncMock()

        await store_signal(sample_signal, mock_redis)

        mock_redis.publish.assert_called_once()
        call_args = mock_redis.publish.call_args
        assert call_args[0][0] == SIGNAL_CHANNEL


class TestEmitSignalToDb:
    """Tests for emit_signal_to_db function."""

    @pytest.mark.anyio
    async def test_stores_signal(self) -> None:
        """Test that signal is stored to database."""
        message = UnifiedMessage(
            external_id="123",
            source_platform=SourcePlatform.telegram,
            source_account="@test",
            text="Test",
            timestamp=datetime.now(timezone.utc),
        )
        extraction = LightClassification(
            news_category=NewsCategory.breaking,
            primary_topics=[PrimaryTopic.monetary_policy],
            summary="Test",
            confidence=0.9,
            primary_entity="Test",
        )
        analysis = SmartAnalysis(
            tickers=["SPY"],
            sectors=[],
            sentiment=Direction.bullish,
            sentiment_score=0.7,
            primary_thesis="Test",
            thesis_confidence=0.8,
        )

        mock_db = MagicMock()
        mock_db.insert_signal = AsyncMock()

        with patch("synesis.agent.pydantic_runner.get_database", return_value=mock_db):
            await emit_signal_to_db(message, extraction, analysis)

        mock_db.insert_signal.assert_called_once()

    @pytest.mark.anyio
    async def test_handles_database_not_available(self) -> None:
        """Test that missing database is handled gracefully."""
        message = UnifiedMessage(
            external_id="123",
            source_platform=SourcePlatform.telegram,
            source_account="@test",
            text="Test",
            timestamp=datetime.now(timezone.utc),
        )
        extraction = LightClassification(
            primary_topics=[PrimaryTopic.other],
            summary="Test",
            confidence=0.5,
            primary_entity="Test",
        )
        analysis = SmartAnalysis(
            tickers=[],
            sectors=[],
            sentiment=Direction.neutral,
            sentiment_score=0.0,
            primary_thesis="Test",
            thesis_confidence=0.5,
        )

        with patch(
            "synesis.agent.pydantic_runner.get_database",
            side_effect=RuntimeError("Not initialized"),
        ):
            # Should not raise
            await emit_signal_to_db(message, extraction, analysis)


class TestEmitPredictionToDb:
    """Tests for emit_prediction_to_db function."""

    @pytest.mark.anyio
    async def test_stores_prediction(self) -> None:
        """Test that prediction is stored."""
        evaluation = MarketEvaluation(
            market_id="mkt_123",
            market_question="Question?",
            is_relevant=True,
            relevance_reasoning="Relevant",
            current_price=0.5,
            verdict="fair",
            confidence=0.7,
            reasoning="Test",
            recommended_side="skip",
        )
        message = UnifiedMessage(
            external_id="123",
            source_platform=SourcePlatform.telegram,
            source_account="@test",
            text="Test",
            timestamp=datetime.now(timezone.utc),
        )

        mock_db = MagicMock()
        mock_db.insert_prediction = AsyncMock()

        with patch("synesis.agent.pydantic_runner.get_database", return_value=mock_db):
            await emit_prediction_to_db(evaluation, message)

        mock_db.insert_prediction.assert_called_once()


class TestEmitRawMessageToDb:
    """Tests for emit_raw_message_to_db function."""

    @pytest.mark.anyio
    async def test_stores_raw_message(self) -> None:
        """Test that raw message is stored."""
        message = UnifiedMessage(
            external_id="123",
            source_platform=SourcePlatform.telegram,
            source_account="@test",
            text="Test message",
            timestamp=datetime.now(timezone.utc),
        )

        mock_db = MagicMock()
        mock_db.insert_raw_message = AsyncMock()

        with patch("synesis.agent.pydantic_runner.get_database", return_value=mock_db):
            await emit_raw_message_to_db(message)

        mock_db.insert_raw_message.assert_called_once_with(message)

    @pytest.mark.anyio
    async def test_handles_database_not_initialized(self) -> None:
        """Test that missing database is handled gracefully."""
        message = UnifiedMessage(
            external_id="123",
            source_platform=SourcePlatform.telegram,
            source_account="@test",
            text="Test",
            timestamp=datetime.now(timezone.utc),
        )

        with patch(
            "synesis.agent.pydantic_runner.get_database",
            side_effect=RuntimeError("Not initialized"),
        ):
            # Should not raise
            await emit_raw_message_to_db(message)


class TestEmitCombinedTelegram:
    """Tests for emit_combined_telegram function."""

    @pytest.mark.anyio
    async def test_sends_all_signals(self) -> None:
        """Test that all signals are sent to Telegram (no confidence threshold)."""
        message = UnifiedMessage(
            external_id="123",
            source_platform=SourcePlatform.telegram,
            source_account="@test",
            text="Test",
            timestamp=datetime.now(timezone.utc),
        )
        analysis = SmartAnalysis(
            tickers=[],
            sectors=[],
            sentiment=Direction.neutral,
            sentiment_score=0.0,
            primary_thesis="Low confidence",
            thesis_confidence=0.3,  # Even low confidence is sent now
        )

        with patch(
            "synesis.agent.pydantic_runner.send_long_telegram", new_callable=AsyncMock
        ) as mock_send:
            await emit_combined_telegram(message, analysis)

        # All signals are sent to Telegram (no threshold)
        mock_send.assert_called_once()

    @pytest.mark.anyio
    async def test_sends_high_confidence(self) -> None:
        """Test that high confidence signals are sent."""
        message = UnifiedMessage(
            external_id="123",
            source_platform=SourcePlatform.telegram,
            source_account="@test",
            text="Test",
            timestamp=datetime.now(timezone.utc),
        )
        analysis = SmartAnalysis(
            tickers=["SPY"],
            sectors=[],
            sentiment=Direction.bullish,
            sentiment_score=0.7,
            primary_thesis="Bullish thesis",
            thesis_confidence=0.8,  # Above 0.5 threshold
        )

        with patch(
            "synesis.agent.pydantic_runner.send_long_telegram", new_callable=AsyncMock
        ) as mock_send:
            await emit_combined_telegram(message, analysis)

        mock_send.assert_called_once()


class TestEnqueueTestMessage:
    """Tests for enqueue_test_message function."""

    @pytest.mark.anyio
    async def test_enqueues_message(self) -> None:
        """Test that test message is enqueued."""
        mock_redis = AsyncMock()
        mock_redis.lpush = AsyncMock()
        mock_redis.close = AsyncMock()

        await enqueue_test_message(redis_client=mock_redis)

        mock_redis.lpush.assert_called_once()
        call_args = mock_redis.lpush.call_args
        assert call_args[0][0] == INCOMING_QUEUE

    @pytest.mark.anyio
    async def test_creates_own_redis_if_not_provided(self) -> None:
        """Test that function creates its own Redis client."""
        mock_redis = AsyncMock()
        mock_redis.lpush = AsyncMock()
        mock_redis.close = AsyncMock()

        with patch("synesis.agent.pydantic_runner.get_settings") as mock_settings:
            mock_settings.return_value.redis_url = "redis://localhost:6379"

            with patch("synesis.agent.pydantic_runner.Redis") as mock_redis_cls:
                mock_redis_cls.from_url.return_value = mock_redis

                await enqueue_test_message()

        mock_redis.close.assert_called_once()  # Should close its own client


class TestEmitStage1Telegram:
    """Tests for emit_stage1_telegram function."""

    @pytest.mark.anyio
    async def test_sends_stage1_notification(self) -> None:
        """Test that Stage 1 signal is sent to Telegram."""
        message = UnifiedMessage(
            external_id="123",
            source_platform=SourcePlatform.telegram,
            source_account="@test",
            text="Breaking: Fed cuts rates",
            timestamp=datetime.now(timezone.utc),
        )
        extraction = LightClassification(
            primary_topics=[PrimaryTopic.monetary_policy],
            summary="Fed cuts rates 25bps",
            confidence=0.9,
            primary_entity="Federal Reserve",
            urgency=UrgencyLevel.critical,
            urgency_reasoning="Surprise Fed cut",
        )

        with patch(
            "synesis.agent.pydantic_runner.send_long_telegram", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = True
            await emit_stage1_telegram(message, extraction)

        mock_send.assert_called_once()
        # Verify the message contains [1st pass] marker
        call_args = mock_send.call_args[0][0]
        assert "[1st pass]" in call_args

    @pytest.mark.anyio
    async def test_stage1_message_includes_entities(self) -> None:
        """Test that Stage 1 message includes entity info."""
        message = UnifiedMessage(
            external_id="124",
            source_platform=SourcePlatform.telegram,
            source_account="@markets",
            text="Apple earnings beat by wide margin",
            timestamp=datetime.now(timezone.utc),
        )
        extraction = LightClassification(
            primary_topics=[PrimaryTopic.earnings],
            summary="Apple Q4 EPS beats estimates",
            confidence=0.95,
            primary_entity="Apple",
            all_entities=["Apple", "Tim Cook"],
            urgency=UrgencyLevel.high,
            urgency_reasoning="Earnings beat",
        )

        with patch(
            "synesis.agent.pydantic_runner.send_long_telegram", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = True
            await emit_stage1_telegram(message, extraction)

        mock_send.assert_called_once()
        call_args = mock_send.call_args[0][0]
        assert "Apple" in call_args
        assert "earnings" in call_args


class TestEmitSignal:
    """Tests for emit_signal function."""

    @pytest.mark.anyio
    async def test_emit_signal_no_analysis(self) -> None:
        """Test emit_signal with no analysis still publishes to Redis."""
        from synesis.core.processor import ProcessingResult

        message = UnifiedMessage(
            external_id="123",
            source_platform=SourcePlatform.telegram,
            source_account="@test",
            text="Test",
            timestamp=datetime.now(timezone.utc),
        )
        extraction = LightClassification(
            primary_topics=[PrimaryTopic.other],
            summary="Test",
            confidence=0.5,
            primary_entity="Test",
        )

        result = ProcessingResult(
            message=message,
            extraction=extraction,
            analysis=None,  # No Stage 2
            is_duplicate=False,
        )

        mock_redis = AsyncMock()

        with patch("synesis.agent.pydantic_runner.emit_signal_to_db", new_callable=AsyncMock):
            await emit_signal(result, mock_redis)

        mock_redis.publish.assert_called_once()

    @pytest.mark.anyio
    async def test_emit_signal_normal_urgency_persists_to_db_no_telegram(self) -> None:
        """Normal urgency: DB persists with analysis=None, no Telegram sent."""
        from synesis.core.processor import ProcessingResult

        message = UnifiedMessage(
            external_id="normal_123",
            source_platform=SourcePlatform.telegram,
            source_account="@test",
            text="Some normal news",
            timestamp=datetime.now(timezone.utc),
        )
        extraction = LightClassification(
            primary_topics=[PrimaryTopic.other],
            summary="Normal news item",
            confidence=0.5,
            primary_entity="Test",
            urgency=UrgencyLevel.normal,
        )
        result = ProcessingResult(
            message=message,
            extraction=extraction,
            analysis=None,
        )

        mock_redis = AsyncMock()

        with (
            patch(
                "synesis.agent.pydantic_runner.emit_signal_to_db", new_callable=AsyncMock
            ) as mock_db,
            patch(
                "synesis.agent.pydantic_runner.emit_stage1_telegram", new_callable=AsyncMock
            ) as mock_stage1,
            patch(
                "synesis.agent.pydantic_runner.emit_combined_telegram", new_callable=AsyncMock
            ) as mock_stage2,
        ):
            await emit_signal(result, mock_redis)

        mock_db.assert_called_once_with(message, extraction, None)
        mock_stage1.assert_not_called()
        mock_stage2.assert_not_called()

    @pytest.mark.anyio
    async def test_emit_signal_high_urgency_stage2_failed_sends_stage1_and_persists(
        self,
    ) -> None:
        """High urgency + Stage 2 failure: Stage 1 Telegram fires, DB persists with analysis=None."""
        from synesis.core.processor import ProcessingResult

        message = UnifiedMessage(
            external_id="high_no_analysis",
            source_platform=SourcePlatform.telegram,
            source_account="@test",
            text="Breaking news",
            timestamp=datetime.now(timezone.utc),
        )
        extraction = LightClassification(
            primary_topics=[PrimaryTopic.monetary_policy],
            summary="Fed cuts rates",
            confidence=0.9,
            primary_entity="Federal Reserve",
            urgency=UrgencyLevel.critical,
        )
        result = ProcessingResult(
            message=message,
            extraction=extraction,
            analysis=None,  # Stage 2 failed
        )

        mock_redis = AsyncMock()

        with (
            patch(
                "synesis.agent.pydantic_runner.emit_signal_to_db", new_callable=AsyncMock
            ) as mock_db,
            patch(
                "synesis.agent.pydantic_runner.emit_stage1_telegram", new_callable=AsyncMock
            ) as mock_stage1,
            patch(
                "synesis.agent.pydantic_runner.emit_combined_telegram", new_callable=AsyncMock
            ) as mock_stage2,
        ):
            await emit_signal(result, mock_redis)

        mock_stage1.assert_called_once_with(message, extraction)
        mock_db.assert_called_once_with(message, extraction, None)
        mock_stage2.assert_not_called()

    @pytest.mark.anyio
    async def test_emit_signal_below_confidence_gate_skips_stage2_telegram(self) -> None:
        """High urgency + low confidence: DB persists + predictions stored, Stage 2 Telegram skipped."""
        from synesis.core.constants import MIN_THESIS_CONFIDENCE_FOR_ALERT
        from synesis.core.processor import ProcessingResult

        message = UnifiedMessage(
            external_id="low_confidence",
            source_platform=SourcePlatform.telegram,
            source_account="@test",
            text="Some high urgency news",
            timestamp=datetime.now(timezone.utc),
        )
        extraction = LightClassification(
            primary_topics=[PrimaryTopic.earnings],
            summary="Earnings report",
            confidence=0.9,
            primary_entity="Apple",
            urgency=UrgencyLevel.high,
        )
        analysis = SmartAnalysis(
            tickers=["AAPL"],
            sectors=[],
            sentiment=Direction.bullish,
            sentiment_score=0.5,
            primary_thesis="Weak thesis",
            thesis_confidence=max(0.0, MIN_THESIS_CONFIDENCE_FOR_ALERT - 0.01),
            market_evaluations=[
                MarketEvaluation(
                    market_id="mkt_abc",
                    market_question="Will AAPL beat earnings?",
                    is_relevant=True,
                    relevance_reasoning="Direct subject",
                    current_price=0.5,
                    verdict="fair",
                    confidence=0.7,
                    reasoning="Uncertain",
                    recommended_side="skip",
                )
            ],
        )
        result = ProcessingResult(
            message=message,
            extraction=extraction,
            analysis=analysis,
        )

        mock_redis = AsyncMock()

        with (
            patch(
                "synesis.agent.pydantic_runner.emit_signal_to_db", new_callable=AsyncMock
            ) as mock_db,
            patch(
                "synesis.agent.pydantic_runner.emit_stage1_telegram", new_callable=AsyncMock
            ) as mock_stage1,
            patch(
                "synesis.agent.pydantic_runner.emit_combined_telegram", new_callable=AsyncMock
            ) as mock_stage2,
            patch(
                "synesis.agent.pydantic_runner.emit_prediction_to_db", new_callable=AsyncMock
            ) as mock_pred,
        ):
            await emit_signal(result, mock_redis)

        mock_stage1.assert_called_once()
        mock_db.assert_called_once_with(message, extraction, analysis)
        mock_pred.assert_called_once()  # predictions still stored
        mock_stage2.assert_not_called()  # Stage 2 Telegram skipped
