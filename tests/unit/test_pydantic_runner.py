"""Tests for PydanticAI agent runner."""

from datetime import datetime, timezone
from pathlib import Path
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
    enqueue_test_message,
    store_signal,
)
from synesis.core.constants import DEFAULT_SIGNALS_OUTPUT_DIR
from synesis.processing.news import (
    Direction,
    EventType,
    NewsSignal,
    ImpactLevel,
    LightClassification,
    MarketEvaluation,
    NewsCategory,
    SmartAnalysis,
    SourcePlatform,
    SourceType,
    UnifiedMessage,
)


class TestConstants:
    """Tests for module constants."""

    def test_queue_keys(self) -> None:
        """Test Redis queue keys are defined."""
        assert INCOMING_QUEUE == "synesis:queue:incoming"
        assert SIGNAL_CHANNEL == "synesis:signals"

    def test_signals_dir(self) -> None:
        """Test signals directory default path."""
        assert DEFAULT_SIGNALS_OUTPUT_DIR == "shared/output"


class TestStoreSignal:
    """Tests for store_signal function."""

    @pytest.fixture
    def sample_signal(self) -> NewsSignal:
        """Create a sample signal for testing."""
        extraction = LightClassification(
            event_type=EventType.macro,
            summary="Fed cuts rates",
            confidence=0.9,
            primary_entity="Federal Reserve",
        )
        return NewsSignal(
            timestamp=datetime.now(timezone.utc),
            source_platform=SourcePlatform.twitter,
            source_account="@test",
            source_type=SourceType.news,
            raw_text="Fed cuts rates",
            external_id="test_123",
            extraction=extraction,
        )

    @pytest.mark.anyio
    async def test_store_signal_creates_dir(
        self, sample_signal: NewsSignal, tmp_path: Path
    ) -> None:
        """Test that store_signal creates output directory."""
        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.signals_output_dir = tmp_path / "signals"

        with patch("synesis.agent.pydantic_runner.get_settings", return_value=mock_settings):
            await store_signal(sample_signal, mock_redis)

        assert (tmp_path / "signals").exists()

    @pytest.mark.anyio
    async def test_store_signal_writes_jsonl(
        self, sample_signal: NewsSignal, tmp_path: Path
    ) -> None:
        """Test that store_signal writes to JSONL file."""
        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock()

        signals_dir = tmp_path / "signals"
        mock_settings = MagicMock()
        mock_settings.signals_output_dir = signals_dir

        with patch("synesis.agent.pydantic_runner.get_settings", return_value=mock_settings):
            await store_signal(sample_signal, mock_redis)

        # Check file was created
        date_str = sample_signal.timestamp.strftime("%Y-%m-%d")
        expected_file = signals_dir / f"signals_{date_str}.jsonl"
        assert expected_file.exists()

        # Check content
        content = expected_file.read_text()
        assert "test_123" in content
        assert "Fed cuts rates" in content

    @pytest.mark.anyio
    async def test_store_signal_publishes_to_redis(
        self, sample_signal: NewsSignal, tmp_path: Path
    ) -> None:
        """Test that store_signal publishes to Redis."""
        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.signals_output_dir = tmp_path / "signals"

        with patch("synesis.agent.pydantic_runner.get_settings", return_value=mock_settings):
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
            source_platform=SourcePlatform.twitter,
            source_account="@test",
            text="Test",
            timestamp=datetime.now(timezone.utc),
            source_type=SourceType.news,
        )
        extraction = LightClassification(
            news_category=NewsCategory.breaking,
            event_type=EventType.macro,
            summary="Test",
            confidence=0.9,
            primary_entity="Test",
        )
        analysis = SmartAnalysis(
            tickers=["SPY"],
            sectors=[],
            predicted_impact=ImpactLevel.high,
            market_direction=Direction.bullish,
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
            source_platform=SourcePlatform.twitter,
            source_account="@test",
            text="Test",
            timestamp=datetime.now(timezone.utc),
            source_type=SourceType.news,
        )
        extraction = LightClassification(
            event_type=EventType.other,
            summary="Test",
            confidence=0.5,
            primary_entity="Test",
        )
        analysis = SmartAnalysis(
            tickers=[],
            sectors=[],
            predicted_impact=ImpactLevel.low,
            market_direction=Direction.neutral,
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
            source_platform=SourcePlatform.twitter,
            source_account="@test",
            text="Test",
            timestamp=datetime.now(timezone.utc),
            source_type=SourceType.news,
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
            source_platform=SourcePlatform.twitter,
            source_account="@test",
            text="Test message",
            timestamp=datetime.now(timezone.utc),
            source_type=SourceType.news,
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
            source_platform=SourcePlatform.twitter,
            source_account="@test",
            text="Test",
            timestamp=datetime.now(timezone.utc),
            source_type=SourceType.news,
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
            source_platform=SourcePlatform.twitter,
            source_account="@test",
            text="Test",
            timestamp=datetime.now(timezone.utc),
            source_type=SourceType.news,
        )
        extraction = LightClassification(
            event_type=EventType.other,
            summary="Test",
            confidence=0.5,
            primary_entity="Test",
        )
        analysis = SmartAnalysis(
            tickers=[],
            sectors=[],
            predicted_impact=ImpactLevel.low,
            market_direction=Direction.neutral,
            primary_thesis="Low confidence",
            thesis_confidence=0.3,  # Even low confidence is sent now
        )

        with patch(
            "synesis.agent.pydantic_runner.send_long_telegram", new_callable=AsyncMock
        ) as mock_send:
            await emit_combined_telegram(message, extraction, analysis)

        # All signals are sent to Telegram (no threshold)
        mock_send.assert_called_once()

    @pytest.mark.anyio
    async def test_sends_high_confidence(self) -> None:
        """Test that high confidence signals are sent."""
        message = UnifiedMessage(
            external_id="123",
            source_platform=SourcePlatform.twitter,
            source_account="@test",
            text="Test",
            timestamp=datetime.now(timezone.utc),
            source_type=SourceType.news,
        )
        extraction = LightClassification(
            event_type=EventType.macro,
            summary="Fed cuts rates",
            confidence=0.9,
            primary_entity="Fed",
        )
        analysis = SmartAnalysis(
            tickers=["SPY"],
            sectors=[],
            predicted_impact=ImpactLevel.high,
            market_direction=Direction.bullish,
            primary_thesis="Bullish thesis",
            thesis_confidence=0.8,  # Above 0.5 threshold
        )

        with patch(
            "synesis.agent.pydantic_runner.send_long_telegram", new_callable=AsyncMock
        ) as mock_send:
            await emit_combined_telegram(message, extraction, analysis)

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


class TestEmitSignal:
    """Tests for emit_signal function."""

    @pytest.mark.anyio
    async def test_emit_signal_no_analysis(self, tmp_path: Path) -> None:
        """Test emit_signal with no analysis."""
        from synesis.core.processor import ProcessingResult

        message = UnifiedMessage(
            external_id="123",
            source_platform=SourcePlatform.twitter,
            source_account="@test",
            text="Test",
            timestamp=datetime.now(timezone.utc),
            source_type=SourceType.news,
        )
        extraction = LightClassification(
            event_type=EventType.other,
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
        mock_redis.publish = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.signals_output_dir = tmp_path / "signals"

        with patch("synesis.agent.pydantic_runner.get_settings", return_value=mock_settings):
            with patch("synesis.agent.pydantic_runner.emit_signal_to_db", new_callable=AsyncMock):
                await emit_signal(result, mock_redis)

        # Should still write to JSONL
        assert (tmp_path / "signals").exists()
