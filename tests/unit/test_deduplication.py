"""Tests for message deduplication."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from synesis.processing.deduplication import (
    DeduplicationResult,
    MessageDeduplicator,
    SIMILARITY_THRESHOLD,
)
from synesis.processing.models import SourcePlatform, SourceType, UnifiedMessage


def create_test_message(
    text: str,
    external_id: str = "test_123",
    platform: SourcePlatform = SourcePlatform.twitter,
) -> UnifiedMessage:
    """Create a test message."""
    return UnifiedMessage(
        external_id=external_id,
        source_platform=platform,
        source_account="@test",
        text=text,
        timestamp=datetime.now(timezone.utc),
        source_type=SourceType.news,
    )


class TestDeduplicationResult:
    """Tests for DeduplicationResult."""

    def test_not_duplicate(self) -> None:
        result = DeduplicationResult(
            is_duplicate=False,
            duplicate_of=None,
            similarity=None,
            processing_time_ms=5.0,
        )
        assert not result.is_duplicate
        assert result.duplicate_of is None

    def test_is_duplicate(self) -> None:
        result = DeduplicationResult(
            is_duplicate=True,
            duplicate_of="original_123",
            similarity=0.92,
            processing_time_ms=3.5,
        )
        assert result.is_duplicate
        assert result.duplicate_of == "original_123"
        assert result.similarity == 0.92


class TestMessageDeduplicator:
    """Tests for MessageDeduplicator."""

    @pytest.fixture
    def mock_redis(self) -> AsyncMock:
        """Create a mock Redis client."""
        redis = AsyncMock()
        redis.scan_iter = MagicMock(return_value=AsyncIteratorMock([]))
        redis.get = AsyncMock(return_value=None)
        redis.setex = AsyncMock()
        return redis

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock Model2Vec model."""
        model = MagicMock()
        # Return consistent embeddings for testing
        model.encode = MagicMock(
            side_effect=lambda texts: np.random.rand(len(texts), 256).astype(np.float32)
        )
        return model

    @pytest.mark.asyncio
    async def test_initialize_loads_model(self, mock_redis: AsyncMock) -> None:
        """Test that initialize loads the Model2Vec model."""
        deduplicator = MessageDeduplicator(redis=mock_redis)

        with patch("synesis.processing.deduplication.StaticModel") as mock_static_model:
            mock_instance = MagicMock()
            mock_instance.encode = MagicMock(return_value=np.zeros((1, 256)))
            mock_static_model.from_pretrained = MagicMock(return_value=mock_instance)

            await deduplicator.initialize()

            mock_static_model.from_pretrained.assert_called_once()
            assert deduplicator._model is not None

    @pytest.mark.asyncio
    async def test_check_duplicate_no_existing(self, mock_redis: AsyncMock) -> None:
        """Test duplicate check when no existing messages."""
        deduplicator = MessageDeduplicator(redis=mock_redis)

        # Mock the model
        deduplicator._model = MagicMock()
        deduplicator._model.encode = MagicMock(
            return_value=np.random.rand(1, 256).astype(np.float32)
        )

        message = create_test_message("This is a test message")
        result = await deduplicator.check_duplicate(message)

        assert not result.is_duplicate
        assert result.duplicate_of is None

    @pytest.mark.asyncio
    async def test_store_message(self, mock_redis: AsyncMock) -> None:
        """Test storing a message embedding."""
        deduplicator = MessageDeduplicator(redis=mock_redis)

        # Mock the model
        deduplicator._model = MagicMock()
        deduplicator._model.encode = MagicMock(
            return_value=np.random.rand(1, 256).astype(np.float32)
        )

        message = create_test_message("Test message", external_id="msg_123")
        await deduplicator.store_message(message)

        # Check that set was called (with nx=True for race condition safety)
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert "dedup:emb:twitter:msg_123" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_process_message_stores_unique(self, mock_redis: AsyncMock) -> None:
        """Test that unique messages are stored."""
        deduplicator = MessageDeduplicator(redis=mock_redis)

        # Mock the model
        deduplicator._model = MagicMock()
        deduplicator._model.encode = MagicMock(
            return_value=np.random.rand(1, 256).astype(np.float32)
        )

        message = create_test_message("Unique message")
        result = await deduplicator.process_message(message)

        assert not result.is_duplicate
        mock_redis.set.assert_called_once()

    def test_cosine_similarity(self, mock_redis: AsyncMock) -> None:
        """Test cosine similarity calculation."""
        deduplicator = MessageDeduplicator(redis=mock_redis)

        # Test identical vectors
        v1 = np.array([1.0, 0.0, 0.0])
        similarity = deduplicator._cosine_similarity(v1, v1)
        assert abs(similarity - 1.0) < 0.001

        # Test orthogonal vectors
        v2 = np.array([0.0, 1.0, 0.0])
        similarity = deduplicator._cosine_similarity(v1, v2)
        assert abs(similarity) < 0.001

        # Test similar vectors
        v3 = np.array([0.9, 0.1, 0.0])
        v3 = v3 / np.linalg.norm(v3)
        v4 = np.array([0.85, 0.15, 0.0])
        v4 = v4 / np.linalg.norm(v4)
        similarity = deduplicator._cosine_similarity(v3, v4)
        assert similarity > 0.9  # Should be very similar

    def test_similarity_threshold(self) -> None:
        """Test that the default similarity threshold is sensible."""
        assert SIMILARITY_THRESHOLD == 0.75
        assert 0.0 < SIMILARITY_THRESHOLD < 1.0


class AsyncIteratorMock:
    """Mock for async iterators."""

    def __init__(self, items: list) -> None:
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item
