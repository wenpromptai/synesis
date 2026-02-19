"""Tests for Stage 1 lightweight LLM classifier (entity extraction)."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synesis.processing.news.classifier import (
    CLASSIFIER_SYSTEM_PROMPT,
    NewsClassifier,
    classify_message,
    create_classifier_agent,
)
from synesis.processing.news import (
    EventType,
    LightClassification,
    NewsCategory,
    SourcePlatform,
    UnifiedMessage,
)


def create_test_message(text: str) -> UnifiedMessage:
    """Create a test message."""
    return UnifiedMessage(
        external_id="test_123",
        source_platform=SourcePlatform.telegram,
        source_account="@DeItaone",
        text=text,
        timestamp=datetime.now(timezone.utc),
    )


def create_mock_classification() -> LightClassification:
    """Create a mock classification result.

    Note: LightClassification is Stage 1 entity extraction only.
    It does NOT contain: sentiment, tickers, sectors.
    Those fields moved to Stage 2 SmartAnalysis.
    """
    return LightClassification(
        news_category=NewsCategory.breaking,
        event_type=EventType.macro,
        summary="Fed announces rate cut of 25 basis points",
        confidence=0.95,
        primary_entity="Federal Reserve",
        all_entities=["Federal Reserve", "Jerome Powell"],
        polymarket_keywords=["Fed", "rate cut", "interest rate"],
        search_keywords=["Fed rate cut analyst forecast", "Fed rate cut market reaction"],
    )


class TestClassifierSystemPrompt:
    """Tests for the classifier system prompt."""

    def test_prompt_contains_key_instructions(self) -> None:
        """Test that the system prompt contains key instructions."""
        assert "entity extractor" in CLASSIFIER_SYSTEM_PROMPT.lower()
        assert "event_type" in CLASSIFIER_SYSTEM_PROMPT.lower()
        assert "polymarket" in CLASSIFIER_SYSTEM_PROMPT.lower()

    def test_prompt_lists_event_types(self) -> None:
        """Test that the prompt lists all event types."""
        assert "macro" in CLASSIFIER_SYSTEM_PROMPT
        assert "earnings" in CLASSIFIER_SYSTEM_PROMPT
        assert "geopolitical" in CLASSIFIER_SYSTEM_PROMPT
        assert "corporate" in CLASSIFIER_SYSTEM_PROMPT
        assert "regulatory" in CLASSIFIER_SYSTEM_PROMPT
        assert "crypto" in CLASSIFIER_SYSTEM_PROMPT

    def test_prompt_clarifies_no_judgment_calls(self) -> None:
        """Test that the prompt clarifies Stage 1 makes no judgment calls."""
        lower = CLASSIFIER_SYSTEM_PROMPT.lower()
        assert "no judgment" in lower or "not to extract" in lower or "stage 2" in lower


class TestNewsClassifier:
    """Tests for NewsClassifier."""

    @pytest.fixture
    def mock_agent_result(self) -> MagicMock:
        """Create a mock agent result."""
        result = MagicMock()
        result.output = create_mock_classification()
        return result

    @pytest.mark.asyncio
    async def test_classify_returns_classification(self, mock_agent_result: MagicMock) -> None:
        """Test that classify returns a LightClassification."""
        classifier = NewsClassifier()

        # Mock the agent
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)
        classifier._agent = mock_agent

        message = create_test_message("Breaking: Fed cuts rates by 25bps")
        result = await classifier.classify(message)

        assert isinstance(result, LightClassification)
        assert result.event_type == EventType.macro
        assert result.confidence == 0.95
        assert result.primary_entity == "Federal Reserve"

    @pytest.mark.asyncio
    async def test_classify_builds_correct_prompt(self, mock_agent_result: MagicMock) -> None:
        """Test that classify builds the correct prompt."""
        classifier = NewsClassifier()

        # Mock the agent to capture the prompt
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)
        classifier._agent = mock_agent

        message = create_test_message("Test message content")
        await classifier.classify(message)

        # Check that run was called with a prompt containing the message
        call_args = mock_agent.run.call_args
        prompt = call_args[0][0]
        assert "Test message content" in prompt
        assert "@DeItaone" in prompt
        assert "telegram" in prompt

    def test_build_prompt_includes_source_info(self) -> None:
        """Test that _build_prompt includes source information."""
        classifier = NewsClassifier()
        message = create_test_message("Fed cuts rates")

        prompt = classifier._build_prompt(message)

        assert "@DeItaone" in prompt
        assert "telegram" in prompt
        assert "Fed cuts rates" in prompt


class TestClassifyMessage:
    """Tests for the classify_message convenience function."""

    @pytest.mark.asyncio
    async def test_classify_message_uses_singleton(self) -> None:
        """Test that classify_message uses the singleton classifier."""
        mock_classification = create_mock_classification()

        with patch("synesis.processing.news.classifier.get_classifier") as mock_get:
            mock_classifier = AsyncMock()
            mock_classifier.classify = AsyncMock(return_value=mock_classification)
            mock_get.return_value = mock_classifier

            message = create_test_message("Test")
            result = await classify_message(message)

            mock_get.assert_called_once()
            mock_classifier.classify.assert_called_once_with(message)
            assert result == mock_classification


class TestCreateClassifierAgent:
    """Tests for create_classifier_agent."""

    def test_creates_agent_with_correct_output_type(self) -> None:
        """Test that the agent has the correct output type."""
        with patch("synesis.processing.news.classifier.create_model") as mock_create_model:
            mock_create_model.return_value = "test"  # Use 'test' model to avoid API key requirement

            agent = create_classifier_agent()

            from pydantic_ai.output import PromptedOutput

            assert isinstance(agent.output_type, PromptedOutput)


class TestLightClassificationModel:
    """Tests for the LightClassification model itself."""

    def test_light_classification_fields(self) -> None:
        """Test that LightClassification has expected Stage 1 fields."""
        classification = create_mock_classification()

        # Stage 1 fields that SHOULD exist
        assert hasattr(classification, "event_type")
        assert hasattr(classification, "summary")
        assert hasattr(classification, "confidence")
        assert hasattr(classification, "primary_entity")
        assert hasattr(classification, "all_entities")
        assert hasattr(classification, "polymarket_keywords")
        assert hasattr(classification, "search_keywords")
        assert hasattr(classification, "news_category")

    def test_light_classification_does_not_have_stage2_fields(self) -> None:
        """Test that LightClassification does NOT have Stage 2 fields.

        These fields moved to SmartAnalysis in the 2-stage architecture.
        """
        classification = create_mock_classification()

        # Stage 2 fields that should NOT exist on LightClassification
        assert not hasattr(classification, "sentiment")
        assert not hasattr(classification, "sentiment_score")
        assert not hasattr(classification, "tickers")
        assert not hasattr(classification, "sectors")

    def test_confidence_bounds(self) -> None:
        """Test that confidence is bounded 0-1."""
        classification = LightClassification(
            event_type=EventType.macro,
            summary="Test",
            confidence=0.0,
            primary_entity="Test",
        )
        assert classification.confidence == 0.0

        classification2 = LightClassification(
            event_type=EventType.macro,
            summary="Test",
            confidence=1.0,
            primary_entity="Test",
        )
        assert classification2.confidence == 1.0

    def test_defaults(self) -> None:
        """Test default values."""
        classification = LightClassification(
            event_type=EventType.other,
            summary="Test",
            confidence=0.5,
            primary_entity="Unknown",
        )

        assert classification.news_category == NewsCategory.other
        assert classification.all_entities == []
        assert classification.polymarket_keywords == []
        assert classification.search_keywords == []
