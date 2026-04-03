"""Tests for Stage 1 instant classifier (no LLM).

Tests that NewsClassifier.classify() produces correct LightClassification
from impact scoring + ticker matching.
"""

from datetime import datetime, timezone

import pytest

from synesis.processing.news.classifier import NewsClassifier
from synesis.processing.news import (
    LightClassification,
    SourcePlatform,
    UnifiedMessage,
    UrgencyLevel,
)


def _msg(text: str, source: str = "FirstSquawk") -> UnifiedMessage:
    return UnifiedMessage(
        external_id=f"test:{source}:1",
        source_platform=SourcePlatform.telegram,
        source_account=source,
        text=text,
        timestamp=datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


class TestNewsClassifier:
    """Tests for the instant (no LLM) classifier."""

    @pytest.fixture
    def classifier(self) -> NewsClassifier:
        return NewsClassifier()

    @pytest.mark.asyncio
    async def test_returns_light_classification(self, classifier: NewsClassifier) -> None:
        result = await classifier.classify(
            _msg("*NVIDIA INVESTS $2B IN MARVELL TECHNOLOGY", "DeItaone")
        )
        assert isinstance(result, LightClassification)

    @pytest.mark.asyncio
    async def test_impact_score_populated(self, classifier: NewsClassifier) -> None:
        result = await classifier.classify(
            _msg("*NVIDIA INVESTS $2B IN MARVELL TECHNOLOGY", "DeItaone")
        )
        assert result.impact_score > 0
        assert len(result.impact_reasons) > 0

    @pytest.mark.asyncio
    async def test_urgency_critical(self, classifier: NewsClassifier) -> None:
        result = await classifier.classify(
            _msg("*NVIDIA INVESTS $2B IN MARVELL TECHNOLOGY", "DeItaone")
        )
        assert result.urgency in (UrgencyLevel.high, UrgencyLevel.critical)

    @pytest.mark.asyncio
    async def test_urgency_low(self, classifier: NewsClassifier) -> None:
        result = await classifier.classify(_msg("Markets open higher today", "unknown"))
        assert result.urgency in (UrgencyLevel.normal, UrgencyLevel.low)

    @pytest.mark.asyncio
    async def test_tickers_matched(self, classifier: NewsClassifier) -> None:
        result = await classifier.classify(
            _msg("*NVIDIA INVESTS $2B IN MARVELL TECHNOLOGY", "DeItaone")
        )
        assert "NVDA" in result.matched_tickers
        assert "MRVL" in result.matched_tickers

    @pytest.mark.asyncio
    async def test_no_tickers_for_macro(self, classifier: NewsClassifier) -> None:
        result = await classifier.classify(
            _msg("US CPI (YOY) ACTUAL: 2.5% VS 2.6% EST", "FirstSquawk")
        )
        assert result.matched_tickers == []

    @pytest.mark.asyncio
    async def test_private_tickers(self, classifier: NewsClassifier) -> None:
        result = await classifier.classify(_msg("JUST IN: OPENAI RAISES $122B", "WatcherGuru"))
        assert "~OPENAI" in result.matched_tickers

    @pytest.mark.asyncio
    async def test_no_llm_fields(self, classifier: NewsClassifier) -> None:
        """LightClassification should NOT have old LLM-dependent fields."""
        result = await classifier.classify(_msg("*NVIDIA INVESTS $2B IN MARVELL", "DeItaone"))
        assert not hasattr(result, "summary")
        assert not hasattr(result, "primary_entity")
        assert not hasattr(result, "all_entities")
        assert not hasattr(result, "primary_topics")
        assert not hasattr(result, "search_keywords")


class TestLightClassificationModel:
    """Tests for the LightClassification model fields."""

    def test_has_stage1_fields(self) -> None:
        lc = LightClassification(
            matched_tickers=["NVDA", "MRVL"],
            impact_score=57,
            impact_reasons=["wire_prefix:+15"],
            urgency=UrgencyLevel.critical,
        )
        assert lc.matched_tickers == ["NVDA", "MRVL"]
        assert lc.impact_score == 57
        assert lc.urgency == UrgencyLevel.critical

    def test_does_not_have_llm_fields(self) -> None:
        lc = LightClassification()
        assert not hasattr(lc, "summary")
        assert not hasattr(lc, "primary_entity")
        assert not hasattr(lc, "all_entities")
        assert not hasattr(lc, "primary_topics")
        assert not hasattr(lc, "secondary_topics")
        assert not hasattr(lc, "confidence")
        assert not hasattr(lc, "search_keywords")
        assert not hasattr(lc, "polymarket_keywords")
        assert not hasattr(lc, "numeric_data")

    def test_defaults(self) -> None:
        lc = LightClassification()
        assert lc.matched_tickers == []
        assert lc.impact_score == 0
        assert lc.urgency == UrgencyLevel.normal
