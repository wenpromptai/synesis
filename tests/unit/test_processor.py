"""Tests for the two-stage NewsProcessor."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synesis.core.processor import NewsProcessor, ProcessingResult
from synesis.processing.news import (
    Direction,
    LightClassification,
    PrimaryTopic,
    MarketEvaluation,
    NewsCategory,
    ResearchQuality,
    SmartAnalysis,
    SourcePlatform,
    TickerAnalysis,
    UnifiedMessage,
    UrgencyLevel,
)


def create_test_message(
    text: str = "Fed cuts rates by 25bps",
) -> UnifiedMessage:
    """Create a test message."""
    return UnifiedMessage(
        external_id="test_123",
        source_platform=SourcePlatform.telegram,
        source_account="@DeItaone",
        text=text,
        timestamp=datetime.now(timezone.utc),
    )


def create_test_extraction(
    urgency: UrgencyLevel = UrgencyLevel.critical,
) -> LightClassification:
    """Create a test Stage 1 extraction.

    By default, creates an extraction that will pass to Stage 2
    (urgency=critical).
    """
    return LightClassification(
        news_category=NewsCategory.breaking,
        primary_topics=[PrimaryTopic.monetary_policy],
        summary="Fed announces rate cut",
        confidence=0.95,
        primary_entity="Federal Reserve",
        all_entities=["Federal Reserve", "Jerome Powell"],
        polymarket_keywords=["Fed", "rate cut"],
        search_keywords=["Fed rate cut forecast"],
        urgency=urgency,
    )


def create_test_analysis(has_edge: bool = False) -> SmartAnalysis:
    """Create a test Stage 2 smart analysis.

    In the 2-stage architecture, SmartAnalysis contains ALL informed judgments:
    - tickers, sectors, sentiment (moved from old LightClassification)
    - ticker analyses, sector implications (from old InvestmentAnalysis)
    - market evaluations (from old EvaluatorOutput)
    """
    market_evaluations = []
    if has_edge:
        market_evaluations.append(
            MarketEvaluation(
                market_id="market_123",
                market_question="Will Fed cut rates in March?",
                is_relevant=True,
                relevance_reasoning="Directly related to news",
                current_price=0.35,
                estimated_fair_price=0.55,
                edge=0.20,
                verdict="undervalued",
                confidence=0.7,
                reasoning="Rate cut is likely",
                recommended_side="yes",
            )
        )

    return SmartAnalysis(
        # Informed judgments (made with research context)
        tickers=["SPY", "QQQ"],
        sectors=["financials"],
        sentiment=Direction.bullish,
        sentiment_score=0.7,
        # Thesis
        primary_thesis="Dovish Fed pivot supports risk assets",
        thesis_confidence=0.75,
        # Ticker analysis
        ticker_analyses=[
            TickerAnalysis(
                ticker="SPY",
                bull_thesis="Rate cuts bullish for equities",
                bear_thesis="Economic weakness",
                net_direction=Direction.bullish,
                conviction=0.8,
                time_horizon="days",
            ),
        ],
        # Market evaluations
        market_evaluations=market_evaluations,
        research_quality=ResearchQuality.high,
    )


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_create_processing_result(self) -> None:
        """Test creating a processing result."""
        message = create_test_message()
        extraction = create_test_extraction()
        analysis = create_test_analysis()

        result = ProcessingResult(
            message=message,
            extraction=extraction,
            analysis=analysis,
        )

        assert result.message == message
        assert result.extraction == extraction
        assert result.analysis == analysis
        assert result.skipped is False

    def test_has_edge_true(self) -> None:
        """Test has_edge returns True when edge exists."""
        result = ProcessingResult(
            message=create_test_message(),
            extraction=create_test_extraction(),
            analysis=create_test_analysis(has_edge=True),
        )

        assert result.has_edge is True

    def test_has_edge_false(self) -> None:
        """Test has_edge returns False when no edge."""
        result = ProcessingResult(
            message=create_test_message(),
            extraction=create_test_extraction(),
            analysis=create_test_analysis(has_edge=False),
        )

        assert result.has_edge is False

    def test_has_edge_false_no_analysis(self) -> None:
        """Test has_edge returns False when no analysis."""
        result = ProcessingResult(
            message=create_test_message(),
            extraction=create_test_extraction(),
            analysis=None,
        )

        assert result.has_edge is False

    def test_best_opportunity(self) -> None:
        """Test best_opportunity returns the best market."""
        result = ProcessingResult(
            message=create_test_message(),
            extraction=create_test_extraction(),
            analysis=create_test_analysis(has_edge=True),
        )

        assert result.best_opportunity is not None
        assert result.best_opportunity.market_id == "market_123"

    def test_to_signal(self) -> None:
        """Test to_signal creates a NewsSignal."""
        message = create_test_message()
        extraction = create_test_extraction()
        analysis = create_test_analysis()

        result = ProcessingResult(
            message=message,
            extraction=extraction,
            analysis=analysis,
        )

        signal = result.to_signal()

        assert signal is not None
        assert signal.external_id == message.external_id
        assert signal.extraction == extraction
        assert signal.analysis == analysis
        # Tickers/sectors come from analysis (Stage 2)
        assert signal.tickers == ["SPY", "QQQ"]
        assert signal.sectors == ["financials"]

    def test_to_signal_none_when_skipped(self) -> None:
        """Test to_signal returns None when skipped."""
        result = ProcessingResult(
            message=create_test_message(),
            skipped=True,
            skip_reason="duplicate",
        )

        assert result.to_signal() is None

    def test_to_signal_none_when_no_extraction(self) -> None:
        """Test to_signal returns None when no extraction."""
        result = ProcessingResult(
            message=create_test_message(),
            extraction=None,
        )

        assert result.to_signal() is None


class TestNewsProcessor:
    """Tests for NewsProcessor class."""

    @pytest.fixture
    def mock_redis(self) -> AsyncMock:
        """Create a mock Redis client."""
        return AsyncMock()

    @pytest.fixture
    def processor(self, mock_redis: AsyncMock) -> NewsProcessor:
        """Create a NewsProcessor instance."""
        return NewsProcessor(mock_redis)

    def test_init(self, processor: NewsProcessor) -> None:
        """Test processor initialization."""
        assert processor._initialized is False
        assert processor._classifier is None
        assert processor._analyzer is None

    @pytest.mark.asyncio
    async def test_initialize(self, processor: NewsProcessor) -> None:
        """Test processor initialization (two-stage architecture)."""
        with (
            patch("synesis.core.processor.create_deduplicator") as mock_dedup,
            patch("synesis.core.processor.PolymarketClient") as mock_poly,
        ):
            mock_dedup.return_value = AsyncMock()
            mock_poly.return_value = AsyncMock()

            await processor.initialize()

            assert processor._initialized is True
            assert processor._classifier is not None
            assert processor._analyzer is not None

    @pytest.mark.asyncio
    async def test_process_message_duplicate(
        self, processor: NewsProcessor, mock_redis: AsyncMock
    ) -> None:
        """Test processing a duplicate message."""
        with patch("synesis.core.processor.create_deduplicator") as mock_dedup:
            # Setup mock deduplicator to return duplicate
            mock_dedup_instance = AsyncMock()
            mock_dedup_result = MagicMock()
            mock_dedup_result.is_duplicate = True
            mock_dedup_result.duplicate_of = "original_123"
            mock_dedup_instance.process_message = AsyncMock(return_value=mock_dedup_result)
            mock_dedup.return_value = mock_dedup_instance

            processor._deduplicator = mock_dedup_instance
            processor._initialized = True

            message = create_test_message()
            result = await processor.process_message(message)

            assert result.skipped is True
            assert result.is_duplicate is True
            assert result.duplicate_of == "original_123"

    @pytest.mark.asyncio
    async def test_process_message_full_pipeline(self, processor: NewsProcessor) -> None:
        """Test full two-stage processing pipeline."""
        # Setup mocks
        mock_dedup = AsyncMock()
        mock_dedup_result = MagicMock()
        mock_dedup_result.is_duplicate = False
        mock_dedup.process_message = AsyncMock(return_value=mock_dedup_result)

        mock_extraction = create_test_extraction()
        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(return_value=mock_extraction)

        mock_analysis = create_test_analysis(has_edge=True)
        mock_analyzer = MagicMock()
        mock_analyzer.search_polymarket = AsyncMock(return_value="Markets found")
        mock_analyzer.analyze = AsyncMock(return_value=mock_analysis)

        mock_polymarket = MagicMock()
        mock_polymarket._get_client = MagicMock(return_value=MagicMock())

        processor._deduplicator = mock_dedup
        processor._classifier = mock_classifier
        processor._analyzer = mock_analyzer
        processor._polymarket = mock_polymarket
        processor._initialized = True

        # Mock web search
        with patch(
            "synesis.core.processor.search_market_impact",
            new_callable=AsyncMock,
            return_value=[{"title": "Test", "snippet": "Content", "url": "http://test.com"}],
        ):
            message = create_test_message()
            result = await processor.process_message(message)

        assert result.skipped is False
        assert result.extraction == mock_extraction
        assert result.analysis == mock_analysis
        assert result.has_edge is True


class TestProcessingResultWithNoneAnalysis:
    """Tests for ProcessingResult when Stage 2 analysis fails (returns None)."""

    def test_to_signal_with_none_analysis(self) -> None:
        """Test to_signal works when analysis is None (Stage 2 failed)."""
        message = create_test_message()
        extraction = create_test_extraction()

        result = ProcessingResult(
            message=message,
            extraction=extraction,
            analysis=None,  # Stage 2 failed
        )

        signal = result.to_signal()

        # Signal should still be created with extraction data
        assert signal is not None
        assert signal.external_id == message.external_id
        assert signal.extraction == extraction
        assert signal.analysis is None
        # Tickers/sectors should be empty when analysis is None
        assert signal.tickers == []
        assert signal.sectors == []

    def test_has_edge_false_with_none_analysis(self) -> None:
        """Test has_edge is False when analysis is None."""
        result = ProcessingResult(
            message=create_test_message(),
            extraction=create_test_extraction(),
            analysis=None,
        )

        assert result.has_edge is False

    def test_best_opportunity_none_with_none_analysis(self) -> None:
        """Test best_opportunity is None when analysis is None."""
        result = ProcessingResult(
            message=create_test_message(),
            extraction=create_test_extraction(),
            analysis=None,
        )

        assert result.best_opportunity is None


class TestFetchWebResults:
    """Tests for _fetch_web_results method."""

    @pytest.fixture
    def processor(self) -> NewsProcessor:
        """Create a processor instance."""
        return NewsProcessor(AsyncMock())

    @pytest.mark.asyncio
    async def test_fetch_web_results_success(self, processor: NewsProcessor) -> None:
        """Test successful web result fetching."""
        with patch(
            "synesis.core.processor.search_market_impact",
            new_callable=AsyncMock,
            return_value=[{"title": "Test", "snippet": "Content", "url": "http://test.com"}],
        ):
            results = await processor._fetch_web_results(["query1", "query2", "query3"])

        # Should only fetch 2 queries (limit)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_fetch_web_results_handles_errors(self, processor: NewsProcessor) -> None:
        """Test that fetch handles errors gracefully."""
        with patch(
            "synesis.core.processor.search_market_impact",
            new_callable=AsyncMock,
            side_effect=Exception("Search failed"),
        ):
            results = await processor._fetch_web_results(["query1"])

        assert len(results) == 1
        assert "failed" in results[0].lower()

    @pytest.mark.asyncio
    async def test_fetch_web_results_handles_search_providers_exhausted(
        self, processor: NewsProcessor
    ) -> None:
        """Test that SearchProvidersExhaustedError is handled with error-level logging."""
        from synesis.processing.common import SearchProvidersExhaustedError

        with patch(
            "synesis.core.processor.search_market_impact",
            new_callable=AsyncMock,
            side_effect=SearchProvidersExhaustedError("All providers failed"),
        ):
            results = await processor._fetch_web_results(["query1"])

        assert len(results) == 1
        assert "unavailable" in results[0].lower() or "failed" in results[0].lower()


class TestUrgencyGate:
    """Tests for urgency-based gate (only critical/high pass to Stage 2)."""

    @pytest.fixture
    def processor(self) -> NewsProcessor:
        """Create a NewsProcessor with mocked internals."""
        proc = NewsProcessor(AsyncMock())
        mock_dedup = AsyncMock()
        mock_dedup_result = MagicMock()
        mock_dedup_result.is_duplicate = False
        mock_dedup.process_message = AsyncMock(return_value=mock_dedup_result)

        mock_classifier = MagicMock()
        mock_analyzer = MagicMock()
        mock_analyzer.search_polymarket = AsyncMock(return_value="Markets found")
        mock_analyzer.analyze = AsyncMock(return_value=create_test_analysis())

        mock_polymarket = MagicMock()
        mock_polymarket._get_client = MagicMock(return_value=MagicMock())

        proc._deduplicator = mock_dedup
        proc._classifier = mock_classifier
        proc._analyzer = mock_analyzer
        proc._polymarket = mock_polymarket
        proc._initialized = True
        return proc

    @pytest.mark.asyncio
    async def test_low_urgency_filtered(self, processor: NewsProcessor) -> None:
        """Low urgency should be filtered (no Stage 2)."""
        processor._classifier.classify = AsyncMock(
            return_value=create_test_extraction(urgency=UrgencyLevel.low)
        )

        message = create_test_message(text="Check out this promo link")
        result = await processor.process_message(message)

        assert result.analysis is None

    @pytest.mark.asyncio
    async def test_normal_urgency_filtered(self, processor: NewsProcessor) -> None:
        """Normal urgency should be filtered (no Stage 2)."""
        processor._classifier.classify = AsyncMock(
            return_value=create_test_extraction(urgency=UrgencyLevel.normal)
        )

        message = create_test_message(text="Minor update on housing data")
        result = await processor.process_message(message)

        assert result.analysis is None

    @pytest.mark.asyncio
    async def test_high_urgency_passes(self, processor: NewsProcessor) -> None:
        """High urgency should pass to Stage 2."""
        processor._classifier.classify = AsyncMock(
            return_value=create_test_extraction(urgency=UrgencyLevel.high)
        )

        with patch(
            "synesis.core.processor.search_market_impact",
            new_callable=AsyncMock,
            return_value=[],
        ):
            message = create_test_message(text="CPI comes in hot at 3.5%")
            result = await processor.process_message(message)

        assert result.analysis is not None

    @pytest.mark.asyncio
    async def test_critical_urgency_calls_polymarket(self, processor: NewsProcessor) -> None:
        """Critical urgency should call search_polymarket."""
        processor._classifier.classify = AsyncMock(
            return_value=create_test_extraction(urgency=UrgencyLevel.critical)
        )

        with patch(
            "synesis.core.processor.search_market_impact",
            new_callable=AsyncMock,
            return_value=[],
        ):
            message = create_test_message(text="*BREAKING* Fed cuts rates by 50bps")
            await processor.process_message(message)

        processor._analyzer.search_polymarket.assert_called_once()


class TestNewsProcessorStage2Failure:
    """Tests for NewsProcessor when Stage 2 analysis fails."""

    @pytest.fixture
    def mock_redis(self) -> AsyncMock:
        """Create a mock Redis client."""
        return AsyncMock()

    @pytest.fixture
    def processor(self, mock_redis: AsyncMock) -> NewsProcessor:
        """Create a NewsProcessor instance."""
        return NewsProcessor(mock_redis)

    @pytest.mark.asyncio
    async def test_process_message_stage2_returns_none(self, processor: NewsProcessor) -> None:
        """Test processing when Stage 2 analyzer returns None (failure)."""
        # Setup mocks
        mock_dedup = AsyncMock()
        mock_dedup_result = MagicMock()
        mock_dedup_result.is_duplicate = False
        mock_dedup.process_message = AsyncMock(return_value=mock_dedup_result)

        mock_extraction = create_test_extraction()
        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(return_value=mock_extraction)

        # Stage 2 returns None (failure)
        mock_analyzer = MagicMock()
        mock_analyzer.search_polymarket = AsyncMock(return_value="Markets found")
        mock_analyzer.analyze = AsyncMock(return_value=None)  # Stage 2 failed!

        mock_polymarket = MagicMock()
        mock_polymarket._get_client = MagicMock(return_value=MagicMock())

        processor._deduplicator = mock_dedup
        processor._classifier = mock_classifier
        processor._analyzer = mock_analyzer
        processor._polymarket = mock_polymarket
        processor._initialized = True

        with patch(
            "synesis.core.processor.search_market_impact",
            new_callable=AsyncMock,
            return_value=[{"title": "Test", "snippet": "Content", "url": "http://test.com"}],
        ):
            message = create_test_message()
            result = await processor.process_message(message)

        # Stage 1 should succeed, Stage 2 failed
        assert result.skipped is False
        assert result.extraction == mock_extraction
        assert result.analysis is None  # Stage 2 failed
        assert result.has_edge is False

        # Signal should still be creatable
        signal = result.to_signal()
        assert signal is not None
        assert signal.extraction == mock_extraction
        assert signal.analysis is None
