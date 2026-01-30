"""Tests for Stage 2 Smart Analyzer."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synesis.processing.models import (
    Direction,
    EventType,
    ImpactLevel,
    LightClassification,
    NewsCategory,
    SmartAnalysis,
    SourcePlatform,
    SourceType,
    UnifiedMessage,
)
from synesis.processing.smart_analyzer import (
    AnalyzerDeps,
    SMART_ANALYZER_SYSTEM_PROMPT,
    SmartAnalyzer,
    get_smart_analyzer,
)


class TestAnalyzerDeps:
    """Tests for AnalyzerDeps dataclass."""

    def test_create_deps(self) -> None:
        """Test creating AnalyzerDeps with required fields."""
        message = UnifiedMessage(
            external_id="123",
            source_platform=SourcePlatform.twitter,
            source_account="@test",
            text="Test message",
            timestamp=datetime.now(timezone.utc),
            source_type=SourceType.news,
        )
        extraction = LightClassification(
            event_type=EventType.macro,
            summary="Test summary",
            confidence=0.9,
            primary_entity="Test Entity",
        )

        deps = AnalyzerDeps(
            message=message,
            extraction=extraction,
            web_results=["Result 1", "Result 2"],
            markets_text="| Market | Price |",
        )

        assert deps.message == message
        assert deps.extraction == extraction
        assert len(deps.web_results) == 2
        assert deps.markets_text == "| Market | Price |"
        assert deps.http_client is None

    def test_deps_with_http_client(self) -> None:
        """Test creating deps with HTTP client."""
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

        mock_client = MagicMock()
        deps = AnalyzerDeps(
            message=message,
            extraction=extraction,
            web_results=[],
            markets_text="",
            http_client=mock_client,
        )

        assert deps.http_client == mock_client


class TestSmartAnalyzerSystemPrompt:
    """Tests for the system prompt."""

    def test_prompt_contains_key_sections(self) -> None:
        """Test that system prompt contains key sections."""
        assert "tickers" in SMART_ANALYZER_SYSTEM_PROMPT.lower()
        assert "sectors" in SMART_ANALYZER_SYSTEM_PROMPT.lower()
        assert "impact" in SMART_ANALYZER_SYSTEM_PROMPT.lower()
        assert "market_evaluations" in SMART_ANALYZER_SYSTEM_PROMPT.lower()
        assert "thesis" in SMART_ANALYZER_SYSTEM_PROMPT.lower()

    def test_prompt_contains_evaluation_guidance(self) -> None:
        """Test that prompt contains market evaluation guidance."""
        assert "undervalued" in SMART_ANALYZER_SYSTEM_PROMPT
        assert "overvalued" in SMART_ANALYZER_SYSTEM_PROMPT
        assert "fair" in SMART_ANALYZER_SYSTEM_PROMPT
        assert "edge" in SMART_ANALYZER_SYSTEM_PROMPT.lower()


class TestSmartAnalyzer:
    """Tests for SmartAnalyzer class."""

    def test_init_no_client(self) -> None:
        """Test initializing analyzer without Polymarket client."""
        analyzer = SmartAnalyzer()
        assert analyzer._polymarket is None
        assert analyzer._own_polymarket is True
        assert analyzer._agent is None

    def test_init_with_client(self) -> None:
        """Test initializing analyzer with Polymarket client."""
        mock_client = MagicMock()
        analyzer = SmartAnalyzer(polymarket_client=mock_client)
        assert analyzer._polymarket == mock_client
        assert analyzer._own_polymarket is False

    def test_agent_property_creates_agent(self) -> None:
        """Test that agent property creates agent lazily."""
        analyzer = SmartAnalyzer()
        assert analyzer._agent is None

        # Mock the entire _create_agent method
        mock_agent = MagicMock()
        with patch.object(analyzer, "_create_agent", return_value=mock_agent):
            agent = analyzer.agent

        assert agent is mock_agent
        assert analyzer._agent is mock_agent

    def test_polymarket_property_creates_client(self) -> None:
        """Test that polymarket property creates client lazily."""
        analyzer = SmartAnalyzer()
        assert analyzer._polymarket is None

        mock_client = MagicMock()
        with patch("synesis.markets.polymarket.PolymarketClient", return_value=mock_client):
            client = analyzer.polymarket

        assert client is not None
        assert analyzer._polymarket is not None
        assert analyzer._own_polymarket is True

    @pytest.mark.anyio
    async def test_close_with_own_client(self) -> None:
        """Test closing analyzer that owns its Polymarket client."""
        mock_client = MagicMock()
        mock_client.close = AsyncMock()

        analyzer = SmartAnalyzer()
        analyzer._polymarket = mock_client
        analyzer._own_polymarket = True

        await analyzer.close()

        mock_client.close.assert_called_once()
        assert analyzer._polymarket is None

    @pytest.mark.anyio
    async def test_close_with_external_client(self) -> None:
        """Test closing analyzer that doesn't own its Polymarket client."""
        mock_client = MagicMock()
        mock_client.close = AsyncMock()

        analyzer = SmartAnalyzer(polymarket_client=mock_client)

        await analyzer.close()

        # Should NOT close external client
        mock_client.close.assert_not_called()

    @pytest.mark.anyio
    async def test_search_polymarket(self) -> None:
        """Test searching Polymarket for markets."""
        mock_market = MagicMock()
        mock_market.id = "market_123"
        mock_market.question = "Will something happen?"
        mock_market.yes_price = 0.65
        mock_market.no_price = 0.35
        mock_market.volume_24h = 10000.0

        mock_client = MagicMock()
        mock_client.search_markets = AsyncMock(return_value=[mock_market])

        analyzer = SmartAnalyzer(polymarket_client=mock_client)

        result = await analyzer.search_polymarket(["test", "keyword"])

        assert "market_123" in result
        assert "Will something happen?" in result
        assert "$0.65" in result
        mock_client.search_markets.assert_called()

    @pytest.mark.anyio
    async def test_search_polymarket_no_results(self) -> None:
        """Test searching Polymarket with no results."""
        mock_client = MagicMock()
        mock_client.search_markets = AsyncMock(return_value=[])

        analyzer = SmartAnalyzer(polymarket_client=mock_client)

        result = await analyzer.search_polymarket(["obscure_keyword"])

        assert "No markets found" in result

    @pytest.mark.anyio
    async def test_search_polymarket_handles_error(self) -> None:
        """Test that search handles errors gracefully."""
        mock_client = MagicMock()
        mock_client.search_markets = AsyncMock(side_effect=Exception("API Error"))

        analyzer = SmartAnalyzer(polymarket_client=mock_client)

        result = await analyzer.search_polymarket(["test"])

        assert "No markets found" in result

    @pytest.mark.anyio
    async def test_analyze_returns_smart_analysis(self) -> None:
        """Test that analyze returns SmartAnalysis."""
        message = UnifiedMessage(
            external_id="123",
            source_platform=SourcePlatform.twitter,
            source_account="@test",
            text="Fed cuts rates",
            timestamp=datetime.now(timezone.utc),
            source_type=SourceType.news,
        )
        extraction = LightClassification(
            news_category=NewsCategory.economic_calendar,
            event_type=EventType.macro,
            summary="Fed rate cut",
            confidence=0.95,
            primary_entity="Federal Reserve",
        )

        mock_output = SmartAnalysis(
            tickers=["SPY"],
            sectors=["financials"],
            predicted_impact=ImpactLevel.high,
            market_direction=Direction.bullish,
            primary_thesis="Rate cut bullish for equities",
            thesis_confidence=0.8,
        )

        mock_result = MagicMock()
        mock_result.output = mock_output

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        analyzer = SmartAnalyzer()
        analyzer._agent = mock_agent

        result = await analyzer.analyze(
            message=message,
            extraction=extraction,
            web_results=["Research result"],
            markets_text="| Market | Price |",
        )

        assert result is not None
        assert result.tickers == ["SPY"]
        assert result.primary_thesis == "Rate cut bullish for equities"

    @pytest.mark.anyio
    async def test_analyze_handles_exception(self) -> None:
        """Test that analyze handles exceptions and returns None."""
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

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=Exception("LLM Error"))

        analyzer = SmartAnalyzer()
        analyzer._agent = mock_agent

        result = await analyzer.analyze(
            message=message,
            extraction=extraction,
            web_results=[],
            markets_text="",
        )

        assert result is None


class TestGetSmartAnalyzer:
    """Tests for get_smart_analyzer singleton."""

    def test_returns_same_instance(self) -> None:
        """Test that get_smart_analyzer returns singleton."""
        # Clear cache first
        get_smart_analyzer.cache_clear()

        analyzer1 = get_smart_analyzer()
        analyzer2 = get_smart_analyzer()

        assert analyzer1 is analyzer2
