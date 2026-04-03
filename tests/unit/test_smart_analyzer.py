"""Tests for Stage 2 Smart Analyzer."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synesis.processing.news import (
    LightClassification,
    SourcePlatform,
    UnifiedMessage,
    UrgencyLevel,
)
from synesis.processing.news.models import ETFImpact
from synesis.processing.news.analyzer import (
    AnalyzerDeps,
    SMART_ANALYZER_SYSTEM_PROMPT,
    SmartAnalyzer,
)


class TestAnalyzerDeps:
    """Tests for AnalyzerDeps dataclass."""

    def test_create_deps(self) -> None:
        """Test creating AnalyzerDeps with required fields."""
        message = UnifiedMessage(
            external_id="123",
            source_platform=SourcePlatform.telegram,
            source_account="@test",
            text="Test message",
            timestamp=datetime.now(timezone.utc),
        )
        extraction = LightClassification(impact_score=50, urgency=UrgencyLevel.high)

        deps = AnalyzerDeps(
            message=message,
            extraction=extraction,
        )

        assert deps.message == message
        assert deps.extraction == extraction
        assert deps.http_client is None

    def test_deps_with_http_client(self) -> None:
        """Test creating deps with HTTP client."""
        message = UnifiedMessage(
            external_id="123",
            source_platform=SourcePlatform.telegram,
            source_account="@test",
            text="Test",
            timestamp=datetime.now(timezone.utc),
        )
        extraction = LightClassification(impact_score=50, urgency=UrgencyLevel.high)

        mock_client = MagicMock()
        deps = AnalyzerDeps(
            message=message,
            extraction=extraction,
            http_client=mock_client,
        )

        assert deps.http_client == mock_client


class TestSmartAnalyzerSystemPrompt:
    """Tests for the system prompt."""

    def test_prompt_contains_key_sections(self) -> None:
        """Test that system prompt contains key sections."""
        assert "search_polymarket" in SMART_ANALYZER_SYSTEM_PROMPT.lower()
        assert "impact" in SMART_ANALYZER_SYSTEM_PROMPT.lower()
        assert "evaluate prediction markets" in SMART_ANALYZER_SYSTEM_PROMPT.lower()
        assert "thesis" in SMART_ANALYZER_SYSTEM_PROMPT.lower()

    def test_prompt_contains_evaluation_guidance(self) -> None:
        """Test that prompt contains market evaluation guidance."""
        assert "undervalued" in SMART_ANALYZER_SYSTEM_PROMPT
        assert "overvalued" in SMART_ANALYZER_SYSTEM_PROMPT
        assert "fair" in SMART_ANALYZER_SYSTEM_PROMPT
        assert "edge" in SMART_ANALYZER_SYSTEM_PROMPT.lower()

    def test_prompt_contains_sector_etf_references(self) -> None:
        """Test that system prompt includes sector ETF tickers."""
        assert "XLK" in SMART_ANALYZER_SYSTEM_PROMPT
        assert "XLF" in SMART_ANALYZER_SYSTEM_PROMPT
        assert "XLRE" in SMART_ANALYZER_SYSTEM_PROMPT

    def test_prompt_contains_sector_etfs_section(self) -> None:
        """Test that system prompt mentions sector ETFs."""
        assert "Sector ETFs" in SMART_ANALYZER_SYSTEM_PROMPT


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
    async def test_analyze_returns_smart_analysis(self) -> None:
        """Test that analyze returns SmartAnalysis."""
        message = UnifiedMessage(
            external_id="123",
            source_platform=SourcePlatform.telegram,
            source_account="@test",
            text="Fed cuts rates",
            timestamp=datetime.now(timezone.utc),
        )
        extraction = LightClassification(
            impact_score=50,
            urgency=UrgencyLevel.high,
        )

        # Use a MagicMock for output so the log.info call in analyze()
        # can access .sentiment.value without AttributeError (the field
        # was removed from SmartAnalysis but the log line still references it).
        mock_output = MagicMock()
        mock_output.all_entities = []
        mock_output.primary_topics = []
        mock_output.sentiment.value = "bullish"
        mock_output.primary_thesis = "Rate cut bullish for equities"
        mock_output.macro_impact = [ETFImpact(ticker="SPY", sentiment_score=0.7)]
        mock_output.sector_impact = []
        mock_output.market_evaluations = []

        mock_result = MagicMock()
        mock_result.output = mock_output

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        analyzer = SmartAnalyzer()
        analyzer._agent = mock_agent

        result = await analyzer.analyze(
            message=message,
            extraction=extraction,
        )

        assert result is not None
        assert result.primary_thesis == "Rate cut bullish for equities"

    @pytest.mark.anyio
    async def test_analyze_handles_exception(self) -> None:
        """Test that analyze handles exceptions and returns None."""
        message = UnifiedMessage(
            external_id="123",
            source_platform=SourcePlatform.telegram,
            source_account="@test",
            text="Test",
            timestamp=datetime.now(timezone.utc),
        )
        extraction = LightClassification(impact_score=50, urgency=UrgencyLevel.high)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=Exception("LLM Error"))

        analyzer = SmartAnalyzer()
        analyzer._agent = mock_agent

        result = await analyzer.analyze(
            message=message,
            extraction=extraction,
        )

        assert result is None
