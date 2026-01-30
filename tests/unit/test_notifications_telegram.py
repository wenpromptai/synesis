"""Tests for Telegram notification service."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from synesis.notifications.telegram import (
    format_combined_signal,
    format_investment_signal,
    format_prediction_alert,
    send_telegram,
)
from synesis.processing.models import (
    Direction,
    EventType,
    ImpactLevel,
    LightClassification,
    MarketEvaluation,
    SmartAnalysis,
    TickerAnalysis,
    SectorImplication,
    SourcePlatform,
    SourceType,
    UnifiedMessage,
)
from datetime import datetime, timezone


class TestSendTelegram:
    """Tests for send_telegram function."""

    @pytest.mark.anyio
    async def test_no_bot_token(self) -> None:
        """Test that missing bot token returns False."""
        with patch("synesis.notifications.telegram.get_settings") as mock_settings:
            mock_settings.return_value.telegram_bot_token = None
            mock_settings.return_value.telegram_chat_id = "123"

            result = await send_telegram("Test message")

        assert result is False

    @pytest.mark.anyio
    async def test_no_chat_id(self) -> None:
        """Test that missing chat ID returns False."""
        with patch("synesis.notifications.telegram.get_settings") as mock_settings:
            mock_settings.return_value.telegram_bot_token = MagicMock()
            mock_settings.return_value.telegram_chat_id = None

            result = await send_telegram("Test message")

        assert result is False

    @pytest.mark.anyio
    async def test_successful_send(self) -> None:
        """Test successful message send."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}
        mock_response.raise_for_status = MagicMock()

        with patch("synesis.notifications.telegram.get_settings") as mock_settings:
            settings = MagicMock()
            settings.telegram_bot_token = MagicMock()
            settings.telegram_bot_token.get_secret_value.return_value = "bot_token"
            settings.telegram_chat_id = "123456"
            mock_settings.return_value = settings

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                result = await send_telegram("Test message")

        assert result is True

    @pytest.mark.anyio
    async def test_api_returns_error(self) -> None:
        """Test handling API error response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "description": "Bad request"}
        mock_response.raise_for_status = MagicMock()

        with patch("synesis.notifications.telegram.get_settings") as mock_settings:
            settings = MagicMock()
            settings.telegram_bot_token = MagicMock()
            settings.telegram_bot_token.get_secret_value.return_value = "bot_token"
            settings.telegram_chat_id = "123456"
            mock_settings.return_value = settings

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                result = await send_telegram("Test message")

        assert result is False

    @pytest.mark.anyio
    async def test_http_error(self) -> None:
        """Test handling HTTP error."""
        with patch("synesis.notifications.telegram.get_settings") as mock_settings:
            settings = MagicMock()
            settings.telegram_bot_token = MagicMock()
            settings.telegram_bot_token.get_secret_value.return_value = "bot_token"
            settings.telegram_chat_id = "123456"
            mock_settings.return_value = settings

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                result = await send_telegram("Test message")

        assert result is False


class TestFormatInvestmentSignal:
    """Tests for format_investment_signal."""

    def test_bullish_signal(self) -> None:
        """Test formatting bullish signal."""
        result = format_investment_signal(
            source="@DeItaone",
            summary="Fed cuts rates",
            primary_thesis="Dovish pivot bullish for equities",
            tickers=["SPY", "QQQ"],
            direction="bullish",
            confidence=0.85,
        )

        assert "@DeItaone" in result
        assert "Fed cuts rates" in result
        assert "Dovish pivot" in result
        assert "$SPY" in result
        assert "$QQQ" in result
        assert "BULLISH" in result
        assert "85%" in result

    def test_bearish_signal(self) -> None:
        """Test formatting bearish signal."""
        result = format_investment_signal(
            source="@test",
            summary="Bad news",
            primary_thesis="Market decline expected",
            tickers=["SPY"],
            direction="bearish",
            confidence=0.7,
        )

        assert "BEARISH" in result

    def test_no_tickers(self) -> None:
        """Test formatting signal without tickers."""
        result = format_investment_signal(
            source="@test",
            summary="News",
            primary_thesis="Thesis",
            tickers=[],
            direction="neutral",
            confidence=0.5,
        )

        assert "N/A" in result


class TestFormatPredictionAlert:
    """Tests for format_prediction_alert."""

    def test_undervalued_market(self) -> None:
        """Test formatting undervalued market alert."""
        result = format_prediction_alert(
            market_question="Will the Fed cut rates in March?",
            verdict="undervalued",
            current_price=0.35,
            fair_price=0.55,
            edge=0.20,
            recommended_side="yes",
            reasoning="Fed guidance suggests cut is likely",
        )

        assert "Fed cut rates" in result
        assert "UNDERVALUED" in result
        assert "$0.35" in result
        assert "$0.55" in result
        assert "+20.0%" in result
        assert "YES" in result

    def test_overvalued_market(self) -> None:
        """Test formatting overvalued market alert."""
        result = format_prediction_alert(
            market_question="Will event happen?",
            verdict="overvalued",
            current_price=0.80,
            fair_price=0.60,
            edge=-0.20,
            recommended_side="no",
            reasoning="Market is overpriced",
        )

        assert "OVERVALUED" in result
        assert "NO" in result

    def test_long_reasoning_truncated(self) -> None:
        """Test that long reasoning is truncated."""
        long_reasoning = "A" * 300

        result = format_prediction_alert(
            market_question="Question?",
            verdict="fair",
            current_price=0.50,
            fair_price=0.50,
            edge=0.0,
            recommended_side="skip",
            reasoning=long_reasoning,
        )

        assert "..." in result


class TestFormatCombinedSignal:
    """Tests for format_combined_signal."""

    def _create_message(self, source: str = "@test", text: str = "Test") -> UnifiedMessage:
        """Helper to create a UnifiedMessage for testing."""
        return UnifiedMessage(
            external_id="test_123",
            source_platform=SourcePlatform.twitter,
            source_account=source,
            text=text,
            timestamp=datetime.now(timezone.utc),
            source_type=SourceType.news,
        )

    def _create_extraction(self, summary: str = "Test summary") -> LightClassification:
        """Helper to create a LightClassification for testing."""
        return LightClassification(
            event_type=EventType.macro,
            summary=summary,
            confidence=0.9,
            primary_entity="Test",
        )

    def test_basic_signal(self) -> None:
        """Test formatting basic combined signal."""
        message = self._create_message(source="@DeItaone", text="Apple beats earnings")
        extraction = self._create_extraction(summary="Apple beats earnings")
        analysis = SmartAnalysis(
            tickers=["AAPL"],
            sectors=["technology"],
            predicted_impact=ImpactLevel.high,
            market_direction=Direction.bullish,
            primary_thesis="Apple earnings beat bullish for tech",
            thesis_confidence=0.8,
        )

        result = format_combined_signal(
            message=message,
            extraction=extraction,
            analysis=analysis,
        )

        assert "SIGNAL ALERT" in result
        assert "@DeItaone" in result
        assert "Apple" in result
        assert "HIGH" in result
        assert "BULLISH" in result
        assert "80%" in result

    def test_signal_with_ticker_analyses(self) -> None:
        """Test signal with ticker analyses."""
        message = self._create_message()
        extraction = self._create_extraction()
        analysis = SmartAnalysis(
            tickers=["AAPL"],
            sectors=[],
            predicted_impact=ImpactLevel.medium,
            market_direction=Direction.bullish,
            primary_thesis="Test thesis",
            thesis_confidence=0.7,
            ticker_analyses=[
                TickerAnalysis(
                    ticker="AAPL",
                    company_name="Apple Inc.",
                    bull_thesis="Strong iPhone sales",
                    bear_thesis="China risk",
                    net_direction=Direction.bullish,
                    conviction=0.8,
                    time_horizon="days",
                ),
            ],
        )

        result = format_combined_signal(
            message=message,
            extraction=extraction,
            analysis=analysis,
        )

        assert "$AAPL" in result
        assert "TICKERS" in result
        assert "Bull:" in result
        assert "Bear:" in result

    def test_signal_with_sector_implications(self) -> None:
        """Test signal with sector implications."""
        message = self._create_message()
        extraction = self._create_extraction(summary="Rate hike")
        analysis = SmartAnalysis(
            tickers=[],
            sectors=["financials"],
            predicted_impact=ImpactLevel.medium,
            market_direction=Direction.bearish,
            primary_thesis="Rate hike negative for banks",
            thesis_confidence=0.6,
            sector_implications=[
                SectorImplication(
                    sector="financials",
                    direction=Direction.bearish,
                    reasoning="NIM compression expected",
                ),
            ],
        )

        result = format_combined_signal(
            message=message,
            extraction=extraction,
            analysis=analysis,
        )

        assert "SECTORS" in result
        assert "financials" in result
        assert "BEARISH" in result

    def test_signal_with_market_evaluations(self) -> None:
        """Test signal with market evaluations showing edge."""
        message = self._create_message()
        extraction = self._create_extraction(summary="Fed news")
        market_eval = MarketEvaluation(
            market_id="mkt_123",
            market_question="Will Fed cut rates?",
            is_relevant=True,
            relevance_reasoning="Directly relevant",
            current_price=0.40,
            estimated_fair_price=0.60,
            edge=0.20,
            verdict="undervalued",
            confidence=0.8,
            reasoning="Strong edge",
            recommended_side="yes",
        )

        analysis = SmartAnalysis(
            tickers=[],
            sectors=[],
            predicted_impact=ImpactLevel.high,
            market_direction=Direction.bullish,
            primary_thesis="Fed pivot",
            thesis_confidence=0.8,
            market_evaluations=[market_eval],
        )

        result = format_combined_signal(
            message=message,
            extraction=extraction,
            analysis=analysis,
        )

        assert "POLYMARKET" in result
        assert "Fed cut" in result
        assert "+20.0%" in result
        assert "YES" in result

    def test_signal_with_historical_context(self) -> None:
        """Test signal with historical context."""
        message = self._create_message()
        extraction = self._create_extraction()
        analysis = SmartAnalysis(
            tickers=[],
            sectors=[],
            predicted_impact=ImpactLevel.medium,
            market_direction=Direction.neutral,
            primary_thesis="Test",
            thesis_confidence=0.5,
            historical_context="In 2019, similar Fed action led to a 10% rally",
        )

        result = format_combined_signal(
            message=message,
            extraction=extraction,
            analysis=analysis,
        )

        assert "HISTORICAL CONTEXT" in result
        assert "2019" in result
