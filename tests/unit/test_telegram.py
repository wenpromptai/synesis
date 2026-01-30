"""Tests for Telegram notification service."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from synesis.notifications.telegram import (
    format_investment_signal,
    format_prediction_alert,
    send_telegram,
)


class TestSendTelegram:
    """Tests for send_telegram function."""

    @pytest.mark.asyncio
    async def test_send_telegram_success(self) -> None:
        """Test successful Telegram message send."""
        mock_settings = MagicMock()
        mock_settings.telegram_bot_token = MagicMock()
        mock_settings.telegram_bot_token.get_secret_value.return_value = "test_token"
        mock_settings.telegram_chat_id = "123456789"

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}
        mock_response.raise_for_status = MagicMock()

        with (
            patch("synesis.notifications.telegram.get_settings", return_value=mock_settings),
            patch("synesis.notifications.telegram.httpx.AsyncClient") as mock_client_class,
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await send_telegram("Test message")

            assert result is True
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert "test_token" in call_args[0][0]
            assert call_args[1]["json"]["text"] == "Test message"
            assert call_args[1]["json"]["chat_id"] == "123456789"

    @pytest.mark.asyncio
    async def test_send_telegram_no_token(self) -> None:
        """Test send_telegram returns False when no token configured."""
        mock_settings = MagicMock()
        mock_settings.telegram_bot_token = None
        mock_settings.telegram_chat_id = "123456789"

        with patch("synesis.notifications.telegram.get_settings", return_value=mock_settings):
            result = await send_telegram("Test message")

            assert result is False

    @pytest.mark.asyncio
    async def test_send_telegram_no_chat_id(self) -> None:
        """Test send_telegram returns False when no chat_id configured."""
        mock_settings = MagicMock()
        mock_settings.telegram_bot_token = MagicMock()
        mock_settings.telegram_bot_token.get_secret_value.return_value = "test_token"
        mock_settings.telegram_chat_id = None

        with patch("synesis.notifications.telegram.get_settings", return_value=mock_settings):
            result = await send_telegram("Test message")

            assert result is False

    @pytest.mark.asyncio
    async def test_send_telegram_api_error(self) -> None:
        """Test send_telegram handles API errors gracefully."""
        mock_settings = MagicMock()
        mock_settings.telegram_bot_token = MagicMock()
        mock_settings.telegram_bot_token.get_secret_value.return_value = "test_token"
        mock_settings.telegram_chat_id = "123456789"

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "description": "Bad Request"}
        mock_response.raise_for_status = MagicMock()

        with (
            patch("synesis.notifications.telegram.get_settings", return_value=mock_settings),
            patch("synesis.notifications.telegram.httpx.AsyncClient") as mock_client_class,
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await send_telegram("Test message")

            assert result is False

    @pytest.mark.asyncio
    async def test_send_telegram_http_error(self) -> None:
        """Test send_telegram handles HTTP errors gracefully."""
        mock_settings = MagicMock()
        mock_settings.telegram_bot_token = MagicMock()
        mock_settings.telegram_bot_token.get_secret_value.return_value = "test_token"
        mock_settings.telegram_chat_id = "123456789"

        with (
            patch("synesis.notifications.telegram.get_settings", return_value=mock_settings),
            patch("synesis.notifications.telegram.httpx.AsyncClient") as mock_client_class,
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await send_telegram("Test message")

            assert result is False

    @pytest.mark.asyncio
    async def test_send_telegram_uses_html_parse_mode(self) -> None:
        """Test send_telegram uses HTML parse mode by default."""
        mock_settings = MagicMock()
        mock_settings.telegram_bot_token = MagicMock()
        mock_settings.telegram_bot_token.get_secret_value.return_value = "test_token"
        mock_settings.telegram_chat_id = "123456789"

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}
        mock_response.raise_for_status = MagicMock()

        with (
            patch("synesis.notifications.telegram.get_settings", return_value=mock_settings),
            patch("synesis.notifications.telegram.httpx.AsyncClient") as mock_client_class,
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await send_telegram("<b>Bold message</b>")

            call_args = mock_client.post.call_args
            assert call_args[1]["json"]["parse_mode"] == "HTML"


class TestFormatInvestmentSignal:
    """Tests for format_investment_signal function."""

    def test_format_investment_signal_bullish(self) -> None:
        """Test formatting a bullish investment signal."""
        result = format_investment_signal(
            source="@DeItaone",
            summary="Fed cuts rates by 25bps",
            primary_thesis="Dovish pivot signals risk-on",
            tickers=["SPY", "QQQ"],
            direction="bullish",
            confidence=0.85,
        )

        assert "@DeItaone" in result
        assert "Fed cuts rates by 25bps" in result
        assert "Dovish pivot signals risk-on" in result
        assert "$SPY" in result
        assert "$QQQ" in result
        assert "BULLISH" in result
        assert "85%" in result
        # Green circle emoji for bullish
        assert "\U0001f7e2" in result

    def test_format_investment_signal_bearish(self) -> None:
        """Test formatting a bearish investment signal."""
        result = format_investment_signal(
            source="@analyst",
            summary="Inflation concerns persist",
            primary_thesis="Higher for longer",
            tickers=["TLT"],
            direction="bearish",
            confidence=0.7,
        )

        assert "BEARISH" in result
        # Red circle emoji for bearish
        assert "\U0001f534" in result

    def test_format_investment_signal_neutral(self) -> None:
        """Test formatting a neutral investment signal."""
        result = format_investment_signal(
            source="@analyst",
            summary="Mixed economic data",
            primary_thesis="Uncertainty remains",
            tickers=[],
            direction="neutral",
            confidence=0.5,
        )

        assert "NEUTRAL" in result
        assert "N/A" in result  # No tickers
        # White circle emoji for neutral
        assert "\u26aa" in result

    def test_format_investment_signal_no_tickers(self) -> None:
        """Test formatting when no tickers."""
        result = format_investment_signal(
            source="@news",
            summary="General market news",
            primary_thesis="Market impact unclear",
            tickers=[],
            direction="neutral",
            confidence=0.4,
        )

        assert "N/A" in result


class TestFormatPredictionAlert:
    """Tests for format_prediction_alert function."""

    def test_format_prediction_alert_undervalued(self) -> None:
        """Test formatting an undervalued prediction alert."""
        result = format_prediction_alert(
            market_question="Will the Fed cut rates in March 2025?",
            verdict="undervalued",
            current_price=0.35,
            fair_price=0.55,
            edge=0.20,
            recommended_side="yes",
            reasoning="Rate cut is likely given economic conditions and Fed guidance.",
        )

        assert "Will the Fed cut rates in March 2025?" in result
        assert "UNDERVALUED" in result
        assert "$0.35" in result
        assert "$0.55" in result
        assert "+20.0%" in result
        assert "YES" in result
        assert "Rate cut is likely" in result
        # Chart up emoji for undervalued
        assert "\U0001f4c8" in result
        # Check mark for yes
        assert "\u2705" in result

    def test_format_prediction_alert_overvalued(self) -> None:
        """Test formatting an overvalued prediction alert."""
        result = format_prediction_alert(
            market_question="Will X happen?",
            verdict="overvalued",
            current_price=0.80,
            fair_price=0.60,
            edge=-0.20,
            recommended_side="no",
            reasoning="Market is overestimating the probability.",
        )

        assert "OVERVALUED" in result
        assert "-20.0%" in result
        assert "NO" in result
        # Chart down emoji for overvalued
        assert "\U0001f4c9" in result
        # X mark for no
        assert "\u274c" in result

    def test_format_prediction_alert_long_reasoning_truncated(self) -> None:
        """Test that long reasoning is truncated."""
        long_reasoning = "A" * 300  # More than 200 chars

        result = format_prediction_alert(
            market_question="Test?",
            verdict="undervalued",
            current_price=0.50,
            fair_price=0.60,
            edge=0.10,
            recommended_side="yes",
            reasoning=long_reasoning,
        )

        assert "..." in result
        # Should have first 200 chars
        assert "A" * 50 in result

    def test_format_prediction_alert_short_reasoning_not_truncated(self) -> None:
        """Test that short reasoning is not truncated."""
        short_reasoning = "Short reason."

        result = format_prediction_alert(
            market_question="Test?",
            verdict="fair",
            current_price=0.50,
            fair_price=0.52,
            edge=0.02,
            recommended_side="skip",
            reasoning=short_reasoning,
        )

        assert short_reasoning in result
        assert "..." not in result or short_reasoning + "..." not in result
