"""Tests for Telegram notification service."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from synesis.notifications.telegram import (
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
