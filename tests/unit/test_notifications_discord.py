"""Tests for Discord webhook routing."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from synesis.notifications.discord import send_discord


class TestSendDiscordWebhookRouting:
    """Tests for send_discord with webhook_url_override."""

    @pytest.mark.asyncio
    async def test_override_uses_provided_url(self) -> None:
        """webhook_url_override should be used instead of default."""
        override = SecretStr("https://discord.com/api/webhooks/override/token")

        with patch("synesis.notifications.discord.httpx.AsyncClient") as mock_client_cls:
            mock_response = AsyncMock()
            mock_response.status_code = 204
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await send_discord([{"title": "test"}], webhook_url_override=override)

        assert result is True
        mock_client.post.assert_called_once()
        called_url = mock_client.post.call_args[0][0]
        assert called_url == "https://discord.com/api/webhooks/override/token"

    @pytest.mark.asyncio
    async def test_no_override_uses_default(self) -> None:
        """Without override, falls back to settings.discord_webhook_url."""
        with (
            patch("synesis.notifications.discord.httpx.AsyncClient") as mock_client_cls,
            patch("synesis.notifications.discord.get_settings") as mock_settings,
        ):
            mock_settings.return_value.discord_webhook_url = SecretStr(
                "https://discord.com/api/webhooks/default/token"
            )
            mock_response = AsyncMock()
            mock_response.status_code = 204
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await send_discord([{"title": "test"}])

        assert result is True
        called_url = mock_client.post.call_args[0][0]
        assert called_url == "https://discord.com/api/webhooks/default/token"

    @pytest.mark.asyncio
    async def test_no_webhook_configured_returns_false(self) -> None:
        """Returns False when no webhook URL is configured."""
        with patch("synesis.notifications.discord.get_settings") as mock_settings:
            mock_settings.return_value.discord_webhook_url = None
            result = await send_discord([{"title": "test"}])

        assert result is False
