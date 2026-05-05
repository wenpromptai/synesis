"""Discord webhook notification service."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from pydantic import SecretStr

from synesis.config import get_settings
from synesis.core.logging import get_logger

logger = get_logger(__name__)

DISCORD_TIMEOUT = 10.0


async def send_discord(
    embeds: list[dict[str, Any]],
    content: str | None = None,
    webhook_url_override: SecretStr | None = None,
) -> bool:
    """Send embed message(s) to a Discord webhook.

    Args:
        embeds: List of embed dicts (max 10 per message)
        content: Optional plain text above the embeds
        webhook_url_override: Use this webhook instead of the default

    Returns:
        True if sent successfully, False otherwise
    """
    webhook_secret = webhook_url_override or get_settings().discord_webhook_url
    is_override = webhook_url_override is not None

    if not webhook_secret:
        logger.warning("Discord webhook URL not configured, skipping notification")
        return False

    webhook_url = webhook_secret.get_secret_value()

    payload: dict[str, Any] = {
        "username": "Synesis",
        "embeds": embeds[:10],
    }
    if content:
        payload["content"] = content[:2000]

    try:
        async with httpx.AsyncClient(timeout=DISCORD_TIMEOUT) as client:
            response = await client.post(webhook_url, json=payload)

            if response.status_code == 204:
                logger.debug("Discord webhook sent successfully")
                return True

            if response.status_code == 429:
                retry_after = 1.0
                try:
                    retry_after = response.json().get("retry_after", 1.0)
                except Exception as e:
                    logger.warning("Could not parse Discord 429 retry_after", error=str(e))
                logger.warning(
                    "Discord rate limited, retrying",
                    retry_after=retry_after,
                )
                await asyncio.sleep(retry_after)
                response = await client.post(webhook_url, json=payload)
                if response.status_code == 204:
                    return True

            logger.error(
                "Discord webhook failed — notification dropped",
                status=response.status_code,
                body=response.text[:200],
            )
            return False

    except httpx.TimeoutException as e:
        logger.error(
            "Discord webhook timed out",
            error_type=type(e).__name__,
            timeout=DISCORD_TIMEOUT,
        )
        return False
    except httpx.ConnectError as e:
        logger.error(
            "Discord webhook connection failed",
            webhook_source="override" if is_override else "DISCORD_WEBHOOK_URL",
            error=str(e),
        )
        return False
    except httpx.HTTPError as e:
        logger.error(
            "Discord webhook HTTP error",
            error_type=type(e).__name__,
            error=str(e),
        )
        return False
