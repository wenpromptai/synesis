"""Daily market brief job — fetches snapshot + movers, sends to Discord."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.notifications.discord import send_discord
from synesis.processing.market.discord_format import format_market_brief_embeds
from synesis.processing.market.snapshot import fetch_market_brief

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)


async def market_brief_job(redis: Redis) -> None:
    """Daily market brief job — fetches snapshot + movers, sends to Discord."""
    settings = get_settings()

    # Reuse the events webhook (same Discord channel)
    webhook = settings.discord_events_webhook_url
    if not webhook:
        webhook = settings.discord_webhook_url
    if not webhook:
        logger.warning("No Discord webhook configured for market brief")
        return

    brief = await fetch_market_brief(redis)
    messages = format_market_brief_embeds(brief)

    sent_ok = 0
    for i, embeds in enumerate(messages):
        ok = await send_discord(embeds, webhook_url_override=webhook)
        if ok:
            sent_ok += 1
        if i < len(messages) - 1:
            await asyncio.sleep(0.5)

    if sent_ok < len(messages):
        logger.warning(
            "Market brief partially sent",
            messages_sent=sent_ok,
            messages_total=len(messages),
        )
    else:
        logger.info(
            "Market brief sent to Discord",
            messages_sent=sent_ok,
            messages_total=len(messages),
            gainers=len(brief.movers.gainers),
            losers=len(brief.movers.losers),
            actives=len(brief.movers.most_actives),
        )
