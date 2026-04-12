"""Daily market movers job — fetches snapshot + movers, sends to Discord."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.notifications.discord import send_discord
from synesis.processing.market.discord_format import format_market_movers_embeds
from synesis.processing.market.snapshot import fetch_market_movers

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from synesis.storage.database import Database

logger = get_logger(__name__)


async def market_movers_job(redis: Redis, db: Database | None = None) -> None:
    """Daily market movers job — fetches snapshot + movers, sends to Discord."""
    settings = get_settings()

    webhook = settings.discord_brief_webhook_url
    if not webhook:
        webhook = settings.discord_webhook_url
    if not webhook:
        logger.warning("No Discord webhook configured for market movers")
        return

    brief = await fetch_market_movers(redis)

    # Send market data embeds
    data_messages = format_market_movers_embeds(brief)
    sent_ok = 0
    for i, embeds in enumerate(data_messages):
        ok = await send_discord(embeds, webhook_url_override=webhook)
        if ok:
            sent_ok += 1
        if i < len(data_messages) - 1:
            await asyncio.sleep(0.5)

    if sent_ok < len(data_messages):
        logger.warning(
            "Market movers partially sent",
            messages_sent=sent_ok,
            messages_total=len(data_messages),
        )
    else:
        logger.info(
            "Market movers sent",
            messages_sent=sent_ok,
            messages_total=len(data_messages),
        )

    # Persist to diary table
    if db:
        try:
            payload: dict[str, Any] = {"market_data": brief.model_dump(mode="json")}
            await db.upsert_diary_entry(
                entry_date=datetime.now(UTC).date(),
                source="market_movers",
                payload=payload,
            )
        except Exception:
            logger.exception("Failed to save market movers to diary")
