"""Notification dispatcher — routes to the configured channel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from synesis.config import get_settings
from synesis.core.logging import get_logger

if TYPE_CHECKING:
    from synesis.processing.news import LightClassification, SmartAnalysis, UnifiedMessage

logger = get_logger(__name__)


async def emit_stage1(message: UnifiedMessage, extraction: LightClassification) -> bool:
    """Send Stage 1 notification via the configured channel.

    Args:
        message: Original unified message
        extraction: Stage 1 light classification output

    Returns:
        True if sent successfully
    """
    channel = get_settings().notification_channel

    try:
        if channel == "discord":
            from synesis.notifications.discord import format_stage1_embed, send_discord

            embeds = format_stage1_embed(message, extraction)
            return await send_discord(embeds)
        else:
            from synesis.notifications.telegram import format_stage1_signal, send_long_telegram

            telegram_msg = format_stage1_signal(message=message, extraction=extraction)
            return await send_long_telegram(telegram_msg)
    except Exception:
        logger.exception("Failed to format/send Stage 1 notification", channel=channel)
        return False


async def emit_stage2(message: UnifiedMessage, analysis: SmartAnalysis) -> bool:
    """Send Stage 2 notification via the configured channel.

    Args:
        message: Original unified message
        analysis: Stage 2 smart analysis

    Returns:
        True if sent successfully
    """
    channel = get_settings().notification_channel

    try:
        if channel == "discord":
            from synesis.notifications.discord import format_stage2_embed, send_discord

            embeds = format_stage2_embed(message, analysis)
            return await send_discord(embeds)
        else:
            from synesis.notifications.telegram import format_condensed_signal, send_long_telegram

            telegram_msg = format_condensed_signal(message=message, analysis=analysis)
            return await send_long_telegram(telegram_msg)
    except Exception:
        logger.exception("Failed to format/send Stage 2 notification", channel=channel)
        return False
