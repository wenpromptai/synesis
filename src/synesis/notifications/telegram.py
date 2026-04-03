"""Telegram notification service.

Provides simple async function to send messages to a Telegram chat.
Used to notify about investment signals and prediction opportunities.
"""

from __future__ import annotations

import json

from typing import TYPE_CHECKING

import httpx

from synesis.config import get_settings
from synesis.core.constants import TELEGRAM_MAX_MESSAGE_LENGTH
from synesis.core.logging import get_logger

if TYPE_CHECKING:
    from synesis.processing.news import ETFImpact, LightClassification, SmartAnalysis, UnifiedMessage

logger = get_logger(__name__)

# Telegram API timeout
TELEGRAM_TIMEOUT = 10.0

# Telegram message limit
TELEGRAM_MAX_LENGTH = TELEGRAM_MAX_MESSAGE_LENGTH

# Section separators for message formatting (10 chars max for mobile)
SECTION_SEPARATOR = "━━━━━━━━━━"
SUBSECTION_SEPARATOR = "──────────"


def _escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _split_message_at_sections(message: str, max_length: int = TELEGRAM_MAX_LENGTH) -> list[str]:
    """Split a long message into chunks at section boundaries.

    Splits at SECTION_SEPARATOR lines so each chunk starts/ends cleanly.
    Multi-part messages get part indicators (e.g. "⋯ 1/3") for context.

    Args:
        message: The full message text
        max_length: Maximum length per chunk (default: Telegram limit)

    Returns:
        List of message chunks, each under max_length
    """
    if len(message) <= max_length:
        return [message]

    # Reserve space for part indicators we'll add later
    effective_max = max_length - 40

    chunks: list[str] = []
    remaining = message

    while remaining:
        if len(remaining) <= effective_max:
            chunks.append(remaining)
            break

        # Find the last section separator within the limit
        search_area = remaining[:effective_max]
        last_separator_pos = search_area.rfind(SECTION_SEPARATOR)

        if last_separator_pos > 0:
            # Split at section boundary — keep separator with the next chunk
            chunk = remaining[:last_separator_pos].rstrip()
            remaining = remaining[last_separator_pos:]
            # Strip only whitespace before the separator, keep the separator itself
            remaining = remaining.lstrip("\n")
        else:
            # No separator found — split at last double-newline or newline
            last_double_nl = search_area.rfind("\n\n")
            last_newline = search_area.rfind("\n")
            split_pos = last_double_nl if last_double_nl > effective_max // 2 else last_newline

            if split_pos > effective_max // 2:
                chunk = remaining[:split_pos].rstrip()
                remaining = remaining[split_pos:].lstrip("\n")
            else:
                # Last resort: hard split at effective_max
                chunk = remaining[:effective_max].rstrip()
                remaining = remaining[effective_max:].lstrip("\n")

        chunks.append(chunk)

    # Add part indicators for multi-part messages
    if len(chunks) > 1:
        total = len(chunks)
        for i in range(len(chunks)):
            if i < total - 1:
                # Not the last chunk — add indicator at bottom
                chunks[i] += f"\n\n<i>⋯ {i + 1}/{total}</i>"
            if i > 0:
                # Not the first chunk — add indicator at top
                chunks[i] = f"<i>⋯ {i + 1}/{total}</i>\n\n" + chunks[i]

    return chunks


async def send_telegram(message: str, parse_mode: str = "HTML") -> bool:
    """Send a message to the configured Telegram chat.

    Args:
        message: The message text to send (supports HTML formatting)
        parse_mode: Parse mode for message formatting (HTML or Markdown)

    Returns:
        True if message was sent successfully, False otherwise
    """
    settings = get_settings()

    if not settings.telegram_bot_token:
        logger.warning("Telegram bot token not configured, skipping notification")
        return False

    if not settings.telegram_chat_id:
        logger.warning("Telegram chat ID not configured, skipping notification")
        return False

    url = (
        f"https://api.telegram.org/bot{settings.telegram_bot_token.get_secret_value()}/sendMessage"
    )

    payload = {
        "chat_id": settings.telegram_chat_id,
        "text": message,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    try:
        async with httpx.AsyncClient(timeout=TELEGRAM_TIMEOUT) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

            result = response.json()
            if result.get("ok"):
                logger.debug("Telegram message sent successfully")
                return True
            else:
                logger.warning(
                    "Telegram API returned error",
                    error=result.get("description"),
                )
                return False

    except httpx.HTTPError as e:
        logger.error("Failed to send Telegram message", error=str(e))
        return False
    except (json.JSONDecodeError, KeyError) as e:
        logger.error("Failed to parse Telegram API response", error=str(e))
        return False


async def send_long_telegram(message: str, parse_mode: str = "HTML") -> bool:
    """Send a potentially long message, splitting into multiple if needed.

    Args:
        message: The message text (may exceed Telegram limits)
        parse_mode: Parse mode for message formatting (HTML or Markdown)

    Returns:
        True if all messages were sent successfully, False otherwise
    """
    chunks = _split_message_at_sections(message)

    all_sent = True
    for i, chunk in enumerate(chunks):
        success = await send_telegram(chunk, parse_mode)
        if not success:
            logger.warning(
                "Failed to send message chunk",
                chunk_index=i,
                total_chunks=len(chunks),
            )
            all_sent = False

    return all_sent


def format_stage1_signal(
    message: "UnifiedMessage",
    extraction: "LightClassification",
) -> str:
    """Format a Stage 1 signal for Telegram — the first-pass notification.

    Sent immediately after gate for high/critical urgency signals.
    Contains impact score + matched tickers only (no LLM output).

    Args:
        message: Original unified message
        extraction: Stage 1 classification (impact score + tickers)

    Returns:
        Formatted HTML message for Telegram
    """
    urgency_prefix = {"critical": "🚨", "high": "⚡"}
    prefix = urgency_prefix.get(extraction.urgency.value, "")

    # Author line: "channel_name · news_source" or just "channel_name"
    from synesis.core.constants import NEWS_SOURCE_RE

    source_match = NEWS_SOURCE_RE.search(message.text)
    news_source = source_match.group(1) if source_match else None
    author = (
        f"{message.source_account} · {news_source}"
        if news_source and news_source != message.source_account
        else message.source_account
    )

    # Truncate original message to ~400 chars for the blockquote
    original_text = message.text
    if len(original_text) > 400:
        original_text = original_text[:397] + "..."

    lines = [
        f"<b>{prefix}[1st pass]</b> — {_escape_html(author)}",
        "",
        f"<blockquote>{_escape_html(original_text)}</blockquote>",
    ]

    # Matched tickers
    if extraction.matched_tickers:
        tickers_str = _escape_html(", ".join(extraction.matched_tickers))
        lines += ["", f"🏷 <b>Tickers</b> {tickers_str}"]

    # Impact score
    lines += ["", f"📊 <b>Impact</b> {extraction.impact_score}/100"]

    return "\n".join(lines)


def format_condensed_signal(
    message: "UnifiedMessage",
    analysis: "SmartAnalysis",
) -> str:
    """Format a condensed single-message signal for Telegram (~1500-2000 chars).

    Includes essential info without splitting:
    - Original message (truncated to ~300 chars)
    - Thesis + direction + confidence
    - All tickers (one line each)
    - All sectors (one line each)
    - Historical context (truncated to ~200 chars)
    - Single most relevant Polymarket market

    Args:
        message: Original unified message with raw text
        analysis: Stage 2 smart analysis with all judgments

    Returns:
        Formatted HTML message for Telegram (single message, ~1500-2000 chars)
    """
    # Truncate original message to ~300 chars
    original_text = message.text
    if len(original_text) > 300:
        original_text = original_text[:297] + "..."

    # Build message
    msg = f"""➕ <b>[add story]</b> — {_escape_html(message.source_account)}

<blockquote>{_escape_html(original_text)}</blockquote>

💡 <b>Thesis:</b> {_escape_html(analysis.primary_thesis)}"""

    # ETF impact with per-ETF sentiment
    def _format_etf_impact(impacts: list[ETFImpact]) -> str:
        lines = []
        for etf in impacts:
            emoji = "🟢" if etf.sentiment_score > 0 else "🔴" if etf.sentiment_score < 0 else "⚪"
            line = f"{emoji} <code>{etf.ticker}</code> ({etf.sentiment_score:+.1f})"
            if etf.reason:
                line += f"\n<blockquote>{_escape_html(etf.reason)}</blockquote>"
            lines.append(line)
        return "\n".join(lines)

    # Macro before sector
    if analysis.macro_impact:
        msg += f"\n\n<b>Macro Impact</b>\n{_format_etf_impact(analysis.macro_impact)}"

    if analysis.sector_impact:
        msg += f"\n\n<b>Sector Impact</b>\n{_format_etf_impact(analysis.sector_impact)}"

    # Historical context (full, no truncation)
    if analysis.historical_context:
        msg += f"\n\n📜 <b>Context:</b> <i>{_escape_html(analysis.historical_context)}</i>"

    # Single most relevant Polymarket (best by relevance, show even without edge)
    relevant_markets = [
        e for e in analysis.market_evaluations if e.is_relevant and e.confidence >= 0.6
    ]
    if relevant_markets:
        # Sort by confidence (most confident first), then by absolute edge
        relevant_markets.sort(key=lambda e: (e.confidence, abs(e.edge or 0)), reverse=True)
        mkt = relevant_markets[0]

        edge = mkt.edge or 0
        side = mkt.recommended_side
        fair_price = mkt.estimated_fair_price if mkt.estimated_fair_price else mkt.current_price

        if side == "yes":
            side_str = "✅ YES"
        elif side == "no":
            side_str = "❌ NO"
        else:
            side_str = "⏭️ SKIP"

        msg += f"""

🎯 <b>Polymarket</b>
{_escape_html(mkt.market_question)}
{side_str} @ ${mkt.current_price:.2f} → ${fair_price:.2f} (edge: {edge:+.1%})"""

    return msg
