"""Discord webhook notification service.

Sends rich embed notifications via Discord webhook.
Used as an alternative to Telegram for Flow 1 news signal output.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import SecretStr

from synesis.config import get_settings
from synesis.core.constants import (
    COLOR_BEARISH,
    COLOR_BULLISH,
    COLOR_CRITICAL,
    COLOR_NEUTRAL,
    COLOR_URGENT,
    SENTIMENT_ICON,
)
from synesis.core.logging import get_logger

if TYPE_CHECKING:
    from synesis.processing.news import LightClassification, SmartAnalysis, UnifiedMessage
    from synesis.processing.news.models import TickerAnalysis
    from synesis.processing.twitter.models import Theme, TwitterAgentAnalysis

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
                except Exception:
                    logger.warning("Could not parse retry_after from rate-limit response")
                logger.warning(
                    "Discord rate limited, retrying after delay",
                    retry_after=retry_after,
                )
                await asyncio.sleep(retry_after)
                response = await client.post(webhook_url, json=payload)
                if response.status_code == 204:
                    logger.debug("Discord webhook sent successfully after retry")
                    return True
                logger.error(
                    "Discord webhook failed after rate-limit retry",
                    status=response.status_code,
                )
                return False

            logger.warning(
                "Discord webhook failed",
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


def format_stage1_embed(
    message: UnifiedMessage,
    extraction: LightClassification,
) -> list[dict[str, Any]]:
    """Format a Stage 1 signal as a Discord embed.

    Sent immediately after Gate 1 for high/critical urgency signals.

    Args:
        message: Original unified message
        extraction: Stage 1 light classification output

    Returns:
        List containing one embed dict
    """
    is_critical = extraction.urgency.value == "critical"
    color = COLOR_CRITICAL if is_critical else COLOR_URGENT
    urgency_label = "CRITICAL" if is_critical else "HIGH"

    # Truncate original message
    original_text = message.text
    if len(original_text) > 400:
        original_text = original_text[:397] + "..."

    # Build description
    description = f"> {original_text}"
    if extraction.summary:
        description += f"\n\n**Summary**\n{extraction.summary}"

    primary = " | ".join(t.value for t in extraction.primary_topics) or "other"
    secondary = " | ".join(t.value for t in extraction.secondary_topics)
    entities = (
        ", ".join(extraction.all_entities[:5])
        if extraction.all_entities
        else extraction.primary_entity
    )

    fields: list[dict[str, Any]] = [
        {"name": "Urgency", "value": urgency_label, "inline": False},
        {
            "name": "Topics",
            "value": primary + (" | " + secondary if secondary else ""),
            "inline": False,
        },
        {"name": "Entities", "value": entities, "inline": False},
    ]

    embed: dict[str, Any] = {
        "author": {"name": message.source_account},
        "title": "\U0001f6a8 1st Pass" if is_critical else "\u26a1 1st Pass",
        "color": color,
        "description": description[:4096],
        "fields": fields,
        "footer": {"text": "Synesis | Stage 1"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return [embed]


def format_stage2_embed(
    message: UnifiedMessage,
    analysis: SmartAnalysis,
) -> list[dict[str, Any]]:
    """Format a Stage 2 analysis as Discord embed(s).

    Args:
        message: Original unified message
        analysis: Stage 2 smart analysis

    Returns:
        List of embed dicts
    """
    from synesis.processing.news.models import MACRO_ETF_TICKERS, SECTOR_ETF_TICKERS

    sentiment = analysis.sentiment.value
    sentiment_colors = {
        "bullish": COLOR_BULLISH,
        "bearish": COLOR_BEARISH,
        "neutral": COLOR_NEUTRAL,
    }
    sentiment_labels = {k: f"{v} {k.upper()}" for k, v in SENTIMENT_ICON.items()}
    color = sentiment_colors.get(sentiment, COLOR_NEUTRAL)
    label = sentiment_labels.get(sentiment, "\u26aa NEUTRAL")

    # Truncate original message
    original_text = message.text
    if len(original_text) > 600:
        original_text = original_text[:597] + "..."

    # Description: quoted source + thesis + context
    description = f"> {original_text}\n\n"
    description += f"**Thesis:** {analysis.primary_thesis}\n"
    if analysis.historical_context:
        description += f"\n**Context:** *{analysis.historical_context}*\n\u200b"

    # Top-level metrics
    fields: list[dict[str, Any]] = [
        {
            "name": "Signal",
            "value": f"**{label}** ({analysis.sentiment_score:+.2f})",
            "inline": True,
        },
        {"name": "Confidence", "value": f"{analysis.thesis_confidence:.0%}", "inline": True},
        {"name": "Source", "value": message.source_account, "inline": True},
    ]

    # Ticker sections
    stock_tickers = [
        t
        for t in analysis.ticker_analyses
        if t.ticker not in MACRO_ETF_TICKERS and t.ticker not in SECTOR_ETF_TICKERS
    ]
    sector_tickers = [t for t in analysis.ticker_analyses if t.ticker in SECTOR_ETF_TICKERS]
    macro_tickers = [t for t in analysis.ticker_analyses if t.ticker in MACRO_ETF_TICKERS]

    def _format_ticker_lines(tickers: list[TickerAnalysis]) -> str:
        lines = []
        for ta in tickers:
            emoji = SENTIMENT_ICON.get(ta.net_direction.value, "\u26aa")
            name = f" - {ta.company_name}" if ta.company_name else ""
            line = f"{emoji} `${ta.ticker}`{name} {ta.net_direction.value} ({ta.conviction:.0%})"
            if ta.relevance_reason:
                line += f"\n> {ta.relevance_reason}"
            lines.append(line)
        return "\n".join(lines)

    if stock_tickers:
        fields.append(
            {
                "name": "Tickers",
                "value": _format_ticker_lines(stock_tickers)[:1024],
                "inline": False,
            }
        )

    if sector_tickers:
        fields.append(
            {
                "name": "Sector Impact",
                "value": _format_ticker_lines(sector_tickers)[:1024],
                "inline": False,
            }
        )

    if macro_tickers:
        fields.append(
            {
                "name": "Macro Impact",
                "value": _format_ticker_lines(macro_tickers)[:1024],
                "inline": False,
            }
        )

    # Polymarket section
    relevant_markets = [
        e for e in analysis.market_evaluations if e.is_relevant and e.confidence >= 0.6
    ]
    if relevant_markets:
        relevant_markets.sort(key=lambda e: (e.confidence, abs(e.edge or 0)), reverse=True)
        mkt = relevant_markets[0]

        edge = mkt.edge or 0
        fair_price = mkt.estimated_fair_price if mkt.estimated_fair_price else mkt.current_price

        side_map = {"yes": "\u2705 YES", "no": "\u274c NO"}
        side_str = side_map.get(mkt.recommended_side, "\u23ed\ufe0f SKIP")

        fields.append(
            {
                "name": "Polymarket",
                "value": (
                    f"**{mkt.market_question}**\n"
                    f"{side_str} @ `${mkt.current_price:.2f}` \u2192 `${fair_price:.2f}` "
                    f"(edge: **{edge:+.1%}**)"
                )[:1024],
                "inline": False,
            }
        )

    embed: dict[str, Any] = {
        "author": {"name": message.source_account},
        "title": f"{label} Signal",
        "color": color,
        "description": description[:4096],
        "fields": fields[:25],
        "footer": {"text": "Synesis | Stage 2 Analysis"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return [embed]


def format_twitter_agent_embeds(analysis: TwitterAgentAnalysis) -> list[list[dict[str, Any]]]:
    """Format a Twitter agent digest as separate Discord messages.

    Returns a list of messages — each message is a list of embeds (sent as one
    webhook call).  This mirrors the Stage 2 news format:

    * Message 1: Header — market overview
    * Messages 2..N: One per theme (own color based on net sentiment)
    * Final message: Summary footer

    Args:
        analysis: The LLM-produced digest analysis

    Returns:
        List of messages, where each message is ``list[dict]`` of embeds
    """
    conviction_emoji = {"high": "\U0001f525", "medium": "\u26a1", "low": "\U0001f4ad"}
    category_emoji = {
        "macro": "\U0001f30d",
        "sector": "\U0001f3ed",
        "earnings": "\U0001f4ca",
        "geopolitical": "\U0001f6e1\ufe0f",
        "trade_idea": "\U0001f4b0",
        "technical": "\U0001f4c8",
    }
    conviction_label = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now_iso = datetime.now(timezone.utc).isoformat()

    messages: list[list[dict[str, Any]]] = []

    # --- Message 1: Header with market overview ---
    header_embed: dict[str, Any] = {
        "title": "\U0001f4f0 Daily X Brief",
        "color": COLOR_NEUTRAL,
        "description": analysis.market_overview[:4096],
        "fields": [
            {"name": "Themes", "value": str(len(analysis.themes)), "inline": True},
            {"name": "Tweets Analyzed", "value": str(analysis.raw_tweet_count), "inline": True},
        ],
        "footer": {"text": f"Synesis X Brief | {today}"},
        "timestamp": now_iso,
    }
    messages.append([header_embed])

    # --- Messages 2..N: One per theme ---
    def _theme_color(theme: Theme) -> int:
        """Pick embed color from the dominant ticker direction."""
        if not theme.tickers:
            return COLOR_NEUTRAL
        directions = [tm.direction for tm in theme.tickers]
        if directions.count("bullish") > directions.count("bearish"):
            return COLOR_BULLISH
        if directions.count("bearish") > directions.count("bullish"):
            return COLOR_BEARISH
        return COLOR_NEUTRAL

    for i, theme in enumerate(analysis.themes, 1):
        cat_icon = category_emoji.get(theme.category, "\U0001f4cb")
        conv_icon = conviction_emoji.get(theme.conviction, "")
        verified_icon = "\u2705" if theme.verified else "\u2753"

        title = f"{cat_icon} {theme.title} {conv_icon}{verified_icon}"

        # Description: summary
        description = theme.summary

        # Fields
        fields: list[dict[str, Any]] = []

        # Conviction + category inline row
        fields.append(
            {
                "name": "Conviction",
                "value": conviction_label.get(theme.conviction, theme.conviction),
                "inline": True,
            }
        )
        fields.append(
            {"name": "Category", "value": theme.category.replace("_", " ").title(), "inline": True}
        )
        fields.append(
            {
                "name": "Verified",
                "value": "Yes" if theme.verified else "No",
                "inline": True,
            }
        )

        # Tickers
        if theme.tickers:
            ticker_lines = []
            for tm in theme.tickers:
                emoji = SENTIMENT_ICON.get(tm.direction, "\u26aa")
                line = f"{emoji} `${tm.ticker}` {tm.direction} — {tm.reasoning}"
                if tm.price_context:
                    line += f"\n> {tm.price_context}"
                if tm.trade_idea:
                    line += f"\n> **Idea:** {tm.trade_idea}"
                ticker_lines.append(line)
            fields.append(
                {"name": "Tickers", "value": "\n".join(ticker_lines)[:1024], "inline": False}
            )

        # Research notes
        if theme.research_notes:
            fields.append(
                {
                    "name": "Research",
                    "value": theme.research_notes[:1024],
                    "inline": False,
                }
            )

        # Risk factors
        if theme.risk_factors:
            fields.append(
                {
                    "name": "Risks",
                    "value": "\n".join(f"\u26a0\ufe0f {r}" for r in theme.risk_factors[:3]),
                    "inline": False,
                }
            )

        # Verification notes
        if theme.verification_notes:
            fields.append(
                {"name": "Verification", "value": theme.verification_notes[:1024], "inline": False}
            )

        # Sources
        sources_str = ", ".join(f"@{s.lstrip('@')}" for s in theme.sources)
        fields.append({"name": "Sources", "value": sources_str[:1024], "inline": False})

        embed: dict[str, Any] = {
            "title": title[:256],
            "color": _theme_color(theme),
            "description": description[:4096],
            "fields": fields[:25],
            "footer": {"text": f"Theme {i}/{len(analysis.themes)}"},
            "timestamp": now_iso,
        }
        messages.append([embed])

    return messages


# Keep old name as alias for backward-compat in tests
def format_twitter_agent_embed(analysis: TwitterAgentAnalysis) -> list[dict[str, Any]]:
    """Legacy wrapper — returns all embeds flat (one per theme + header).

    Prefer ``format_twitter_agent_embeds`` for production sending.
    """
    all_embeds: list[dict[str, Any]] = []
    for msg in format_twitter_agent_embeds(analysis):
        all_embeds.extend(msg)
    return all_embeds
