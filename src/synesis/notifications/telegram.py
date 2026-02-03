"""Telegram notification service.

Provides simple async function to send messages to a Telegram chat.
Used to notify about investment signals and prediction opportunities.
"""

import json
from typing import TYPE_CHECKING

import httpx

from synesis.config import get_settings
from synesis.core.constants import TELEGRAM_MAX_MESSAGE_LENGTH
from synesis.core.logging import get_logger

if TYPE_CHECKING:
    from synesis.processing.news import LightClassification, SmartAnalysis, UnifiedMessage
    from synesis.processing.sentiment import SentimentSignal

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

    Args:
        message: The full message text
        max_length: Maximum length per chunk (default: Telegram limit)

    Returns:
        List of message chunks, each under max_length
    """
    if len(message) <= max_length:
        return [message]

    chunks: list[str] = []
    remaining = message

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        # Find the last section separator within the limit
        search_area = remaining[:max_length]
        last_separator_pos = search_area.rfind(SECTION_SEPARATOR)

        if last_separator_pos > 0:
            # Split at section boundary
            chunk = remaining[:last_separator_pos].rstrip()
            remaining = remaining[last_separator_pos:].lstrip()
        else:
            # No separator found - split at last newline
            last_newline = search_area.rfind("\n")
            if last_newline > max_length // 2:
                chunk = remaining[:last_newline].rstrip()
                remaining = remaining[last_newline:].lstrip()
            else:
                # Last resort: hard split
                chunk = remaining[: max_length - 20] + "\n\n<i>(continued...)</i>"
                remaining = remaining[max_length - 20 :]

        chunks.append(chunk)

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
        logger.debug("Telegram bot token not configured, skipping notification")
        return False

    if not settings.telegram_chat_id:
        logger.debug("Telegram chat ID not configured, skipping notification")
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
        logger.warning("Failed to send Telegram message", error=str(e))
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


def format_investment_signal(
    source: str,
    summary: str,
    primary_thesis: str,
    tickers: list[str],
    direction: str,
    confidence: float,
) -> str:
    """Format an investment signal for Telegram notification.

    Args:
        source: News source (e.g., @DeItaone)
        summary: One-line summary of the news
        primary_thesis: Investment thesis from Stage 2A
        tickers: List of affected tickers
        direction: Market direction (bullish/bearish/neutral)
        confidence: Thesis confidence (0.0 to 1.0)

    Returns:
        Formatted HTML message for Telegram
    """
    # Direction emoji
    direction_emoji = {
        "bullish": "\U0001f7e2",  # Green circle
        "bearish": "\U0001f534",  # Red circle
        "neutral": "\u26aa",  # White circle
    }.get(direction, "\u26aa")

    # Format tickers
    tickers_str = ", ".join(f"${t}" for t in tickers) if tickers else "N/A"

    return f"""<b>{direction_emoji} Investment Signal</b>

<b>Source:</b> {_escape_html(source)}
<b>Summary:</b> {_escape_html(summary)}

<b>Thesis:</b> {_escape_html(primary_thesis)}

<b>Tickers:</b> {tickers_str}
<b>Direction:</b> {direction.upper()}
<b>Confidence:</b> {confidence:.0%}"""


def format_prediction_alert(
    market_question: str,
    verdict: str,
    current_price: float,
    fair_price: float,
    edge: float,
    recommended_side: str,
    reasoning: str,
) -> str:
    """Format a prediction market alert for Telegram notification.

    Args:
        market_question: The prediction market question
        verdict: undervalued/overvalued/fair
        current_price: Current YES price
        fair_price: Estimated fair price
        edge: Edge percentage
        recommended_side: yes/no/skip
        reasoning: Brief reasoning

    Returns:
        Formatted HTML message for Telegram
    """
    # Verdict emoji
    verdict_emoji = {
        "undervalued": "\U0001f4c8",  # Chart up
        "overvalued": "\U0001f4c9",  # Chart down
        "fair": "\u2696\ufe0f",  # Balance scale
    }.get(verdict, "\U0001f4ca")  # Bar chart

    # Side emoji
    side_emoji = {
        "yes": "\u2705",  # Check mark
        "no": "\u274c",  # X mark
        "skip": "\u23ed\ufe0f",  # Skip
    }.get(recommended_side, "\U0001f914")  # Thinking

    reasoning_display = f"{reasoning[:200]}..." if len(reasoning) > 200 else reasoning

    return f"""<b>{verdict_emoji} Prediction Opportunity</b>

<b>Market:</b> {_escape_html(market_question)}

<b>Verdict:</b> {verdict.upper()}
<b>Current:</b> ${current_price:.2f}
<b>Fair:</b> ${fair_price:.2f}
<b>Edge:</b> {edge:+.1%}

<b>{side_emoji} Recommended:</b> {recommended_side.upper()}

<i>{_escape_html(reasoning_display)}</i>"""


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
    direction_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}
    impact_emoji = {"high": "🔥", "medium": "⚡", "low": "ℹ️"}

    direction = analysis.market_direction.value
    impact = analysis.predicted_impact.value

    # Truncate original message to ~300 chars
    original_text = message.text
    if len(original_text) > 300:
        original_text = original_text[:297] + "..."
        logger.debug(
            "Message truncated for condensed format",
            message_id=message.external_id,
            original_length=len(message.text),
        )

    # Build message
    msg = f"""📢 <b>SIGNAL</b> {direction_emoji.get(direction, "⚪")} {direction.upper()} | {impact_emoji.get(impact, "ℹ️")} {impact.upper()}

<i>{_escape_html(original_text)}</i>

💡 <b>Thesis:</b> {_escape_html(analysis.primary_thesis)}
<i>Confidence: {analysis.thesis_confidence:.0%}</i>"""

    # All tickers (one line each)
    if analysis.ticker_analyses:
        msg += "\n\n📊 <b>Tickers</b>"
        for ta in analysis.ticker_analyses:
            ticker_dir = direction_emoji.get(ta.net_direction.value, "⚪")
            company_name = f" - {_escape_html(ta.company_name)}" if ta.company_name else ""
            msg += f"\n{ticker_dir} <code>${ta.ticker}</code>{company_name} {ta.net_direction.value} ({ta.conviction:.0%})"
            if ta.relevance_reason:
                msg += f"\n   ↳ {_escape_html(ta.relevance_reason)}"

    # All sectors (one line each)
    if analysis.sector_implications:
        msg += "\n\n🏭 <b>Sectors</b>"
        for si in analysis.sector_implications:
            sec_dir = direction_emoji.get(si.direction.value, "⚪")
            msg += f"\n{sec_dir} {_escape_html(si.sector)} {si.direction.value}"

    # Historical context (full, no truncation)
    if analysis.historical_context:
        msg += f"\n\n📜 <b>Context:</b> <i>{_escape_html(analysis.historical_context)}</i>"

    # Single most relevant Polymarket (best by relevance, show even without edge)
    relevant_markets = [e for e in analysis.market_evaluations if e.is_relevant]
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


def format_combined_signal(
    message: "UnifiedMessage",
    extraction: "LightClassification",
    analysis: "SmartAnalysis",
) -> str:
    """Format a combined investment signal + polymarket edge for Telegram.

    Includes ALL available data without truncation for comprehensive notifications.

    Args:
        message: Original unified message with raw text
        extraction: Stage 1 entity extraction with metadata
        analysis: Stage 2 smart analysis with all judgments

    Returns:
        Formatted HTML message for Telegram (may be long, use send_long_telegram)
    """
    # Emoji mappings
    direction_emoji = {
        "bullish": "🟢",
        "bearish": "🔴",
        "neutral": "⚪",
    }
    impact_emoji = {
        "high": "🔥",
        "medium": "⚡",
        "low": "ℹ️",
    }
    urgency_emoji = {
        "critical": "🔥",
        "high": "⚡",
        "normal": "📌",
        "low": "💤",
    }
    category_emoji = {
        "breaking": "🔴 BREAKING",
        "economic_calendar": "📅 SCHEDULED",
        "other": "ℹ️ NEWS",
    }
    beat_miss_emoji = {
        "beat": "✅ BEAT",
        "miss": "❌ MISS",
        "inline": "➖ INLINE",
        "unknown": "❓",
    }
    research_quality_emoji = {
        "high": "🟢 HIGH",
        "medium": "🟡 MEDIUM",
        "low": "🔴 LOW",
    }

    direction = analysis.market_direction.value
    impact = analysis.predicted_impact.value
    urgency = extraction.urgency.value
    category = extraction.news_category.value

    # =========================================================================
    # HEADER
    # =========================================================================
    msg = f"""📢 <b>SIGNAL ALERT</b>
{SECTION_SEPARATOR}

📌 <b>METADATA</b>
Source: <code>{_escape_html(message.source_account)}</code>
Category: {category_emoji.get(category, category)} | Type: <code>{extraction.event_type.value}</code>
Urgency: {urgency_emoji.get(urgency, urgency)} {urgency.upper()}"""

    if extraction.urgency_reasoning:
        msg += f" - {_escape_html(extraction.urgency_reasoning)}"

    # =========================================================================
    # ORIGINAL MESSAGE
    # =========================================================================
    msg += f"""

📝 <b>ORIGINAL MESSAGE</b>
<i>{_escape_html(message.text)}</i>

📊 <b>SUMMARY</b>
{_escape_html(extraction.summary)}

💡 <b>THESIS</b>
{_escape_html(analysis.primary_thesis)} (confidence: {analysis.thesis_confidence:.0%})
Impact: {impact_emoji.get(impact, impact)} {impact.upper()} | Direction: {direction_emoji.get(direction, direction)} {direction.upper()}"""

    # =========================================================================
    # NUMERIC DATA (for economic/earnings)
    # =========================================================================
    if extraction.numeric_data and extraction.numeric_data.metrics:
        msg += f"""

{SECTION_SEPARATOR}

📈 <b>NUMERIC DATA</b>
┌{SUBSECTION_SEPARATOR}"""

        overall = extraction.numeric_data.overall_beat_miss.value
        msg += f"\n│ Overall: {beat_miss_emoji.get(overall, overall)}"

        for metric in extraction.numeric_data.metrics:
            status = beat_miss_emoji.get(metric.beat_miss.value, metric.beat_miss.value)
            line = f"\n│ {_escape_html(metric.metric_name)}: {metric.actual}{metric.unit}"
            if metric.estimate is not None:
                line += f" (Est: {metric.estimate}{metric.unit}) {status}"
            if metric.surprise_magnitude is not None:
                line += f"\n│   Surprise: {metric.surprise_magnitude:+}{metric.unit}"
            if metric.previous is not None:
                line += f" | Prev: {metric.previous}{metric.unit}"
            msg += line

        msg += f"\n└{SUBSECTION_SEPARATOR}"

    # =========================================================================
    # ENTITIES
    # =========================================================================
    if extraction.all_entities:
        entities_str = ", ".join(_escape_html(e) for e in extraction.all_entities)
        msg += f"""

{SECTION_SEPARATOR}

🏷️ <b>ENTITIES</b>
{entities_str}"""

    # =========================================================================
    # TICKERS (ALL, not top 3)
    # =========================================================================
    if analysis.ticker_analyses:
        msg += f"""

{SECTION_SEPARATOR}

📊 <b>TICKERS</b>"""

        for ta in analysis.ticker_analyses:
            ticker_dir = direction_emoji.get(ta.net_direction.value, "⚪")
            company_name = f" - {_escape_html(ta.company_name)}" if ta.company_name else ""
            msg += f"""
{SUBSECTION_SEPARATOR}
{ticker_dir} <code>${ta.ticker}</code>{company_name} ({ta.net_direction.value.upper()}, {ta.conviction:.0%} conviction)"""
            if ta.relevance_reason:
                msg += f"\n   ↳ {_escape_html(ta.relevance_reason)}"
            msg += f"""
   Time horizon: {ta.time_horizon}

   🟢 Bull: {_escape_html(ta.bull_thesis)}
   🔴 Bear: {_escape_html(ta.bear_thesis)}"""

            if ta.catalysts:
                msg += "\n\n   ✨ <b>Catalysts:</b>"
                for catalyst in ta.catalysts:
                    msg += f"\n   • {_escape_html(catalyst)}"

            if ta.risk_factors:
                msg += "\n\n   ⚠️ <b>Risk Factors:</b>"
                for risk in ta.risk_factors:
                    msg += f"\n   • {_escape_html(risk)}"

    # =========================================================================
    # SECTORS (ALL, not top 3)
    # =========================================================================
    if analysis.sector_implications:
        msg += f"""

{SECTION_SEPARATOR}

🏭 <b>SECTORS</b>"""

        for si in analysis.sector_implications:
            sec_dir = direction_emoji.get(si.direction.value, "⚪")
            msg += f"""
{SUBSECTION_SEPARATOR}
{sec_dir} <b>{_escape_html(si.sector)}</b>: {si.direction.value.upper()}
   {_escape_html(si.reasoning)}"""

            if si.subsectors:
                subsectors = ", ".join(_escape_html(s) for s in si.subsectors)
                msg += f"\n\n   Subsectors: {subsectors}"

    # =========================================================================
    # HISTORICAL CONTEXT (untruncated)
    # =========================================================================
    if analysis.historical_context:
        msg += f"""

{SECTION_SEPARATOR}

📜 <b>HISTORICAL CONTEXT</b>
<i>{_escape_html(analysis.historical_context)}</i>"""

        if analysis.typical_market_reaction:
            msg += f"""

<b>Typical reaction:</b> {_escape_html(analysis.typical_market_reaction)}"""

    # =========================================================================
    # POLYMARKET (all relevant markets, no edge filter)
    # =========================================================================
    relevant_markets = [e for e in analysis.market_evaluations if e.is_relevant]
    if relevant_markets:
        msg += f"""

{SECTION_SEPARATOR}

🎯 <b>POLYMARKET</b>"""

        # Sort by absolute edge (best first), but show all
        relevant_markets.sort(key=lambda e: abs(e.edge or 0), reverse=True)

        for i, mkt in enumerate(relevant_markets, 1):
            edge = mkt.edge or 0
            side = mkt.recommended_side
            confidence_pct = f"{mkt.confidence:.0%}"

            # Determine edge status
            if abs(edge) < 0.01:
                edge_label = "FAIR"
            elif edge > 0:
                edge_label = "UNDERVALUED"
            else:
                edge_label = "OVERVALUED"

            # Side emoji
            if side == "yes":
                side_emoji_char = "✅ YES"
            elif side == "no":
                side_emoji_char = "❌ NO"
            else:
                side_emoji_char = "⏭️ SKIP"

            # Fair price
            fair_price = mkt.estimated_fair_price if mkt.estimated_fair_price else mkt.current_price

            msg += f"""
{SUBSECTION_SEPARATOR}
<b>{i}.</b> {_escape_html(mkt.market_question)}
   Edge: {edge:+.1%} ({edge_label}) | Confidence: {confidence_pct}
   {side_emoji_char} @ ${mkt.current_price:.2f} → Fair: ${fair_price:.2f}

   <i>{_escape_html(mkt.reasoning)}</i>"""

    # =========================================================================
    # FOOTER
    # =========================================================================
    research_quality = analysis.research_quality.value
    msg += f"""

{SECTION_SEPARATOR}
Research Quality: {research_quality_emoji.get(research_quality, research_quality)}"""

    return msg


def format_sentiment_signal(signal: "SentimentSignal") -> str:
    """Format sentiment signal for Telegram.

    Args:
        signal: SentimentSignal with 6-hour sentiment data

    Returns:
        HTML-formatted message for Telegram
    """
    # Overall sentiment emoji mapping
    sentiment_emoji = {
        "bullish": "🟢 BULLISH",
        "bearish": "🔴 BEARISH",
        "neutral": "⚪ NEUTRAL",
        "mixed": "🟡 MIXED",
    }

    # Ticker sentiment emoji
    ticker_emoji = {
        "bullish": "🟢",
        "bearish": "🔴",
        "neutral": "⚪",
    }

    # Build header
    overall = signal.overall_sentiment
    msg = f"""📊 <b>REDDIT SENTIMENT ({signal.signal_period})</b> {sentiment_emoji.get(overall, overall)}
{SECTION_SEPARATOR}"""

    # Narrative section
    if signal.narrative_summary:
        msg += f"""

💬 <b>Narrative</b>
<i>{_escape_html(signal.narrative_summary)}</i>"""

    # Tickers section (sorted by mention count)
    if signal.ticker_sentiments:
        msg += """

📈 <b>Tickers</b> (by mention volume)"""

        # Sort by mention count descending
        sorted_tickers = sorted(
            signal.ticker_sentiments,
            key=lambda t: t.mention_count,
            reverse=True,
        )

        for ts in sorted_tickers:
            # Determine sentiment label from avg_sentiment
            if ts.avg_sentiment > 0.1:
                sentiment_label = "bullish"
            elif ts.avg_sentiment < -0.1:
                sentiment_label = "bearish"
            else:
                sentiment_label = "neutral"

            emoji = ticker_emoji.get(sentiment_label, "⚪")
            company_part = f" - {_escape_html(ts.company_name)}" if ts.company_name else ""

            msg += f"""
{emoji} <code>${ts.ticker}</code>{company_part} ({ts.mention_count} mentions) {sentiment_label}"""

            # Add catalysts if present
            if ts.key_catalysts:
                catalysts_str = ", ".join(_escape_html(c) for c in ts.key_catalysts)
                msg += f"""
   ↳ Catalysts: {catalysts_str}"""

            # Add extreme sentiment badge
            if ts.is_extreme_bullish:
                msg += """
   🔥 EXTREME BULLISH"""
            elif ts.is_extreme_bearish:
                msg += """
   🔥 EXTREME BEARISH"""

    # Watchlist changes section (only if there are changes)
    if signal.watchlist_added or signal.watchlist_removed:
        msg += """

📋 <b>Watchlist Changes</b>"""
        if signal.watchlist_added:
            added_str = ", ".join(signal.watchlist_added)
            msg += f"""
➕ Added: {added_str}"""
        if signal.watchlist_removed:
            removed_str = ", ".join(signal.watchlist_removed)
            msg += f"""
➖ Removed: {removed_str}"""

    # Key themes section
    if signal.key_themes:
        msg += """

🏷️ <b>Key Themes</b>"""
        for theme in signal.key_themes:
            msg += f"""
• {_escape_html(theme)}"""

    # Stats section (condensed)
    msg += f"""

📊 <b>Stats</b>
Posts: {signal.total_posts_analyzed} analyzed | {signal.high_quality_posts} high quality | {signal.spam_posts} spam filtered"""

    # Subreddit breakdown
    if signal.subreddits:
        sorted_subs = sorted(signal.subreddits.items(), key=lambda x: x[1], reverse=True)
        subs_str = ", ".join(f"r/{sub} ({count})" for sub, count in sorted_subs)
        msg += f"""
Sources: {subs_str}"""

    # Footer
    msg += f"""
{SECTION_SEPARATOR}"""

    return msg
