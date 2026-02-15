"""Telegram notification service.

Provides simple async function to send messages to a Telegram chat.
Used to notify about investment signals and prediction opportunities.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx

from synesis.config import get_settings
from synesis.core.constants import TELEGRAM_MAX_MESSAGE_LENGTH
from synesis.core.logging import get_logger

if TYPE_CHECKING:
    from synesis.markets.models import CrossPlatformArb
    from synesis.processing.mkt_intel.models import MarketIntelSignal
    from synesis.processing.news import LightClassification, SmartAnalysis, UnifiedMessage
    from synesis.processing.sentiment import SentimentSignal
    from synesis.processing.watchlist.models import WatchlistSignal

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
    sentiment_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}

    sentiment = analysis.sentiment.value

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
    msg = f"""📢 <b>SIGNAL</b> {sentiment_emoji.get(sentiment, "⚪")} {sentiment.upper()} ({analysis.sentiment_score:+.2f})

<blockquote>{_escape_html(original_text)}</blockquote>

💡 <b>Thesis:</b> {_escape_html(analysis.primary_thesis)}
<i>Confidence: {analysis.thesis_confidence:.0%}</i>"""

    # All tickers (one line each)
    if analysis.ticker_analyses:
        msg += "\n\n📊 <b>Tickers</b>"
        for ta in analysis.ticker_analyses:
            ticker_dir = sentiment_emoji.get(ta.net_direction.value, "⚪")
            company_name = f" - {_escape_html(ta.company_name)}" if ta.company_name else ""
            msg += f"\n{ticker_dir} <code>${ta.ticker}</code>{company_name} {ta.net_direction.value} ({ta.conviction:.0%})"
            if ta.relevance_reason:
                msg += f"\n   ↳ {_escape_html(ta.relevance_reason)}"

    # All sectors (one line each)
    if analysis.sector_implications:
        msg += "\n\n🏭 <b>Sectors</b>"
        for si in analysis.sector_implications:
            sec_dir = sentiment_emoji.get(si.direction.value, "⚪")
            msg += f"\n{sec_dir} {_escape_html(si.sector)} {si.direction.value}"

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
    sentiment_emoji = {
        "bullish": "🟢",
        "bearish": "🔴",
        "neutral": "⚪",
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

    sentiment = analysis.sentiment.value
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
<blockquote>{_escape_html(message.text)}</blockquote>

📊 <b>SUMMARY</b>
{_escape_html(extraction.summary)}

💡 <b>THESIS</b>
{_escape_html(analysis.primary_thesis)} (confidence: {analysis.thesis_confidence:.0%})
Sentiment: {sentiment_emoji.get(sentiment, sentiment)} {sentiment.upper()} ({analysis.sentiment_score:+.2f})"""

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
            ticker_dir = sentiment_emoji.get(ta.net_direction.value, "⚪")
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
            sec_dir = sentiment_emoji.get(si.direction.value, "⚪")
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
    relevant_markets = [
        e for e in analysis.market_evaluations if e.is_relevant and e.confidence >= 0.6
    ]
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
    msg = f"""📊 <b>REDDIT SENTIMENT ({signal.signal_period})</b> {sentiment_emoji.get(overall, overall)}"""

    # Narrative section
    if signal.narrative_summary:
        msg += f"""

💬 <b>Narrative</b>
<i>{_escape_html(signal.narrative_summary)}</i>"""

    # Tickers section (sorted by mention count)
    if signal.ticker_sentiments:
        msg += f"""

{SECTION_SEPARATOR}
📈 <b>Tickers</b> (by mention volume)"""

        # Sort by mention count descending
        sorted_tickers = sorted(
            signal.ticker_sentiments,
            key=lambda t: t.mention_count,
            reverse=True,
        )

        # Split into shown (actionable) vs collapsed (noise) tickers
        shown = [t for t in sorted_tickers if t.mention_count >= 2 or abs(t.avg_sentiment) > 0.1]
        collapsed_count = len(sorted_tickers) - len(shown)

        for ts in shown:
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
            if ts.bullish_ratio >= 0.85:
                msg += """
   🔥 EXTREME BULLISH"""
            elif ts.bearish_ratio >= 0.85:
                msg += """
   🔥 EXTREME BEARISH"""

        if collapsed_count:
            msg += f"""
⚪ <i>+ {collapsed_count} other tickers (1 mention, neutral)</i>"""

    # Watchlist changes section (only if there are changes)
    if signal.watchlist_added or signal.watchlist_removed:
        msg += f"""

{SECTION_SEPARATOR}
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

{SECTION_SEPARATOR}
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


def format_arb_alert(arb: "CrossPlatformArb") -> str:
    """Format a real-time cross-platform arbitrage alert for Telegram."""
    return f"""💱 <b>ARB ALERT</b>

<b>"{_escape_html(arb.polymarket.question)}"</b>
Polymarket: ${arb.polymarket.yes_price:.2f} | Kalshi: ${arb.kalshi.yes_price:.2f} | Gap: ${arb.price_gap:.2f}
→ Buy {arb.suggested_side.upper()} on {arb.suggested_buy_platform.title()}
Match confidence: {arb.match_similarity:.0%}"""


def format_mkt_intel_signal(signal: "MarketIntelSignal") -> str:
    """Format market intelligence signal for Telegram.

    Args:
        signal: MarketIntelSignal with scan results

    Returns:
        HTML-formatted message for Telegram
    """
    # Header
    ws_status = "🟢 LIVE" if signal.ws_connected else "🔴 REST-only"
    msg = f"""🎯 <b>MARKET INTEL ({signal.signal_period})</b> {ws_status}
Markets scanned: {signal.total_markets_scanned}"""

    # Opportunities (top 5)
    if signal.opportunities:
        msg += f"""

{SECTION_SEPARATOR}
💰 <b>Opportunities</b>"""
        for i, opp in enumerate(signal.opportunities[:5], 1):
            direction = "✅ YES" if opp.suggested_direction == "yes" else "❌ NO"
            triggers = ", ".join(opp.triggers)
            question = _escape_html(opp.market.question)
            if opp.market.outcome_label:
                question += f" → {_escape_html(opp.market.outcome_label)}"
            price_str = f"${opp.market.yes_price:.2f}"
            if opp.market.yes_outcome:
                price_str = f"{_escape_html(opp.market.yes_outcome)} {price_str}"
            msg += f"""
{i}. {question}
   {direction} @ {price_str} | Conf: {opp.confidence:.0%}
   Triggers: {triggers}"""
            if opp.reasoning:
                msg += f"""
   ↳ {_escape_html(opp.reasoning[:150])}"""

    # Odds movements
    if signal.odds_movements:
        msg += f"""

{SECTION_SEPARATOR}
📊 <b>Odds Movements</b>"""
        for om in signal.odds_movements[:5]:
            arrow = "⬆️" if om.direction == "up" else "⬇️"
            question = _escape_html(om.market.question)
            if om.market.outcome_label:
                question += f" → {_escape_html(om.market.outcome_label)}"
            current = om.market.yes_price
            prev_1h = current - om.price_change_1h
            changes = f"YES {prev_1h:.0%} → {current:.0%} (1h)"
            if om.price_change_6h is not None:
                prev_6h = current - om.price_change_6h
                changes += f" | YES {prev_6h:.0%} → {current:.0%} (6h)"
            msg += f"""
{arrow} {question}
   {changes}"""

    # Volume spikes
    if signal.volume_spikes:
        msg += f"""

{SECTION_SEPARATOR}
📈 <b>Volume Spikes</b>"""
        for vs in signal.volume_spikes[:5]:
            question = _escape_html(vs.market.question)
            if vs.market.outcome_label:
                question += f" → {_escape_html(vs.market.outcome_label)}"
            msg += f"""
{question}
   +{vs.pct_change:.0%} ({vs.volume_previous:,.0f} → {vs.volume_current:,.0f})"""

    # Insider activity
    if signal.insider_activity:
        msg += f"""

{SECTION_SEPARATOR}
🕵️ <b>Insider Activity</b>"""
        for ia in signal.insider_activity[:5]:
            specialty = f" [{_escape_html(ia.wallet_specialty)}]" if ia.wallet_specialty else ""
            reason_tag = " ⚡" if ia.watch_reason == "high_conviction" else ""
            direction = (
                "✅ YES"
                if ia.trade_direction == "yes"
                else "❌ NO"
                if ia.trade_direction == "no"
                else ""
            )
            direction_str = f" | {direction}" if direction else ""
            msg += f"""
👤 {ia.wallet_address[:8]}...{specialty} on {_escape_html(ia.market.question)}
   Score: {ia.insider_score:.2f}{reason_tag}{direction_str} ${ia.trade_size:,.0f}"""

    # Cross-platform arbs omitted — pushed live via format_arb_alert()

    # High-conviction trades
    if signal.high_conviction_trades:
        msg += f"""

{SECTION_SEPARATOR}
🎯 <b>High-Conviction Trades</b>"""
        for hc in signal.high_conviction_trades[:5]:
            specialty = f" [{_escape_html(hc.wallet_specialty)}]" if hc.wallet_specialty else ""
            hc_dir = (
                "✅ YES"
                if hc.trade_direction == "yes"
                else "❌ NO"
                if hc.trade_direction == "no"
                else ""
            )
            hc_dir_str = f" | {hc_dir}" if hc_dir else ""
            entry_parts: list[str] = []
            if hc.avg_entry_price > 0:
                entry_parts.append(f"entered @ {hc.avg_entry_price:.0%}")
            if hc.entry_cost > 0:
                entry_parts.append(f"${hc.entry_cost:,.0f}")
            if hc.entry_date:
                delta = datetime.now(UTC) - hc.entry_date
                days = delta.days
                if days >= 1:
                    entry_parts.append(f"{days}d ago")
                else:
                    hours = int(delta.total_seconds() / 3600)
                    entry_parts.append(f"{hours}h ago")
            entry_str = f" ({', '.join(entry_parts)})" if entry_parts else ""
            msg += f"""
👤 {hc.wallet_address[:8]}...{specialty} on {_escape_html(hc.market.question)}
   {hc_dir_str} ${hc.position_size:,.0f}{entry_str} ({hc.concentration_pct:.0%} of portfolio, {hc.total_positions} positions)"""

    # Expiring markets (only show markets that haven't expired yet)
    if signal.expiring_soon:
        expiring_lines: list[str] = []
        for m in signal.expiring_soon[:10]:
            if m.end_date:
                hours_left = (m.end_date - datetime.now(UTC)).total_seconds() / 3600
                if hours_left > 0:
                    question = _escape_html(m.question)
                    if m.outcome_label:
                        question += f" → {_escape_html(m.outcome_label)}"
                    price_str = f"${m.yes_price:.2f}"
                    if m.yes_outcome:
                        price_str = f"{_escape_html(m.yes_outcome)} {price_str}"
                    expiring_lines.append(f"\n{question} @ {price_str} ({hours_left:.1f}h left)")
        if expiring_lines:
            msg += f"""

{SECTION_SEPARATOR}
⏰ <b>Expiring Soon</b>"""
            for line in expiring_lines[:5]:
                msg += line

    return msg


def format_watchlist_signal(signal: "WatchlistSignal") -> str:
    """Format watchlist intelligence signal for Telegram.

    Args:
        signal: WatchlistSignal with fundamental analysis results

    Returns:
        HTML-formatted message for Telegram
    """
    outlook_emoji = {
        "bullish": "🟢",
        "bearish": "🔴",
        "neutral": "⚪",
    }
    severity_emoji = {
        "high": "🔴",
        "medium": "🟡",
        "low": "⚪",
    }

    msg = f"""📋 <b>WATCHLIST INTEL</b>
Tickers analyzed: {signal.tickers_analyzed}"""

    # Alerts section
    if signal.alerts:
        msg += f"""

{SECTION_SEPARATOR}
⚡ <b>Alerts</b>"""
        for alert in signal.alerts:
            sev = severity_emoji.get(alert.severity, "⚪")
            msg += f"""
{sev} <code>${alert.ticker}</code> — {alert.alert_type.replace("_", " ")}
   {_escape_html(alert.summary)}"""

    # Ticker reports section
    if signal.ticker_reports:
        msg += f"""

{SECTION_SEPARATOR}
📊 <b>Ticker Reports</b>"""
        for report in signal.ticker_reports:
            emoji = outlook_emoji.get(report.overall_outlook, "⚪")
            company = f" - {_escape_html(report.company_name)}" if report.company_name else ""
            msg += f"""
{SUBSECTION_SEPARATOR}
{emoji} <code>${report.ticker}</code>{company} ({report.overall_outlook.upper()}, {report.confidence:.0%})
{_escape_html(report.fundamental_summary)}"""
            if report.catalyst_flags:
                cats = ", ".join(_escape_html(c) for c in report.catalyst_flags)
                msg += f"""
   ✨ Catalysts: {cats}"""
            if report.risk_flags:
                risks = ", ".join(_escape_html(r) for r in report.risk_flags)
                msg += f"""
   ⚠️ Risks: {risks}"""

    # Summary
    if signal.summary:
        msg += f"""

{SECTION_SEPARATOR}
💬 <i>{_escape_html(signal.summary)}</i>"""

    return msg
