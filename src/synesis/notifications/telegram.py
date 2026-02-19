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
    from synesis.processing.news import SmartAnalysis, UnifiedMessage
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
