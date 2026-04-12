"""Discord embed formatting for market movers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from synesis.core.constants import COLOR_HEADER
from synesis.processing.market.models import MarketMoversData, TickerChange
from synesis.providers.yfinance.models import MarketMover

# Discord limits
_FIELD_VALUE_LIMIT = 1024
_EMBED_TOTAL_LIMIT = 6000
_EMBED_MAX_FIELDS = 25


def format_market_movers_embeds(brief: MarketMoversData) -> list[list[dict[str, Any]]]:
    """Format MarketMoversData into Discord embed messages.

    Splits across multiple messages if the total embed size exceeds Discord's
    6000-char limit.  Returns a list of messages (each message is a list of embeds).
    """
    now_iso = datetime.now(UTC).isoformat()
    today_str = datetime.now(UTC).strftime("%b %d")
    title = f"\U0001f4ca Market Movers \u2014 {today_str}"

    fields: list[dict[str, Any]] = []

    # Equities — each on its own line with price
    eq_val = _format_ticker_lines(brief.equities)
    if eq_val:
        fields.append(
            {"name": "\U0001f4c8 Equities", "value": eq_val[:_FIELD_VALUE_LIMIT], "inline": True}
        )

    # Rates / FX
    rf_val = _format_ticker_lines(brief.rates_fx)
    if rf_val:
        fields.append(
            {"name": "\U0001f4b5 Rates / FX", "value": rf_val[:_FIELD_VALUE_LIMIT], "inline": True}
        )

    # Commodities
    cm_val = _format_ticker_lines(brief.commodities)
    if cm_val:
        fields.append(
            {
                "name": "\U0001f6e2\ufe0f Commodities",
                "value": cm_val[:_FIELD_VALUE_LIMIT],
                "inline": True,
            }
        )

    # Volatility
    if brief.volatility is not None:
        vix_last = brief.volatility.last
        vix_prev = brief.volatility.prev_close
        if vix_last is not None:
            if vix_prev is not None:
                change = vix_last - vix_prev
                sign = "+" if change >= 0 else ""
                vix_val = f"VIX **{vix_last:.2f}** ({sign}{change:.2f})"
            else:
                vix_val = f"VIX **{vix_last:.2f}**"
            fields.append({"name": "\U0001f321\ufe0f Volatility", "value": vix_val, "inline": True})

    # Spacer for alignment
    fields.append({"name": "\u200b", "value": "\u200b", "inline": True})

    # Sectors — each on its own row
    sector_val = _format_sector_lines(brief.sectors)
    if sector_val:
        fields.append(
            {
                "name": "\U0001f4ca Sectors",
                "value": sector_val[:_FIELD_VALUE_LIMIT],
                "inline": False,
            }
        )

    # Top Movers
    movers = brief.movers
    if movers.gainers:
        val = _format_movers(movers.gainers[:10])
        if val:
            fields.append(
                {
                    "name": "\U0001f7e2 Top Gainers",
                    "value": val[:_FIELD_VALUE_LIMIT],
                    "inline": False,
                }
            )

    if movers.losers:
        val = _format_movers(movers.losers[:10])
        if val:
            fields.append(
                {
                    "name": "\U0001f534 Top Losers",
                    "value": val[:_FIELD_VALUE_LIMIT],
                    "inline": False,
                }
            )

    if movers.most_actives:
        val = _format_movers(movers.most_actives[:10])
        if val:
            fields.append(
                {
                    "name": "\U0001f525 Most Active",
                    "value": val[:_FIELD_VALUE_LIMIT],
                    "inline": False,
                }
            )

    # Split fields across multiple embeds if total size exceeds Discord limit
    return _split_into_messages(fields, title, now_iso)


def _embed_size(embed: dict[str, Any]) -> int:
    """Estimate total character count for a Discord embed."""
    size = len(embed.get("title", ""))
    size += len(embed.get("description", ""))
    footer = embed.get("footer")
    if footer:
        size += len(footer.get("text", ""))
    for field in embed.get("fields", []):
        size += len(field.get("name", ""))
        size += len(field.get("value", ""))
    return size


def _split_into_messages(
    fields: list[dict[str, Any]],
    title: str,
    now_iso: str,
) -> list[list[dict[str, Any]]]:
    """Pack fields into embeds, splitting when Discord limits are reached."""
    footer_text = "Synesis Market Movers"
    # Overhead from title + footer + timestamp (only on first/last embed)
    base_overhead = len(title) + len(footer_text) + 50  # buffer

    messages: list[list[dict[str, Any]]] = []
    current_fields: list[dict[str, Any]] = []
    current_size = base_overhead

    for field in fields:
        field_size = len(field.get("name", "")) + len(field.get("value", ""))

        # Would adding this field exceed the embed limit or field count?
        if current_fields and (
            current_size + field_size > _EMBED_TOTAL_LIMIT
            or len(current_fields) >= _EMBED_MAX_FIELDS
        ):
            # Flush current embed
            embed: dict[str, Any] = {
                "color": COLOR_HEADER,
                "fields": current_fields,
                "timestamp": now_iso,
            }
            if not messages:
                embed["title"] = title
            messages.append([embed])
            current_fields = []
            current_size = len(footer_text) + 50

        current_fields.append(field)
        current_size += field_size

    # Flush remaining fields
    if current_fields:
        embed = {
            "color": COLOR_HEADER,
            "fields": current_fields,
            "footer": {"text": footer_text},
            "timestamp": now_iso,
        }
        if not messages:
            embed["title"] = title
        messages.append([embed])

    # If we split, put the footer only on the last message
    if len(messages) > 1:
        for msg in messages[:-1]:
            msg[0].pop("footer", None)

    return messages


def _format_ticker_lines(tickers: list[TickerChange]) -> str:
    """Format tickers as rows: `$SPY` **$520.00** +0.97%."""
    lines: list[str] = []
    for tc in tickers:
        if tc.change_pct is None and tc.last is None:
            continue
        parts: list[str] = [f"`${tc.ticker}`"]
        if tc.last is not None:
            parts.append(f"**${tc.last:.2f}**")
        if tc.change_pct is not None:
            sign = "+" if tc.change_pct >= 0 else ""
            parts.append(f"{sign}{tc.change_pct:.1f}%")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _format_sector_lines(sectors: list[TickerChange]) -> str:
    """Format sectors as rows: Tech `$XLK` **$200.00** +1.5%."""
    lines: list[str] = []
    for tc in sectors:
        if tc.change_pct is None and tc.last is None:
            continue
        label = tc.label or tc.ticker
        parts: list[str] = [f"{label} `${tc.ticker}`"]
        if tc.last is not None:
            parts.append(f"**${tc.last:.2f}**")
        if tc.change_pct is not None:
            sign = "+" if tc.change_pct >= 0 else ""
            parts.append(f"{sign}{tc.change_pct:.1f}%")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _format_movers(movers: list[MarketMover]) -> str:
    """Format movers as rows with company name and sector."""
    lines: list[str] = []
    for m in movers:
        if m.change_pct is None:
            continue
        sign = "+" if m.change_pct >= 0 else ""
        parts: list[str] = [f"`${m.ticker}`"]
        if m.name:
            parts.append(f"{m.name}")
        if m.price is not None:
            parts.append(f"**${m.price:.2f}**")
        parts.append(f"{sign}{m.change_pct:.1f}%")
        if m.sector:
            parts.append(f"({m.sector})")
        lines.append(" ".join(parts))
    return "\n".join(lines)
