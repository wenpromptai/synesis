"""Discord embed formatter for the daily intelligence brief."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from synesis.core.constants import (
    COLOR_BEARISH,
    COLOR_BULLISH,
    COLOR_HEADER,
    COLOR_NEUTRAL,
)

# Macro regime → embed color
_REGIME_COLORS: dict[str, int] = {
    "risk_on": COLOR_BULLISH,
    "risk_off": COLOR_BEARISH,
    "transitioning": COLOR_NEUTRAL,
    "uncertain": COLOR_HEADER,
}

# Macro regime → display label
_REGIME_LABELS: dict[str, str] = {
    "risk_on": "\U0001f7e2 Risk-On",
    "risk_off": "\U0001f534 Risk-Off",
    "transitioning": "\U0001f7e1 Transitioning",
    "uncertain": "\u26aa Uncertain",
}


def format_intelligence_brief(brief: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Format compiled brief into Discord embed batches.

    Returns a list of embed batches (each batch <= 10 embeds) to send
    as separate webhook calls.
    """
    embeds: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()

    # ── Embed 1: Header + Macro ─────────────────────────────────
    macro = brief.get("macro", {})
    regime = macro.get("regime", "uncertain")
    color = _REGIME_COLORS.get(regime, COLOR_HEADER)

    brief_date = brief.get("date", "")
    try:
        dt = datetime.strptime(brief_date, "%Y-%m-%d")
        display_date = f"{dt.strftime('%b')} {dt.day}, {dt.year}"
    except (ValueError, TypeError):
        display_date = brief_date

    macro_fields: list[dict[str, Any]] = [
        {
            "name": "Regime",
            "value": _REGIME_LABELS.get(regime, regime),
            "inline": True,
        },
    ]

    sentiment = macro.get("sentiment_score")
    if sentiment is not None:
        macro_fields.append({"name": "Sentiment", "value": f"{sentiment:+.2f}", "inline": True})

    drivers = macro.get("key_drivers", [])
    if drivers:
        macro_fields.append(
            {
                "name": "Key Drivers",
                "value": "\n".join(f"\u2022 {d}" for d in drivers[:5])[:1024],
                "inline": False,
            }
        )

    tilts = macro.get("sector_tilts", [])
    if tilts:
        tilt_lines = []
        for t in tilts[:6]:
            score = t.get("sentiment_score", 0)
            emoji = "\U0001f7e2" if score > 0 else "\U0001f534" if score < 0 else "\u26aa"
            tilt_lines.append(f"{emoji} **{t.get('sector', '?')}** ({score:+.1f})")
        macro_fields.append(
            {"name": "Sector Tilts", "value": "\n".join(tilt_lines)[:1024], "inline": False}
        )

    risks = macro.get("risks", [])
    if risks:
        macro_fields.append(
            {
                "name": "Risks",
                "value": "\n".join(f"\u26a0\ufe0f {r}" for r in risks[:4])[:1024],
                "inline": False,
            }
        )

    embeds.append(
        {
            "title": f"\U0001f4ca Daily Brief \u2014 {display_date}",
            "color": color,
            "fields": macro_fields,
            "timestamp": now,
        }
    )

    # ── Layer 1 summaries ───────────────────────────────────────
    l1 = brief.get("l1_summary", {})
    social_summary = l1.get("social", "")
    news_summary = l1.get("news", "")
    if social_summary or news_summary:
        l1_fields: list[dict[str, Any]] = []
        if social_summary:
            l1_fields.append({"name": "Social", "value": social_summary[:1024], "inline": False})
        if news_summary:
            l1_fields.append({"name": "News", "value": news_summary[:1024], "inline": False})
        embeds.append(
            {
                "title": "\U0001f4e1 Signal Summary",
                "color": COLOR_HEADER,
                "fields": l1_fields,
            }
        )

    # ── Per-ticker Debate embeds ────────────────────────────────
    debates = brief.get("debates", [])
    trade_ideas = brief.get("trade_ideas", [])

    # Index single-ticker trade ideas by ticker (multi-ticker shown in portfolio section)
    ideas_by_ticker: dict[str, list[dict[str, Any]]] = {}
    for idea in trade_ideas:
        tickers_list = idea.get("tickers", [])
        if len(tickers_list) == 1:
            ideas_by_ticker.setdefault(tickers_list[0], []).append(idea)

    for debate in debates:
        ticker = debate.get("ticker", "?")
        fields: list[dict[str, Any]] = []

        bull = debate.get("bull", {})
        bear = debate.get("bear", {})

        # Bull argument — single argument string + key_evidence list
        bull_arg = bull.get("argument", "")
        if bull_arg:
            bull_text = _format_debate_side(bull_arg, bull.get("key_evidence", []))
            fields.append(
                {"name": "\U0001f7e2 Bull Case", "value": bull_text[:1024], "inline": False}
            )

        # Bear argument
        bear_arg = bear.get("argument", "")
        if bear_arg:
            bear_text = _format_debate_side(bear_arg, bear.get("key_evidence", []))
            fields.append(
                {"name": "\U0001f534 Bear Case", "value": bear_text[:1024], "inline": False}
            )

        # Trade ideas for this ticker (single-ticker only)
        ticker_ideas = ideas_by_ticker.get(ticker, [])
        for idea in ticker_ideas:
            idea_text = f"**{idea.get('trade_structure', '')}**"
            thesis = idea.get("thesis", "")
            if thesis:
                idea_text += f"\n{thesis}"
            meta_parts = []
            catalyst = idea.get("catalyst", "")
            if catalyst:
                meta_parts.append(f"\U0001f4a5 {catalyst}")
            timeframe = idea.get("timeframe", "")
            if timeframe:
                meta_parts.append(f"\u23f0 {timeframe}")
            key_risk = idea.get("key_risk", "")
            if key_risk:
                meta_parts.append(f"\u26a0\ufe0f {key_risk}")
            if meta_parts:
                idea_text += "\n" + " \u2022 ".join(meta_parts)
            fields.append(
                {"name": "\U0001f4a1 Trade Idea", "value": idea_text[:1024], "inline": False}
            )

        embeds.append(
            {
                "title": f"\u2694\ufe0f {ticker}",
                "color": color,
                "fields": fields[:25],
            }
        )

    # ── Portfolio-level trade ideas (multi-ticker) ──────────────
    portfolio_ideas = [idea for idea in trade_ideas if len(idea.get("tickers", [])) > 1]
    if portfolio_ideas:
        pf_fields: list[dict[str, Any]] = []
        for idea in portfolio_ideas:
            tickers_str = " / ".join(idea.get("tickers", []))
            text = f"**{idea.get('trade_structure', '')}**"
            catalyst = idea.get("catalyst", "")
            if catalyst:
                text += f"\nCatalyst: {catalyst}"
            timeframe = idea.get("timeframe", "")
            if timeframe:
                text += f"\n\u23f0 {timeframe}"
            pf_fields.append({"name": tickers_str, "value": text[:1024], "inline": False})
        embeds.append(
            {
                "title": "\U0001f4bc Portfolio Ideas",
                "color": color,
                "fields": pf_fields[:25],
            }
        )

    # ── Errors footer (if any) ──────────────────────────────────
    errors = brief.get("errors", {})
    error_lines = _format_errors(errors)
    if error_lines:
        embeds.append(
            {
                "color": COLOR_BEARISH,
                "description": ("\u26a0\ufe0f **Pipeline Errors**\n" + "\n".join(error_lines))[
                    :4096
                ],
                "footer": {"text": "Synesis Intelligence"},
            }
        )

    return _split_into_batches(embeds)


# Discord limits
_MAX_EMBEDS_PER_MSG = 10
_MAX_CHARS_PER_MSG = 6000


def _embed_char_count(embed: dict[str, Any]) -> int:
    """Count characters that Discord counts toward the 6000-char message limit."""
    total = len(embed.get("title", ""))
    total += len(embed.get("description", ""))
    total += len(embed.get("footer", {}).get("text", ""))
    total += len(embed.get("author", {}).get("name", ""))
    for field in embed.get("fields", []):
        total += len(field.get("name", ""))
        total += len(field.get("value", ""))
    return total


def _split_into_batches(embeds: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split embeds into batches respecting Discord limits.

    Each batch has at most 10 embeds and at most 6000 total characters.
    """
    if not embeds:
        return []

    batches: list[list[dict[str, Any]]] = []
    current_batch: list[dict[str, Any]] = []
    current_chars = 0

    for embed in embeds:
        chars = _embed_char_count(embed)
        would_exceed_chars = current_chars + chars > _MAX_CHARS_PER_MSG
        would_exceed_count = len(current_batch) >= _MAX_EMBEDS_PER_MSG

        if current_batch and (would_exceed_chars or would_exceed_count):
            batches.append(current_batch)
            current_batch = []
            current_chars = 0

        current_batch.append(embed)
        current_chars += chars

    if current_batch:
        batches.append(current_batch)

    return batches


def _format_debate_side(argument: str, key_evidence: list[str]) -> str:
    """Format a single debate side (argument + evidence) into a readable string."""
    lines = [argument]
    for ev in key_evidence[:4]:
        lines.append(f"\u203a {ev}")
    return "\n".join(lines)


def _format_errors(errors: dict[str, Any]) -> list[str]:
    """Format pipeline errors into human-readable lines."""
    lines: list[str] = []
    if errors.get("social_failed"):
        lines.append("\u2022 Social analysis failed")
    if errors.get("news_failed"):
        lines.append("\u2022 News analysis failed")
    if errors.get("macro_failed"):
        lines.append("\u2022 Macro analysis failed")
    for label, key in [
        ("Company", "company_failures"),
        ("Price", "price_failures"),
        ("Bull", "bull_failures"),
        ("Bear", "bear_failures"),
        ("Trader", "trader_failures"),
    ]:
        tickers = errors.get(key, [])
        if tickers:
            lines.append(f"\u2022 {label} failed: {', '.join(tickers)}")
    return lines
