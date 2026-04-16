"""Discord embed formatter for intelligence briefs."""

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


def _build_watchlist_embed(brief: dict[str, Any], color: int) -> dict[str, Any] | None:
    """Build a watchlist Discord embed from the brief's watchlist data.

    Returns None if the watchlist has no selected picks.
    """
    watchlist = brief.get("watchlist", {})
    picks = watchlist.get("selected", [])
    if not picks:
        return None

    l1_pool = watchlist.get("l1_tickers", [])
    pick_lines = []
    for pick in picks:
        direction = pick.get("direction_lean", "?")
        emoji = "\U0001f7e2" if direction == "bullish" else "\U0001f534"
        wildcard = " \u26a1 wildcard" if pick.get("is_wildcard") else ""
        line = f"{emoji} **{pick.get('ticker', '?')}** \u2014 {pick.get('thematic_angle', '')}{wildcard}"
        note = pick.get("signal_strength", "")
        if note:
            line += f"\n   {note}"
        pick_lines.append(line)

    fields: list[dict[str, Any]] = _split_field("Tickers", "\n".join(pick_lines))

    dropped = watchlist.get("dropped", [])
    drop_reasons = watchlist.get("drop_reasons", [])
    if dropped:
        drop_parts = []
        for i, t in enumerate(dropped):
            reason = drop_reasons[i] if i < len(drop_reasons) else ""
            drop_parts.append(f"{t} ({reason})" if reason else t)
        fields.extend(_split_field("Dropped", ", ".join(drop_parts)))

    title_str = f"\U0001f4cb Watchlist \u2014 {len(picks)} tickers ({len(l1_pool)} from signals)"
    embed: dict[str, Any] = {
        "title": title_str,
        "color": color,
        "fields": fields,
    }
    themes = watchlist.get("themes", [])
    if themes:
        embed["description"] = " | ".join(themes[:5])
    return embed


def format_scan_brief(brief: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Format scan brief into Discord embed batches.

    Scan-only output: macro regime, signal summaries, and watchlist.
    No debates or trade ideas.
    """
    embeds: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()

    macro = brief.get("macro", {})
    regime = macro.get("regime", "uncertain")
    color = _REGIME_COLORS.get(regime, COLOR_HEADER)

    brief_date = brief.get("date", "")
    try:
        dt = datetime.strptime(brief_date, "%Y-%m-%d")
        display_date = f"{dt.strftime('%b')} {dt.day}, {dt.year}"
    except (ValueError, TypeError):
        display_date = brief_date

    # ── Header + Macro ─────────────────────────────────────────
    macro_fields: list[dict[str, Any]] = [
        {"name": "Regime", "value": _REGIME_LABELS.get(regime, regime), "inline": True},
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
    tilts = macro.get("thematic_tilts", [])
    if tilts:
        tilt_lines = []
        for t in tilts[:15]:
            score = t.get("sentiment_score", 0)
            emoji = "\U0001f7e2" if score > 0 else "\U0001f534" if score < 0 else "\u26aa"
            etf = t.get("etf")
            etf_str = f" [{etf}]" if etf else ""
            tilt_lines.append(f"{emoji} **{t.get('theme', '?')}**{etf_str} ({score:+.1f})")
        macro_fields.extend(_split_field("Thematic Tilts", "\n".join(tilt_lines)))
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
            "title": f"\U0001f4e1 Market Scan \u2014 {display_date}",
            "color": color,
            "fields": macro_fields,
            "timestamp": now,
        }
    )

    # ── L1 Signal Summary ──────────────────────────────────────
    l1 = brief.get("l1_summary", {})
    social_summary = l1.get("social", "")
    news_summary = l1.get("news", "")
    if social_summary or news_summary:
        l1_fields: list[dict[str, Any]] = []
        if social_summary:
            l1_fields.extend(_split_field("Social", social_summary))
        if news_summary:
            l1_fields.extend(_split_field("News", news_summary))
        embeds.append(
            {"title": "\U0001f4e1 Signal Summary", "color": COLOR_HEADER, "fields": l1_fields}
        )

    # ── Thematic Research ─────────────────────────────────────
    research = brief.get("social_research_context", [])
    discovered = brief.get("social_discovered_themes", [])
    if research or discovered:
        research_fields: list[dict[str, Any]] = []
        if research:
            research_fields.extend(
                _split_field("Research", "\n".join(f"\u2022 {r}" for r in research))
            )
        if discovered:
            research_fields.extend(
                _split_field("Discovered Themes", "\n".join(f"\u2022 {d}" for d in discovered))
            )
        embeds.append(
            {
                "title": "\U0001f50d Thematic Research",
                "color": COLOR_HEADER,
                "fields": research_fields,
            }
        )

    # ── Watchlist ──────────────────────────────────────────────
    wl_embed = _build_watchlist_embed(brief, color)
    if wl_embed is not None:
        embeds.append(wl_embed)

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
    Oversized single embeds are split by moving excess fields into new embeds.
    """
    if not embeds:
        return []

    # Pre-pass: split any single embed that exceeds the char limit
    safe_embeds: list[dict[str, Any]] = []
    for embed in embeds:
        if _embed_char_count(embed) <= _MAX_CHARS_PER_MSG:
            safe_embeds.append(embed)
        else:
            safe_embeds.extend(_split_oversized_embed(embed))

    batches: list[list[dict[str, Any]]] = []
    current_batch: list[dict[str, Any]] = []
    current_chars = 0

    for embed in safe_embeds:
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


def _split_oversized_embed(embed: dict[str, Any]) -> list[dict[str, Any]]:
    """Split an embed that exceeds 6000 chars into multiple embeds by fields."""
    fields = embed.get("fields", [])
    if not fields:
        # No fields to split — truncate description as last resort
        return [{**embed, "description": embed.get("description", "")[:4096]}]

    title = embed.get("title", "")
    color = embed.get("color")
    # Base overhead: title + description + footer + author
    base_chars = (
        len(title)
        + len(embed.get("description", ""))
        + len(embed.get("footer", {}).get("text", ""))
        + len(embed.get("author", {}).get("name", ""))
    )

    result: list[dict[str, Any]] = []
    current_fields: list[dict[str, Any]] = []
    current_chars = base_chars

    for field in fields:
        field_chars = len(field.get("name", "")) + len(field.get("value", ""))
        if current_fields and current_chars + field_chars > _MAX_CHARS_PER_MSG:
            # Emit current embed
            part = {**embed, "fields": current_fields}
            if result:
                # Continuation — keep title for context but drop description
                part = {"title": f"{title} (cont.)", "color": color, "fields": current_fields}
            result.append(part)
            current_fields = []
            current_chars = len(title) + 8  # " (cont.)" overhead
        current_fields.append(field)
        current_chars += field_chars

    if current_fields:
        part = {**embed, "fields": current_fields}
        if result:
            part = {"title": f"{title} (cont.)", "color": color, "fields": current_fields}
        result.append(part)

    return result or [embed]


def _split_field(name: str, value: str, *, inline: bool = False) -> list[dict[str, Any]]:
    """Split a field value into multiple fields if it exceeds Discord's 1024 char limit.

    Continuation fields use "{name} (cont.)" as the name for context.
    """
    if len(value) <= 1024:
        return [{"name": name, "value": value, "inline": inline}]

    fields: list[dict[str, Any]] = []
    remaining = value
    first = True
    while remaining:
        chunk = remaining[:1024]
        # Try to break at a newline for cleaner splits
        if len(remaining) > 1024:
            last_nl = chunk.rfind("\n")
            if last_nl > 512:
                chunk = remaining[:last_nl]
        fields.append(
            {
                "name": name if first else f"{name} (cont.)",
                "value": chunk,
                "inline": inline,
            }
        )
        remaining = remaining[len(chunk) :]
        first = False
    return fields


def _format_debate_side_with_variant(side: dict[str, Any]) -> str:
    """Format a debate side with variant perception fields."""
    lines: list[str] = []
    variant = side.get("variant_vs_consensus", "")
    if variant:
        lines.append(f"**Variant:** {variant}")
    target = side.get("estimated_upside_downside", "")
    if target:
        lines.append(f"**Target:** {target}")
    lines.append(side.get("argument", ""))
    for ev in side.get("key_evidence", [])[:4]:
        lines.append(f"\u203a {ev}")
    catalyst = side.get("catalyst", "")
    if catalyst:
        timeline = side.get("catalyst_timeline", "")
        tl_str = f" ({timeline})" if timeline else ""
        lines.append(f"**Catalyst:** {catalyst}{tl_str}")
    invalidation = side.get("what_would_change_my_mind", "")
    if invalidation:
        lines.append(f"**Invalidation:** {invalidation}")
    return "\n".join(lines)
