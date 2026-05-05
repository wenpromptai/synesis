"""Discord embed formatter for intelligence briefs."""

from __future__ import annotations

from typing import Any


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
        lines.append(f"› {ev}")
    catalyst = side.get("catalyst", "")
    if catalyst:
        timeline = side.get("catalyst_timeline", "")
        tl_str = f" ({timeline})" if timeline else ""
        lines.append(f"**Catalyst:** {catalyst}{tl_str}")
    invalidation = side.get("what_would_change_my_mind", "")
    if invalidation:
        lines.append(f"**Invalidation:** {invalidation}")
    return "\n".join(lines)
