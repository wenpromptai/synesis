"""Daily Event Radar digest — forward-looking calendar Discord message.

Message: "What's Coming" — forward-looking calendar for next N days.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from synesis.config import get_settings
from synesis.core.constants import (
    CATEGORY_EMOJI,
    COLOR_CALENDAR,
    DAY_NAMES,
    DIGEST_WHATS_COMING_DAYS,
    LAST_DIGEST_KEY,
    LAST_DIGEST_TTL,
    SEC_13F_DEADLINES,
)
from synesis.core.logging import get_logger
from synesis.notifications.discord import send_discord
from synesis.processing.events.fetchers import load_hedge_fund_registry

if TYPE_CHECKING:
    from pydantic import SecretStr
    from redis.asyncio import Redis

    from synesis.storage.database import Database

logger = get_logger(__name__)


async def send_event_digest(
    db: Database,
    redis: Redis | None = None,
) -> bool:
    """Build and send the daily event digest to Discord.

    Returns True if at least one message sent successfully.
    """
    settings = get_settings()
    webhook = settings.discord_brief_webhook_url
    if not webhook:
        webhook = settings.discord_webhook_url
    if not webhook:
        logger.warning("No Discord webhook configured for event digest")
        return False

    return await _send_whats_coming(db, redis, webhook)


# ─────────────────────────────────────────────────────────────
# What's Coming
# ─────────────────────────────────────────────────────────────


async def _send_whats_coming(
    db: Database,
    redis: Redis | None,
    webhook: SecretStr,
) -> bool:
    """Send the forward-looking calendar message."""
    today = date.today()
    end = today + timedelta(days=DIGEST_WHATS_COMING_DAYS)

    rows = await db.get_events_by_date_range(today, end)
    if not rows:
        logger.info("No upcoming events for What's Coming")
        return False

    # Determine newly discovered events
    newly_discovered: set[int] = set()
    if redis:
        last_digest_ts = await redis.get(LAST_DIGEST_KEY)
        if last_digest_ts:
            try:
                since = datetime.fromisoformat(
                    last_digest_ts.decode() if isinstance(last_digest_ts, bytes) else last_digest_ts
                )
            except (ValueError, TypeError):
                logger.warning("Failed to parse last digest timestamp", exc_info=True)
                since = None
            if since is not None:
                new_ids = await db.get_events_discovered_since(since)
                newly_discovered = set(new_ids)

    # Check 13F deadline proximity
    deadline_reminder = _get_13f_deadline_reminder(today, DIGEST_WHATS_COMING_DAYS)

    # Build calendar content grouped by date
    now_iso = datetime.now(timezone.utc).isoformat()
    embeds = _format_whats_coming_embeds(rows, newly_discovered, deadline_reminder, now_iso, today)

    sent_ok = 0
    for i, embed_batch in enumerate(embeds):
        ok = await send_discord(embed_batch, webhook_url_override=webhook)
        if ok:
            sent_ok += 1
        if i < len(embeds) - 1:
            await asyncio.sleep(0.5)

    # Update last digest timestamp
    if redis and sent_ok > 0:
        try:
            await redis.set(
                LAST_DIGEST_KEY, datetime.now(timezone.utc).isoformat(), ex=LAST_DIGEST_TTL
            )
        except Exception:
            logger.warning("Failed to update last digest timestamp", exc_info=True)

    logger.info("What's Coming sent", messages=sent_ok, events=len(rows))
    return sent_ok > 0


def _get_13f_deadline_reminder(today: date, lookahead_days: int) -> str | None:
    """Check if a 13F deadline falls within the lookahead window."""
    end = today + timedelta(days=lookahead_days)

    for quarter_month, (deadline_month, deadline_day) in SEC_13F_DEADLINES.items():
        # Try current year and next year
        for year in (today.year, today.year + 1):
            try:
                deadline = date(year, deadline_month, deadline_day)
            except ValueError:
                continue
            if today <= deadline <= end:
                quarter_year = year if deadline_month > quarter_month else year - 1
                quarter_label = (
                    f"Q{quarter_month // 3 if quarter_month != 12 else 4} {quarter_year}"
                )

                # List top-tier funds by name, then count the rest
                fund_registry, top_tier_ciks = load_hedge_fund_registry()
                top_tier_names = [
                    fund_registry[cik] for cik in top_tier_ciks if cik in fund_registry
                ]
                other_count = len(fund_registry) - len(top_tier_names)
                fund_list = ", ".join(top_tier_names)
                if other_count > 0:
                    fund_list += f", ... (+{other_count} more)"

                return (
                    f"\u23f0 **13F Filing Deadline: "
                    f"{deadline.strftime('%b %d')} ({quarter_label})**\n"
                    f"Filings due from: {fund_list}"
                )
    return None


def _format_whats_coming_embeds(
    rows: list[Any],
    newly_discovered: set[int],
    deadline_reminder: str | None,
    now_iso: str,
    today: date,
) -> list[list[dict[str, Any]]]:
    """Format upcoming events into calendar-style Discord embeds."""
    # Group events by date
    by_date: dict[date, list[Any]] = {}
    for row in rows:
        d = row["event_date"]
        by_date.setdefault(d, []).append(row)

    lines: list[str] = []

    # Add 13F deadline reminder at the top
    if deadline_reminder:
        lines.append(deadline_reminder)
        lines.append("")

    for event_date in sorted(by_date.keys()):
        day_name = DAY_NAMES[event_date.weekday()]
        lines.append(f"**{event_date.strftime('%b %d')} ({day_name})**")
        for row in by_date[event_date]:
            emoji = CATEGORY_EMOJI.get(row["category"], "\U0001f4cb")
            regions = ", ".join(row["region"]) if row["region"] else ""
            tickers = row.get("tickers") or []
            ticker_str = " ".join(f"`${t}`" for t in tickers[:5]) if tickers else ""

            badge = "\U0001f195 " if row.get("id") in newly_discovered else ""
            time_label = row.get("time_label") or ""
            parts = [p for p in [regions, time_label, ticker_str] if p]
            suffix = f" \u2014 {' | '.join(parts)}" if parts else ""
            lines.append(f"{badge}{emoji} {row['title']}{suffix}")
        lines.append("")

    content = "\n".join(lines).strip()

    # Split into multiple embeds if content exceeds Discord's 4096 char limit
    messages: list[list[dict[str, Any]]] = []
    chunks = _split_content(content, 4096)

    for i, chunk in enumerate(chunks):
        embed: dict[str, Any] = {
            "color": COLOR_CALENDAR,
            "description": chunk,
            "timestamp": now_iso,
        }
        if i == 0:
            embed["title"] = f"\U0001f4c5 What's Coming \u2014 Next {DIGEST_WHATS_COMING_DAYS} Days"
        if i == len(chunks) - 1:
            embed["footer"] = {"text": f"{len(rows)} events | {today.strftime('%Y-%m-%d')}"}
        messages.append([embed])

    return messages


def _split_content(text: str, max_len: int) -> list[str]:
    """Split text into chunks at line boundaries."""
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in text.split("\n"):
        line_len = len(line) + 1  # +1 for newline
        if current_len + line_len > max_len and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += line_len

    if current:
        chunks.append("\n".join(current))

    return chunks
