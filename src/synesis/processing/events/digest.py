"""Daily Event Radar digest — two-part Discord messages.

Message 1: "What's Coming" — forward-looking calendar for next N days.
Message 2: "Yesterday's Brief" — LLM analysis of yesterday's events.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from synesis.config import get_settings
from synesis.core.constants import (
    CATEGORY_EMOJI,
    COLOR_CALENDAR,
    COLOR_HEADER,
    DAY_NAMES,
    DIGEST_WHATS_COMING_DAYS,
    DIRECTION_ICON,
    FRED_OUTCOME_SERIES,
    LAST_DIGEST_KEY,
    LAST_DIGEST_TTL,
    SEC_13F_DEADLINES,
    SECTOR_COLORS,
    SENTIMENT_ICON,
    SOURCE_LABEL,
    THEME_EMOJI,
)
from synesis.processing.events.fetchers import load_hedge_fund_registry
from synesis.core.logging import get_logger
from synesis.notifications.discord import send_discord
from synesis.processing.events.models import YesterdayBriefAnalysis, YesterdayTheme
from synesis.processing.events.yesterday import synthesize_yesterday_brief
from synesis.processing.market.snapshot import fetch_market_brief, format_market_data_for_llm

if TYPE_CHECKING:
    from pydantic import SecretStr
    from redis.asyncio import Redis

    from synesis.providers.crawler.crawl4ai import Crawl4AICrawlerProvider
    from synesis.providers.fred.client import FREDClient
    from synesis.providers.sec_edgar.client import SECEdgarClient
    from synesis.storage.database import Database

logger = get_logger(__name__)


async def send_event_digest(
    db: Database,
    redis: Redis | None = None,
    sec_edgar: SECEdgarClient | None = None,
    crawler: Crawl4AICrawlerProvider | None = None,
    fred: FREDClient | None = None,
) -> bool:
    """Build and send the daily event digest to Discord (two messages).

    Returns True if at least one message sent successfully.
    """
    settings = get_settings()
    webhook = settings.discord_events_webhook_url
    if not webhook:
        webhook = settings.discord_webhook_url
    if not webhook:
        logger.warning("No Discord webhook configured for event digest")
        return False

    sent_coming = await _send_whats_coming(db, redis, webhook)
    # Small delay between messages
    if sent_coming:
        await asyncio.sleep(1.0)
    sent_brief = await _send_yesterday_brief(db, redis, sec_edgar, webhook, crawler, fred)

    return sent_coming or sent_brief


# ─────────────────────────────────────────────────────────────
# Message 1: What's Coming
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


# ─────────────────────────────────────────────────────────────
# Message 2: Yesterday's Brief
# ─────────────────────────────────────────────────────────────


async def _fetch_outcomes(
    events: list[dict[str, Any]],
    redis: Redis | None,
    sec_edgar: SECEdgarClient | None = None,
    crawler: Crawl4AICrawlerProvider | None = None,
    fred: FREDClient | None = None,
    db: Database | None = None,
) -> list[dict[str, Any]]:
    """Enrich calendar events with actual outcomes via specialized sources."""
    sem = asyncio.Semaphore(5)

    async def _fetch_one(ev: dict[str, Any]) -> None:
        category = ev.get("category", "")
        # 13F filings already have QoQ diff in description — no outcome to fetch
        if category == "13f_filing":
            return

        async with sem:
            try:
                if category == "earnings":
                    outcome = await _get_earnings_outcome(ev, sec_edgar)
                elif category == "economic_data":
                    outcome = await _get_economic_data_outcome(ev, fred)
                elif category == "fed" and "minute" not in ev.get("title", "").lower():
                    # Rate decisions: crawl date-specific Fed statement URL
                    outcome = await _get_crawled_outcome(ev, crawler)
                elif category == "fed":
                    # Minutes: look up meeting date from DB, crawl Fed minutes URL
                    outcome = await _get_fomc_minutes_outcome(ev, db, crawler)
                else:
                    outcome = ""
            except Exception:
                logger.warning(
                    "Outcome fetch failed",
                    title=ev.get("title"),
                    category=category,
                    exc_info=True,
                )
                outcome = ""

        if outcome:
            ev["outcome"] = outcome

    await asyncio.gather(*[_fetch_one(ev) for ev in events])
    return events


async def _get_earnings_outcome(
    ev: dict[str, Any],
    sec_edgar: SECEdgarClient | None,
) -> str:
    """Get earnings outcome from SEC 8-K Item 2.02 press release."""
    tickers = ev.get("tickers") or []
    if not tickers or not sec_edgar:
        return ""

    ticker = tickers[0]
    try:
        releases = await sec_edgar.get_earnings_releases(ticker, limit=1)
        if releases and releases[0].content:
            return releases[0].content[:3000]
    except Exception:
        logger.warning("SEC earnings release fetch failed", ticker=ticker, exc_info=True)
    return ""


async def _get_economic_data_outcome(
    ev: dict[str, Any],
    fred: FREDClient | None,
) -> str:
    """Get economic data outcome from FRED API observations."""
    if not fred:
        return ""

    title = ev.get("title", "")
    # Match longest key first to prefer "Core CPI" over "CPI"
    matched_key = ""
    for key in sorted(FRED_OUTCOME_SERIES.keys(), key=len, reverse=True):
        if key.lower() in title.lower():
            matched_key = key
            break

    if not matched_key:
        return ""

    series_id, units = FRED_OUTCOME_SERIES[matched_key]
    try:
        obs = await fred.get_observations(series_id, sort_order="desc", limit=2, units=units)
        if not obs.observations:
            return ""
        latest = obs.observations[0]
        if latest.value is None:
            return ""
        if len(obs.observations) >= 2 and obs.observations[1].value is not None:
            prev = obs.observations[1].value
            return f"{matched_key}: {latest.value:.1f}{'%' if units != 'lin' else ''} (prev {prev:.1f}{'%' if units != 'lin' else ''})"
        return f"{matched_key}: {latest.value:.1f}{'%' if units != 'lin' else ''}"
    except Exception:
        logger.warning("FRED outcome fetch failed", series_id=series_id, exc_info=True)
    return ""


async def _get_fomc_minutes_outcome(
    ev: dict[str, Any],
    db: Database | None,
    crawler: Crawl4AICrawlerProvider | None,
) -> str:
    """Get FOMC minutes by looking up the meeting date and crawling the Fed URL."""
    if not crawler or not db:
        return ""

    release_date = ev.get("event_date")
    if not release_date:
        return ""

    meeting_date = await db.get_last_fomc_meeting_date(release_date)
    if not meeting_date:
        return ""

    ds = (
        meeting_date.strftime("%Y%m%d")
        if isinstance(meeting_date, date)
        else str(meeting_date).replace("-", "")[:8]
    )
    url = f"https://www.federalreserve.gov/monetarypolicy/fomcminutes{ds}.htm"

    try:
        result = await crawler.crawl(url)
        if result.success and result.markdown.strip():
            logger.debug("FOMC minutes crawled", url=url, chars=len(result.markdown))
            return result.markdown
    except Exception:
        logger.warning("FOMC minutes crawl failed", url=url, exc_info=True)
    return ""


async def _get_crawled_outcome(
    ev: dict[str, Any],
    crawler: Crawl4AICrawlerProvider | None,
) -> str:
    """Get outcome by crawling the event's source URL or Fed press releases."""
    if not crawler:
        return ""

    source_urls = ev.get("source_urls") or []
    url = source_urls[0] if source_urls else None

    # For fed: construct a date-specific Fed statement URL
    if ev.get("category") == "fed":
        event_date = ev.get("event_date")
        if event_date is not None:
            if isinstance(event_date, date):
                ds = event_date.strftime("%Y%m%d")
            else:
                ds = str(event_date).replace("-", "")[:8]
            # Fed FOMC statements follow this URL pattern
            url = f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{ds}a.htm"
        elif not url:
            url = "https://www.federalreserve.gov/newsevents/pressreleases.htm"

    if not url:
        return ""

    try:
        result = await crawler.crawl(url)
        if result.success and result.markdown.strip():
            logger.debug("Outcome crawled", url=url, chars=len(result.markdown))
            return result.markdown
    except Exception:
        logger.warning("Crawl4AI outcome fetch failed", url=url, exc_info=True)
    return ""


async def _send_yesterday_brief(
    db: Database,
    redis: Redis | None,
    sec_edgar: SECEdgarClient | None,
    webhook: SecretStr,
    crawler: Crawl4AICrawlerProvider | None = None,
    fred: FREDClient | None = None,
) -> bool:
    """Send the backward-looking LLM analysis message."""
    yesterday = date.today() - timedelta(days=1)

    # Gather yesterday's calendar events
    yesterday_rows = await db.get_events_by_date_range(yesterday, yesterday)
    yesterday_events = [dict(r) for r in yesterday_rows]

    # Enrich calendar events with actual outcomes
    yesterday_events = await _fetch_outcomes(yesterday_events, redis, sec_edgar, crawler, fred, db)

    # Gather SEC data: 13F filings
    filing_briefs: list[dict[str, Any]] = []
    if sec_edgar:
        filing_briefs = await _get_yesterday_13f_briefs(sec_edgar, yesterday_rows)

    # Skip if nothing happened
    if not yesterday_events and not filing_briefs:
        logger.info("No yesterday events for brief")
        return False

    # Fetch market data for LLM context
    market_data_text = ""
    sector_display = ""
    if redis:
        try:
            brief = await fetch_market_brief(redis)
            market_data_text, sector_display = format_market_data_for_llm(brief)
        except Exception:
            logger.warning("Market data fetch failed, proceeding without", exc_info=True)

    # LLM synthesis
    analysis = await synthesize_yesterday_brief(
        yesterday_events,
        filing_briefs,
        market_data=market_data_text,
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    if analysis:
        # Override LLM sector field with our labeled + sorted string
        if sector_display:
            analysis.market_snapshot.sector_performance = sector_display
        messages = _format_yesterday_brief_rich(analysis, yesterday, now_iso)
    else:
        logger.error("Yesterday brief LLM synthesis failed, using fallback")
        messages = _format_yesterday_brief_fallback(yesterday_events, yesterday, now_iso)

    sent_ok = 0
    for i, embeds in enumerate(messages):
        ok = await send_discord(embeds, webhook_url_override=webhook)
        if ok:
            sent_ok += 1
        if i < len(messages) - 1:
            await asyncio.sleep(0.5)

    logger.info(
        "Yesterday's Brief sent",
        messages=sent_ok,
        calendar_events=len(yesterday_events),
        filings=len(filing_briefs),
        llm_synthesis=analysis is not None,
    )
    return sent_ok > 0


async def _get_yesterday_13f_briefs(
    sec_edgar: SECEdgarClient,
    yesterday_rows: list[Any],
) -> list[dict[str, Any]]:
    """Check if any yesterday events were 13F filings and get their data."""
    briefs: list[dict[str, Any]] = []
    for row in yesterday_rows:
        if row.get("category") != "13f_filing":
            continue
        # Try to extract CIK from the event title or description
        title = row.get("title", "")
        fund_registry, _ = load_hedge_fund_registry()
        for cik, fund_name in fund_registry.items():
            if fund_name.lower() in title.lower():
                try:
                    diff = await sec_edgar.compare_13f_quarters(cik, fund_name)
                    if diff:
                        diff["fund_name"] = fund_name
                        briefs.append(diff)
                except Exception:
                    logger.warning("13F comparison failed for brief", fund=fund_name, exc_info=True)
                break
    return briefs


def _format_yesterday_brief_rich(
    analysis: YesterdayBriefAnalysis,
    yesterday: date,
    now_iso: str,
) -> list[list[dict[str, Any]]]:
    """Format LLM analysis into rich Discord embed messages."""
    messages: list[list[dict[str, Any]]] = []

    # Header + Market Snapshot embed
    snap = analysis.market_snapshot
    snapshot_desc = f"\U0001f4ca **Market Snapshot**\n{snap.summary}"

    movers_str = " | ".join(f"`${t}`" for t in analysis.top_movers[:10])
    fields: list[dict[str, Any]] = [
        {"name": "\U0001f4c8 Equities", "value": snap.equities, "inline": True},
        {"name": "\U0001f4b5 Rates / FX", "value": snap.rates_fx, "inline": True},
        {"name": "\U0001f6e2\ufe0f Commodities", "value": snap.commodities, "inline": True},
        {"name": "\U0001f321\ufe0f Volatility", "value": snap.volatility, "inline": True},
        {"name": "\u200b", "value": "\u200b", "inline": True},  # alignment spacer
        {"name": "\U0001f4ca Sectors", "value": snap.sector_performance, "inline": False},
    ]
    if movers_str:
        fields.append({"name": "Top Movers", "value": movers_str, "inline": False})
    header: dict[str, Any] = {
        "title": f"\U0001f4f0 Yesterday's Brief \u2014 {yesterday.strftime('%b %d')}",
        "color": COLOR_HEADER,
        "description": f"**{analysis.headline}**\n\n{snapshot_desc}"[:4096],
        "footer": {"text": f"Synesis Event Radar | {yesterday.strftime('%Y-%m-%d')}"},
        "timestamp": now_iso,
        "fields": fields,
    }
    messages.append([header])

    # Per-theme embeds
    for i, theme in enumerate(analysis.themes, 1):
        embed = _format_yesterday_theme_embed(theme, i, len(analysis.themes), now_iso)
        messages.append([embed])

    # Synthesis + Actionables + Risk Radar embed
    synth_embed = _format_synthesis_embed(analysis, now_iso)
    messages.append([synth_embed])

    return messages


def _format_yesterday_theme_embed(
    theme: YesterdayTheme,
    index: int,
    total: int,
    now_iso: str,
) -> dict[str, Any]:
    """Format a single yesterday theme into a Discord embed."""
    emoji = THEME_EMOJI.get(theme.category, "\U0001f4cb")
    sent_icon = SENTIMENT_ICON.get(theme.sentiment, "\u26aa")
    color = SECTOR_COLORS.get(theme.category, 0x5865F2)

    source_label = SOURCE_LABEL.get(theme.source, "\u26a1 Surprise")

    fields: list[dict[str, Any]] = []

    if theme.outcome:
        fields.append({"name": "Outcome", "value": theme.outcome[:1024], "inline": False})

    fields.extend(
        [
            {"name": "Category", "value": theme.category.replace("_", " ").title(), "inline": True},
            {
                "name": "Sentiment",
                "value": f"{sent_icon} {theme.sentiment.title()}",
                "inline": True,
            },
            {"name": "Source", "value": source_label, "inline": True},
        ]
    )

    if theme.key_events:
        event_lines = "\n".join(f"\u2022 {e}" for e in theme.key_events[:8])
        fields.append({"name": "Key Events", "value": event_lines[:1024], "inline": False})

    if theme.tickers:
        ticker_str = " | ".join(f"`${t}`" for t in theme.tickers[:10])
        fields.append({"name": "Tickers", "value": ticker_str[:1024], "inline": False})

    if theme.market_reaction:
        fields.append(
            {"name": "Market Reaction", "value": theme.market_reaction[:1024], "inline": False}
        )

    return {
        "title": f"{emoji} {theme.title} {sent_icon}"[:256],
        "color": color,
        "description": theme.analysis[:4096],
        "fields": fields[:25],
        "footer": {"text": f"Theme {index}/{total}"},
        "timestamp": now_iso,
    }


def _format_synthesis_embed(
    analysis: YesterdayBriefAnalysis,
    now_iso: str,
) -> dict[str, Any]:
    """Format synthesis, actionables, and risk radar into a single embed."""
    parts = [f"\U0001f517 **Summary**\n{analysis.synthesis}"]

    if analysis.actionables:
        action_lines = []
        for a in analysis.actionables:
            icon = DIRECTION_ICON.get(a.direction, "\u2022")
            tickers = ", ".join(a.tickers[:5])
            action_lines.append(
                f"{icon} **{a.direction.upper()} {tickers}** \u2014 {a.action} ({a.timeframe})"
            )
        parts.append("\n\u26a1 **Actionables**\n" + "\n".join(action_lines))

    if analysis.risk_radar:
        risk_lines = [f"\u2022 {r}" for r in analysis.risk_radar]
        parts.append("\n\u26a0\ufe0f **Risk Radar**\n" + "\n".join(risk_lines))

    return {
        "color": COLOR_HEADER,
        "description": "\n".join(parts)[:4096],
        "timestamp": now_iso,
    }


def _format_yesterday_brief_fallback(
    yesterday_events: list[dict[str, Any]],
    yesterday: date,
    now_iso: str,
) -> list[list[dict[str, Any]]]:
    """Fallback format when LLM synthesis fails."""
    messages: list[list[dict[str, Any]]] = []

    lines: list[str] = []
    if yesterday_events:
        lines.append("**Calendar Events**")
        for ev in yesterday_events:
            emoji = CATEGORY_EMOJI.get(ev.get("category", "other"), "\U0001f4cb")
            lines.append(f"{emoji} {ev.get('title', '')}")
        lines.append("")

    embed: dict[str, Any] = {
        "title": f"\U0001f4f0 Yesterday's Brief \u2014 {yesterday.strftime('%b %d')}",
        "color": COLOR_HEADER,
        "description": "\n".join(lines)[:4096] if lines else "No notable events.",
        "timestamp": now_iso,
    }
    messages.append([embed])
    return messages
