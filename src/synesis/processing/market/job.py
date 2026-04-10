"""Daily market brief job — fetches snapshot + movers, gathers context, runs LLM analysis, sends to Discord."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

import orjson

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.notifications.discord import send_discord
from synesis.processing.common.web_search import search_market_impact
from synesis.processing.market.analyze import analyze_market_brief, identify_context_gaps
from synesis.processing.market.discord_format import (
    format_analysis_embeds,
    format_market_brief_embeds,
)
from synesis.processing.market.snapshot import fetch_market_brief, format_market_data_for_llm

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from synesis.storage.database import Database

logger = get_logger(__name__)


async def _gather_context(
    db: Database,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Fetch events YTD, twitter today, and news signals last 24hrs from DB."""
    today = date.today()
    jan1 = date(today.year, 1, 1)

    events_rows, twitter_rows, signal_rows = await asyncio.gather(
        db.get_diary_entries("events", jan1, today),
        db.get_diary_entries("twitter", today, today),
        db.get_recent_signals(hours=24),
    )

    events: list[dict[str, Any]] = []
    for r in events_rows:
        try:
            events.append(
                {"entry_date": str(r["entry_date"]), "payload": orjson.loads(r["payload"])}
            )
        except Exception:
            logger.warning(
                "Corrupt diary row skipped", source="events", entry_date=r.get("entry_date")
            )

    twitter: list[dict[str, Any]] = []
    for r in twitter_rows:
        try:
            twitter.append(
                {"entry_date": str(r["entry_date"]), "payload": orjson.loads(r["payload"])}
            )
        except Exception:
            logger.warning(
                "Corrupt diary row skipped", source="twitter", entry_date=r.get("entry_date")
            )

    signals: list[dict[str, Any]] = []
    for r in signal_rows:
        try:
            signals.append(
                {
                    "time": str(r["time"]),
                    "payload": orjson.loads(r["payload"]),
                    "tickers": r["tickers"],
                    "entities": r["entities"],
                }
            )
        except Exception:
            logger.warning("Corrupt signal row skipped", time=r.get("time"))

    return events, twitter, signals


async def _search_one(query: str) -> list[dict[str, Any]]:
    """Run a single web search query, returning [] on failure."""
    try:
        return await search_market_impact(query, count=3, recency="day")
    except Exception:
        logger.warning("Web search failed", query=query, exc_info=True)
        return []


async def _search_for_gaps(queries: list[str]) -> list[dict[str, Any]]:
    """Run targeted web searches for LLM-identified context gaps."""
    if not queries:
        return []

    results_per_query = await asyncio.gather(*[_search_one(q) for q in queries])
    # Flatten and deduplicate by URL
    seen_urls: set[str] = set()
    all_results: list[dict[str, Any]] = []
    for batch in results_per_query:
        for r in batch:
            url = r.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)
    return all_results


async def market_brief_job(redis: Redis, db: Database | None = None) -> None:
    """Daily market brief job — fetches snapshot + movers, gathers context, runs LLM analysis, sends to Discord."""
    settings = get_settings()

    webhook = settings.discord_brief_webhook_url
    if not webhook:
        webhook = settings.discord_webhook_url
    if not webhook:
        logger.warning("No Discord webhook configured for market brief")
        return

    brief = await fetch_market_brief(redis)

    # Send market data embeds first (fast, no LLM wait)
    data_messages = format_market_brief_embeds(brief)
    sent_ok = 0
    for i, embeds in enumerate(data_messages):
        ok = await send_discord(embeds, webhook_url_override=webhook)
        if ok:
            sent_ok += 1
        if i < len(data_messages) - 1:
            await asyncio.sleep(0.5)

    if sent_ok < len(data_messages):
        logger.warning(
            "Market brief data partially sent",
            messages_sent=sent_ok,
            messages_total=len(data_messages),
        )
    else:
        logger.info(
            "Market brief data sent",
            messages_sent=sent_ok,
            messages_total=len(data_messages),
        )

    # Gather context and run LLM analysis (requires DB)
    analysis = None
    if db:
        # Stage 1: format market data + gather context from DB
        market_data_text: str | None = None
        events: list[dict[str, Any]] = []
        twitter: list[dict[str, Any]] = []
        signals: list[dict[str, Any]] = []
        search_results: list[dict[str, Any]] = []

        try:
            market_data_text, _ = format_market_data_for_llm(brief)
            events, twitter, signals = await _gather_context(db)

            logger.info(
                "Market brief context gathered",
                events=len(events),
                twitter=len(twitter),
                signals=len(signals),
            )
        except Exception:
            logger.exception("Market brief context gathering failed")

        # Stage 2: identify gaps — fast LLM decides what needs web search
        if market_data_text:
            try:
                gaps = await identify_context_gaps(
                    market_data_text=market_data_text,
                    events_context=events,
                    twitter_context=twitter,
                    signals_context=signals,
                )
                if gaps.gaps:
                    search_results = await _search_for_gaps([g.query for g in gaps.gaps])
                    logger.info(
                        "Gap searches complete",
                        queries=len(gaps.gaps),
                        results=len(search_results),
                    )
            except Exception:
                logger.warning("Context gap search failed, continuing without", exc_info=True)

        # Stage 3: full LLM analysis (analyze_market_brief handles errors internally, returns None on failure)
        if market_data_text:
            analysis = await analyze_market_brief(
                market_data_text=market_data_text,
                events_context=events,
                twitter_context=twitter,
                signals_context=signals,
                search_results=search_results,
            )
    else:
        logger.debug("Skipping market brief analysis (no database connection)")

    # Send analysis embeds
    if analysis:
        await asyncio.sleep(0.5)
        analysis_messages = format_analysis_embeds(analysis)
        analysis_sent = 0
        for i, embeds in enumerate(analysis_messages):
            ok = await send_discord(embeds, webhook_url_override=webhook)
            if ok:
                analysis_sent += 1
            if i < len(analysis_messages) - 1:
                await asyncio.sleep(0.5)

        if analysis_sent < len(analysis_messages):
            logger.warning(
                "Market brief analysis partially sent",
                messages_sent=analysis_sent,
                messages_total=len(analysis_messages),
            )
        else:
            logger.info(
                "Market brief analysis sent",
                messages_sent=analysis_sent,
                messages_total=len(analysis_messages),
            )

    # Persist to diary table
    if db:
        try:
            payload: dict[str, Any] = {"market_data": brief.model_dump(mode="json")}
            if analysis:
                payload["analysis"] = analysis.model_dump(mode="json")
            await db.upsert_diary_entry(
                entry_date=datetime.now(UTC).date(),
                source="market_brief",
                payload=payload,
            )
        except Exception:
            logger.exception("Failed to save market brief to diary")
