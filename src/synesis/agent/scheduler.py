"""Centralized job scheduler for periodic processing flows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from synesis.config import get_settings
from synesis.core.logging import get_logger

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from synesis.providers.crawler.crawl4ai import Crawl4AICrawlerProvider
    from synesis.providers.fred import FREDClient
    from synesis.providers.nasdaq import NasdaqClient
    from synesis.providers.sec_edgar.client import SECEdgarClient
    from synesis.storage.database import Database

__all__ = [
    "create_scheduler",
    "event_digest_job",
    "event_fetch_job",
    "market_brief_job",
    "refresh_tickers_job",
    "watchlist_cleanup_job",
]

logger = get_logger(__name__)


def create_scheduler() -> AsyncIOScheduler:
    """Create a new scheduler instance."""
    return AsyncIOScheduler(timezone="UTC")


async def watchlist_cleanup_job(db: Database) -> None:
    """Deactivate expired watchlist tickers in PostgreSQL."""
    try:
        expired = await db.deactivate_expired_watchlist()
        if expired:
            logger.info("Deactivated expired watchlist tickers", tickers=expired)
    except Exception:
        logger.exception("Watchlist cleanup job failed")


async def event_fetch_job(
    db: Database,
    redis: Redis,
    fred: FREDClient | None = None,
    nasdaq: NasdaqClient | None = None,
    sec_edgar: SECEdgarClient | None = None,
) -> None:
    """Fetch events from structured APIs (FRED, NASDAQ, FOMC, 13F)."""
    from synesis.processing.events.runner import run_structured_sources

    try:
        stored = await run_structured_sources(
            db, redis, fred=fred, nasdaq=nasdaq, sec_edgar=sec_edgar
        )
        logger.info("Event fetch job complete", stored=stored)
    except Exception:
        logger.exception("Event fetch job failed")


async def market_brief_job(redis: Redis, db: Database | None = None) -> None:
    """Send daily market brief with LLM analysis to Discord and persist to diary."""
    from synesis.processing.market.job import market_brief_job as _market_brief_job

    try:
        await _market_brief_job(redis, db=db)
        logger.info("Market brief job complete")
    except Exception:
        logger.exception("Market brief job failed")


async def event_digest_job(
    db: Database,
    redis: Redis | None = None,
    sec_edgar: SECEdgarClient | None = None,
    crawler: Crawl4AICrawlerProvider | None = None,
    fred: FREDClient | None = None,
) -> None:
    """Send daily Event Radar digest to Discord."""
    from synesis.processing.events.digest import send_event_digest

    try:
        sent = await send_event_digest(
            db, redis=redis, sec_edgar=sec_edgar, crawler=crawler, fred=fred
        )
        logger.info("Event digest job complete", sent=sent)
    except Exception:
        logger.exception("Event digest job failed")


async def refresh_tickers_job() -> None:
    """Refresh data/us_tickers.json from Finnhub /stock/symbol?exchange=US.

    Runs weekly. On failure, keeps the existing file intact.
    """
    settings = get_settings()
    if not settings.finnhub_api_key:
        logger.warning("FINNHUB_API_KEY not configured, skipping ticker refresh")
        return

    tickers_file = Path(__file__).resolve().parents[2] / "data" / "us_tickers.json"
    api_key = settings.finnhub_api_key.get_secret_value()

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(
                f"https://finnhub.io/api/v1/stock/symbol?exchange=US&token={api_key}"
            )
            resp.raise_for_status()
            data = resp.json()

        if not isinstance(data, list) or len(data) < 1000:
            logger.error(
                "Finnhub returned unexpected data", count=len(data) if isinstance(data, list) else 0
            )
            return

        # Filter to useful types: Common Stock, ETF, ETP, ADR, REIT
        valid_types = {"Common Stock", "ETF", "ETP", "ADR", "REIT", ""}
        lookup = {}
        for item in data:
            sym = item.get("symbol", "")
            desc = item.get("description", "")
            sym_type = item.get("type", "")
            if sym and sym_type in valid_types:
                lookup[sym.upper()] = desc

        # Write atomically (write to temp, then rename)
        tmp_file = tickers_file.with_suffix(".tmp")
        tmp_file.write_text(json.dumps(lookup, separators=(",", ":")))
        tmp_file.rename(tickers_file)

        logger.info("Ticker list refreshed", tickers=len(lookup), file=str(tickers_file))
    except Exception:
        logger.exception("Ticker refresh job failed — keeping existing file")
