"""NASDAQ API client for earnings calendar.

Free API — no key required.
- Earnings calendar: https://api.nasdaq.com/api/calendar/earnings?date=YYYY-MM-DD
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import TYPE_CHECKING

import httpx
import orjson

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.providers.nasdaq.models import EarningsEvent

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)

NASDAQ_API_URL = "https://api.nasdaq.com/api"
CACHE_PREFIX = "synesis:nasdaq"

# NASDAQ time label mapping
_TIME_MAP: dict[str, str] = {
    "time-pre-market": "pre-market",
    "time-after-hours": "after-hours",
    "time-not-supplied": "during-market",
}


class NasdaqClient:
    """Client for the NASDAQ earnings calendar API.

    Usage:
        client = NasdaqClient(redis=redis_client)
        events = await client.get_earnings_by_date(date(2026, 2, 13))
        await client.close()
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        self._http_client: httpx.AsyncClient | None = None

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; Synesis/1.0)",
                    "Accept": "application/json",
                },
            )
        return self._http_client

    async def get_earnings_by_date(self, target_date: date) -> list[EarningsEvent]:
        """Get all earnings reports for a specific date.

        Args:
            target_date: Date to look up earnings for

        Returns:
            List of EarningsEvent objects for that date
        """
        settings = get_settings()
        date_str = target_date.isoformat()
        cache_key = f"{CACHE_PREFIX}:earnings:{date_str}"

        # Check cache
        cached = await self._redis.get(cache_key)
        if cached:
            try:
                return [EarningsEvent.model_validate(e) for e in orjson.loads(cached)]
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        client = self._get_http_client()
        try:
            resp = await client.get(
                f"{NASDAQ_API_URL}/calendar/earnings",
                params={"date": date_str},
            )
            resp.raise_for_status()
            data = orjson.loads(resp.content)
        except Exception as e:
            logger.warning("Failed to fetch NASDAQ earnings", date=date_str, error=str(e))
            return []

        rows = data.get("data", {}).get("rows", []) if isinstance(data.get("data"), dict) else []
        if not rows:
            return []

        events: list[EarningsEvent] = []
        for row in rows:
            symbol = row.get("symbol", "").strip()
            if not symbol:
                continue

            # Parse market cap (e.g., "$1,234,567,890")
            market_cap = _parse_market_cap(row.get("marketCap", ""))

            # Parse EPS forecast
            eps_forecast = _parse_float(row.get("epsForecast"))

            # Parse number of estimates
            num_estimates = 0
            try:
                num_estimates = int(row.get("noOfEsts", 0))
            except (ValueError, TypeError) as e:
                logger.debug("Failed to parse num_estimates", symbol=symbol, error=str(e))

            events.append(
                EarningsEvent(
                    ticker=symbol,
                    company_name=row.get("name", ""),
                    earnings_date=target_date,
                    time=_TIME_MAP.get(row.get("time", ""), "during-market"),
                    eps_forecast=eps_forecast,
                    num_estimates=num_estimates,
                    market_cap=market_cap,
                    fiscal_quarter=row.get("fiscalQuarterEnding", ""),
                )
            )

        # Cache
        await self._redis.set(
            cache_key,
            orjson.dumps([e.model_dump(mode="json") for e in events]),
            ex=settings.nasdaq_cache_ttl_earnings,
        )
        logger.debug("Fetched NASDAQ earnings", date=date_str, count=len(events))

        return events

    async def get_upcoming_earnings(
        self,
        tickers: list[str],
        days: int | None = None,
    ) -> list[EarningsEvent]:
        """Get upcoming earnings for specific tickers.

        Checks the next N days of earnings calendars and filters for
        the given tickers.

        Args:
            tickers: Tickers to look for
            days: Number of days to look ahead (defaults to config)

        Returns:
            List of EarningsEvent for matching tickers
        """
        settings = get_settings()
        if days is None:
            days = settings.nasdaq_earnings_lookahead_days

        ticker_set = {t.upper() for t in tickers}
        today = date.today()
        targets = [today + timedelta(days=i) for i in range(days)]

        sem = asyncio.Semaphore(5)

        async def _fetch(target_date: date) -> list[EarningsEvent]:
            async with sem:
                return await self.get_earnings_by_date(target_date)

        all_events = await asyncio.gather(*(_fetch(d) for d in targets))

        matches: list[EarningsEvent] = []
        for events in all_events:
            for event in events:
                if event.ticker.upper() in ticker_set:
                    matches.append(event)

        return matches

    async def close(self) -> None:
        """Clean up resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.debug("NasdaqClient closed")


def _parse_market_cap(value: str) -> float | None:
    """Parse market cap string like '$1,234,567,890' to float."""
    if not value or value == "N/A":
        return None
    cleaned = value.replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_float(value: str | float | None) -> float | None:
    """Parse a string or float value to float."""
    if value is None or value == "N/A" or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
