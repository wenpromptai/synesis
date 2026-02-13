"""Finnhub fundamentals provider — shell preserved for future use.

All fundamentals methods have been migrated to standalone providers:
- FactSet: Basic financials, market cap, margins, ratios
- SEC EDGAR: Filings, insider transactions/sentiment, historical EPS/revenue (XBRL)
- NASDAQ: Earnings calendar, EPS forecasts

The class shell, init, close, and HTTP helpers are kept in case Finnhub
adds unique endpoints not available from other sources.

Rate limiting: Free tier = 60 calls/min across ALL endpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from synesis.core.logging import get_logger
from synesis.providers.finnhub.prices import get_rate_limiter

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)

# Redis key prefix
CACHE_PREFIX = "synesis:finnhub"


class FinnhubFundamentalsProvider:
    """Finnhub fundamentals provider — shell preserved for future use.

    All 6 protocol methods have been migrated to standalone providers:
    - FactSetProvider for financials
    - SECEdgarClient for filings/insiders/historical EPS
    - NasdaqClient for earnings calendar
    """

    def __init__(self, api_key: str, redis: Redis) -> None:
        self._api_key = api_key
        self._redis = redis
        self._http_client: httpx.AsyncClient | None = None

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def _get_cached(self, key: str) -> bytes | str | None:
        result: bytes | str | None = await self._redis.get(key)
        return result

    async def _set_cached(self, key: str, value: str, ttl: int) -> None:
        await self._redis.set(key, value, ex=ttl)

    async def _fetch_finnhub(
        self,
        endpoint: str,
        params: dict[str, str | int],
    ) -> object:
        import orjson

        from synesis.config import get_settings

        await get_rate_limiter().acquire()

        settings = get_settings()
        client = self._get_http_client()
        url = f"{settings.finnhub_api_url}{endpoint}"
        params["token"] = self._api_key

        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return orjson.loads(response.content)
        except httpx.HTTPStatusError as e:
            logger.warning(
                "Finnhub API error",
                endpoint=endpoint,
                status=e.response.status_code,
            )
            return None
        except Exception as e:
            logger.warning("Finnhub API request failed", endpoint=endpoint, error=str(e))
            return None

    # ─────────────────────────────────────────────────────────────
    # FundamentalsProvider Protocol Implementation
    # Migrated to FactSet / SEC EDGAR / NASDAQ standalone providers.
    # ─────────────────────────────────────────────────────────────

    # async def get_basic_financials(self, ticker: str) -> dict[str, Any] | None:
    #     """Replaced by FactSetProvider.get_fundamentals() + get_market_cap()."""
    #     ...

    # async def get_insider_transactions(self, ticker: str, limit: int = 10) -> list[dict[str, Any]]:
    #     """Replaced by SECEdgarClient.get_insider_transactions()."""
    #     ...

    # async def get_insider_sentiment(self, ticker: str) -> dict[str, Any] | None:
    #     """Replaced by SECEdgarClient.get_insider_sentiment()."""
    #     ...

    # async def get_sec_filings(self, ticker: str, limit: int = 5) -> list[dict[str, Any]]:
    #     """Replaced by SECEdgarClient.get_filings()."""
    #     ...

    # async def get_eps_surprises(self, ticker: str, limit: int = 4) -> list[dict[str, Any]]:
    #     """Replaced by SECEdgarClient.get_historical_eps() (XBRL)."""
    #     ...

    # async def get_earnings_calendar(self, ticker: str) -> dict[str, Any] | None:
    #     """Replaced by NasdaqClient.get_upcoming_earnings()."""
    #     ...

    # ─────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Clean up resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.debug("FinnhubFundamentalsProvider closed")
