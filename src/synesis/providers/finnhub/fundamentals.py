"""Finnhub fundamentals provider and WatchlistDataProvider adapter.

FinnhubFundamentalsProvider: Shell preserved for legacy protocol methods.
FinnhubWatchlistAdapter: Implements WatchlistDataProvider for Flow 4 using
    /stock/profile2, /stock/metric, and /quote endpoints.

Rate limiting: Free tier = 60 calls/min across ALL endpoints.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import httpx

from synesis.core.logging import get_logger
from synesis.providers.base import CompanyInfo, FundamentalsSnapshot, PriceSnapshot
from synesis.providers.finnhub.prices import get_rate_limiter

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)

# Redis key prefix + cache TTL
CACHE_PREFIX = "synesis:finnhub"
WATCHLIST_CACHE_TTL = 300  # 5 minutes


class FinnhubFundamentalsProvider:
    """Finnhub fundamentals provider -- shell preserved for future use."""

    def __init__(self, api_key: str, redis: Redis) -> None:
        self._api_key = api_key
        self._redis = redis
        self._http_client: httpx.AsyncClient | None = None

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
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

    async def close(self) -> None:
        """Clean up resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.debug("FinnhubFundamentalsProvider closed")


# =============================================================================
# WatchlistDataProvider adapter
# =============================================================================


class FinnhubWatchlistAdapter:
    """Implements WatchlistDataProvider using Finnhub REST API.

    Endpoints used:
        /stock/profile2  -> resolve_company(), get_market_cap()
        /stock/metric     -> get_fundamentals()
        /quote            -> get_price() (day change)
    """

    def __init__(self, api_key: str, redis: Redis) -> None:
        self._api_key = api_key
        self._redis = redis
        self._http_client: httpx.AsyncClient | None = None

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        return self._http_client

    async def _fetch(self, endpoint: str, params: dict[str, str | int]) -> Any:
        """Fetch from Finnhub API with rate limiting and Redis caching."""
        import orjson

        cache_key = f"{CACHE_PREFIX}:wl:{endpoint}:{params.get('symbol', '')}"
        cached = await self._redis.get(cache_key)
        if cached:
            try:
                return orjson.loads(cached)
            except Exception:
                logger.warning("Corrupt cache entry, refetching", cache_key=cache_key)
                await self._redis.delete(cache_key)

        from synesis.config import get_settings

        await get_rate_limiter().acquire()

        settings = get_settings()
        client = self._get_http_client()
        url = f"{settings.finnhub_api_url}{endpoint}"
        params["token"] = self._api_key

        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = orjson.loads(response.content)
            # Cache successful responses
            await self._redis.set(cache_key, orjson.dumps(data), ex=WATCHLIST_CACHE_TTL)
            return data
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

    # -- profile cache (shared between resolve_company and get_market_cap) --

    async def _get_profile(self, ticker: str) -> dict[str, Any] | None:
        data = await self._fetch("/stock/profile2", {"symbol": ticker})
        if not data or not isinstance(data, dict) or not data.get("name"):
            return None
        result: dict[str, Any] = data
        return result

    async def resolve_company(self, ticker: str) -> CompanyInfo | None:
        profile = await self._get_profile(ticker)
        if not profile:
            return None
        return CompanyInfo(name=profile["name"])

    async def get_market_cap(self, ticker: str) -> float | None:
        profile = await self._get_profile(ticker)
        if not profile:
            return None
        mc = profile.get("marketCapitalization")
        if mc is None:
            return None
        # Finnhub returns market cap in millions
        return float(mc) * 1_000_000

    async def get_fundamentals(self, ticker: str) -> FundamentalsSnapshot | None:
        data = await self._fetch("/stock/metric", {"symbol": ticker, "metric": "all"})
        if not data or not isinstance(data, dict):
            return None
        m: dict[str, Any] = data.get("metric", {})
        if not m:
            return None
        return FundamentalsSnapshot(
            eps_diluted=m.get("epsBasicExclExtraItemsTTM"),
            price_to_book=m.get("pbAnnual"),
            price_to_sales=m.get("psAnnual"),
            ev_to_ebitda=m.get("currentEv/ebitdaTTM"),
            roe=m.get("roeTTM"),
            net_margin=m.get("netProfitMarginTTM"),
            gross_margin=m.get("grossMarginTTM"),
            debt_to_equity=m.get("totalDebtToEquityAnnual"),
            period_type="ttm",
            period_end=date.today(),
        )

    async def get_price(self, ticker: str) -> PriceSnapshot | None:
        # Get day change from /quote
        quote = await self._fetch("/quote", {"symbol": ticker})
        if not quote or not isinstance(quote, dict) or quote.get("c") is None:
            return None

        # Get month-to-date return from /stock/metric
        metric_data = await self._fetch("/stock/metric", {"symbol": ticker, "metric": "all"})
        one_mth_pct: float | None = None
        if metric_data and isinstance(metric_data, dict):
            m = metric_data.get("metric", {})
            one_mth_pct = m.get("monthToDatePriceReturnDaily")

        return PriceSnapshot(
            one_day_pct=quote.get("dp"),
            one_mth_pct=one_mth_pct,
            price_date=date.today(),
        )

    async def close(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.debug("FinnhubWatchlistAdapter closed")
