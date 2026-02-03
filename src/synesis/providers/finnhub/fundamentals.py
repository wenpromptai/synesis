"""Finnhub fundamentals provider implementation.

This module provides the Finnhub implementation of the FundamentalsProvider protocol.
It supports:
- Basic financials (P/E, market cap, 52w range)
- Insider transactions and sentiment
- SEC filings
- EPS surprises and earnings calendar

Rate limiting: Free tier = 60 calls/min across ALL endpoints.
Caching strategy:
- Financials: 1 hour (slow-changing)
- Insider txns/sentiment: 6 hours (filed daily)
- Earnings calendar: 24 hours (known in advance)
- SEC filings: 6 hours
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

from synesis.config import get_settings
from synesis.core.constants import (
    FINNHUB_CACHE_TTL_EARNINGS,
    FINNHUB_CACHE_TTL_FILINGS,
    FINNHUB_CACHE_TTL_FINANCIALS,
    FINNHUB_CACHE_TTL_INSIDER,
)
from synesis.core.logging import get_logger
from synesis.providers.finnhub.prices import get_rate_limiter

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)

# Redis key prefix
CACHE_PREFIX = "synesis:finnhub"


class FinnhubFundamentalsProvider:
    """Finnhub fundamentals provider for company financial data.

    This provider implements the FundamentalsProvider protocol and provides:
    - get_basic_financials(ticker) - P/E, market cap, etc.
    - get_insider_transactions(ticker) - Recent insider buys/sells
    - get_insider_sentiment(ticker) - MSPR score
    - get_earnings_calendar(ticker) - Next earnings date
    - get_eps_surprises(ticker) - Historical EPS surprises
    - get_sec_filings(ticker) - Recent SEC filings

    Uses Redis caching to minimize API calls within rate limits.

    Usage:
        provider = FinnhubFundamentalsProvider(api_key="your_key", redis=redis_client)
        financials = await provider.get_basic_financials("AAPL")
        await provider.close()
    """

    def __init__(self, api_key: str, redis: Redis) -> None:
        """Initialize FinnhubFundamentalsProvider.

        Args:
            api_key: Finnhub API key
            redis: Redis client for caching
        """
        self._api_key = api_key
        self._redis = redis
        self._http_client: httpx.AsyncClient | None = None

    def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def _get_cached(self, key: str) -> bytes | str | None:
        """Get value from Redis cache."""
        result: bytes | str | None = await self._redis.get(key)
        return result

    async def _set_cached(self, key: str, value: str, ttl: int) -> None:
        """Set value in Redis cache with TTL."""
        await self._redis.set(key, value, ex=ttl)

    async def _fetch_finnhub(
        self,
        endpoint: str,
        params: dict[str, str | int],
    ) -> Any:
        """Fetch data from Finnhub API.

        Args:
            endpoint: API endpoint path (e.g., "/stock/metric")
            params: Query parameters

        Returns:
            Parsed JSON response, or None if request failed
        """
        import orjson

        # Use global rate limiter to prevent exceeding Finnhub API limits
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
    # ─────────────────────────────────────────────────────────────

    async def get_basic_financials(self, ticker: str) -> dict[str, Any] | None:
        """Get key financial metrics (Protocol method).

        Endpoint: /stock/metric?symbol={ticker}&metric=all

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            Dict with financial metrics, or None if not available.
            Keys include: peRatio, marketCap, 52WeekHigh, 52WeekLow, etc.
        """
        import orjson

        ticker = ticker.upper()
        cache_key = f"{CACHE_PREFIX}:financials:{ticker}"

        # Check cache
        cached = await self._get_cached(cache_key)
        if cached:
            try:
                result: dict[str, Any] = orjson.loads(cached)
                return result
            except Exception:
                pass

        # Fetch from API
        data = await self._fetch_finnhub("/stock/metric", {"symbol": ticker, "metric": "all"})
        if data is None or not isinstance(data, dict):
            return None

        # Extract relevant metrics from the nested structure
        metrics = data.get("metric", {})
        if not metrics:
            logger.debug("No metrics available", ticker=ticker)
            return None

        # Build simplified result
        result = {
            "ticker": ticker,
            "peRatio": metrics.get("peBasicExclExtraTTM"),
            "marketCap": metrics.get("marketCapitalization"),  # in millions
            "52WeekHigh": metrics.get("52WeekHigh"),
            "52WeekLow": metrics.get("52WeekLow"),
            "beta": metrics.get("beta"),
            "eps": metrics.get("epsBasicExclExtraItemsTTM"),
            "revenueGrowth": metrics.get("revenueGrowthTTMYoy"),
            "dividendYield": metrics.get("dividendYieldIndicatedAnnual"),
            "priceToBook": metrics.get("pbAnnual"),
            "priceToSales": metrics.get("psAnnual"),
            "roeTTM": metrics.get("roeTTM"),
            "debtToEquity": metrics.get("totalDebt/totalEquityAnnual"),
        }

        # Cache the result
        await self._set_cached(
            cache_key, orjson.dumps(result).decode(), FINNHUB_CACHE_TTL_FINANCIALS
        )
        logger.debug("Fetched basic financials", ticker=ticker)

        return result

    async def get_insider_transactions(self, ticker: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent insider transactions (Protocol method).

        Endpoint: /stock/insider-transactions?symbol={ticker}

        Args:
            ticker: Stock ticker symbol
            limit: Maximum transactions to return (default 10)

        Returns:
            List of insider transactions with: name, share, change, filingDate, transactionCode
        """
        import orjson

        ticker = ticker.upper()
        cache_key = f"{CACHE_PREFIX}:insider_txns:{ticker}"

        # Check cache
        cached = await self._get_cached(cache_key)
        if cached:
            try:
                cached_list: list[dict[str, Any]] = orjson.loads(cached)
                return cached_list[:limit]
            except Exception:
                pass

        # Fetch from API
        data = await self._fetch_finnhub("/stock/insider-transactions", {"symbol": ticker})
        if data is None or not isinstance(data, dict):
            return []

        transactions = data.get("data", [])
        if not transactions:
            return []

        # Simplify the data
        result: list[dict[str, Any]] = []
        for txn in transactions[:limit]:
            result.append(
                {
                    "name": txn.get("name"),
                    "shares": txn.get("share"),
                    "change": txn.get("change"),
                    "filingDate": txn.get("filingDate"),
                    "transactionCode": txn.get("transactionCode"),  # P=purchase, S=sale
                    "transactionPrice": txn.get("transactionPrice"),
                }
            )

        # Cache the result
        await self._set_cached(cache_key, orjson.dumps(result).decode(), FINNHUB_CACHE_TTL_INSIDER)
        logger.debug("Fetched insider transactions", ticker=ticker, count=len(result))

        return result

    async def get_insider_sentiment(self, ticker: str) -> dict[str, Any] | None:
        """Get aggregate insider sentiment (MSPR score) (Protocol method).

        Endpoint: /stock/insider-sentiment?symbol={ticker}

        The MSPR (Monthly Share Purchase Ratio) indicates net insider sentiment:
        - Positive: More buying than selling
        - Negative: More selling than buying

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with: mspr, change, symbol, or None if not available
        """
        import orjson

        ticker = ticker.upper()
        cache_key = f"{CACHE_PREFIX}:insider_sentiment:{ticker}"

        # Check cache
        cached = await self._get_cached(cache_key)
        if cached:
            try:
                cached_result: dict[str, Any] = orjson.loads(cached)
                return cached_result
            except Exception:
                pass

        # Fetch from API
        data = await self._fetch_finnhub(
            "/stock/insider-sentiment", {"symbol": ticker, "from": "2020-01-01"}
        )
        if data is None or not isinstance(data, dict):
            return None

        sentiment_data = data.get("data", [])
        if not sentiment_data:
            return None

        # Get the most recent sentiment data
        latest = sentiment_data[-1] if sentiment_data else None
        if not latest:
            return None

        result: dict[str, Any] = {
            "ticker": ticker,
            "mspr": latest.get("mspr"),  # Monthly Share Purchase Ratio
            "change": latest.get("change"),  # Net change in shares
            "year": latest.get("year"),
            "month": latest.get("month"),
        }

        # Cache the result
        await self._set_cached(cache_key, orjson.dumps(result).decode(), FINNHUB_CACHE_TTL_INSIDER)
        logger.debug("Fetched insider sentiment", ticker=ticker, mspr=result.get("mspr"))

        return result

    async def get_sec_filings(self, ticker: str, limit: int = 5) -> list[dict[str, Any]]:
        """Get recent SEC filings (Protocol method).

        Endpoint: /stock/filings?symbol={ticker}

        Args:
            ticker: Stock ticker symbol
            limit: Maximum filings to return (default 5)

        Returns:
            List of SEC filings with: form, filedDate, acceptedDate, accessNumber
        """
        import orjson

        ticker = ticker.upper()
        cache_key = f"{CACHE_PREFIX}:sec_filings:{ticker}"

        # Check cache
        cached = await self._get_cached(cache_key)
        if cached:
            try:
                cached_list: list[dict[str, Any]] = orjson.loads(cached)
                return cached_list[:limit]
            except Exception:
                pass

        # Fetch from API
        data = await self._fetch_finnhub("/stock/filings", {"symbol": ticker})
        if data is None or not isinstance(data, list):
            return []

        # Simplify and limit the data
        result: list[dict[str, Any]] = []
        for filing in data[:limit]:
            result.append(
                {
                    "form": filing.get("form"),  # 10-K, 10-Q, 8-K, etc.
                    "filedDate": filing.get("filedDate"),
                    "acceptedDate": filing.get("acceptedDate"),
                    "accessNumber": filing.get("accessNumber"),
                    "reportUrl": filing.get("reportUrl"),
                }
            )

        # Cache the result
        await self._set_cached(cache_key, orjson.dumps(result).decode(), FINNHUB_CACHE_TTL_FILINGS)
        logger.debug("Fetched SEC filings", ticker=ticker, count=len(result))

        return result

    async def get_eps_surprises(self, ticker: str, limit: int = 4) -> list[dict[str, Any]]:
        """Get historical EPS surprises (Protocol method).

        Endpoint: /stock/earnings?symbol={ticker}

        Args:
            ticker: Stock ticker symbol
            limit: Number of quarters to return (default 4)

        Returns:
            List of earnings with: period, actual, estimate, surprise, surprisePercent
        """
        import orjson

        ticker = ticker.upper()
        cache_key = f"{CACHE_PREFIX}:eps_surprises:{ticker}"

        # Check cache
        cached = await self._get_cached(cache_key)
        if cached:
            try:
                cached_list: list[dict[str, Any]] = orjson.loads(cached)
                return cached_list[:limit]
            except Exception:
                pass

        # Fetch from API
        data = await self._fetch_finnhub("/stock/earnings", {"symbol": ticker})
        if data is None or not isinstance(data, list):
            return []

        # Data is already in a good format, just limit it
        result: list[dict[str, Any]] = []
        for earning in data[:limit]:
            result.append(
                {
                    "period": earning.get("period"),
                    "actual": earning.get("actual"),
                    "estimate": earning.get("estimate"),
                    "surprise": earning.get("surprise"),
                    "surprisePercent": earning.get("surprisePercent"),
                }
            )

        # Cache the result
        await self._set_cached(cache_key, orjson.dumps(result).decode(), FINNHUB_CACHE_TTL_EARNINGS)
        logger.debug("Fetched EPS surprises", ticker=ticker, count=len(result))

        return result

    async def get_earnings_calendar(self, ticker: str) -> dict[str, Any] | None:
        """Get next earnings date (Protocol method).

        Endpoint: /calendar/earnings?symbol={ticker}

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with: date, epsEstimate, hour (bmo=before market, amc=after market)
        """
        import orjson

        ticker = ticker.upper()
        cache_key = f"{CACHE_PREFIX}:earnings_calendar:{ticker}"

        # Check cache
        cached = await self._get_cached(cache_key)
        if cached:
            try:
                cached_result: dict[str, Any] = orjson.loads(cached)
                return cached_result
            except Exception:
                pass

        # Get date range for next 90 days
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        future = datetime.now(UTC).replace(day=1)
        # Add 3 months
        if future.month > 9:
            future = future.replace(year=future.year + 1, month=future.month - 9)
        else:
            future = future.replace(month=future.month + 3)
        to_date = future.strftime("%Y-%m-%d")

        # Fetch from API
        data = await self._fetch_finnhub(
            "/calendar/earnings", {"symbol": ticker, "from": today, "to": to_date}
        )
        if data is None or not isinstance(data, dict):
            return None

        earnings_data = data.get("earningsCalendar", [])
        if not earnings_data:
            return None

        # Get the next upcoming earnings (first in the list)
        next_earning = earnings_data[0]

        result: dict[str, Any] = {
            "ticker": ticker,
            "date": next_earning.get("date"),
            "epsEstimate": next_earning.get("epsEstimate"),
            "epsActual": next_earning.get("epsActual"),
            "revenueEstimate": next_earning.get("revenueEstimate"),
            "hour": next_earning.get("hour"),  # bmo, amc, dmh
        }

        # Cache the result
        await self._set_cached(cache_key, orjson.dumps(result).decode(), FINNHUB_CACHE_TTL_EARNINGS)
        logger.debug("Fetched earnings calendar", ticker=ticker, date=result.get("date"))

        return result

    # ─────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Clean up resources (Protocol method)."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.debug("FinnhubFundamentalsProvider closed")
