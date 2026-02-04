"""Finnhub ticker provider implementation.

This module provides the Finnhub implementation of the TickerProvider protocol.
It supports:
- Ticker verification (check if symbol exists on major exchanges)
- Symbol search (find tickers matching a query)
- Redis caching to minimize API calls
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx

from synesis.config import get_settings
from synesis.core.constants import FINNHUB_CACHE_TTL_SYMBOL, FINNHUB_CACHE_TTL_US_SYMBOLS
from synesis.core.logging import get_logger
from synesis.providers.finnhub.prices import get_rate_limiter

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)

# Redis key prefix
CACHE_PREFIX = "synesis:finnhub"


class FinnhubTickerProvider:
    """Finnhub ticker provider for symbol validation and search.

    This provider implements the TickerProvider protocol and provides:
    - verify_ticker(ticker) - Check if a ticker exists on major exchanges
    - search_symbol(query) - Search for symbols matching a query

    Uses bulk US symbol list for instant ticker verification (no API call per ticker).
    Falls back to search endpoint if bulk list unavailable.

    Usage:
        provider = FinnhubTickerProvider(api_key="your_key", redis=redis_client)
        is_valid, company = await provider.verify_ticker("AAPL")
        results = await provider.search_symbol("Apple")
        await provider.close()
    """

    # Cache key for bulk US symbols
    US_SYMBOLS_CACHE_KEY = f"{CACHE_PREFIX}:us_symbols"

    def __init__(self, api_key: str, redis: Redis) -> None:
        """Initialize FinnhubTickerProvider.

        Args:
            api_key: Finnhub API key
            redis: Redis client for caching
        """
        self._api_key = api_key
        self._redis = redis
        self._http_client: httpx.AsyncClient | None = None
        # In-memory cache for US symbols (symbol -> company name)
        self._us_symbols: dict[str, str] | None = None
        # Lock to prevent concurrent bulk symbol fetches
        self._symbols_lock = asyncio.Lock()

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
            endpoint: API endpoint path (e.g., "/search")
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

    async def _load_us_symbols(self) -> dict[str, str]:
        """Load all US stock symbols from Finnhub (cached for 24h).

        Fetches /stock/symbol?exchange=US which returns ~10k symbols.
        Caches in Redis and in-memory for instant ticker verification.
        Uses a lock to prevent concurrent fetches.

        Returns:
            Dict mapping symbol -> company name
        """
        import orjson

        # Return in-memory cache if available (fast path, no lock needed)
        if self._us_symbols is not None:
            return self._us_symbols

        # Use lock to prevent concurrent fetches
        async with self._symbols_lock:
            # Double-check after acquiring lock
            if self._us_symbols is not None:
                return self._us_symbols

            # Check Redis cache
            cached = await self._get_cached(self.US_SYMBOLS_CACHE_KEY)
            if cached:
                try:
                    self._us_symbols = orjson.loads(cached)
                    logger.info(
                        "Loaded US symbols from cache",
                        count=len(self._us_symbols) if self._us_symbols else 0,
                    )
                    return self._us_symbols or {}
                except Exception as e:
                    logger.warning(
                        "Failed to parse US symbols cache, will fetch from API", error=str(e)
                    )

            # Fetch from Finnhub API
            logger.info("Fetching all US symbols from Finnhub...")
            data = await self._fetch_finnhub("/stock/symbol", {"exchange": "US"})

            if data is None or not isinstance(data, list):
                logger.warning("Failed to fetch US symbols, falling back to search")
                self._us_symbols = {}
                return {}

            # Build symbol -> company name mapping
            symbols: dict[str, str] = {}
            for item in data:
                symbol = item.get("symbol", "")
                description = item.get("description", "")
                symbol_type = item.get("type", "")
                # Include common stocks, ETFs, ADRs, REITs
                if symbol and symbol_type in ("Common Stock", "ETF", "ADR", "REIT", ""):
                    symbols[symbol.upper()] = description

            # Cache in Redis (24 hours)
            await self._set_cached(
                self.US_SYMBOLS_CACHE_KEY,
                orjson.dumps(symbols).decode(),
                FINNHUB_CACHE_TTL_US_SYMBOLS,
            )

            # Cache in memory
            self._us_symbols = symbols
            logger.info("Loaded US symbols from Finnhub API", count=len(symbols))

            return symbols

    # ─────────────────────────────────────────────────────────────
    # TickerProvider Protocol Implementation
    # ─────────────────────────────────────────────────────────────

    async def search_symbol(self, query: str) -> list[dict[str, str]]:
        """Search for stock symbols matching a query (Protocol method).

        Endpoint: /search?q={query}

        Args:
            query: Search query (ticker or company name)

        Returns:
            List of matching symbols with: symbol, description, type
        """
        import orjson

        query = query.upper()
        cache_key = f"{CACHE_PREFIX}:symbol_search:{query}"

        # Check cache
        cached = await self._get_cached(cache_key)
        if cached:
            try:
                cached_list: list[dict[str, str]] = orjson.loads(cached)
                return cached_list
            except Exception as e:
                logger.warning(
                    "Failed to parse symbol search cache, will fetch from API",
                    error=str(e),
                    query=query,
                )

        # Fetch from API
        data = await self._fetch_finnhub("/search", {"q": query})
        if data is None or not isinstance(data, dict):
            return []

        results = data.get("result", [])
        if not results:
            return []

        # Filter to common stock types and simplify
        result: list[dict[str, str]] = []
        for item in results[:10]:  # Limit to top 10
            symbol_type = item.get("type", "")
            # Include common stocks, ETFs, ADRs
            if symbol_type in ("Common Stock", "ETF", "ADR", "REIT", ""):
                result.append(
                    {
                        "symbol": item.get("symbol", ""),
                        "description": item.get("description", ""),
                        "type": symbol_type,
                    }
                )

        # Cache the result
        await self._set_cached(cache_key, orjson.dumps(result).decode(), FINNHUB_CACHE_TTL_SYMBOL)
        logger.debug("Symbol search complete", query=query, count=len(result))

        return result

    async def verify_ticker(self, ticker: str) -> tuple[bool, str | None]:
        """Verify if a ticker symbol exists on a major exchange (Protocol method).

        Uses bulk US symbol list for instant lookup (no API call per ticker).
        Falls back to search endpoint if bulk list unavailable.

        Args:
            ticker: Stock ticker symbol to verify (e.g., "AAPL")

        Returns:
            Tuple of (is_valid, company_name):
            - is_valid: True if ticker exists on major exchange
            - company_name: Company name if found, None otherwise
        """
        ticker = ticker.upper()

        # Try bulk symbol lookup first (instant, no API call)
        us_symbols = await self._load_us_symbols()
        if us_symbols:
            if ticker in us_symbols:
                company_name = us_symbols[ticker]
                logger.debug("Ticker verified (bulk)", ticker=ticker, company=company_name)
                return True, company_name
            else:
                logger.debug("Ticker not in US symbols", ticker=ticker)
                return False, None

        # Fallback to search endpoint if bulk list unavailable
        results = await self.search_symbol(ticker)

        if not results:
            logger.debug("Ticker not found", ticker=ticker)
            return False, None

        # Look for exact symbol match
        for item in results:
            if item.get("symbol", "").upper() == ticker:
                company_name = item.get("description", "")
                logger.debug(
                    "Ticker verified (search)",
                    ticker=ticker,
                    company=company_name,
                    type=item.get("type"),
                )
                return True, company_name

        # No exact match found
        logger.debug(
            "No exact ticker match",
            ticker=ticker,
            similar=[r.get("symbol") for r in results[:3]],
        )
        return False, None

    # ─────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Clean up resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.debug("FinnhubTickerProvider closed")
