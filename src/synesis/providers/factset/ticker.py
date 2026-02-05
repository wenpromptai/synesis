"""FactSet ticker provider implementation.

Bulk-loads all tickers from FactSet SQL Server into memory + Redis
for instant dict-based ticker verification. Replaces Finnhub for
ticker validation.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from synesis.core.logging import get_logger

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from synesis.providers.factset.client import FactSetClient

logger = get_logger(__name__)

# Redis key + TTL
CACHE_KEY = "synesis:factset:all_tickers"
CACHE_TTL = 86400  # 24 hours


class FactSetTickerProvider:
    """FactSet ticker provider — bulk-load + instant dict lookup.

    Implements the TickerProvider protocol:
    - verify_ticker(ticker) → (bool, company_name | None)
    - search_symbol(query) → list of matches

    On first call, fetches all equity-like tickers from FactSet SQL Server,
    builds a dict[bare_ticker, company_name] (US-preferred for conflicts),
    and caches in Redis (24h) + in-memory dict.

    Usage:
        provider = FactSetTickerProvider(factset_client, redis)
        is_valid, company = await provider.verify_ticker("AAPL")
    """

    def __init__(self, client: FactSetClient, redis: Redis) -> None:
        self._client = client
        self._redis = redis
        self._symbols: dict[str, str] | None = None
        self._lock = asyncio.Lock()

    async def _load_symbols(self) -> dict[str, str]:
        """Load all tickers from FactSet (cached in Redis + memory).

        Returns:
            Dict mapping bare ticker (e.g. "AAPL") → company name.
        """
        import orjson

        # Fast path: in-memory cache
        if self._symbols is not None:
            return self._symbols

        async with self._lock:
            # Double-check after lock
            if self._symbols is not None:
                return self._symbols

            # Check Redis
            cached = await self._redis.get(CACHE_KEY)
            if cached:
                try:
                    self._symbols = orjson.loads(cached)
                    logger.info(
                        "Loaded tickers from Redis cache",
                        count=len(self._symbols) if self._symbols else 0,
                    )
                    return self._symbols if self._symbols is not None else {}
                except Exception as e:
                    logger.warning("Failed to parse ticker cache", error=str(e))

            # Fetch from FactSet SQL Server
            logger.info("Fetching all tickers from FactSet...")
            from synesis.providers.factset.queries import ALL_TICKERS

            rows = await self._client.execute_query(ALL_TICKERS)

            # Build bare_ticker → company_name, preferring US region
            symbols: dict[str, str] = {}
            us_tickers: set[str] = set()

            for row in rows:
                ticker_region: str = row.get("ticker_region", "")
                proper_name: str = row.get("proper_name", "")

                if not ticker_region or "-" not in ticker_region:
                    continue

                bare, region = ticker_region.rsplit("-", 1)
                bare = bare.upper()

                if region == "US":
                    # US always wins
                    symbols[bare] = proper_name
                    us_tickers.add(bare)
                elif bare not in us_tickers:
                    # Non-US only if no US version exists
                    symbols[bare] = proper_name

            # Cache in Redis
            await self._redis.set(
                CACHE_KEY,
                orjson.dumps(symbols).decode(),
                ex=CACHE_TTL,
            )

            self._symbols = symbols
            logger.info("Loaded tickers from FactSet", count=len(symbols))
            return symbols

    # ─────────────────────────────────────────────────────────────
    # TickerProvider Protocol
    # ─────────────────────────────────────────────────────────────

    async def verify_ticker(self, ticker: str) -> tuple[bool, str | None]:
        """Check if a ticker exists. Instant dict lookup."""
        ticker = ticker.upper()
        symbols = await self._load_symbols()

        if ticker in symbols:
            company = symbols[ticker]
            logger.debug("Ticker verified", ticker=ticker, company=company)
            return True, company

        logger.debug("Ticker not found", ticker=ticker)
        return False, None

    async def search_symbol(self, query: str) -> list[dict[str, str]]:
        """Filter cached symbols matching query (prefix or substring)."""
        query = query.upper()
        symbols = await self._load_symbols()

        results: list[dict[str, str]] = []
        for symbol, name in symbols.items():
            if symbol.startswith(query) or query in name.upper():
                results.append({"symbol": symbol, "description": name, "type": ""})
                if len(results) >= 10:
                    break
        return results

    async def refresh(self) -> None:
        """Clear cache and re-fetch from FactSet."""
        self._symbols = None
        await self._redis.delete(CACHE_KEY)
        await self._load_symbols()
        logger.info("Ticker cache refreshed")

    async def close(self) -> None:
        """No resources to clean up (client lifecycle managed externally)."""
        self._symbols = None
        logger.debug("FactSetTickerProvider closed")
