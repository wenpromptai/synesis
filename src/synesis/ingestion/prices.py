"""Finnhub price service with WebSocket + REST + Redis cache.

This module provides real-time stock price data for Flow 1 and Flow 2 signal tracking:
- WebSocket: Persistent connection for real-time price updates
- REST: Fallback for cache misses and outcome verification
- Redis: Fast cache for instant price lookups when signals fire

Architecture:
    ┌──────────────────────────────────────────────────────┐
    │ FINNHUB WEBSOCKET (persistent connection)            │
    │   Subscribe to watchlist tickers                     │
    │     ↓                                                │
    │   On trade: update Redis cache                       │
    └──────────────────────────────────────────────────────┘
                       ↓ (read from cache)
    ┌──────────────────────────────────────────────────────┐
    │ Signal fires → read prices → store on signal         │
    └──────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING

import httpx
import orjson
import websockets
import websockets.exceptions
from websockets.asyncio.client import ClientConnection

from synesis.config import get_settings
from synesis.core.constants import (
    FINNHUB_RATE_LIMIT_CALLS_PER_MINUTE,
    PRICE_CACHE_TTL_SECONDS,
)
from synesis.core.logging import get_logger

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)

# Redis key prefix for price cache
PRICE_CACHE_PREFIX = "synesis:prices"


class RateLimiter:
    """Token bucket rate limiter for API calls.

    Ensures we don't exceed Finnhub's rate limit even with parallel calls.
    """

    def __init__(self, calls_per_minute: int = FINNHUB_RATE_LIMIT_CALLS_PER_MINUTE) -> None:
        self._calls_per_minute = calls_per_minute
        self._calls: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire permission to make an API call, waiting if necessary."""
        async with self._lock:
            now = asyncio.get_event_loop().time()

            # Remove calls older than 60 seconds
            self._calls = [t for t in self._calls if t > now - 60]

            if len(self._calls) >= self._calls_per_minute:
                # Wait until oldest call expires
                sleep_time = 60 - (now - self._calls[0]) + 0.1
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    # Re-check after sleep
                    now = asyncio.get_event_loop().time()
                    self._calls = [t for t in self._calls if t > now - 60]

            self._calls.append(now)


# Global rate limiter instance for Finnhub REST API (eager init to avoid race condition)
_finnhub_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    """Get the global Finnhub rate limiter."""
    return _finnhub_rate_limiter


class PriceService:
    """Finnhub price service with WebSocket + REST + Redis cache.

    This service provides stock price data for signal tracking:
    - get_cached_price(ticker) - instant lookup from Redis
    - get_cached_prices(tickers) - batch lookup for multiple tickers
    - fetch_quote(ticker) - REST API fallback
    - start_websocket() - real-time price streaming

    Usage:
        service = PriceService(api_key="your_key", redis=redis_client)

        # Start WebSocket for real-time updates (optional)
        await service.start_websocket()
        await service.subscribe(["AAPL", "TSLA"])

        # Get prices when signal fires
        prices = await service.get_cached_prices(["AAPL", "TSLA"])

        # Cleanup
        await service.close()
    """

    def __init__(self, api_key: str, redis: Redis) -> None:
        """Initialize PriceService.

        Args:
            api_key: Finnhub API key
            redis: Redis client for price caching
        """
        self._api_key = api_key
        self._redis = redis
        self._ws: ClientConnection | None = None
        self._ws_task: asyncio.Task[None] | None = None
        self._subscribed_tickers: set[str] = set()
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        self._running = False
        self._http_client: httpx.AsyncClient | None = None

    # ─────────────────────────────────────────────────────────────
    # Redis Cache
    # ─────────────────────────────────────────────────────────────

    async def get_cached_price(self, ticker: str) -> Decimal | None:
        """Get latest price from Redis cache.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            Price as Decimal, or None if not cached
        """
        key = f"{PRICE_CACHE_PREFIX}:{ticker.upper()}"
        price_str = await self._redis.get(key)
        if price_str:
            try:
                return Decimal(price_str)
            except Exception:
                logger.warning("Invalid cached price", ticker=ticker, value=price_str)
        return None

    async def get_cached_prices(self, tickers: list[str]) -> dict[str, Decimal]:
        """Get latest prices for multiple tickers from Redis cache.

        Args:
            tickers: List of stock ticker symbols

        Returns:
            Dict mapping ticker to price (only includes tickers with cached prices)
        """
        if not tickers:
            return {}

        prices: dict[str, Decimal] = {}
        for ticker in tickers:
            price = await self.get_cached_price(ticker)
            if price is not None:
                prices[ticker.upper()] = price
        return prices

    async def set_cached_price(
        self,
        ticker: str,
        price: Decimal | float,
        ttl: int = PRICE_CACHE_TTL_SECONDS,
    ) -> None:
        """Update price in Redis cache.

        Args:
            ticker: Stock ticker symbol
            price: Current price
            ttl: Cache TTL in seconds (default 30 min)
        """
        key = f"{PRICE_CACHE_PREFIX}:{ticker.upper()}"
        await self._redis.set(key, str(price), ex=ttl)

    async def set_cached_prices(
        self,
        prices: dict[str, Decimal | float],
        ttl: int = PRICE_CACHE_TTL_SECONDS,
    ) -> None:
        """Update multiple prices in Redis cache.

        Args:
            prices: Dict mapping ticker to price
            ttl: Cache TTL in seconds
        """
        for ticker, price in prices.items():
            await self.set_cached_price(ticker, price, ttl)

    # ─────────────────────────────────────────────────────────────
    # WebSocket (real-time)
    # ─────────────────────────────────────────────────────────────

    async def start_websocket(self) -> None:
        """Start WebSocket connection for real-time prices.

        This runs the WebSocket in a background task. Call subscribe()
        to add tickers to the real-time feed.
        """
        if self._running:
            logger.warning("WebSocket already running")
            return

        self._running = True
        self._ws_task = asyncio.create_task(self._ws_loop())
        logger.info("PriceService WebSocket started")

    async def stop_websocket(self) -> None:
        """Stop WebSocket connection."""
        self._running = False

        if self._ws:
            await self._ws.close()
            self._ws = None

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None

        logger.info("PriceService WebSocket stopped")

    async def _ws_loop(self) -> None:
        """Main WebSocket loop with reconnection."""
        while self._running:
            try:
                await self._connect_and_listen()
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning("WebSocket connection closed", code=e.code, reason=e.reason)
            except Exception as e:
                logger.error("WebSocket error", error=str(e))

            if self._running:
                logger.info("Reconnecting WebSocket", delay=self._reconnect_delay)
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)

    async def _connect_and_listen(self) -> None:
        """Connect to Finnhub WebSocket and listen for trades."""
        settings = get_settings()
        url = f"{settings.finnhub_ws_url}?token={self._api_key}"

        try:
            async with websockets.connect(url) as ws:
                self._ws = ws
                self._reconnect_delay = 1.0  # Reset on successful connect

                logger.info(
                    "Connected to Finnhub WebSocket",
                    subscribed_tickers=len(self._subscribed_tickers),
                )

                # Re-subscribe to all tickers after reconnect
                for ticker in self._subscribed_tickers:
                    await self._send_subscribe(ticker)

                # Listen for messages
                async for message in ws:
                    await self._handle_ws_message(message)
        finally:
            self._ws = None  # Always clear reference to avoid dangling socket

    async def _handle_ws_message(self, message: str | bytes) -> None:
        """Handle incoming WebSocket message.

        Finnhub trade message format:
        {
            "type": "trade",
            "data": [
                {"s": "AAPL", "p": 150.25, "t": 1706812800000, "v": 100}
            ]
        }
        """
        try:
            if isinstance(message, bytes):
                data = orjson.loads(message)
            else:
                data = orjson.loads(message.encode())

            msg_type = data.get("type")

            if msg_type == "trade":
                trades = data.get("data", [])
                for trade in trades:
                    ticker = trade.get("s")
                    price = trade.get("p")
                    if ticker and price:
                        await self.set_cached_price(ticker, Decimal(str(price)))

            elif msg_type == "ping":
                # Finnhub sends periodic pings, no action needed
                pass

            elif msg_type == "error":
                logger.error("Finnhub WebSocket error", message=data)

        except Exception as e:
            logger.warning("Failed to parse WebSocket message", error=str(e))

    async def _send_subscribe(self, ticker: str) -> None:
        """Send subscribe message to WebSocket."""
        if self._ws:
            msg = orjson.dumps({"type": "subscribe", "symbol": ticker.upper()})
            await self._ws.send(msg)

    async def _send_unsubscribe(self, ticker: str) -> None:
        """Send unsubscribe message to WebSocket."""
        if self._ws:
            msg = orjson.dumps({"type": "unsubscribe", "symbol": ticker.upper()})
            await self._ws.send(msg)

    async def subscribe(self, tickers: list[str]) -> None:
        """Subscribe to real-time price updates for tickers.

        Args:
            tickers: List of ticker symbols to subscribe to
        """
        for ticker in tickers:
            ticker_upper = ticker.upper()
            if ticker_upper not in self._subscribed_tickers:
                self._subscribed_tickers.add(ticker_upper)
                if self._ws:
                    await self._send_subscribe(ticker_upper)
                logger.debug("Subscribed to ticker", ticker=ticker_upper)

    async def unsubscribe(self, tickers: list[str]) -> None:
        """Unsubscribe from real-time price updates.

        Args:
            tickers: List of ticker symbols to unsubscribe from
        """
        for ticker in tickers:
            ticker_upper = ticker.upper()
            if ticker_upper in self._subscribed_tickers:
                self._subscribed_tickers.discard(ticker_upper)
                if self._ws:
                    await self._send_unsubscribe(ticker_upper)
                logger.debug("Unsubscribed from ticker", ticker=ticker_upper)

    # ─────────────────────────────────────────────────────────────
    # REST API (fallback + outcome verification)
    # ─────────────────────────────────────────────────────────────

    def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def fetch_quote(self, ticker: str) -> Decimal | None:
        """Fetch current price via REST API.

        This is a fallback for when the cache is empty.
        Note: Rate limited to 60 calls/min on free tier.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Current price as Decimal, or None if fetch failed
        """
        # Use global rate limiter to prevent exceeding API limits
        rate_limiter = get_rate_limiter()
        await rate_limiter.acquire()

        settings = get_settings()
        client = self._get_http_client()
        url = f"{settings.finnhub_api_url}/quote"
        params = {"symbol": ticker.upper(), "token": self._api_key}

        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Finnhub quote response: {"c": 150.25, "h": 151, "l": 149, ...}
            current_price = data.get("c")
            if current_price and current_price > 0:
                price = Decimal(str(current_price))
                # Cache the fetched price
                await self.set_cached_price(ticker, price)
                logger.debug("Fetched quote via REST", ticker=ticker, price=str(price))
                return price
            else:
                logger.warning("Invalid quote response", ticker=ticker, data=data)
                return None

        except httpx.HTTPStatusError as e:
            logger.warning(
                "Quote fetch HTTP error",
                ticker=ticker,
                status=e.response.status_code,
            )
            return None
        except Exception as e:
            logger.warning("Quote fetch failed", ticker=ticker, error=str(e))
            return None

    async def fetch_quotes(self, tickers: list[str]) -> dict[str, Decimal]:
        """Fetch prices for multiple tickers via REST API.

        Includes rate limiting (60 calls/min on free tier).

        Args:
            tickers: List of ticker symbols

        Returns:
            Dict mapping ticker to price (only includes successful fetches)
        """
        prices: dict[str, Decimal] = {}

        for ticker in tickers:
            # Rate limiting is handled globally by fetch_quote()
            price = await self.fetch_quote(ticker)
            if price is not None:
                prices[ticker.upper()] = price

        return prices

    async def get_prices(
        self,
        tickers: list[str],
        fallback_to_rest: bool = True,
    ) -> dict[str, Decimal]:
        """Get prices for tickers, with optional REST fallback.

        First checks Redis cache, then falls back to REST API if needed.

        Args:
            tickers: List of ticker symbols
            fallback_to_rest: Whether to use REST API for cache misses

        Returns:
            Dict mapping ticker to price
        """
        if not tickers:
            return {}

        # First, try cache
        prices = await self.get_cached_prices(tickers)

        # Find missing tickers
        missing = [t for t in tickers if t.upper() not in prices]

        if missing and fallback_to_rest:
            logger.debug("Fetching missing prices via REST", missing=missing)
            rest_prices = await self.fetch_quotes(missing)
            prices.update(rest_prices)

        return prices

    # ─────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Clean up resources."""
        await self.stop_websocket()

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        logger.info("PriceService closed")


# Global price service instance (initialized in lifespan)
_price_service: PriceService | None = None


def get_price_service() -> PriceService:
    """Get the global price service instance."""
    if _price_service is None:
        raise RuntimeError("PriceService not initialized")
    return _price_service


async def init_price_service(api_key: str, redis: Redis) -> PriceService:
    """Initialize the global price service instance.

    Args:
        api_key: Finnhub API key
        redis: Redis client

    Returns:
        Initialized PriceService
    """
    global _price_service
    _price_service = PriceService(api_key, redis)
    return _price_service


async def close_price_service() -> None:
    """Close the global price service instance."""
    global _price_service
    if _price_service:
        await _price_service.close()
        _price_service = None
