"""Multi-platform WebSocket manager for market data streams.

Manages WebSocket connections for both Polymarket and Kalshi.
Handles subscription lifecycle — subscribes/unsubscribes markets
as the tracked set changes based on scan results from the caller.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from synesis.core.constants import MARKET_INTEL_MAX_TRACKED_MARKETS, MARKET_INTEL_REDIS_PREFIX
from synesis.core.logging import get_logger
from synesis.markets.kalshi_ws import KalshiWSClient
from synesis.markets.models import UnifiedMarket
from synesis.markets.polymarket_ws import PolymarketWSClient

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)

_PRICE_PREFIX = f"{MARKET_INTEL_REDIS_PREFIX}:ws:price"
_VOLUME_PREFIX = f"{MARKET_INTEL_REDIS_PREFIX}:ws:volume_1h"


class MarketWSManager:
    """Manages WebSocket connections for both platforms.

    Handles subscription lifecycle — subscribes/unsubscribes markets
    as the tracked set changes based on scan results from the caller.
    """

    def __init__(
        self,
        poly_ws: PolymarketWSClient,
        kalshi_ws: KalshiWSClient,
        redis: Redis,
    ) -> None:
        self._poly_ws = poly_ws
        self._kalshi_ws = kalshi_ws
        self._redis = redis

        # Track current subscriptions by platform
        self._poly_markets: dict[str, str] = {}  # external_id -> condition_id/token
        self._kalshi_markets: dict[str, str] = {}  # external_id -> ticker

    @property
    def is_connected(self) -> bool:
        return self._poly_ws.is_connected or self._kalshi_ws.is_connected

    @property
    def poly_connected(self) -> bool:
        return self._poly_ws.is_connected

    @property
    def kalshi_connected(self) -> bool:
        return self._kalshi_ws.is_connected

    @property
    def total_subscribed(self) -> int:
        return self._poly_ws.subscribed_count + self._kalshi_ws.subscribed_count

    async def start(self) -> None:
        """Start both WebSocket clients."""
        await asyncio.gather(self._poly_ws.start(), self._kalshi_ws.start())
        logger.info("MarketWSManager started")

    async def stop(self) -> None:
        """Stop both WebSocket clients."""
        results = await asyncio.gather(
            self._poly_ws.stop(), self._kalshi_ws.stop(), return_exceptions=True
        )
        for r in results:
            if isinstance(r, Exception):
                logger.error("WS client stop failed", error=str(r))
        logger.info("MarketWSManager stopped")

    async def update_subscriptions(self, markets: list[UnifiedMarket]) -> None:
        """Update which markets we're tracking based on latest scan.

        Called by processor after each 15-min REST scan.
        Subscribes to newly interesting markets, unsubscribes from stale ones.
        Respects MARKET_INTEL_MAX_TRACKED_MARKETS limit.
        """
        # Split by platform
        poly_new: dict[str, str] = {}
        kalshi_new: dict[str, str] = {}

        for m in markets[:MARKET_INTEL_MAX_TRACKED_MARKETS]:
            if m.platform == "polymarket" and m.condition_id:
                poly_new[m.external_id] = m.condition_id
            elif m.platform == "kalshi" and m.ticker:
                kalshi_new[m.external_id] = m.ticker

        # Polymarket: subscribe new, unsubscribe stale
        poly_to_add = [cid for eid, cid in poly_new.items() if eid not in self._poly_markets]
        poly_to_remove = [cid for eid, cid in self._poly_markets.items() if eid not in poly_new]
        if poly_to_remove:
            await self._poly_ws.unsubscribe(poly_to_remove)
        if poly_to_add:
            await self._poly_ws.subscribe(poly_to_add)
        self._poly_markets = poly_new

        # Kalshi: subscribe new, unsubscribe stale
        kalshi_to_add = [
            ticker for eid, ticker in kalshi_new.items() if eid not in self._kalshi_markets
        ]
        kalshi_to_remove = [
            ticker for eid, ticker in self._kalshi_markets.items() if eid not in kalshi_new
        ]
        if kalshi_to_remove:
            await self._kalshi_ws.unsubscribe(kalshi_to_remove)
        if kalshi_to_add:
            await self._kalshi_ws.subscribe(kalshi_to_add)
        self._kalshi_markets = kalshi_new

        logger.debug(
            "WS subscriptions updated",
            poly=len(self._poly_markets),
            kalshi=len(self._kalshi_markets),
            added_poly=len(poly_to_add),
            added_kalshi=len(kalshi_to_add),
            removed_poly=len(poly_to_remove),
            removed_kalshi=len(kalshi_to_remove),
        )

    async def get_realtime_volume(self, platform: str, market_id: str) -> float | None:
        """Read accumulated 1h volume from Redis (non-destructive)."""
        key = f"{_VOLUME_PREFIX}:{platform}:{market_id}"
        val = await self._redis.get(key)
        if val:
            try:
                return float(val)
            except (ValueError, TypeError):
                logger.warning("Volume parse failed", key=key, raw_value=repr(val))
        return None

    async def read_and_reset_volume(self, platform: str, market_id: str) -> float | None:
        """Atomically read and delete accumulated volume from Redis (GETDEL).

        Used at snapshot time to capture the hour's volume and reset the counter.
        """
        key = f"{_VOLUME_PREFIX}:{platform}:{market_id}"
        val = await self._redis.getdel(key)
        if val:
            try:
                return float(val)
            except (ValueError, TypeError):
                logger.error(
                    "GETDEL volume parse failed (data lost)",
                    key=key,
                    raw_value=repr(val),
                )
        return None

    async def get_realtime_price(self, platform: str, market_id: str) -> tuple[float, float] | None:
        """Read latest price from Redis (written by WebSocket).

        Returns (yes_price, no_price) or None if no data.
        """
        key = f"{_PRICE_PREFIX}:{platform}:{market_id}"
        data = await self._redis.hgetall(key)  # type: ignore[misc]
        if data and b"price" in data:
            try:
                yes_price = float(data[b"price"])
                no_price = 1.0 - yes_price
                return (yes_price, no_price)
            except (ValueError, TypeError):
                logger.warning("Price parse failed", key=key, raw_value=repr(data))
        return None
