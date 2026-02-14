"""Polymarket CLOB WebSocket client for real-time market data.

Streams price changes and last trade prices for subscribed markets.
No authentication required.

Pattern: Same as FinnhubPriceProvider._ws_loop() — connect, subscribe,
listen, reconnect with exponential backoff, cache in Redis.

WebSocket URL: wss://ws-subscriptions-clob.polymarket.com/ws/market
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import orjson
import websockets
import websockets.exceptions
from websockets.asyncio.client import ClientConnection

from synesis.config import get_settings
from synesis.core.constants import MARKET_INTEL_REDIS_PREFIX, PRICE_UPDATE_CHANNEL
from synesis.core.logging import get_logger

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)

# Redis key prefixes for real-time data
_PRICE_PREFIX = f"{MARKET_INTEL_REDIS_PREFIX}:ws:price:polymarket"
_TRADES_PREFIX = f"{MARKET_INTEL_REDIS_PREFIX}:ws:trades:polymarket"
_VOLUME_PREFIX = f"{MARKET_INTEL_REDIS_PREFIX}:ws:volume_1h:polymarket"

# TTLs
_PRICE_TTL = 300  # 5 min
_VOLUME_TTL = 7200  # 2 hour safety net for unsubscribed markets


class PolymarketWSClient:
    """Real-time Polymarket data via CLOB WebSocket.

    No auth required. Streams price changes and last trade prices
    for subscribed markets.
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        self._ws: ClientConnection | None = None
        self._ws_task: asyncio.Task[None] | None = None
        self._subscribed_tokens: set[str] = set()
        self._running = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._running

    @property
    def subscribed_count(self) -> int:
        return len(self._subscribed_tokens)

    async def start(self) -> None:
        """Start WebSocket in background task."""
        if self._running:
            logger.warning("Polymarket WS already running")
            return
        self._running = True
        self._ws_task = asyncio.create_task(self._ws_loop())
        logger.debug("Polymarket WS started")

    async def stop(self) -> None:
        """Stop WebSocket."""
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
        logger.debug("Polymarket WS stopped")

    async def subscribe(self, token_ids: list[str]) -> None:
        """Subscribe to market token IDs."""
        new_tokens = [t for t in token_ids if t not in self._subscribed_tokens]
        if not new_tokens:
            return
        self._subscribed_tokens.update(new_tokens)
        if self._ws:
            msg = orjson.dumps({"type": "market", "assets_ids": new_tokens})
            try:
                await self._ws.send(msg)
                logger.debug("Polymarket WS subscribed", count=len(new_tokens))
            except Exception as e:
                logger.warning("Polymarket WS subscribe failed", error=str(e))

    async def unsubscribe(self, token_ids: list[str]) -> None:
        """Unsubscribe from market token IDs."""
        to_remove = [t for t in token_ids if t in self._subscribed_tokens]
        for tid in to_remove:
            self._subscribed_tokens.discard(tid)
        if to_remove and self._ws:
            try:
                msg = orjson.dumps(
                    {"type": "market", "assets_ids": to_remove, "action": "unsubscribe"}
                )
                await self._ws.send(msg)
            except Exception as e:
                logger.warning("Polymarket WS unsubscribe failed", error=str(e))

    async def _ws_loop(self) -> None:
        """Main WebSocket loop with reconnection."""
        while self._running:
            try:
                await self._connect_and_listen()
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(
                    "Polymarket WS connection closed",
                    code=e.code,
                    reason=e.reason,
                )
            except Exception as e:
                logger.error("Polymarket WS error", error=str(e))

            if self._running:
                logger.debug(
                    "Polymarket WS reconnecting",
                    delay=self._reconnect_delay,
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)

    async def _connect_and_listen(self) -> None:
        """Connect to WebSocket and listen for messages."""
        settings = get_settings()
        url = settings.polymarket_clob_ws_url

        try:
            async with websockets.connect(url) as ws:
                self._ws = ws
                self._reconnect_delay = 1.0  # Reset on success
                logger.debug(
                    "Polymarket WS connected",
                    subscribed=len(self._subscribed_tokens),
                )

                # Re-subscribe after reconnect
                if self._subscribed_tokens:
                    await self._subscribe_all()

                async for message in ws:
                    await self._handle_message(message)
        finally:
            self._ws = None

    async def _subscribe_all(self) -> None:
        """Send subscription for all tracked tokens."""
        if self._ws and self._subscribed_tokens:
            # Send in batches of 50
            tokens = list(self._subscribed_tokens)
            for i in range(0, len(tokens), 50):
                batch = tokens[i : i + 50]
                msg = orjson.dumps({"type": "market", "assets_ids": batch})
                try:
                    await self._ws.send(msg)
                except Exception as e:
                    logger.warning("Polymarket WS re-subscribe failed", error=str(e))
                    break

    async def _handle_message(self, raw: str | bytes) -> None:
        """Handle CLOB WebSocket events and cache in Redis."""
        try:
            data = orjson.loads(raw)

            # Handle array of events
            events = data if isinstance(data, list) else [data]

            for event in events:
                event_type = event.get("event_type")
                asset_id = event.get("asset_id", "")

                if not asset_id:
                    continue

                if event_type == "last_trade_price":
                    price = float(event.get("price", 0))
                    if price > 0:
                        # Cache price
                        key = f"{_PRICE_PREFIX}:{asset_id}"
                        await self._redis.hset(  # type: ignore[misc]
                            key,
                            mapping={
                                "price": str(price),
                                "ts": event.get("timestamp", ""),
                            },
                        )
                        await self._redis.expire(key, _PRICE_TTL)

                        # Publish for real-time arb detection
                        await self._redis.publish(
                            PRICE_UPDATE_CHANNEL,
                            f"polymarket:{asset_id}:{price}",
                        )

                        # Increment volume counter
                        size = float(event.get("size", 0))
                        if size > 0:
                            vol_key = f"{_VOLUME_PREFIX}:{asset_id}"
                            await self._redis.incrbyfloat(vol_key, size)
                            await self._redis.expire(vol_key, _VOLUME_TTL)

                elif event_type == "price_change":
                    # Update best bid/ask
                    price = event.get("price")
                    if price:
                        key = f"{_PRICE_PREFIX}:{asset_id}"
                        await self._redis.hset(  # type: ignore[misc]
                            key,
                            mapping={
                                "price": str(price),
                                "ts": event.get("timestamp", ""),
                            },
                        )
                        await self._redis.expire(key, _PRICE_TTL)

                        # Publish for real-time arb detection
                        await self._redis.publish(
                            PRICE_UPDATE_CHANNEL,
                            f"polymarket:{asset_id}:{price}",
                        )

        except Exception as e:
            logger.warning("Polymarket WS message parse error", error=str(e))
