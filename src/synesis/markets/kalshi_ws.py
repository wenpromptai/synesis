"""Kalshi WebSocket client for real-time market data.

Streams ticker updates and trade executions for subscribed markets.
Requires RSA-PSS authentication headers on the WebSocket handshake.

Pattern: Same as FinnhubPriceProvider._ws_loop() — connect, subscribe,
listen, reconnect with exponential backoff, cache in Redis.

WebSocket URL: wss://api.elections.kalshi.com/trade-api/ws/v2
"""

from __future__ import annotations

import asyncio
import itertools
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

# Redis key prefixes
_PRICE_PREFIX = f"{MARKET_INTEL_REDIS_PREFIX}:ws:price:kalshi"
_TRADES_PREFIX = f"{MARKET_INTEL_REDIS_PREFIX}:ws:trades:kalshi"
_VOLUME_PREFIX = f"{MARKET_INTEL_REDIS_PREFIX}:ws:volume_1h:kalshi"

# TTLs
_PRICE_TTL = 300  # 5 min
_VOLUME_TTL = 7200  # 2 hour safety net for unsubscribed markets

# Subscription message ID counter
_id_counter = itertools.count(1)


class KalshiWSClient:
    """Real-time Kalshi data via authenticated WebSocket channels."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        self._ws: ClientConnection | None = None
        self._ws_task: asyncio.Task[None] | None = None
        self._subscribed_tickers: set[str] = set()
        self._running = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._running

    @property
    def subscribed_count(self) -> int:
        return len(self._subscribed_tickers)

    async def start(self) -> None:
        """Start WebSocket in background task."""
        if self._running:
            logger.warning("Kalshi WS already running")
            return
        self._running = True
        self._ws_task = asyncio.create_task(self._ws_loop())
        logger.debug("Kalshi WS started")

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
        logger.debug("Kalshi WS stopped")

    async def subscribe(self, market_tickers: list[str]) -> None:
        """Subscribe to market tickers."""
        new_tickers = [t for t in market_tickers if t not in self._subscribed_tickers]
        if not new_tickers:
            return
        self._subscribed_tickers.update(new_tickers)
        if self._ws:
            await self._send_subscribe(new_tickers)

    async def unsubscribe(self, market_tickers: list[str]) -> None:
        """Unsubscribe from market tickers."""
        to_remove = [t for t in market_tickers if t in self._subscribed_tickers]
        if not to_remove:
            return
        for t in to_remove:
            self._subscribed_tickers.discard(t)
        if self._ws:
            await self._send_unsubscribe(to_remove)

    async def _ws_loop(self) -> None:
        """Main WebSocket loop with reconnection."""
        while self._running:
            try:
                await self._connect_and_listen()
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(
                    "Kalshi WS connection closed",
                    code=e.code,
                    reason=e.reason,
                )
            except Exception as e:
                logger.error("Kalshi WS error", error=str(e))

            if self._running:
                logger.debug("Kalshi WS reconnecting", delay=self._reconnect_delay)
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)

    async def _connect_and_listen(self) -> None:
        """Connect to WebSocket with RSA-PSS auth and listen for messages."""
        from urllib.parse import urlparse

        from synesis.markets.kalshi_auth import load_private_key, make_kalshi_headers

        settings = get_settings()
        url = settings.kalshi_ws_url

        # Build auth headers if credentials are configured
        extra_headers: dict[str, str] = {}
        api_key = settings.kalshi_api_key
        key_path = settings.kalshi_private_key_path

        if api_key and key_path:
            try:
                private_key = load_private_key(key_path)
                ws_path = urlparse(url).path or "/trade-api/ws/v2"
                extra_headers = make_kalshi_headers(
                    api_key=api_key.get_secret_value(),
                    private_key=private_key,
                    method="GET",
                    path=ws_path,
                )
            except Exception as e:
                logger.error("Failed to load Kalshi auth credentials", error=str(e))
                return
        else:
            logger.warning(
                "Kalshi WS credentials not configured — "
                "set KALSHI_API_KEY and KALSHI_PRIVATE_KEY_PATH"
            )
            return

        try:
            async with websockets.connect(url, additional_headers=extra_headers) as ws:
                self._ws = ws
                self._reconnect_delay = 1.0
                logger.debug(
                    "Kalshi WS connected",
                    subscribed=len(self._subscribed_tickers),
                )

                # Re-subscribe after reconnect
                if self._subscribed_tickers:
                    await self._send_subscribe(list(self._subscribed_tickers))

                async for message in ws:
                    await self._handle_message(message)
        finally:
            self._ws = None

    async def _send_cmd(self, cmd: str, tickers: list[str]) -> None:
        """Send subscribe/unsubscribe command for tickers."""
        if not self._ws or not tickers:
            return
        for channel in ("ticker", "trade"):
            msg = orjson.dumps(
                {
                    "id": next(_id_counter),
                    "cmd": cmd,
                    "params": {
                        "channels": [channel],
                        "market_tickers": tickers,
                    },
                }
            )
            try:
                await self._ws.send(msg)
            except Exception as e:
                logger.warning(
                    f"Kalshi WS {cmd} failed",
                    channel=channel,
                    error=str(e),
                )
        if cmd == "subscribe":
            logger.debug("Kalshi WS subscribed", count=len(tickers))

    async def _send_subscribe(self, tickers: list[str]) -> None:
        """Send subscription for tickers."""
        await self._send_cmd("subscribe", tickers)

    async def _send_unsubscribe(self, tickers: list[str]) -> None:
        """Send unsubscription for tickers."""
        await self._send_cmd("unsubscribe", tickers)

    async def _handle_message(self, raw: str | bytes) -> None:
        """Handle Kalshi events and cache in Redis."""
        try:
            data = orjson.loads(raw)

            msg_type = data.get("type")

            if msg_type == "ticker":
                msg_data = data.get("msg", {})
                ticker = msg_data.get("market_ticker", "")
                if not ticker:
                    return

                yes_bid = float(msg_data.get("yes_bid_dollars", 0) or 0)
                yes_ask = float(msg_data.get("yes_ask_dollars", 0) or 0)
                volume = int(msg_data.get("volume", 0) or 0)

                # Cache price
                key = f"{_PRICE_PREFIX}:{ticker}"
                mapping: dict[str, str] = {}
                if yes_bid > 0 and yes_ask > 0:
                    mid = (yes_bid + yes_ask) / 2
                    mapping["price"] = str(mid)
                    mapping["yes_bid"] = str(yes_bid)
                    mapping["yes_ask"] = str(yes_ask)
                if volume > 0:
                    mapping["volume"] = str(volume)
                if mapping:
                    await self._redis.hset(key, mapping=mapping)  # type: ignore[misc]
                    await self._redis.expire(key, _PRICE_TTL)
                    # Publish for real-time arb detection
                    if "price" in mapping:
                        await self._redis.publish(
                            PRICE_UPDATE_CHANNEL,
                            f"kalshi:{ticker}:{mapping['price']}",
                        )

            elif msg_type == "trade":
                msg_data = data.get("msg", {})
                ticker = msg_data.get("market_ticker", "")
                if not ticker:
                    return

                count = int(msg_data.get("count", 0) or 0)
                yes_price = float(msg_data.get("yes_price_dollars", 0) or 0)

                # Update price
                if yes_price > 0:
                    key = f"{_PRICE_PREFIX}:{ticker}"
                    await self._redis.hset(key, mapping={"price": str(yes_price)})  # type: ignore[misc]
                    await self._redis.expire(key, _PRICE_TTL)
                    # Publish for real-time arb detection
                    await self._redis.publish(
                        PRICE_UPDATE_CHANNEL,
                        f"kalshi:{ticker}:{yes_price}",
                    )

                # Increment volume
                if count > 0:
                    vol_key = f"{_VOLUME_PREFIX}:{ticker}"
                    await self._redis.incrbyfloat(vol_key, float(count))
                    await self._redis.expire(vol_key, _VOLUME_TTL)

        except Exception as e:
            logger.warning("Kalshi WS message parse error", error=str(e))
