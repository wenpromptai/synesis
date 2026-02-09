"""Kalshi REST API client for market discovery (read-only).

The Kalshi Trade API v2 provides public market data without authentication.
Trading requires RSA-PSS signed requests (future phase).

API Base: https://api.elections.kalshi.com/trade-api/v2

IMPORTANT: Uses *_dollars and *_fp fields exclusively.
Cent-based fields are deprecated as of Feb 19, 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from typing import Any, Literal

import httpx

from synesis.config import get_settings
from synesis.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class KalshiMarket:
    """Kalshi market representation."""

    ticker: str
    event_ticker: str
    title: str
    subtitle: str | None
    status: Literal["open", "closed", "settled"]
    yes_bid: float  # Best YES bid (dollars, 0.0-1.0)
    yes_ask: float  # Best YES ask (dollars, 0.0-1.0)
    no_bid: float
    no_ask: float
    last_price: float
    volume: int  # Total contracts traded
    volume_24h: int
    open_interest: int
    close_time: datetime | None
    category: str | None
    result: Literal["yes", "no"] | None

    @property
    def yes_price(self) -> float:
        """Mid-price for YES."""
        if self.yes_bid > 0 and self.yes_ask > 0:
            return (self.yes_bid + self.yes_ask) / 2
        return self.last_price

    @property
    def no_price(self) -> float:
        """Mid-price for NO."""
        if self.no_bid > 0 and self.no_ask > 0:
            return (self.no_bid + self.no_ask) / 2
        return 1.0 - self.yes_price

    @property
    def url(self) -> str:
        return f"https://kalshi.com/markets/{self.event_ticker}/{self.ticker}"

    @property
    def is_active(self) -> bool:
        return self.status == "open"


@dataclass(frozen=True, slots=True)
class KalshiTrade:
    """A single Kalshi trade."""

    trade_id: str
    ticker: str
    count: int
    yes_price: float
    taker_side: Literal["yes", "no"]
    created_time: datetime


@dataclass(frozen=True, slots=True)
class KalshiEvent:
    """A Kalshi event (group of related markets)."""

    event_ticker: str
    title: str
    category: str | None
    status: str
    markets: list[KalshiMarket]


@dataclass(frozen=True, slots=True)
class OrderBookLevel:
    """A single orderbook level."""

    price: float
    quantity: int


@dataclass(frozen=True, slots=True)
class OrderBook:
    """Kalshi orderbook snapshot."""

    ticker: str
    yes_bids: list[OrderBookLevel]
    yes_asks: list[OrderBookLevel]
    no_bids: list[OrderBookLevel]
    no_asks: list[OrderBookLevel]


@dataclass
class KalshiClient:
    """Read-only client for Kalshi Trade API v2."""

    base_url: str = dataclass_field(default_factory=lambda: get_settings().kalshi_api_url)
    timeout: float = 30.0

    _client: httpx.AsyncClient | None = dataclass_field(default=None, init=False, repr=False)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Accept": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> KalshiClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def _parse_market(self, data: dict[str, Any]) -> KalshiMarket:
        """Parse market from API response using dollar fields."""
        close_time = None
        if data.get("close_time"):
            try:
                close_time = datetime.fromisoformat(data["close_time"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        return KalshiMarket(
            ticker=data.get("ticker", ""),
            event_ticker=data.get("event_ticker", ""),
            title=data.get("title", ""),
            subtitle=data.get("subtitle"),
            status=data.get("status", ""),
            yes_bid=float(data.get("yes_bid_dollars", 0) or 0),
            yes_ask=float(data.get("yes_ask_dollars", 0) or 0),
            no_bid=float(data.get("no_bid_dollars", 0) or 0),
            no_ask=float(data.get("no_ask_dollars", 0) or 0),
            last_price=float(data.get("last_price_dollars", 0) or 0),
            volume=int(data.get("volume", 0) or 0),
            volume_24h=int(data.get("volume_24h", 0) or 0),
            open_interest=int(data.get("open_interest", 0) or 0),
            close_time=close_time,
            category=data.get("category"),
            result=data.get("result"),
        )

    def _parse_trade(self, data: dict[str, Any]) -> KalshiTrade:
        """Parse trade from API response."""
        created_time = datetime.now(timezone.utc)
        if data.get("created_time"):
            try:
                created_time = datetime.fromisoformat(data["created_time"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        return KalshiTrade(
            trade_id=data.get("trade_id", ""),
            ticker=data.get("ticker", ""),
            count=int(data.get("count", 0) or 0),
            yes_price=float(data.get("yes_price_dollars", 0) or 0),
            taker_side=data.get("taker_side", ""),
            created_time=created_time,
        )

    async def get_markets(
        self,
        status: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
    ) -> list[KalshiMarket]:
        """Get markets with optional filters.

        Args:
            status: Filter by status ('open', 'closed', 'settled')
            limit: Max results (up to 1000)
            cursor: Pagination cursor
            min_close_ts: Min close timestamp (unix seconds)
            max_close_ts: Max close timestamp (unix seconds)
        """
        client = self._get_client()
        params: dict[str, Any] = {"limit": min(limit, 1000)}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        if min_close_ts:
            params["min_close_ts"] = min_close_ts
        if max_close_ts:
            params["max_close_ts"] = max_close_ts

        try:
            response = await client.get("/markets", params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            logger.error("Kalshi API error", status_code=e.response.status_code)
            return []
        except httpx.RequestError as e:
            logger.error("Kalshi request error", error=str(e))
            return []

        markets = []
        for market_data in data.get("markets", []):
            try:
                markets.append(self._parse_market(market_data))
            except (KeyError, ValueError) as e:
                logger.warning("Failed to parse Kalshi market", error=str(e))
        return markets

    async def get_market(self, ticker: str) -> KalshiMarket | None:
        """Get a specific market by ticker."""
        client = self._get_client()
        try:
            response = await client.get(f"/markets/{ticker}")
            response.raise_for_status()
            data = response.json()
            return self._parse_market(data.get("market", data))
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.error("Kalshi API error", ticker=ticker, status=e.response.status_code)
            return None
        except httpx.RequestError as e:
            logger.error("Kalshi request error", error=str(e))
            return None

    async def get_orderbook(self, ticker: str, depth: int = 10) -> OrderBook | None:
        """Get orderbook for a market."""
        client = self._get_client()
        try:
            response = await client.get(
                f"/markets/{ticker}/orderbook",
                params={"depth": depth},
            )
            response.raise_for_status()
            data = response.json()
            ob = data.get("orderbook", data)

            def parse_levels(levels: list[list[Any]]) -> list[OrderBookLevel]:
                return [
                    OrderBookLevel(price=float(lvl[0]), quantity=int(lvl[1]))
                    for lvl in levels
                    if len(lvl) >= 2
                ]

            return OrderBook(
                ticker=ticker,
                yes_bids=parse_levels(ob.get("yes", {}).get("bids", [])),
                yes_asks=parse_levels(ob.get("yes", {}).get("asks", [])),
                no_bids=parse_levels(ob.get("no", {}).get("bids", [])),
                no_asks=parse_levels(ob.get("no", {}).get("asks", [])),
            )
        except httpx.HTTPStatusError as e:
            logger.error("Kalshi orderbook error", ticker=ticker, status=e.response.status_code)
            return None
        except httpx.RequestError as e:
            logger.error("Kalshi orderbook request error", error=str(e))
            return None

    async def get_trades(
        self,
        ticker: str,
        limit: int = 100,
        min_ts: int | None = None,
        max_ts: int | None = None,
    ) -> list[KalshiTrade]:
        """Get recent trades for a market."""
        client = self._get_client()
        params: dict[str, Any] = {"ticker": ticker, "limit": min(limit, 1000)}
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts

        try:
            response = await client.get("/markets/trades", params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            logger.error("Kalshi trades error", ticker=ticker, status=e.response.status_code)
            return []
        except httpx.RequestError as e:
            logger.error("Kalshi trades request error", error=str(e))
            return []

        trades = []
        for trade_data in data.get("trades", []):
            try:
                trades.append(self._parse_trade(trade_data))
            except (KeyError, ValueError) as e:
                logger.warning("Failed to parse Kalshi trade", error=str(e))
        return trades

    async def get_events(
        self,
        status: str | None = None,
        limit: int = 100,
        with_nested_markets: bool = True,
    ) -> list[KalshiEvent]:
        """Get events (groups of related markets)."""
        client = self._get_client()
        params: dict[str, Any] = {
            "limit": min(limit, 200),
            "with_nested_markets": with_nested_markets,
        }
        if status:
            params["status"] = status

        try:
            response = await client.get("/events", params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            logger.error("Kalshi events error", status_code=e.response.status_code)
            return []
        except httpx.RequestError as e:
            logger.error("Kalshi events request error", error=str(e))
            return []

        events = []
        for event_data in data.get("events", []):
            try:
                nested_markets = []
                for m in event_data.get("markets", []):
                    try:
                        nested_markets.append(self._parse_market(m))
                    except (KeyError, ValueError):
                        continue

                events.append(
                    KalshiEvent(
                        event_ticker=event_data.get("event_ticker", ""),
                        title=event_data.get("title", ""),
                        category=event_data.get("category"),
                        status=event_data.get("status", ""),
                        markets=nested_markets,
                    )
                )
            except (KeyError, ValueError) as e:
                logger.warning("Failed to parse Kalshi event", error=str(e))
        return events

    async def get_expiring_markets(self, hours: int = 24) -> list[KalshiMarket]:
        """Get markets expiring within the specified hours."""
        import time

        now_ts = int(time.time())
        max_ts = now_ts + (hours * 3600)
        return await self.get_markets(
            status="open",
            min_close_ts=now_ts,
            max_close_ts=max_ts,
            limit=100,
        )
