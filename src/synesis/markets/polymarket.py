"""Polymarket Gamma API client for market discovery.

The Gamma API provides read-only access to market data:
- Market search and discovery
- Current prices and volumes
- Market metadata

API Base: https://gamma-api.polymarket.com

Note: This is for market discovery only. Trading uses the CLOB API.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from synesis.core.logging import get_logger
from synesis.processing.models import MarketOpportunity

logger = get_logger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"


@dataclass
class SimpleMarket:
    """Simplified market representation from Gamma API."""

    id: str
    condition_id: str
    question: str
    slug: str
    description: str | None
    category: str | None

    # Prices (0.0 to 1.0)
    yes_price: float
    no_price: float

    # Volume
    volume_24h: float
    volume_total: float

    # Dates
    end_date: datetime | None
    created_at: datetime | None

    # Status
    is_active: bool
    is_closed: bool

    @property
    def url(self) -> str:
        """Get the Polymarket URL for this market."""
        return f"https://polymarket.com/event/{self.slug}"


@dataclass
class PolymarketClient:
    """Client for Polymarket Gamma API (read-only market discovery)."""

    base_url: str = GAMMA_API_BASE
    timeout: float = 30.0

    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Accept": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "PolymarketClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    def _parse_market(self, data: dict[str, Any]) -> SimpleMarket:
        """Parse market data from API response."""
        import json

        yes_price = 0.5
        no_price = 0.5

        # Method 1: Parse outcomePrices JSON string (from search/events endpoint)
        # Format: outcomePrices = "[\"0.9965\", \"0.0035\"]" with outcomes = "[\"Yes\", \"No\"]"
        outcome_prices_str = data.get("outcomePrices")
        outcomes_str = data.get("outcomes")

        if outcome_prices_str and outcomes_str:
            try:
                prices = json.loads(outcome_prices_str)
                outcomes = json.loads(outcomes_str)
                for i, outcome in enumerate(outcomes):
                    if i < len(prices):
                        price = float(prices[i])
                        if outcome.lower() == "yes":
                            yes_price = price
                        elif outcome.lower() == "no":
                            no_price = price
            except (json.JSONDecodeError, ValueError, IndexError) as e:
                logger.debug("Failed to parse outcomePrices", error=str(e))

        # Method 2: Handle tokens array (from direct market endpoint)
        if yes_price == 0.5 and no_price == 0.5:
            tokens = data.get("tokens", [])
            for token in tokens:
                outcome = token.get("outcome", "").lower()
                price = float(token.get("price", 0.5))
                if outcome == "yes":
                    yes_price = price
                elif outcome == "no":
                    no_price = price

        # Parse dates
        end_date = None
        if data.get("endDate"):
            try:
                end_date = datetime.fromisoformat(data["endDate"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        created_at = None
        if data.get("createdAt"):
            try:
                created_at = datetime.fromisoformat(data["createdAt"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        return SimpleMarket(
            id=data.get("id", ""),
            condition_id=data.get("conditionId", ""),
            question=data.get("question", ""),
            slug=data.get("slug", ""),
            description=data.get("description"),
            category=data.get("category"),
            yes_price=yes_price,
            no_price=no_price,
            volume_24h=float(data.get("volume24hr", 0)),
            volume_total=float(data.get("volume", 0)),
            end_date=end_date,
            created_at=created_at,
            is_active=data.get("active", False),
            is_closed=data.get("closed", False),
        )

    async def search_markets(
        self,
        query: str,
        limit: int = 10,
        active_only: bool = True,
    ) -> list[SimpleMarket]:
        """Search for markets by keyword using public-search endpoint.

        Args:
            query: Search query
            limit: Maximum results to return
            active_only: Only return active (open) markets

        Returns:
            List of matching markets
        """
        client = self._get_client()

        params: dict[str, Any] = {
            "q": query,
            "limit_per_type": limit,
        }
        if active_only:
            params["events_status"] = "active"

        log = logger.bind(query=query, limit=limit)

        try:
            response = await client.get("/public-search", params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            log.error("Polymarket search API error", status_code=e.response.status_code)
            return []
        except httpx.RequestError as e:
            log.error("Polymarket search request error", error=str(e))
            return []

        # Parse events → markets from response
        # Note: events_status=active filters at EVENT level, not MARKET level
        # A resolved market inside an "active" event still gets returned
        # So we must filter at market level here
        markets = []
        for event in data.get("events", []):
            for market_data in event.get("markets", []):
                try:
                    market = self._parse_market(market_data)
                    # Filter: only include active, non-closed markets with a question
                    if market.question and market.is_active and not market.is_closed:
                        markets.append(market)
                except (KeyError, ValueError) as e:
                    log.warning("Failed to parse market", error=str(e))
                    continue

        log.debug("Search complete", results=len(markets))
        return markets[:limit]

    async def get_market(self, market_id: str) -> SimpleMarket | None:
        """Get a specific market by ID.

        Args:
            market_id: The market ID

        Returns:
            Market data or None if not found
        """
        client = self._get_client()

        try:
            response = await client.get(f"/markets/{market_id}")
            response.raise_for_status()
            data = response.json()
            return self._parse_market(data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.error("Polymarket API error", market_id=market_id, status=e.response.status_code)
            return None
        except httpx.RequestError as e:
            logger.error("Polymarket request error", error=str(e))
            return None

    async def get_trending_markets(
        self,
        limit: int = 20,
    ) -> list[SimpleMarket]:
        """Get trending markets by volume.

        Args:
            limit: Maximum results

        Returns:
            List of trending markets sorted by 24h volume
        """
        client = self._get_client()

        params: dict[str, Any] = {
            "limit": limit,
            "active": "true",
            "closed": "false",
            "order": "volume24hr",
            "ascending": "false",
        }

        try:
            response = await client.get("/markets", params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            logger.error("Polymarket API error", status_code=e.response.status_code)
            return []
        except httpx.RequestError as e:
            logger.error("Polymarket request error", error=str(e))
            return []

        markets = []
        for market_data in data:
            try:
                market = self._parse_market(market_data)
                if market.question:
                    markets.append(market)
            except (KeyError, ValueError) as e:
                logger.warning("Failed to parse market", error=str(e))
                continue

        return markets


async def find_market_opportunities(
    keywords: list[str],
    client: PolymarketClient | None = None,
) -> list[MarketOpportunity]:
    """Search Polymarket for opportunities based on keywords.

    Args:
        keywords: List of search keywords
        client: Optional client (creates one if not provided)

    Returns:
        List of market opportunities
    """
    own_client = client is None
    active_client: PolymarketClient
    if own_client or client is None:
        active_client = PolymarketClient()
    else:
        active_client = client

    opportunities = []

    try:
        for keyword in keywords[:5]:  # Limit to 5 keywords to avoid rate limits
            markets = await active_client.search_markets(keyword, limit=5)

            for market in markets:
                # Skip closed or inactive markets
                if market.is_closed or not market.is_active:
                    continue

                # Create opportunity
                opportunity = MarketOpportunity(
                    market_id=market.id,
                    platform="polymarket",
                    question=market.question,
                    slug=market.slug,
                    yes_price=market.yes_price,
                    no_price=market.no_price,
                    volume_24h=market.volume_24h,
                    suggested_direction="yes" if market.yes_price < 0.5 else "no",
                    reason=f"Found via keyword: {keyword}",
                    end_date=market.end_date,
                )
                opportunities.append(opportunity)

    finally:
        if own_client:
            await active_client.close()

    # Deduplicate by market_id
    seen = set()
    unique_opportunities = []
    for opp in opportunities:
        if opp.market_id not in seen:
            seen.add(opp.market_id)
            unique_opportunities.append(opp)

    return unique_opportunities
