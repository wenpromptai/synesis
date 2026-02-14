"""Polymarket Gamma API client for market discovery.

The Gamma API provides read-only access to market data:
- Market search and discovery
- Current prices and volumes
- Market metadata

API Base: https://gamma-api.polymarket.com

Note: This is for market discovery only. Trading uses the CLOB API.
"""

import asyncio
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime
from typing import Any

import httpx

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.processing.news import MarketOpportunity

logger = get_logger(__name__)

# Maps frozenset of lowercased outcome names → the "positive" outcome (maps to yes_price).
# Used by _match_known_pair() to assign prices semantically instead of positionally.
_KNOWN_OUTCOME_PAIRS: dict[frozenset[str], str] = {
    frozenset({"up", "down"}): "up",
    frozenset({"over", "under"}): "over",
    frozenset({"higher", "lower"}): "higher",
    frozenset({"above", "below"}): "above",
}


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

    # Multi-outcome event support
    group_item_title: str | None = None

    # Non-Yes/No outcome label for yes_price (e.g. "Up", "Over", "Pistons")
    yes_outcome: str | None = None

    # CLOB token ID for the YES outcome (used to resolve holder direction)
    yes_token_id: str | None = None

    # Price parsing metadata
    prices_are_default: bool = False

    # Event linkage (for category enrichment)
    event_id: str | None = None

    @property
    def url(self) -> str:
        """Get the Polymarket URL for this market."""
        return f"https://polymarket.com/event/{self.slug}"


@dataclass
class PolymarketClient:
    """Client for Polymarket Gamma API (read-only market discovery)."""

    base_url: str = dataclass_field(default_factory=lambda: get_settings().polymarket_gamma_api_url)
    timeout: float = 30.0

    _client: httpx.AsyncClient | None = dataclass_field(default=None, init=False, repr=False)

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

    @staticmethod
    def _match_known_pair(
        outcomes: list[str], prices: list[float]
    ) -> tuple[float, float, str | None] | None:
        """Match outcomes against known semantic pairs (Up/Down, Over/Under, etc.).

        Returns (yes_price, no_price, yes_outcome) if matched, else None.
        """
        if len(outcomes) != 2 or len(prices) < 2:
            return None
        pair_key = frozenset(o.lower() for o in outcomes)
        positive = _KNOWN_OUTCOME_PAIRS.get(pair_key)
        if positive is None:
            return None
        # Find which index holds the positive outcome
        if outcomes[0].lower() == positive:
            return prices[0], prices[1], outcomes[0]
        return prices[1], prices[0], outcomes[1]

    def _parse_market(self, data: dict[str, Any]) -> SimpleMarket | None:
        """Parse market data from API response."""
        import json

        yes_price = 0.5
        no_price = 0.5
        yes_outcome: str | None = None

        # Method 1: Parse outcomePrices JSON string (from search/events endpoint)
        # Format: outcomePrices = "[\"0.9965\", \"0.0035\"]" with outcomes = "[\"Yes\", \"No\"]"
        outcome_prices_str = data.get("outcomePrices")
        outcomes_str = data.get("outcomes")

        if outcome_prices_str and outcomes_str:
            try:
                prices = json.loads(outcome_prices_str)
                outcomes = json.loads(outcomes_str)

                # Skip markets with 3+ outcomes — they don't fit the binary model
                if len(outcomes) > 2:
                    logger.info(
                        "Skipping market with 3+ outcomes",
                        outcome_count=len(outcomes),
                        outcomes=outcomes[:5],
                        market_id=data.get("id"),
                    )
                    return None

                for i, outcome in enumerate(outcomes):
                    if i < len(prices):
                        price = float(prices[i])
                        # Validate price is within valid range (0.0-1.0)
                        if not (0.0 <= price <= 1.0):
                            logger.warning(
                                "Invalid price value from API",
                                price=price,
                                market_id=data.get("id"),
                            )
                            continue
                        if outcome.lower() == "yes":
                            yes_price = price
                        elif outcome.lower() == "no":
                            no_price = price

                # Semantic matching for non-Yes/No outcomes (e.g. Up/Down, Over/Under)
                if yes_price == 0.5 and no_price == 0.5 and len(prices) >= 2:
                    float_prices = [float(prices[0]), float(prices[1])]
                    if all(0.0 <= p <= 1.0 for p in float_prices):
                        matched = self._match_known_pair(outcomes, float_prices)
                        if matched:
                            yes_price, no_price, yes_outcome = matched
                        else:
                            # Positional fallback for unknown pairs (e.g. team names)
                            yes_price = float_prices[0]
                            no_price = float_prices[1]
                            yes_outcome = (
                                outcomes[0] if outcomes[0].lower() not in ("yes", "no") else None
                            )
                            logger.info(
                                "Unknown outcome pair, using positional fallback",
                                outcomes=outcomes,
                                market_id=data.get("id"),
                            )
            except (json.JSONDecodeError, ValueError, IndexError) as e:
                logger.warning("Failed to parse outcomePrices", error=str(e))

        # Method 2: Handle tokens array (from direct market endpoint)
        if yes_price == 0.5 and no_price == 0.5:
            tokens = data.get("tokens", [])

            # Skip markets with 3+ token outcomes
            if len(tokens) > 2:
                logger.info(
                    "Skipping market with 3+ token outcomes",
                    token_count=len(tokens),
                    market_id=data.get("id"),
                )
                return None

            for token in tokens:
                outcome = token.get("outcome", "").lower()
                price = float(token.get("price", 0.5))
                # Validate price is within valid range (0.0-1.0)
                if not (0.0 <= price <= 1.0):
                    logger.warning(
                        "Invalid price value from API",
                        price=price,
                        market_id=data.get("id"),
                    )
                    continue
                if outcome == "yes":
                    yes_price = price
                elif outcome == "no":
                    no_price = price

            # Semantic matching for non-Yes/No token outcomes
            if yes_price == 0.5 and no_price == 0.5 and len(tokens) >= 2:
                p0 = float(tokens[0].get("price", 0.5))
                p1 = float(tokens[1].get("price", 0.5))
                if (0.0 <= p0 <= 1.0) and (0.0 <= p1 <= 1.0):
                    token_outcomes = [
                        tokens[0].get("outcome", ""),
                        tokens[1].get("outcome", ""),
                    ]
                    matched = self._match_known_pair(token_outcomes, [p0, p1])
                    if matched:
                        yes_price, no_price, yes_outcome = matched
                    else:
                        yes_price = p0
                        no_price = p1
                        yes_outcome = (
                            token_outcomes[0]
                            if token_outcomes[0].lower() not in ("yes", "no")
                            else None
                        )
                        logger.info(
                            "Unknown outcome pair, using positional fallback",
                            outcomes=token_outcomes,
                            market_id=data.get("id"),
                        )

        # Parse dates
        end_date = None
        if data.get("endDate"):
            try:
                end_date = datetime.fromisoformat(data["endDate"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                logger.warning(
                    "Failed to parse endDate",
                    market_id=data.get("id"),
                    raw_value=data.get("endDate"),
                )

        created_at = None
        if data.get("createdAt"):
            try:
                created_at = datetime.fromisoformat(data["createdAt"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                logger.warning(
                    "Failed to parse createdAt",
                    market_id=data.get("id"),
                    raw_value=data.get("createdAt"),
                )

        # Sum-to-one validation: prices should sum to ~1.0 (allow 0.85–1.15 for spread)
        price_sum = yes_price + no_price
        if not (0.85 <= price_sum <= 1.15) and not (yes_price == 0.5 and no_price == 0.5):
            logger.warning(
                "Price sum outside valid range, resetting to defaults",
                yes_price=yes_price,
                no_price=no_price,
                price_sum=price_sum,
                market_id=data.get("id"),
            )
            yes_price = 0.5
            no_price = 0.5
            yes_outcome = None

        # Extract YES token ID from clobTokenIds for holder direction resolution
        yes_token_id: str | None = None
        clob_ids_str = data.get("clobTokenIds")
        outcomes_list = None
        if outcomes_str:
            try:
                outcomes_list = json.loads(outcomes_str)
            except (json.JSONDecodeError, ValueError):
                pass
        if clob_ids_str:
            try:
                clob_ids = json.loads(clob_ids_str)
                if outcomes_list and len(clob_ids) == len(outcomes_list):
                    # Match by outcome name
                    for i, outcome in enumerate(outcomes_list):
                        if outcome.lower() == "yes":
                            yes_token_id = str(clob_ids[i])
                            break
                    # For non-Yes/No markets, first token = yes_price side
                    if not yes_token_id and clob_ids:
                        yes_token_id = str(clob_ids[0])
                elif clob_ids:
                    yes_token_id = str(clob_ids[0])
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "Failed to parse clobTokenIds, holder direction will be unknown",
                    condition_id=data.get("conditionId"),
                    clob_ids_raw=clob_ids_str[:100] if clob_ids_str else None,
                    error=str(e),
                )

        prices_are_default = yes_price == 0.5 and no_price == 0.5
        if prices_are_default:
            logger.debug(
                "Prices remain at default after all parsing methods",
                market_id=data.get("id"),
                question=data.get("question", "")[:80],
            )

        # Suppress groupItemTitle when it duplicates the question
        question = data.get("question", "")
        group_item_title = data.get("groupItemTitle")
        if group_item_title and group_item_title.strip() == question.strip():
            group_item_title = None

        # Extract event ID for category enrichment
        event_id = None
        events = data.get("events", [])
        if events and isinstance(events, list):
            event_id = str(events[0].get("id", "")) or None

        return SimpleMarket(
            id=data.get("id", ""),
            condition_id=data.get("conditionId", ""),
            question=question,
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
            group_item_title=group_item_title,
            yes_outcome=yes_outcome,
            yes_token_id=yes_token_id,
            prices_are_default=prices_are_default,
            event_id=event_id,
        )

    async def _fetch_event_category(self, event_id: str) -> tuple[str, str | None]:
        """Fetch category from an event's tags. Returns (event_id, category)."""
        client = self._get_client()
        try:
            response = await client.get(f"/events/{event_id}")
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.warning("Event tag fetch failed", event_id=event_id, error=str(e))
            return event_id, None

        tags = data.get("tags", [])
        if not tags:
            return event_id, None

        # Prefer tags with forceShow=True, then fall back to first tag
        for tag in tags:
            if tag.get("forceShow"):
                return event_id, tag.get("label")
        return event_id, tags[0].get("label")

    async def _enrich_categories(self, markets: list[SimpleMarket]) -> None:
        """Batch-fetch event tags to fill missing categories on newer markets."""
        needs_enrichment = [m for m in markets if not m.category and m.event_id]
        if not needs_enrichment:
            return

        # Collect unique event IDs
        event_ids = list({m.event_id for m in needs_enrichment if m.event_id})
        if not event_ids:
            return

        # Fetch event tags in parallel
        results = await asyncio.gather(
            *[self._fetch_event_category(eid) for eid in event_ids],
            return_exceptions=True,
        )

        # Build event_id → category map
        event_categories: dict[str, str | None] = {}
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Event category fetch failed", error=str(result))
            elif isinstance(result, tuple):
                event_categories[result[0]] = result[1]

        # Apply categories
        for market in needs_enrichment:
            if market.event_id and market.event_id in event_categories:
                cat = event_categories[market.event_id]
                if cat:
                    market.category = cat

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
                    if market and market.question and market.is_active and not market.is_closed:
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
                if market and market.question:
                    markets.append(market)
            except (KeyError, ValueError) as e:
                logger.warning("Failed to parse market", error=str(e))
                continue

        await self._enrich_categories(markets)
        return markets

    async def get_expiring_markets(self, hours: int = 24) -> list[SimpleMarket]:
        """Get markets expiring within the specified hours.

        Args:
            hours: Hours until expiration to filter by

        Returns:
            List of markets expiring soon
        """
        from datetime import datetime, timedelta, timezone

        client = self._get_client()
        now = datetime.now(timezone.utc)
        end_max = now + timedelta(hours=hours)

        params: dict[str, Any] = {
            "limit": 50,
            "active": "true",
            "closed": "false",
            "end_date_min": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_date_max": end_max.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "order": "endDate",
            "ascending": "true",
        }

        try:
            response = await client.get("/markets", params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            logger.error("Polymarket expiring markets error", status_code=e.response.status_code)
            return []
        except httpx.RequestError as e:
            logger.error("Polymarket expiring markets request error", error=str(e))
            return []

        markets = []
        for market_data in data:
            try:
                market = self._parse_market(market_data)
                if market and market.question and market.is_active and not market.is_closed:
                    if market.prices_are_default:
                        logger.debug(
                            "Filtering default-priced expiring market",
                            market_id=market.id,
                            question=market.question[:80],
                        )
                        continue
                    markets.append(market)
            except (KeyError, ValueError) as e:
                logger.warning("Failed to parse expiring market", error=str(e))
        await self._enrich_categories(markets)
        return markets


@dataclass
class PolymarketDataClient:
    """Client for Polymarket Data API (wallet positions, trades, holders).

    API Base: https://data-api.polymarket.com
    """

    base_url: str = dataclass_field(default_factory=lambda: get_settings().polymarket_data_api_url)
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

    async def __aenter__(self) -> "PolymarketDataClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def get_wallet_positions(self, proxy_address: str) -> list[dict[str, Any]]:
        """Get wallet positions from Data API.

        Args:
            proxy_address: Wallet proxy address

        Returns:
            List of position data dicts
        """
        client = self._get_client()
        try:
            response = await client.get("/positions", params={"user": proxy_address})
            response.raise_for_status()
            result: list[dict[str, Any]] = response.json()
            return result
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.error(
                "Data API positions error",
                address=proxy_address,
                error=str(e),
            )
            return []

    async def get_wallet_trades(self, proxy_address: str, limit: int = 500) -> list[dict[str, Any]]:
        """Get wallet trade history.

        Args:
            proxy_address: Wallet proxy address
            limit: Max trades to return

        Returns:
            List of trade data dicts
        """
        client = self._get_client()
        try:
            response = await client.get(
                "/trades",
                params={"user": proxy_address, "limit": limit},
            )
            response.raise_for_status()
            result: list[dict[str, Any]] = response.json()
            return result
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.error(
                "Data API trades error",
                address=proxy_address,
                error=str(e),
            )
            return []

    async def get_top_holders(
        self,
        condition_id: str,
        limit: int = 20,
        yes_token_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get top holders for a market.

        The API returns nested ``[{token, holders: [...]}, ...]``.
        This method flattens the response and normalises ``proxyWallet`` → ``address``
        so downstream code can use ``holder.get("address")`` uniformly.

        Each holder gets an ``outcome`` field ("yes" or "no") resolved by matching
        the token group's ``token`` field against ``yes_token_id``.

        Args:
            condition_id: Polymarket condition ID
            limit: Max holders to return
            yes_token_id: CLOB token ID for the YES outcome (from Gamma API).
                Used to deterministically resolve holder direction.

        Returns:
            Flat list of holder dicts with ``address``, ``amount``, ``outcome``, etc.
        """
        client = self._get_client()
        try:
            response = await client.get(
                "/holders",
                params={"market": condition_id, "limit": limit},
            )
            response.raise_for_status()
            raw: list[dict[str, Any]] = response.json()

            # Flatten nested {token, holders: [...]} into flat list
            flat: list[dict[str, Any]] = []
            for token_group in raw:
                token_id = str(token_group.get("token", ""))
                # Resolve outcome by matching token ID
                if yes_token_id and token_id:
                    outcome = "yes" if token_id == yes_token_id else "no"
                else:
                    outcome = ""
                holders = token_group.get("holders")
                if isinstance(holders, list):
                    for holder in holders:
                        holder["address"] = holder.pop("proxyWallet", holder.get("address", ""))
                        holder.setdefault("outcome", outcome)
                        flat.append(holder)
                else:
                    # Already flat format (e.g. from mocks or future API changes)
                    token_group["address"] = token_group.pop(
                        "proxyWallet", token_group.get("address", "")
                    )
                    token_group.setdefault("outcome", outcome)
                    flat.append(token_group)

            # Sort by amount descending so truncation doesn't bias toward one token group
            flat.sort(key=lambda h: float(h.get("amount", 0) or 0), reverse=True)
            return flat[:limit]
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.error(
                "Data API holders error",
                condition_id=condition_id,
                error=str(e),
            )
            return []

    async def get_open_interest(self, condition_id: str) -> float | None:
        """Get open interest for a market.

        Args:
            condition_id: Polymarket condition ID

        Returns:
            Open interest value or None
        """
        client = self._get_client()
        try:
            response = await client.get("/oi", params={"market": condition_id})
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                # API returns list of {market, value} records
                return sum(float(r.get("value", 0) or 0) for r in data) if data else None
            return float(data) if data else None
        except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as e:
            logger.error(
                "Data API open interest error",
                condition_id=condition_id,
                error=str(e),
            )
            return None


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
    if client is None:
        active_client = PolymarketClient()
        own_client = True
    else:
        active_client = client
        own_client = False

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
