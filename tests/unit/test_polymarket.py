"""Tests for Polymarket client."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from synesis.markets.polymarket import (
    PolymarketClient,
    SimpleMarket,
    find_market_opportunities,
)


class TestSimpleMarket:
    """Tests for SimpleMarket dataclass."""

    def test_create_market(self) -> None:
        market = SimpleMarket(
            id="market_123",
            condition_id="cond_456",
            question="Will the Fed cut rates?",
            slug="fed-rate-cut",
            description="Fed rate cut prediction",
            category="economics",
            yes_price=0.65,
            no_price=0.35,
            volume_24h=50000.0,
            volume_total=1000000.0,
            end_date=datetime(2025, 3, 15, tzinfo=timezone.utc),
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            is_active=True,
            is_closed=False,
        )

        assert market.id == "market_123"
        assert market.yes_price == 0.65
        assert market.is_active

    def test_market_url(self) -> None:
        market = SimpleMarket(
            id="market_123",
            condition_id="cond_456",
            question="Test",
            slug="test-market",
            description=None,
            category=None,
            yes_price=0.5,
            no_price=0.5,
            volume_24h=0.0,
            volume_total=0.0,
            end_date=None,
            created_at=None,
            is_active=True,
            is_closed=False,
        )

        assert market.url == "https://polymarket.com/event/test-market"


class TestPolymarketClient:
    """Tests for PolymarketClient."""

    @pytest.fixture
    def mock_search_response(self) -> dict:
        """Sample search response from /public-search endpoint."""
        return {
            "events": [
                {
                    "title": "Test Event",
                    "markets": [
                        {
                            "id": "market_1",
                            "conditionId": "cond_1",
                            "question": "Will X happen?",
                            "slug": "will-x-happen",
                            "description": "Test description",
                            "category": "politics",
                            "tokens": [
                                {"outcome": "Yes", "price": 0.60},
                                {"outcome": "No", "price": 0.40},
                            ],
                            "volume24hr": 10000,
                            "volume": 100000,
                            "endDate": "2025-03-15T00:00:00Z",
                            "createdAt": "2025-01-01T00:00:00Z",
                            "active": True,
                            "closed": False,
                        },
                        {
                            "id": "market_2",
                            "conditionId": "cond_2",
                            "question": "Will Y happen?",
                            "slug": "will-y-happen",
                            "description": "Another test",
                            "category": "economics",
                            "tokens": [
                                {"outcome": "Yes", "price": 0.30},
                                {"outcome": "No", "price": 0.70},
                            ],
                            "volume24hr": 5000,
                            "volume": 50000,
                            "active": True,
                            "closed": False,
                        },
                    ],
                }
            ]
        }

    @pytest.fixture
    def mock_market_data(self) -> dict:
        """Sample single market data for get_market endpoint."""
        return {
            "id": "market_1",
            "conditionId": "cond_1",
            "question": "Will X happen?",
            "slug": "will-x-happen",
            "description": "Test description",
            "category": "politics",
            "tokens": [
                {"outcome": "Yes", "price": 0.60},
                {"outcome": "No", "price": 0.40},
            ],
            "volume24hr": 10000,
            "volume": 100000,
            "endDate": "2025-03-15T00:00:00Z",
            "createdAt": "2025-01-01T00:00:00Z",
            "active": True,
            "closed": False,
        }

    @pytest.mark.asyncio
    async def test_search_markets(self, mock_search_response: dict) -> None:
        """Test searching for markets."""
        client = PolymarketClient()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_search_response
            mock_response.raise_for_status = MagicMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            markets = await client.search_markets("test", limit=10)

            assert len(markets) == 2
            assert markets[0].question == "Will X happen?"
            assert markets[0].yes_price == 0.60
            assert markets[1].yes_price == 0.30

    @pytest.mark.asyncio
    async def test_search_markets_empty(self) -> None:
        """Test searching with no results."""
        client = PolymarketClient()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"events": []}
            mock_response.raise_for_status = MagicMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            markets = await client.search_markets("nonexistent")

            assert len(markets) == 0

    @pytest.mark.asyncio
    async def test_search_markets_handles_error(self) -> None:
        """Test that search handles HTTP errors gracefully."""
        client = PolymarketClient()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Server error",
                    request=httpx.Request("GET", "http://test"),
                    response=httpx.Response(500),
                )
            )
            mock_get_client.return_value = mock_http

            markets = await client.search_markets("test")

            assert len(markets) == 0

    @pytest.mark.asyncio
    async def test_get_market(self, mock_market_data: dict) -> None:
        """Test getting a specific market."""
        client = PolymarketClient()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_market_data
            mock_response.raise_for_status = MagicMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            market = await client.get_market("market_1")

            assert market is not None
            assert market.id == "market_1"
            assert market.question == "Will X happen?"

    @pytest.mark.asyncio
    async def test_get_market_not_found(self) -> None:
        """Test getting a market that doesn't exist."""
        client = PolymarketClient()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Not found",
                    request=httpx.Request("GET", "http://test"),
                    response=httpx.Response(404),
                )
            )
            mock_get_client.return_value = mock_http

            market = await client.get_market("nonexistent")

            assert market is None

    def test_parse_market(self) -> None:
        """Test parsing market data."""
        client = PolymarketClient()
        data = {
            "id": "test_id",
            "conditionId": "test_cond",
            "question": "Test question",
            "slug": "test-slug",
            "description": "Test desc",
            "category": "test",
            "tokens": [
                {"outcome": "Yes", "price": 0.75},
                {"outcome": "No", "price": 0.25},
            ],
            "volume24hr": 1000,
            "volume": 10000,
            "active": True,
            "closed": False,
        }

        market = client._parse_market(data)

        assert market.id == "test_id"
        assert market.yes_price == 0.75
        assert market.no_price == 0.25
        assert market.is_active


class TestFindMarketOpportunities:
    """Tests for find_market_opportunities function."""

    @pytest.mark.asyncio
    async def test_finds_opportunities(self) -> None:
        """Test finding opportunities from keywords."""
        mock_market = SimpleMarket(
            id="opp_1",
            condition_id="cond_1",
            question="Fed rate cut?",
            slug="fed-rate-cut",
            description=None,
            category="economics",
            yes_price=0.40,
            no_price=0.60,
            volume_24h=10000,
            volume_total=100000,
            end_date=None,
            created_at=None,
            is_active=True,
            is_closed=False,
        )

        with patch("synesis.markets.polymarket.PolymarketClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.search_markets = AsyncMock(return_value=[mock_market])
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client

            opportunities = await find_market_opportunities(["Fed", "rate cut"])

            assert len(opportunities) >= 1
            assert opportunities[0].platform == "polymarket"

    @pytest.mark.asyncio
    async def test_deduplicates_markets(self) -> None:
        """Test that duplicate markets are removed."""
        mock_market = SimpleMarket(
            id="same_id",  # Same ID
            condition_id="cond_1",
            question="Test?",
            slug="test",
            description=None,
            category=None,
            yes_price=0.50,
            no_price=0.50,
            volume_24h=0,
            volume_total=0,
            end_date=None,
            created_at=None,
            is_active=True,
            is_closed=False,
        )

        with patch("synesis.markets.polymarket.PolymarketClient") as mock_client_class:
            mock_client = AsyncMock()
            # Return same market for multiple keywords
            mock_client.search_markets = AsyncMock(return_value=[mock_market])
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client

            opportunities = await find_market_opportunities(["keyword1", "keyword2", "keyword3"])

            # Should only have 1 unique opportunity
            assert len(opportunities) == 1
