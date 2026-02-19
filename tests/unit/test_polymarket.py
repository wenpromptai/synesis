"""Tests for Polymarket client."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from synesis.markets.polymarket import (
    PolymarketClient,
    SimpleMarket,
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
        assert not market.prices_are_default

    def test_parse_market_up_down_outcomes(self) -> None:
        """Test positional fallback for Up/Down outcomes via outcomePrices."""
        import json

        client = PolymarketClient()
        data = {
            "id": "crypto_up_down",
            "conditionId": "cond_crypto",
            "question": "Will BTC go up or down?",
            "slug": "btc-up-down",
            "description": None,
            "category": "crypto",
            "outcomes": json.dumps(["Up", "Down"]),
            "outcomePrices": json.dumps(["0.49", "0.51"]),
            "volume24hr": 5000,
            "volume": 50000,
            "active": True,
            "closed": False,
        }

        market = client._parse_market(data)

        assert market.yes_price == 0.49
        assert market.no_price == 0.51
        assert not market.prices_are_default
        assert market.yes_outcome == "Up"

    def test_parse_market_over_under_tokens(self) -> None:
        """Test positional fallback for Over/Under outcomes via tokens array."""
        client = PolymarketClient()
        data = {
            "id": "sports_ou",
            "conditionId": "cond_sports",
            "question": "1H O/U 114.5",
            "slug": "sports-ou",
            "description": None,
            "category": "sports",
            "tokens": [
                {"outcome": "Over", "price": 0.62},
                {"outcome": "Under", "price": 0.38},
            ],
            "volume24hr": 3000,
            "volume": 30000,
            "active": True,
            "closed": False,
        }

        market = client._parse_market(data)

        assert market.yes_price == 0.62
        assert market.no_price == 0.38
        assert not market.prices_are_default
        assert market.yes_outcome == "Over"

    def test_parse_market_group_item_title_suppressed_when_equals_question(self) -> None:
        """Test groupItemTitle is None when it duplicates the question."""
        import json

        client = PolymarketClient()
        data = {
            "id": "dup_title",
            "conditionId": "cond_dup",
            "question": "Total Kills Over/Under 55.5 in Game 1?",
            "slug": "kills-ou",
            "description": None,
            "category": "esports",
            "groupItemTitle": "Total Kills Over/Under 55.5 in Game 1?",
            "outcomes": json.dumps(["Over", "Under"]),
            "outcomePrices": json.dumps(["0.55", "0.45"]),
            "volume24hr": 2000,
            "volume": 20000,
            "active": True,
            "closed": False,
        }

        market = client._parse_market(data)

        assert market.group_item_title is None
        assert market.yes_outcome == "Over"

    def test_parse_market_group_item_title(self) -> None:
        """Test groupItemTitle is captured as group_item_title."""
        import json

        client = PolymarketClient()
        data = {
            "id": "multi_outcome",
            "conditionId": "cond_multi",
            "question": "What will Hochul say during Roundtable?",
            "slug": "hochul-roundtable",
            "description": None,
            "category": "politics",
            "groupItemTitle": "Venezuela",
            "outcomes": json.dumps(["Yes", "No"]),
            "outcomePrices": json.dumps(["0.12", "0.88"]),
            "volume24hr": 1000,
            "volume": 10000,
            "active": True,
            "closed": False,
        }

        market = client._parse_market(data)

        assert market.group_item_title == "Venezuela"
        assert market.yes_price == 0.12
        assert market.no_price == 0.88

    def test_parse_market_prices_are_default_flag(self) -> None:
        """Test prices_are_default is set when no price data available."""
        client = PolymarketClient()
        data = {
            "id": "no_prices",
            "conditionId": "cond_no_prices",
            "question": "Unknown market?",
            "slug": "unknown",
            "description": None,
            "category": None,
            "volume24hr": 0,
            "volume": 0,
            "active": True,
            "closed": False,
        }

        market = client._parse_market(data)

        assert market.yes_price == 0.5
        assert market.no_price == 0.5
        assert market.prices_are_default


class TestOutcomeParsing:
    """Tests for robust outcome parsing (known pairs, sum validation, 3+ outcomes)."""

    def test_known_pair_reversed_order(self) -> None:
        """Known pair with reversed API order maps correctly (Down first, Up second)."""
        import json

        client = PolymarketClient()
        data = {
            "id": "btc_reversed",
            "conditionId": "cond_btc",
            "question": "Will BTC price go up?",
            "slug": "btc-up",
            "description": None,
            "category": "crypto",
            "outcomes": json.dumps(["Down", "Up"]),
            "outcomePrices": json.dumps(["0.35", "0.65"]),
            "volume24hr": 5000,
            "volume": 50000,
            "active": True,
            "closed": False,
        }

        market = client._parse_market(data)

        assert market is not None
        # "Up" is the positive outcome → maps to yes_price regardless of position
        assert market.yes_price == 0.65
        assert market.no_price == 0.35
        assert market.yes_outcome == "Up"

    def test_known_pair_higher_lower(self) -> None:
        """Higher/Lower known pair maps correctly."""
        import json

        client = PolymarketClient()
        data = {
            "id": "rate_higher",
            "conditionId": "cond_rate",
            "question": "Will the rate be higher or lower?",
            "slug": "rate-higher-lower",
            "description": None,
            "category": "economics",
            "outcomes": json.dumps(["Lower", "Higher"]),
            "outcomePrices": json.dumps(["0.40", "0.60"]),
            "volume24hr": 3000,
            "volume": 30000,
            "active": True,
            "closed": False,
        }

        market = client._parse_market(data)

        assert market is not None
        assert market.yes_price == 0.60
        assert market.no_price == 0.40
        assert market.yes_outcome == "Higher"

    def test_known_pair_via_tokens(self) -> None:
        """Known pair matching works through the tokens array path."""
        client = PolymarketClient()
        data = {
            "id": "token_ou",
            "conditionId": "cond_ou",
            "question": "Total points O/U 210.5",
            "slug": "points-ou",
            "description": None,
            "category": "sports",
            "tokens": [
                {"outcome": "Under", "price": 0.45},
                {"outcome": "Over", "price": 0.55},
            ],
            "volume24hr": 8000,
            "volume": 80000,
            "active": True,
            "closed": False,
        }

        market = client._parse_market(data)

        assert market is not None
        # "Over" is the positive outcome
        assert market.yes_price == 0.55
        assert market.no_price == 0.45
        assert market.yes_outcome == "Over"

    def test_sum_validation_rejects_bad_prices(self) -> None:
        """Prices that sum far from 1.0 are reset to defaults."""
        import json

        client = PolymarketClient()
        data = {
            "id": "bad_sum",
            "conditionId": "cond_bad",
            "question": "Garbled market?",
            "slug": "garbled",
            "description": None,
            "category": None,
            "outcomes": json.dumps(["Yes", "No"]),
            "outcomePrices": json.dumps(["0.90", "0.90"]),
            "volume24hr": 1000,
            "volume": 10000,
            "active": True,
            "closed": False,
        }

        market = client._parse_market(data)

        assert market is not None
        assert market.yes_price == 0.5
        assert market.no_price == 0.5
        assert market.prices_are_default

    def test_sum_validation_accepts_normal_spread(self) -> None:
        """Normal bid-ask spread (sum ~1.03) passes validation."""
        import json

        client = PolymarketClient()
        data = {
            "id": "normal_spread",
            "conditionId": "cond_spread",
            "question": "Normal market?",
            "slug": "normal",
            "description": None,
            "category": "politics",
            "outcomes": json.dumps(["Yes", "No"]),
            "outcomePrices": json.dumps(["0.62", "0.41"]),
            "volume24hr": 20000,
            "volume": 200000,
            "active": True,
            "closed": False,
        }

        market = client._parse_market(data)

        assert market is not None
        assert market.yes_price == 0.62
        assert market.no_price == 0.41
        assert not market.prices_are_default

    def test_three_plus_outcomes_returns_none(self) -> None:
        """Markets with 3+ outcomes return None from _parse_market."""
        import json

        client = PolymarketClient()
        data = {
            "id": "multi_outcome",
            "conditionId": "cond_multi",
            "question": "Who will win the election?",
            "slug": "election-winner",
            "description": None,
            "category": "politics",
            "outcomes": json.dumps(["Alice", "Bob", "Carol"]),
            "outcomePrices": json.dumps(["0.40", "0.35", "0.25"]),
            "volume24hr": 50000,
            "volume": 500000,
            "active": True,
            "closed": False,
        }

        market = client._parse_market(data)

        assert market is None


class TestCategoryEnrichment:
    """Tests for category enrichment from event tags."""

    @pytest.mark.asyncio
    async def test_enrich_fills_missing_category_from_event_tags(self) -> None:
        """Markets without category get it from event tags."""
        client = PolymarketClient()

        # Market response: no category, but has event_id in events array
        market_data = [
            {
                "id": "new_market",
                "conditionId": "cond_new",
                "question": "Will something happen?",
                "slug": "something-happen",
                "description": None,
                "category": None,
                "tokens": [
                    {"outcome": "Yes", "price": 0.60},
                    {"outcome": "No", "price": 0.40},
                ],
                "volume24hr": 50000,
                "volume": 200000,
                "active": True,
                "closed": False,
                "events": [{"id": "99999", "ticker": "something-event"}],
            }
        ]

        # Event response: has tags with forceShow
        event_response = {
            "tags": [
                {"label": "maduro", "forceShow": False},
                {"label": "Geopolitics", "forceShow": True},
                {"label": "Venezuela", "forceShow": False},
            ]
        }

        call_count = 0

        async def mock_get(url: str, **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "/events/" in url:
                resp.json.return_value = event_response
            else:
                resp.json.return_value = market_data
            return resp

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(side_effect=mock_get)
            mock_get_client.return_value = mock_http

            markets = await client.get_trending_markets(limit=1)

        assert len(markets) == 1
        assert markets[0].category == "Geopolitics"
        assert markets[0].event_id == "99999"

    @pytest.mark.asyncio
    async def test_enrich_skips_markets_with_existing_category(self) -> None:
        """Markets with category already set are not enriched."""
        client = PolymarketClient()

        market_data = [
            {
                "id": "old_market",
                "conditionId": "cond_old",
                "question": "Old market?",
                "slug": "old-market",
                "description": None,
                "category": "Crypto",
                "tokens": [
                    {"outcome": "Yes", "price": 0.50},
                    {"outcome": "No", "price": 0.50},
                ],
                "volume24hr": 1000,
                "volume": 10000,
                "active": True,
                "closed": False,
                "events": [{"id": "11111"}],
            }
        ]

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = market_data
            mock_response.raise_for_status = MagicMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            markets = await client.get_trending_markets(limit=1)

        assert len(markets) == 1
        assert markets[0].category == "Crypto"
        # Event endpoint should NOT have been called (only /markets was called)
        assert mock_http.get.await_count == 1

    @pytest.mark.asyncio
    async def test_enrich_uses_first_tag_when_no_forceshow(self) -> None:
        """Falls back to first tag label when none have forceShow."""
        client = PolymarketClient()

        market_data = [
            {
                "id": "m1",
                "conditionId": "c1",
                "question": "Test?",
                "slug": "test",
                "description": None,
                "category": None,
                "tokens": [
                    {"outcome": "Yes", "price": 0.70},
                    {"outcome": "No", "price": 0.30},
                ],
                "volume24hr": 5000,
                "volume": 50000,
                "active": True,
                "closed": False,
                "events": [{"id": "22222"}],
            }
        ]

        event_response = {
            "tags": [
                {"label": "Politics", "forceShow": False},
                {"label": "Trump", "forceShow": False},
            ]
        }

        async def mock_get(url: str, **kwargs: object) -> MagicMock:
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "/events/" in url:
                resp.json.return_value = event_response
            else:
                resp.json.return_value = market_data
            return resp

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(side_effect=mock_get)
            mock_get_client.return_value = mock_http

            markets = await client.get_trending_markets(limit=1)

        assert markets[0].category == "Politics"

    def test_parse_market_extracts_event_id(self) -> None:
        """_parse_market extracts event_id from events array."""
        client = PolymarketClient()
        data = {
            "id": "test",
            "conditionId": "cond",
            "question": "Q?",
            "slug": "q",
            "description": None,
            "category": None,
            "volume24hr": 0,
            "volume": 0,
            "active": True,
            "closed": False,
            "events": [{"id": "12345", "ticker": "test-event"}],
        }

        market = client._parse_market(data)
        assert market.event_id == "12345"

    def test_parse_market_no_events(self) -> None:
        """_parse_market sets event_id to None when no events."""
        client = PolymarketClient()
        data = {
            "id": "test",
            "conditionId": "cond",
            "question": "Q?",
            "slug": "q",
            "description": None,
            "category": None,
            "volume24hr": 0,
            "volume": 0,
            "active": True,
            "closed": False,
        }

        market = client._parse_market(data)
        assert market.event_id is None


class TestCategoryEnrichmentGatherException:
    """Test 8: One event fetch raises in asyncio.gather → others still succeed."""

    @pytest.mark.asyncio
    async def test_one_event_fetch_raises_others_succeed(self) -> None:
        client = PolymarketClient()

        # Two markets needing enrichment, different event IDs
        market_data = [
            {
                "id": "m_ok",
                "conditionId": "c_ok",
                "question": "Good market?",
                "slug": "good-market",
                "description": None,
                "category": None,
                "tokens": [
                    {"outcome": "Yes", "price": 0.60},
                    {"outcome": "No", "price": 0.40},
                ],
                "volume24hr": 5000,
                "volume": 50000,
                "active": True,
                "closed": False,
                "events": [{"id": "event_ok"}],
            },
            {
                "id": "m_fail",
                "conditionId": "c_fail",
                "question": "Fail market?",
                "slug": "fail-market",
                "description": None,
                "category": None,
                "tokens": [
                    {"outcome": "Yes", "price": 0.70},
                    {"outcome": "No", "price": 0.30},
                ],
                "volume24hr": 3000,
                "volume": 30000,
                "active": True,
                "closed": False,
                "events": [{"id": "event_fail"}],
            },
        ]

        event_ok_response = {"tags": [{"label": "Politics", "forceShow": True}]}

        async def mock_get(url: str, **kwargs: object) -> MagicMock:
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "/events/event_ok" in url:
                resp.json.return_value = event_ok_response
            elif "/events/event_fail" in url:
                raise httpx.RequestError("Connection refused")
            else:
                resp.json.return_value = market_data
            return resp

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(side_effect=mock_get)
            mock_get_client.return_value = mock_http

            markets = await client.get_trending_markets(limit=10)

        # Both markets should be returned
        assert len(markets) == 2

        # The one with the successful event fetch should have a category
        ok_market = next(m for m in markets if m.id == "m_ok")
        assert ok_market.category == "Politics"

        # The one with the failed fetch should still exist, category stays None
        fail_market = next(m for m in markets if m.id == "m_fail")
        assert fail_market.category is None
