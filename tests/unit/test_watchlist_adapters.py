"""Tests for WatchlistDataProvider adapters (FactSet + Finnhub)."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synesis.providers.base import CompanyInfo, FundamentalsSnapshot, PriceSnapshot


# =============================================================================
# FactSetWatchlistAdapter
# =============================================================================


class TestFactSetWatchlistAdapter:
    """Tests for FactSetWatchlistAdapter."""

    @pytest.fixture
    def mock_provider(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture
    def adapter(self, mock_provider: AsyncMock):  # noqa: ANN201
        from synesis.providers.factset.provider import FactSetWatchlistAdapter

        return FactSetWatchlistAdapter(provider=mock_provider)

    @pytest.mark.asyncio
    async def test_resolve_company_found(self, adapter, mock_provider: AsyncMock) -> None:  # noqa: ANN001
        """Test resolving a company returns CompanyInfo."""
        security = MagicMock()
        security.configure_mock(name="Apple Inc.")
        mock_provider.resolve_ticker.return_value = security

        result = await adapter.resolve_company("AAPL-US")

        assert isinstance(result, CompanyInfo)
        assert result.name == "Apple Inc."
        mock_provider.resolve_ticker.assert_awaited_once_with("AAPL-US")

    @pytest.mark.asyncio
    async def test_resolve_company_not_found(self, adapter, mock_provider: AsyncMock) -> None:  # noqa: ANN001
        """Test resolving unknown ticker returns None."""
        mock_provider.resolve_ticker.return_value = None

        result = await adapter.resolve_company("UNKNOWN")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_market_cap(self, adapter, mock_provider: AsyncMock) -> None:  # noqa: ANN001
        """Test get_market_cap delegates to provider."""
        mock_provider.get_market_cap.return_value = 3e12

        result = await adapter.get_market_cap("AAPL-US")

        assert result == 3e12
        mock_provider.get_market_cap.assert_awaited_once_with("AAPL-US")

    @pytest.mark.asyncio
    async def test_get_market_cap_none(self, adapter, mock_provider: AsyncMock) -> None:  # noqa: ANN001
        """Test get_market_cap returns None when provider returns None."""
        mock_provider.get_market_cap.return_value = None

        result = await adapter.get_market_cap("AAPL-US")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_fundamentals(self, adapter, mock_provider: AsyncMock) -> None:  # noqa: ANN001
        """Test get_fundamentals maps FactSet model to FundamentalsSnapshot."""
        today = date.today()
        factset_fund = MagicMock(
            eps_diluted=6.5,
            price_to_book=45.0,
            price_to_sales=8.0,
            ev_to_ebitda=25.0,
            roe=150.0,
            net_margin=25.0,
            gross_margin=45.0,
            debt_to_equity=1.5,
            period_type="ltm",
            period_end=today,
        )
        mock_provider.get_fundamentals.return_value = [factset_fund]

        result = await adapter.get_fundamentals("AAPL-US")

        assert isinstance(result, FundamentalsSnapshot)
        assert result.eps_diluted == 6.5
        assert result.price_to_book == 45.0
        assert result.ev_to_ebitda == 25.0
        assert result.roe == 150.0
        assert result.net_margin == 25.0
        assert result.gross_margin == 45.0
        assert result.debt_to_equity == 1.5
        assert result.period_type == "ltm"
        assert result.period_end == today
        mock_provider.get_fundamentals.assert_awaited_once_with("AAPL-US", "ltm", 1)

    @pytest.mark.asyncio
    async def test_get_fundamentals_empty(self, adapter, mock_provider: AsyncMock) -> None:  # noqa: ANN001
        """Test get_fundamentals returns None when no data."""
        mock_provider.get_fundamentals.return_value = []

        result = await adapter.get_fundamentals("AAPL-US")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_price(self, adapter, mock_provider: AsyncMock) -> None:  # noqa: ANN001
        """Test get_price maps FactSet price to PriceSnapshot."""
        today = date.today()
        factset_price = MagicMock(
            one_day_pct=1.2,
            one_mth_pct=5.5,
            price_date=today,
        )
        mock_provider.get_price.return_value = factset_price

        result = await adapter.get_price("AAPL-US")

        assert isinstance(result, PriceSnapshot)
        assert result.one_day_pct == 1.2
        assert result.one_mth_pct == 5.5
        assert result.price_date == today

    @pytest.mark.asyncio
    async def test_get_price_none(self, adapter, mock_provider: AsyncMock) -> None:  # noqa: ANN001
        """Test get_price returns None when no data."""
        mock_provider.get_price.return_value = None

        result = await adapter.get_price("AAPL-US")
        assert result is None

    @pytest.mark.asyncio
    async def test_close(self, adapter, mock_provider: AsyncMock) -> None:  # noqa: ANN001
        """Test close delegates to provider."""
        await adapter.close()
        mock_provider.close.assert_awaited_once()


# =============================================================================
# FinnhubWatchlistAdapter
# =============================================================================


class TestFinnhubWatchlistAdapter:
    """Tests for FinnhubWatchlistAdapter."""

    @pytest.fixture
    def mock_redis(self) -> AsyncMock:
        redis = AsyncMock()
        redis.get.return_value = None  # No cache by default
        return redis

    @pytest.fixture
    def adapter(self, mock_redis: AsyncMock):  # noqa: ANN201
        from synesis.providers.finnhub.fundamentals import FinnhubWatchlistAdapter

        return FinnhubWatchlistAdapter(api_key="test_key", redis=mock_redis)

    @pytest.mark.asyncio
    async def test_resolve_company(self, adapter, mock_redis: AsyncMock) -> None:  # noqa: ANN001
        """Test resolve_company uses /stock/profile2."""
        with patch.object(adapter, "_fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = {"name": "Apple Inc.", "marketCapitalization": 3000000}

            result = await adapter.resolve_company("AAPL")

            assert isinstance(result, CompanyInfo)
            assert result.name == "Apple Inc."
            mock_fetch.assert_awaited_once_with("/stock/profile2", {"symbol": "AAPL"})

    @pytest.mark.asyncio
    async def test_resolve_company_not_found(self, adapter) -> None:  # noqa: ANN001
        """Test resolve_company returns None for unknown ticker."""
        with patch.object(adapter, "_fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = {}

            result = await adapter.resolve_company("UNKNOWN")
            assert result is None

    @pytest.mark.asyncio
    async def test_resolve_company_api_failure(self, adapter) -> None:  # noqa: ANN001
        """Test resolve_company returns None on API failure."""
        with patch.object(adapter, "_fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = None

            result = await adapter.resolve_company("AAPL")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_market_cap(self, adapter) -> None:  # noqa: ANN001
        """Test get_market_cap converts millions to dollars."""
        with patch.object(adapter, "_fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = {"name": "Apple Inc.", "marketCapitalization": 3000000}

            result = await adapter.get_market_cap("AAPL")

            # Finnhub returns market cap in millions
            assert result == 3_000_000 * 1_000_000

    @pytest.mark.asyncio
    async def test_get_market_cap_none(self, adapter) -> None:  # noqa: ANN001
        """Test get_market_cap returns None when profile missing."""
        with patch.object(adapter, "_fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = None

            result = await adapter.get_market_cap("AAPL")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_market_cap_no_field(self, adapter) -> None:  # noqa: ANN001
        """Test get_market_cap returns None when field is missing."""
        with patch.object(adapter, "_fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = {"name": "Apple Inc."}

            result = await adapter.get_market_cap("AAPL")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_fundamentals(self, adapter) -> None:  # noqa: ANN001
        """Test get_fundamentals maps Finnhub metric fields."""
        with patch.object(adapter, "_fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = {
                "metric": {
                    "epsBasicExclExtraItemsTTM": 6.5,
                    "pbAnnual": 45.0,
                    "psAnnual": 8.0,
                    "currentEv/ebitdaTTM": 25.0,
                    "roeTTM": 150.0,
                    "netProfitMarginTTM": 25.0,
                    "grossMarginTTM": 45.0,
                    "totalDebtToEquityAnnual": 1.5,
                }
            }

            result = await adapter.get_fundamentals("AAPL")

            assert isinstance(result, FundamentalsSnapshot)
            assert result.eps_diluted == 6.5
            assert result.price_to_book == 45.0
            assert result.price_to_sales == 8.0
            assert result.ev_to_ebitda == 25.0
            assert result.roe == 150.0
            assert result.net_margin == 25.0
            assert result.gross_margin == 45.0
            assert result.debt_to_equity == 1.5
            assert result.period_type == "ttm"
            assert result.period_end == date.today()

    @pytest.mark.asyncio
    async def test_get_fundamentals_none(self, adapter) -> None:  # noqa: ANN001
        """Test get_fundamentals returns None on API failure."""
        with patch.object(adapter, "_fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = None

            result = await adapter.get_fundamentals("AAPL")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_fundamentals_empty_metric(self, adapter) -> None:  # noqa: ANN001
        """Test get_fundamentals returns None when metric is empty."""
        with patch.object(adapter, "_fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = {"metric": {}}

            result = await adapter.get_fundamentals("AAPL")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_price(self, adapter) -> None:  # noqa: ANN001
        """Test get_price combines /quote and /stock/metric data."""
        with patch.object(adapter, "_fetch", new_callable=AsyncMock) as mock_fetch:
            # First call = /quote, second call = /stock/metric
            mock_fetch.side_effect = [
                {"c": 190.5, "dp": 1.2},
                {"metric": {"monthToDatePriceReturnDaily": 5.5}},
            ]

            result = await adapter.get_price("AAPL")

            assert isinstance(result, PriceSnapshot)
            assert result.one_day_pct == 1.2
            assert result.one_mth_pct == 5.5
            assert result.price_date == date.today()

    @pytest.mark.asyncio
    async def test_get_price_quote_only(self, adapter) -> None:  # noqa: ANN001
        """Test get_price works when metric data unavailable."""
        with patch.object(adapter, "_fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = [
                {"c": 190.5, "dp": 1.2},
                None,
            ]

            result = await adapter.get_price("AAPL")

            assert isinstance(result, PriceSnapshot)
            assert result.one_day_pct == 1.2
            assert result.one_mth_pct is None

    @pytest.mark.asyncio
    async def test_get_price_none(self, adapter) -> None:  # noqa: ANN001
        """Test get_price returns None when quote fails."""
        with patch.object(adapter, "_fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = None

            result = await adapter.get_price("AAPL")
            assert result is None

    @pytest.mark.asyncio
    async def test_close(self, adapter) -> None:  # noqa: ANN001
        """Test close cleans up HTTP client."""
        mock_client = AsyncMock()
        adapter._http_client = mock_client
        await adapter.close()
        mock_client.aclose.assert_awaited_once()
        assert adapter._http_client is None

    @pytest.mark.asyncio
    async def test_close_no_client(self, adapter) -> None:  # noqa: ANN001
        """Test close is safe when no HTTP client exists."""
        assert adapter._http_client is None
        await adapter.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_fetch_uses_redis_cache(self, adapter, mock_redis: AsyncMock) -> None:  # noqa: ANN001
        """Test that _fetch checks Redis cache before calling API."""
        import orjson

        cached_data = {"name": "Cached Corp", "marketCapitalization": 1000}
        mock_redis.get.return_value = orjson.dumps(cached_data)

        result = await adapter.resolve_company("CACHED")

        assert isinstance(result, CompanyInfo)
        assert result.name == "Cached Corp"

    @pytest.mark.asyncio
    async def test_fetch_caches_api_response(self, adapter, mock_redis: AsyncMock) -> None:  # noqa: ANN001
        """Test that _fetch stores API response in Redis."""
        import orjson

        mock_redis.get.return_value = None  # No cache

        mock_response = MagicMock()
        mock_response.content = orjson.dumps({"name": "Apple Inc.", "marketCapitalization": 3e6})
        mock_response.raise_for_status = MagicMock()

        with (
            patch.object(adapter, "_get_http_client") as mock_client_getter,
            patch("synesis.providers.finnhub.fundamentals.get_rate_limiter") as mock_rl,
            patch("synesis.config.get_settings") as mock_settings,
        ):
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_getter.return_value = mock_client
            mock_rl.return_value.acquire = AsyncMock()
            mock_settings.return_value.finnhub_api_url = "https://finnhub.io/api/v1"

            result = await adapter.resolve_company("AAPL")

            assert result is not None
            assert result.name == "Apple Inc."
            # Verify Redis set was called to cache the response
            mock_redis.set.assert_called()
