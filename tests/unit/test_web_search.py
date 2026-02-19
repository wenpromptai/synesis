"""Tests for web search utility."""

from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from synesis.processing.common.web_search import (
    SearchProvidersExhaustedError,
    _get_date_range,
    format_search_results,
    search_market_impact,
    search_ticker_analysis,
)


class TestGetDateRange:
    """Tests for _get_date_range helper."""

    def test_day_range(self) -> None:
        """Test day recency range."""
        start, end = _get_date_range("day")
        assert end == date.today()
        assert start == date.today() - timedelta(days=1)

    def test_week_range(self) -> None:
        """Test week recency range."""
        start, end = _get_date_range("week")
        assert end == date.today()
        assert start == date.today() - timedelta(weeks=1)

    def test_month_range(self) -> None:
        """Test month recency range."""
        start, end = _get_date_range("month")
        assert end == date.today()
        assert start == date.today() - timedelta(days=30)

    def test_year_range(self) -> None:
        """Test year recency range."""
        start, end = _get_date_range("year")
        assert end == date.today()
        assert start == date.today() - timedelta(days=365)

    def test_none_range(self) -> None:
        """Test no recency filter."""
        start, end = _get_date_range("none")
        assert end == date.today()
        assert start is None


class TestFormatSearchResults:
    """Tests for format_search_results."""

    def test_format_empty_results(self) -> None:
        """Test formatting empty results."""
        result = format_search_results([])
        assert result == "No search results found."

    def test_format_single_result(self) -> None:
        """Test formatting single result."""
        results = [
            {
                "title": "Test Article",
                "snippet": "This is a test snippet",
                "url": "https://example.com",
            }
        ]
        result = format_search_results(results)
        assert "Test Article" in result
        assert "This is a test snippet" in result

    def test_format_multiple_results(self) -> None:
        """Test formatting multiple results."""
        results = [
            {"title": "Article 1", "snippet": "Snippet 1", "url": "https://example1.com"},
            {"title": "Article 2", "snippet": "Snippet 2", "url": "https://example2.com"},
        ]
        result = format_search_results(results)
        assert "Article 1" in result
        assert "Article 2" in result
        assert "Snippet 1" in result
        assert "Snippet 2" in result

    def test_format_result_without_snippet(self) -> None:
        """Test formatting result without snippet."""
        results = [{"title": "Title Only", "snippet": "", "url": "https://example.com"}]
        result = format_search_results(results)
        assert "Title Only" in result


class TestSearchMarketImpact:
    """Tests for search_market_impact function."""

    @pytest.mark.anyio
    async def test_searxng_success(self) -> None:
        """Test successful SearXNG search."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": "Fed News", "content": "Fed cuts rates", "url": "https://example.com"}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("synesis.processing.common.web_search.get_settings") as mock_settings:
            mock_settings.return_value.searxng_url = "http://localhost:8080"
            mock_settings.return_value.exa_api_key = None
            mock_settings.return_value.brave_api_key = None

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                results = await search_market_impact("Fed rate cut")

        assert len(results) == 1
        assert results[0]["title"] == "Fed News"

    @pytest.mark.anyio
    async def test_fallback_to_exa(self) -> None:
        """Test fallback to Exa when SearXNG fails."""
        mock_exa_response = MagicMock()
        mock_exa_response.json.return_value = {
            "results": [{"title": "Exa Result", "text": "Content", "url": "https://exa.ai"}]
        }
        mock_exa_response.raise_for_status = MagicMock()

        with patch("synesis.processing.common.web_search.get_settings") as mock_settings:
            settings = MagicMock()
            settings.searxng_url = "http://localhost:8080"
            settings.exa_api_key = MagicMock()
            settings.exa_api_key.get_secret_value.return_value = "exa-key"
            settings.brave_api_key = None
            mock_settings.return_value = settings

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                # First call (SearXNG) fails, second call (Exa) succeeds
                mock_client.get = AsyncMock(side_effect=httpx.RequestError("SearXNG down"))
                mock_client.post = AsyncMock(return_value=mock_exa_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                results = await search_market_impact("test query")

        assert len(results) == 1
        assert results[0]["title"] == "Exa Result"

    @pytest.mark.anyio
    async def test_all_providers_exhausted(self) -> None:
        """Test exception when all providers fail."""
        with patch("synesis.processing.common.web_search.get_settings") as mock_settings:
            settings = MagicMock()
            settings.searxng_url = None
            settings.exa_api_key = None
            settings.brave_api_key = None
            mock_settings.return_value = settings

            with pytest.raises(SearchProvidersExhaustedError):
                await search_market_impact("test query")

    @pytest.mark.anyio
    async def test_recency_parameter_passed(self) -> None:
        """Test that recency parameter is used."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"title": "Test", "content": "Content", "url": "http://test.com"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("synesis.processing.common.web_search.get_settings") as mock_settings:
            mock_settings.return_value.searxng_url = "http://localhost:8080"
            mock_settings.return_value.exa_api_key = None
            mock_settings.return_value.brave_api_key = None

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                results = await search_market_impact("test", recency="week")

                # Verify results returned
                assert len(results) == 1
                # Verify get was called with params
                mock_client.get.assert_called_once()


class TestSearchTickerAnalysis:
    """Tests for search_ticker_analysis function."""

    @pytest.mark.anyio
    async def test_returns_results(self) -> None:
        """Test successful ticker analysis search."""
        expected = [
            {"title": "AAPL upgrade", "snippet": "Analyst upgrades", "url": "https://ex.com"}
        ]
        with patch(
            "synesis.processing.common.web_search.search_market_impact",
            new_callable=AsyncMock,
            return_value=expected,
        ) as mock_search:
            results = await search_ticker_analysis("AAPL", company_name="Apple Inc.")

        assert results == expected
        # Verify the query includes ticker, company name, and keywords
        call_args = mock_search.call_args
        query = call_args[0][0]
        assert "AAPL" in query
        assert "Apple Inc." in query
        assert "analyst" in query
        # Verify recency is set to month
        assert call_args[1]["recency"] == "month"

    @pytest.mark.anyio
    async def test_without_company_name(self) -> None:
        """Test ticker analysis search without company name."""
        with patch(
            "synesis.processing.common.web_search.search_market_impact",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_search:
            results = await search_ticker_analysis("NVDA")

        assert results == []
        query = mock_search.call_args[0][0]
        assert "NVDA" in query

    @pytest.mark.anyio
    async def test_returns_empty_on_exhausted(self) -> None:
        """Test returns empty list when all search providers fail."""
        with patch(
            "synesis.processing.common.web_search.search_market_impact",
            new_callable=AsyncMock,
            side_effect=SearchProvidersExhaustedError("All providers failed"),
        ):
            results = await search_ticker_analysis("AAPL")

        assert results == []

    @pytest.mark.anyio
    async def test_custom_count(self) -> None:
        """Test custom count parameter is passed through."""
        with patch(
            "synesis.processing.common.web_search.search_market_impact",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_search:
            await search_ticker_analysis("AAPL", count=5)

        assert mock_search.call_args[1]["count"] == 5


class TestSearchProvidersExhaustedError:
    """Tests for SearchProvidersExhaustedError."""

    def test_error_message(self) -> None:
        """Test error has descriptive message."""
        error = SearchProvidersExhaustedError("All providers failed")
        assert "All providers failed" in str(error)

    def test_is_exception(self) -> None:
        """Test error is an Exception."""
        error = SearchProvidersExhaustedError("Test")
        assert isinstance(error, Exception)
