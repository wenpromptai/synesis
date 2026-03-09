"""Tests for web search utility."""

from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from synesis.processing.common.web_search import (
    SearchProvidersExhaustedError,
    _crawl_top_results,
    _extract_article_content,
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


class TestExtractArticleContent:
    """Tests for _extract_article_content."""

    def test_skips_nav_before_first_heading(self) -> None:
        """Nav content before the first heading is stripped."""
        md = "Home | About | Contact\n\n## Article Title\n\nReal content here."
        result = _extract_article_content(md)
        assert "Home" not in result
        assert "Article Title" in result
        assert "Real content here." in result

    def test_no_heading_starts_from_beginning(self) -> None:
        """When no heading is found, content starts from position 0."""
        md = "No headings here, just plain text."
        result = _extract_article_content(md)
        assert "No headings" in result

    def test_strips_image_lines(self) -> None:
        """Standalone image lines are removed."""
        md = "## Article\n\n![logo](https://example.com/img.png)\n\nActual article text."
        result = _extract_article_content(md)
        assert "![logo]" not in result
        assert "Actual article text." in result

    def test_strips_bare_social_share_links(self) -> None:
        """Bare social-share link lines ([ ](url)) are removed."""
        md = "## Article\n\n[ ](https://twitter.com/share)\n* [ ](https://fb.com)\n\nContent."
        result = _extract_article_content(md)
        assert "twitter.com" not in result
        assert "fb.com" not in result
        assert "Content." in result

    def test_preserves_inline_links(self) -> None:
        """Inline links [text](url) within sentences are preserved."""
        md = "## Article\n\nRead the [full report](https://example.com/report) here."
        result = _extract_article_content(md)
        assert "[full report](https://example.com/report)" in result

    def test_collapses_excessive_blank_lines(self) -> None:
        """Three or more consecutive blank lines are collapsed to two."""
        md = "## Article\n\nParagraph one.\n\n\n\n\nParagraph two."
        result = _extract_article_content(md)
        assert "\n\n\n" not in result
        assert "Paragraph one." in result
        assert "Paragraph two." in result

    def test_truncates_to_max_chars(self) -> None:
        """Content is truncated to max_chars."""
        md = "## Article\n\n" + "x" * 5000
        result = _extract_article_content(md, max_chars=100)
        assert len(result) <= 100

    def test_exact_max_chars_boundary(self) -> None:
        """Content equal to max_chars is not empty."""
        content = "x" * 2000
        md = content
        result = _extract_article_content(md, max_chars=2000)
        assert len(result) > 0


class TestEnrichTopResult:
    """Tests for _crawl_top_results."""

    @pytest.mark.anyio
    async def test_no_crawl4ai_url_returns_unchanged(self) -> None:
        """When crawl4ai_url is not configured, results are returned unchanged."""
        results = [{"title": "T", "snippet": "S", "url": "https://example.com"}]
        with patch("synesis.processing.common.web_search.get_settings") as mock_settings:
            mock_settings.return_value.crawl4ai_url = None
            enriched = await _crawl_top_results(results)
        assert enriched == results

    @pytest.mark.anyio
    async def test_no_urls_in_results_returns_unchanged(self) -> None:
        """When results have no url keys, returns unchanged."""
        results = [{"title": "T", "snippet": "S"}]
        with patch("synesis.processing.common.web_search.get_settings") as mock_settings:
            mock_settings.return_value.crawl4ai_url = "http://crawl4ai:11235"
            enriched = await _crawl_top_results(results)
        assert enriched == results

    @pytest.mark.anyio
    async def test_crawl_success_replaces_snippet(self) -> None:
        """Successful crawl replaces the snippet of the top result."""
        results = [
            {"title": "T1", "snippet": "short", "url": "https://example.com/1"},
            {"title": "T2", "snippet": "short2", "url": "https://example.com/2"},
            {"title": "T3", "snippet": "short3", "url": "https://example.com/3"},
        ]

        crawl_result = MagicMock()
        crawl_result.success = True
        crawl_result.markdown = "## Article\n\nFull article content here."

        mock_crawler = AsyncMock()
        mock_crawler.crawl = AsyncMock(return_value=crawl_result)
        mock_crawler.close = AsyncMock()

        with (
            patch("synesis.processing.common.web_search.get_settings") as mock_settings,
            patch(
                "synesis.providers.crawler.crawl4ai.Crawl4AICrawlerProvider",
                return_value=mock_crawler,
            ),
        ):
            mock_settings.return_value.crawl4ai_url = "http://crawl4ai:11235"
            enriched = await _crawl_top_results(results)

        # Top 2 get enriched, index 2 unchanged
        assert enriched[0]["snippet"] != "short"
        assert enriched[1]["snippet"] != "short2"
        assert enriched[2]["snippet"] == "short3"
        # Other fields preserved
        assert enriched[0]["title"] == "T1"
        assert enriched[0]["url"] == "https://example.com/1"

    @pytest.mark.anyio
    async def test_partial_crawl_failure_others_still_enriched(self) -> None:
        """When one URL fails, the other is still enriched."""
        results = [
            {"title": "T1", "snippet": "short1", "url": "https://example.com/1"},
            {"title": "T2", "snippet": "short2", "url": "https://example.com/2"},
        ]

        good_crawl = MagicMock()
        good_crawl.success = True
        good_crawl.markdown = "## Title\n\nGood content."

        async def crawl_side_effect(url: str):
            if "1" in url:
                raise httpx.RequestError("timeout")
            return good_crawl

        mock_crawler = AsyncMock()
        mock_crawler.crawl = AsyncMock(side_effect=crawl_side_effect)
        mock_crawler.close = AsyncMock()

        with (
            patch("synesis.processing.common.web_search.get_settings") as mock_settings,
            patch(
                "synesis.providers.crawler.crawl4ai.Crawl4AICrawlerProvider",
                return_value=mock_crawler,
            ),
        ):
            mock_settings.return_value.crawl4ai_url = "http://crawl4ai:11235"
            enriched = await _crawl_top_results(results)

        assert enriched[0]["snippet"] == "short1"  # failed, kept original
        assert enriched[1]["snippet"] != "short2"  # enriched

    @pytest.mark.anyio
    async def test_crawler_close_called_even_on_error(self) -> None:
        """crawler.close() is called in finally even if crawl raises."""
        results = [{"title": "T", "snippet": "S", "url": "https://example.com"}]

        mock_crawler = AsyncMock()
        mock_crawler.crawl = AsyncMock(side_effect=Exception("boom"))
        mock_crawler.close = AsyncMock()

        with (
            patch("synesis.processing.common.web_search.get_settings") as mock_settings,
            patch(
                "synesis.providers.crawler.crawl4ai.Crawl4AICrawlerProvider",
                return_value=mock_crawler,
            ),
        ):
            mock_settings.return_value.crawl4ai_url = "http://crawl4ai:11235"
            await _crawl_top_results(results)

        mock_crawler.close.assert_called_once()


class TestSearchMarketImpact:
    """Tests for search_market_impact function."""

    @pytest.mark.anyio
    async def test_brave_success(self) -> None:
        """Brave is the primary provider and returns results with enrichment skipped."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Brave Result",
                        "description": "Content",
                        "url": "https://example.com",
                    }
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch("synesis.processing.common.web_search.get_settings") as mock_settings,
            patch(
                "synesis.processing.common.web_search._crawl_top_results", new_callable=AsyncMock
            ) as mock_enrich,
        ):
            settings = MagicMock()
            settings.brave_api_key = MagicMock()
            settings.brave_api_key.get_secret_value.return_value = "brave-key"
            settings.brave_min_interval = 0.0
            settings.crawl4ai_url = None
            mock_settings.return_value = settings
            mock_enrich.return_value = [
                {"title": "Brave Result", "snippet": "Content", "url": "https://example.com"}
            ]

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                results = await search_market_impact("Fed rate cut")

        assert len(results) == 1
        assert results[0]["title"] == "Brave Result"
        mock_enrich.assert_called_once()

    @pytest.mark.anyio
    async def test_brave_fails_falls_back_to_exa(self) -> None:
        """When Brave fails, Exa is tried next."""
        mock_exa_response = MagicMock()
        mock_exa_response.json.return_value = {
            "results": [{"title": "Exa Result", "text": "Content", "url": "https://exa.ai"}]
        }
        mock_exa_response.raise_for_status = MagicMock()

        with patch("synesis.processing.common.web_search.get_settings") as mock_settings:
            settings = MagicMock()
            settings.brave_api_key = MagicMock()
            settings.brave_api_key.get_secret_value.return_value = "brave-key"
            settings.brave_min_interval = 0.0
            settings.exa_api_keys = ["exa-key"]
            mock_settings.return_value = settings

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(side_effect=httpx.RequestError("Brave down"))
                mock_client.post = AsyncMock(return_value=mock_exa_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                results = await search_market_impact("test query")

        assert len(results) == 1
        assert results[0]["title"] == "Exa Result"

    @pytest.mark.anyio
    async def test_all_providers_exhausted(self) -> None:
        """Exception when all providers fail."""
        with patch("synesis.processing.common.web_search.get_settings") as mock_settings:
            settings = MagicMock()
            settings.brave_api_key = None
            settings.exa_api_keys = []
            mock_settings.return_value = settings

            with pytest.raises(SearchProvidersExhaustedError):
                await search_market_impact("test query")

    @pytest.mark.anyio
    async def test_exa_only_when_no_brave(self) -> None:
        """When brave_api_key is None, Exa is tried directly."""
        mock_exa_response = MagicMock()
        mock_exa_response.json.return_value = {
            "results": [{"title": "Exa Result", "text": "Content", "url": "https://exa.ai"}]
        }
        mock_exa_response.raise_for_status = MagicMock()

        with patch("synesis.processing.common.web_search.get_settings") as mock_settings:
            settings = MagicMock()
            settings.brave_api_key = None
            settings.exa_api_keys = ["exa-key"]
            mock_settings.return_value = settings

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_exa_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                results = await search_market_impact("test query")

        assert results[0]["title"] == "Exa Result"


class TestSearchTickerAnalysis:
    """Tests for search_ticker_analysis function."""

    @pytest.mark.anyio
    async def test_returns_results(self) -> None:
        """Test successful ticker analysis search via SearXNG."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "AAPL upgrade",
                    "content": "Analyst upgrades Apple",
                    "url": "https://ex.com",
                }
            ]
        }

        with patch("synesis.processing.common.web_search.get_settings") as mock_settings:
            mock_settings.return_value.searxng_url = "http://localhost:8080"

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                results = await search_ticker_analysis("AAPL", company_name="Apple Inc.")

        assert len(results) == 1
        assert results[0]["title"] == "AAPL upgrade"
        mock_client.get.assert_called_once()
        call_kwargs = mock_client.get.call_args
        params = call_kwargs[1]["params"] if "params" in call_kwargs[1] else call_kwargs[0][1]
        assert "AAPL" in params["q"]
        assert "Apple Inc." in params["q"]
        assert "analyst" in params["q"]

    @pytest.mark.anyio
    async def test_without_company_name(self) -> None:
        """Test ticker analysis search without company name."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"results": []}

        with patch("synesis.processing.common.web_search.get_settings") as mock_settings:
            mock_settings.return_value.searxng_url = "http://localhost:8080"

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                results = await search_ticker_analysis("NVDA")

        assert results == []

    @pytest.mark.anyio
    async def test_returns_empty_when_searxng_not_configured(self) -> None:
        """Test returns empty list when SearXNG is not configured."""
        with patch("synesis.processing.common.web_search.get_settings") as mock_settings:
            mock_settings.return_value.searxng_url = None

            results = await search_ticker_analysis("AAPL")

        assert results == []

    @pytest.mark.anyio
    async def test_returns_empty_on_http_error(self) -> None:
        """Test returns empty list when SearXNG request fails."""
        with patch("synesis.processing.common.web_search.get_settings") as mock_settings:
            mock_settings.return_value.searxng_url = "http://localhost:8080"

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(side_effect=httpx.RequestError("connection failed"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                results = await search_ticker_analysis("AAPL")

        assert results == []

    @pytest.mark.anyio
    async def test_custom_count(self) -> None:
        """Test custom count parameter limits results."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": f"Result {i}", "content": "snippet", "url": f"https://ex.com/{i}"}
                for i in range(10)
            ]
        }

        with patch("synesis.processing.common.web_search.get_settings") as mock_settings:
            mock_settings.return_value.searxng_url = "http://localhost:8080"

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                results = await search_ticker_analysis("AAPL", count=3)

        assert len(results) == 3


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
