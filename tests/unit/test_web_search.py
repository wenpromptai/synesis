"""Tests for web search utility."""

from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import synesis.processing.common.web_search as ws_module
from synesis.processing.common.web_search import (
    _extract_article_content,
    _get_date_range,
    format_search_results,
    search_market_impact,
)


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Reset circuit breaker state between tests."""
    ws_module._brave_tripped = False
    ws_module._brave_fail_count = 0
    ws_module._brave_last_call = 0.0
    ws_module._brave_lock = None
    yield
    ws_module._brave_tripped = False
    ws_module._brave_fail_count = 0
    ws_module._brave_last_call = 0.0
    ws_module._brave_lock = None


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


class TestSearchMarketImpact:
    """Tests for search_market_impact function."""

    @pytest.mark.anyio
    async def test_brave_success(self) -> None:
        """Brave returns results successfully."""
        mock_response = MagicMock()
        mock_response.status_code = 200
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

        with patch("synesis.processing.common.web_search.get_settings") as mock_settings:
            settings = MagicMock()
            settings.brave_api_key = MagicMock()
            settings.brave_api_key.get_secret_value.return_value = "brave-key"
            settings.brave_min_interval = 0.0
            mock_settings.return_value = settings

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                results = await search_market_impact("Fed rate cut")

        assert len(results) == 1
        assert results[0]["title"] == "Brave Result"

    @pytest.mark.anyio
    async def test_brave_failure_returns_empty(self) -> None:
        """When Brave fails, empty list is returned."""
        with patch("synesis.processing.common.web_search.get_settings") as mock_settings:
            settings = MagicMock()
            settings.brave_api_key = MagicMock()
            settings.brave_api_key.get_secret_value.return_value = "brave-key"
            settings.brave_min_interval = 0.0
            mock_settings.return_value = settings

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(side_effect=httpx.RequestError("Brave down"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                results = await search_market_impact("test query")

        assert results == []

    @pytest.mark.anyio
    async def test_no_brave_key_returns_empty(self) -> None:
        """When brave_api_key is not configured, returns empty list."""
        with patch("synesis.processing.common.web_search.get_settings") as mock_settings:
            settings = MagicMock()
            settings.brave_api_key = None
            mock_settings.return_value = settings

            results = await search_market_impact("test query")

        assert results == []

    @pytest.mark.anyio
    async def test_circuit_breaker_trips_after_threshold(self) -> None:
        """Circuit breaker trips after 3 consecutive 429 responses."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("429", request=MagicMock(), response=mock_response)
        )

        with patch("synesis.processing.common.web_search.get_settings") as mock_settings:
            settings = MagicMock()
            settings.brave_api_key = MagicMock()
            settings.brave_api_key.get_secret_value.return_value = "brave-key"
            settings.brave_min_interval = 0.0
            mock_settings.return_value = settings

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                # First 3 calls trip the breaker
                for _ in range(3):
                    results = await search_market_impact("test query")
                    assert results == []

                assert ws_module._brave_tripped is True

                # 4th call skips Brave entirely (no HTTP call)
                mock_client.get.reset_mock()
                results = await search_market_impact("test query")
                assert results == []
                mock_client.get.assert_not_called()

    @pytest.mark.anyio
    async def test_circuit_breaker_resets_on_success(self) -> None:
        """Successful call resets the failure counter."""
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("429", request=MagicMock(), response=mock_429)
        )

        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.json.return_value = {
            "web": {"results": [{"title": "OK", "description": "d", "url": "u"}]}
        }
        mock_200.raise_for_status = MagicMock()

        with patch("synesis.processing.common.web_search.get_settings") as mock_settings:
            settings = MagicMock()
            settings.brave_api_key = MagicMock()
            settings.brave_api_key.get_secret_value.return_value = "brave-key"
            settings.brave_min_interval = 0.0
            mock_settings.return_value = settings

            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                # 2 failures then a success
                mock_client.get = AsyncMock(side_effect=[mock_429, mock_429, mock_200])
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                await search_market_impact("q1")  # fail 1
                await search_market_impact("q2")  # fail 2
                results = await search_market_impact("q3")  # success

                assert ws_module._brave_fail_count == 0
                assert ws_module._brave_tripped is False
                assert len(results) == 1
