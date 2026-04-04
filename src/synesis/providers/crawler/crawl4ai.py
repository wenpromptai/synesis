"""Crawl4AI crawler provider implementation.

This module provides a crawler implementation using the crawl4ai service.
It supports:
- Single URL crawling with markdown extraction
- Batch URL crawling
- Configurable extraction strategies
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from synesis.config import get_settings
from synesis.core.logging import get_logger

logger = get_logger(__name__)

_BASIC_BROWSER_CFG: dict[str, Any] = {"headless": True, "type": "chromium"}
_BASIC_CRAWLER_CFG: dict[str, Any] = {"output_formats": ["markdown"]}

_STEALTH_BROWSER_CFG: dict[str, Any] = {
    "headless": True,
    "type": "chromium",
    "enable_stealth": True,
    "user_agent_mode": "random",
    "extra_args": ["--disable-blink-features=AutomationControlled"],
}
_STEALTH_CRAWLER_CFG: dict[str, Any] = {
    "output_formats": ["markdown"],
    "magic": True,
    "page_timeout": 60000,
}


@dataclass
class CrawlResult:
    """Result of crawling a URL."""

    url: str
    success: bool
    markdown: str
    html: str
    cleaned_html: str
    error: str | None = None
    metadata: dict[str, Any] | None = None
    links: list[dict[str, str]] | None = None
    tables: list[dict[str, Any]] | None = None


class Crawl4AICrawlerProvider:
    """Crawl4AI-based web crawler for fetching and parsing web content.

    This provider uses the crawl4ai service to fetch web pages and extract
    clean markdown content. Ideal for scraping SEC filings, news articles,
    and other web content for LLM analysis.

    Usage:
        crawler = Crawl4AICrawlerProvider()
        result = await crawler.crawl("https://example.com")
        print(result.markdown)

        # Batch crawling
        results = await crawler.crawl_many(["https://a.com", "https://b.com"])

        await crawler.close()
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_token: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Initialize Crawl4AICrawlerProvider.

        Args:
            base_url: Crawl4AI service URL (default: from settings or localhost:11235)
            api_token: Optional API token for authentication
            timeout: Request timeout in seconds (default: 60)
        """
        settings = get_settings()
        self._base_url = (
            base_url or getattr(settings, "crawl4ai_url", None) or "http://localhost:11235"
        )
        self._api_token = api_token
        self._timeout = timeout
        self._http_client: httpx.AsyncClient | None = None

    def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            headers = {}
            if self._api_token:
                headers["Authorization"] = f"Bearer {self._api_token}"
            self._http_client = httpx.AsyncClient(
                timeout=self._timeout,
                headers=headers,
            )
        return self._http_client

    async def health_check(self) -> bool:
        """Check if crawl4ai service is healthy.

        Returns:
            True if service is available, False otherwise
        """
        client = self._get_http_client()
        try:
            response = await client.get(f"{self._base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning("Crawl4AI health check failed", error=str(e))
            return False

    async def crawl(
        self,
        url: str,
        *,
        stealth: bool = True,
        wait_for: str | None = None,
        js_code: str | None = None,
        css_selector: str | None = None,
        screenshot: bool = False,
    ) -> CrawlResult:
        """Crawl a single URL and extract content.

        Args:
            url: URL to crawl
            stealth: Use stealth browser config for anti-bot bypass (default True)
            wait_for: CSS selector to wait for before extracting (for JS-heavy pages)
            js_code: JavaScript code to execute before extraction
            css_selector: CSS selector to focus extraction on
            screenshot: Whether to capture a screenshot

        Returns:
            CrawlResult with markdown, html, and metadata
        """
        results = await self.crawl_many(
            [url],
            stealth=stealth,
            wait_for=wait_for,
            js_code=js_code,
            css_selector=css_selector,
            screenshot=screenshot,
        )
        return (
            results[0]
            if results
            else CrawlResult(
                url=url,
                success=False,
                markdown="",
                html="",
                cleaned_html="",
                error="No results returned",
            )
        )

    async def crawl_many(
        self,
        urls: list[str],
        *,
        stealth: bool = True,
        wait_for: str | None = None,
        js_code: str | None = None,
        css_selector: str | None = None,
        screenshot: bool = False,
    ) -> list[CrawlResult]:
        """Crawl multiple URLs and extract content.

        Args:
            urls: List of URLs to crawl
            stealth: Use stealth browser config for anti-bot bypass (default True)
            wait_for: CSS selector to wait for before extracting
            js_code: JavaScript code to execute before extraction
            css_selector: CSS selector to focus extraction on
            screenshot: Whether to capture screenshots

        Returns:
            List of CrawlResult objects
        """
        client = self._get_http_client()

        if stealth:
            browser_cfg = _STEALTH_BROWSER_CFG
            crawler_cfg = _STEALTH_CRAWLER_CFG
        else:
            browser_cfg = _BASIC_BROWSER_CFG
            crawler_cfg = _BASIC_CRAWLER_CFG

        crawler_cfg_local = dict(crawler_cfg)
        if wait_for:
            crawler_cfg_local["wait_for"] = wait_for
        if js_code:
            crawler_cfg_local["js_code"] = js_code
        if css_selector:
            crawler_cfg_local["css_selector"] = css_selector
        if screenshot:
            crawler_cfg_local["screenshot"] = True

        payload: dict[str, Any] = {
            "urls": urls,
            "browser_config": browser_cfg,
            "crawler_config": crawler_cfg_local,
        }

        try:
            response = await client.post(
                f"{self._base_url}/crawl",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            results: list[CrawlResult] = []
            for item in data.get("results", []):
                # Handle markdown being a dict or string
                markdown_data = item.get("markdown", "")
                if isinstance(markdown_data, dict):
                    markdown = markdown_data.get("raw_markdown", "")
                else:
                    markdown = str(markdown_data) if markdown_data else ""

                results.append(
                    CrawlResult(
                        url=item.get("url", ""),
                        success=item.get("success", False),
                        markdown=markdown,
                        html=item.get("html", ""),
                        cleaned_html=item.get("cleaned_html", ""),
                        error=item.get("error_message"),
                        metadata=item.get("metadata"),
                        links=item.get("links"),
                        tables=item.get("tables"),
                    )
                )

            logger.debug(
                "Crawled URLs",
                count=len(results),
                successful=sum(1 for r in results if r.success),
            )
            return results

        except httpx.HTTPStatusError as e:
            logger.error(
                "Crawl4AI HTTP error",
                status=e.response.status_code,
                urls=urls,
            )
            return [
                CrawlResult(
                    url=url,
                    success=False,
                    markdown="",
                    html="",
                    cleaned_html="",
                    error=f"HTTP {e.response.status_code}",
                )
                for url in urls
            ]
        except Exception as e:
            logger.error("Crawl4AI request failed", error=str(e), urls=urls)
            return [
                CrawlResult(
                    url=url,
                    success=False,
                    markdown="",
                    html="",
                    cleaned_html="",
                    error=str(e),
                )
                for url in urls
            ]

    async def crawl_sec_filing(self, url: str) -> CrawlResult:
        """Crawl an SEC filing URL with basic config (no stealth needed)."""
        return await self.crawl(url, stealth=False)

    async def close(self) -> None:
        """Clean up resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.debug("Crawl4AICrawlerProvider closed")
