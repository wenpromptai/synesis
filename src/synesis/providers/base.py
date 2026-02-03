"""Abstract provider protocols for financial data.

This module defines the abstract interfaces (Protocols) that financial data
providers must implement. This allows swapping between providers (Finnhub,
Polygon, Yahoo Finance, etc.) without changing consumer code.

Provider Types:
- PriceProvider: Real-time and historical price data with WebSocket support
- TickerProvider: Ticker validation and symbol search
- FundamentalsProvider: Company fundamental data (financials, filings, etc.)
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PriceProvider(Protocol):
    """Protocol for real-time and historical price data.

    Providers implementing this protocol must support:
    - Cached price lookups (fast, from local cache)
    - REST API fallback for cache misses
    - Optional WebSocket subscription for real-time updates
    """

    async def get_price(self, ticker: str) -> Decimal | None:
        """Get current price for a single ticker.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            Price as Decimal, or None if not available
        """
        ...

    async def get_prices(
        self,
        tickers: list[str],
        fallback_to_rest: bool = True,
    ) -> dict[str, Decimal]:
        """Get prices for multiple tickers.

        Args:
            tickers: List of stock ticker symbols
            fallback_to_rest: Whether to use REST API for cache misses

        Returns:
            Dict mapping ticker to price (only includes available prices)
        """
        ...

    async def subscribe(self, tickers: list[str]) -> None:
        """Subscribe to real-time price updates for tickers.

        Args:
            tickers: List of ticker symbols to subscribe to
        """
        ...

    async def unsubscribe(self, tickers: list[str]) -> None:
        """Unsubscribe from real-time price updates.

        Args:
            tickers: List of ticker symbols to unsubscribe from
        """
        ...

    async def start(self) -> None:
        """Start the price provider (e.g., WebSocket connection)."""
        ...

    async def close(self) -> None:
        """Clean up resources."""
        ...


@runtime_checkable
class TickerProvider(Protocol):
    """Protocol for ticker validation and symbol search.

    Providers implementing this protocol must support:
    - Ticker verification (check if symbol exists on major exchanges)
    - Symbol search (find tickers matching a query)
    """

    async def verify_ticker(self, ticker: str) -> tuple[bool, str | None]:
        """Verify if a ticker symbol exists on a major exchange.

        Args:
            ticker: Stock ticker symbol to verify (e.g., "AAPL")

        Returns:
            Tuple of (is_valid, company_name):
            - is_valid: True if ticker exists on major exchange
            - company_name: Company name if found, None otherwise
        """
        ...

    async def search_symbol(self, query: str) -> list[dict[str, str]]:
        """Search for stock symbols matching a query.

        Args:
            query: Search query (ticker or company name)

        Returns:
            List of matching symbols with keys: symbol, description, type
        """
        ...


@runtime_checkable
class FundamentalsProvider(Protocol):
    """Protocol for company fundamental data.

    Providers implementing this protocol must support:
    - Basic financials (P/E, market cap, 52-week range)
    - Insider transactions and sentiment
    - Earnings data (calendar, EPS surprises)
    - SEC filings
    """

    async def get_basic_financials(self, ticker: str) -> dict[str, Any] | None:
        """Get key financial metrics for a stock.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            Dict with financial metrics, or None if not available.
            Expected keys: peRatio, marketCap, 52WeekHigh, 52WeekLow,
            beta, eps, revenueGrowth, dividendYield, etc.
        """
        ...

    async def get_insider_transactions(self, ticker: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent insider transactions.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum transactions to return

        Returns:
            List of insider transactions with: name, shares, change,
            filingDate, transactionCode, transactionPrice
        """
        ...

    async def get_insider_sentiment(self, ticker: str) -> dict[str, Any] | None:
        """Get aggregate insider sentiment.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with: ticker, mspr (Monthly Share Purchase Ratio),
            change, year, month. Or None if not available.
        """
        ...

    async def get_earnings_calendar(self, ticker: str) -> dict[str, Any] | None:
        """Get next earnings date and estimates.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with: ticker, date, epsEstimate, hour. Or None if not available.
        """
        ...

    async def get_eps_surprises(self, ticker: str, limit: int = 4) -> list[dict[str, Any]]:
        """Get historical EPS surprises.

        Args:
            ticker: Stock ticker symbol
            limit: Number of quarters to return

        Returns:
            List of earnings with: period, actual, estimate, surprise, surprisePercent
        """
        ...

    async def get_sec_filings(self, ticker: str, limit: int = 5) -> list[dict[str, Any]]:
        """Get recent SEC filings.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum filings to return

        Returns:
            List of SEC filings with: form, filedDate, acceptedDate, reportUrl
        """
        ...

    async def close(self) -> None:
        """Clean up resources."""
        ...


@runtime_checkable
class CrawlerProvider(Protocol):
    """Protocol for web crawling and content extraction.

    Providers implementing this protocol must support:
    - Single URL crawling with content extraction
    - Batch URL crawling
    - Health check for service availability
    """

    async def crawl(self, url: str) -> Any:
        """Crawl a single URL and extract content.

        Args:
            url: URL to crawl

        Returns:
            CrawlResult or similar with extracted content
        """
        ...

    async def crawl_many(self, urls: list[str]) -> list[Any]:
        """Crawl multiple URLs and extract content.

        Args:
            urls: List of URLs to crawl

        Returns:
            List of crawl results
        """
        ...

    async def health_check(self) -> bool:
        """Check if crawler service is available.

        Returns:
            True if service is healthy, False otherwise
        """
        ...

    async def close(self) -> None:
        """Clean up resources."""
        ...
