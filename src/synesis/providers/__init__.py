"""Financial data provider abstractions.

## Provider Types

- **TickerProvider**: Ticker validation and symbol search
- **CrawlerProvider**: Web crawling and content extraction

## Standalone Providers

- **Finnhub**: Real-time prices, ticker verification
- **SEC EDGAR**: Filings, insider transactions/sentiment, historical EPS/revenue (XBRL)
- **NASDAQ**: Earnings calendar, EPS forecasts
"""

# Protocols (abstract interfaces)
from synesis.providers.base import (
    CrawlerProvider,
    FundamentalsProvider,
    TickerProvider,
)

# Factory functions
from synesis.providers.factory import (
    create_ticker_provider,
)

# Finnhub implementations
from synesis.providers.finnhub import (
    FinnhubFundamentalsProvider,
    FinnhubPriceProvider,
    FinnhubTickerProvider,
    # Price service utilities
    PriceService,
    RateLimiter,
    close_price_service,
    get_price_service,
    get_rate_limiter,
    init_price_service,
)

# Crawler implementations
from synesis.providers.crawler import Crawl4AICrawlerProvider
from synesis.providers.crawler.crawl4ai import CrawlResult

# SEC EDGAR implementation
from synesis.providers.sec_edgar import (
    InsiderTransaction,
    SECEdgarClient,
    SECFiling,
)

# NASDAQ implementation
from synesis.providers.nasdaq import (
    EarningsEvent,
    NasdaqClient,
)

__all__ = [
    # Protocols
    "TickerProvider",
    "FundamentalsProvider",
    "CrawlerProvider",
    # Factory functions
    "create_ticker_provider",
    # Finnhub
    "FinnhubPriceProvider",
    "FinnhubTickerProvider",
    "FinnhubFundamentalsProvider",
    # SEC EDGAR
    "SECEdgarClient",
    "SECFiling",
    "InsiderTransaction",
    # NASDAQ
    "NasdaqClient",
    "EarningsEvent",
    # Price service utilities
    "PriceService",
    "RateLimiter",
    "get_price_service",
    "init_price_service",
    "close_price_service",
    "get_rate_limiter",
    # Crawler
    "Crawl4AICrawlerProvider",
    "CrawlResult",
]
