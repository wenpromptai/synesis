"""Financial data provider abstractions.

This module provides abstract protocols and factory functions for financial
data providers. It allows swapping between providers (Finnhub, Polygon,
Yahoo Finance, etc.) without changing consumer code.

## Provider Types

- **PriceProvider**: Real-time and historical price data with WebSocket support
- **TickerProvider**: Ticker validation and symbol search
- **FundamentalsProvider**: Company fundamental data (financials, filings, etc.)
- **CrawlerProvider**: Web crawling and content extraction

## Usage

### Using Factory Functions (Recommended)

```python
from synesis.providers import create_price_provider, create_ticker_provider

# Create providers based on settings
price_provider = await create_price_provider(redis)
ticker_provider = await create_ticker_provider(redis)

# Use the providers
price = await price_provider.get_price("AAPL")
is_valid, company = await ticker_provider.verify_ticker("AAPL")
```

### Using Direct Imports (For Specific Implementations)

```python
from synesis.providers.finnhub import FinnhubPriceProvider, FinnhubTickerProvider

provider = FinnhubPriceProvider(api_key="...", redis=redis)
```

### Backwards Compatibility

Existing code using `FinnhubService` can continue to work:

```python
from synesis.providers import FinnhubService

service = FinnhubService(api_key="...", redis=redis)
is_valid, company = await service.verify_ticker("AAPL")
financials = await service.get_basic_financials("AAPL")
```

## Configuration

Provider selection is controlled via settings:

```python
# config.py
price_provider: Literal["finnhub", "polygon", "yahoo"] = "finnhub"
ticker_provider: Literal["finnhub", "polygon", "sec"] = "finnhub"
fundamentals_provider: Literal["finnhub", "polygon", "none"] = "finnhub"
```
"""

# Protocols (abstract interfaces)
from synesis.providers.base import (
    CrawlerProvider,
    FundamentalsProvider,
    PriceProvider,
    TickerProvider,
)

# Factory functions
from synesis.providers.factory import (
    FinnhubService,  # Backwards compat combined service
    create_fundamentals_provider,
    create_price_provider,
    create_ticker_provider,
)

# Finnhub implementations (for direct use if needed)
from synesis.providers.finnhub import (
    FinnhubFundamentalsProvider,
    FinnhubPriceProvider,
    FinnhubTickerProvider,
    # Price service utilities (backwards compat)
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

# FactSet implementation
from synesis.providers.factset import (
    FactSetClient,
    FactSetCorporateAction,
    FactSetFundamentals,
    FactSetPrice,
    FactSetProvider,
    FactSetSecurity,
    FactSetSharesOutstanding,
    FactSetTickerProvider,
)

__all__ = [
    # Protocols
    "PriceProvider",
    "TickerProvider",
    "FundamentalsProvider",
    "CrawlerProvider",
    # Factory functions
    "create_price_provider",
    "create_ticker_provider",
    "create_fundamentals_provider",
    # Finnhub implementations
    "FinnhubPriceProvider",
    "FinnhubTickerProvider",
    "FinnhubFundamentalsProvider",
    "FinnhubService",
    # Price service utilities
    "PriceService",
    "RateLimiter",
    "get_price_service",
    "init_price_service",
    "close_price_service",
    "get_rate_limiter",
    # Crawler implementations
    "Crawl4AICrawlerProvider",
    "CrawlResult",
    # FactSet implementation
    "FactSetProvider",
    "FactSetTickerProvider",
    "FactSetClient",
    "FactSetPrice",
    "FactSetSecurity",
    "FactSetFundamentals",
    "FactSetCorporateAction",
    "FactSetSharesOutstanding",
]
