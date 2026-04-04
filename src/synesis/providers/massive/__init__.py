"""Massive.com provider for stocks and options market data.

API docs: https://massive.com/docs/rest
Free tier: 5 API calls/minute. API key required.
"""

from synesis.providers.massive.client import MassiveClient
from synesis.providers.massive.models import (
    Bar,
    BarsResponse,
    DailySummary,
    Dividend,
    FinancialResult,
    IndicatorValue,
    MACDValue,
    MarketHoliday,
    MarketStatus,
    NewsArticle,
    NewsInsight,
    OptionsContractRef,
    ShortInterest,
    ShortVolume,
    Split,
    TickerEvent,
    TickerInfo,
    TickerOverview,
)

__all__ = [
    "MassiveClient",
    "Bar",
    "BarsResponse",
    "DailySummary",
    "Dividend",
    "FinancialResult",
    "IndicatorValue",
    "MACDValue",
    "MarketHoliday",
    "MarketStatus",
    "NewsArticle",
    "NewsInsight",
    "OptionsContractRef",
    "ShortInterest",
    "ShortVolume",
    "Split",
    "TickerEvent",
    "TickerInfo",
    "TickerOverview",
]
