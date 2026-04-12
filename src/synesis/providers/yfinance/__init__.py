"""yfinance provider — equity/ETF/index quotes, FX rates, OHLCV history, and options chains.

Free data, no API key required. ~15 min delayed for US equities during market hours.
"""

from synesis.providers.yfinance.client import YFinanceClient
from synesis.providers.yfinance.models import (
    AnalystPriceTargets,
    AnalystRatings,
    EquityQuote,
    FXRate,
    OHLCBar,
    OptionsChain,
    OptionsContract,
    OptionsGreeks,
    OptionsSnapshot,
    RecommendationTrend,
    UpgradeDowngrade,
)

__all__ = [
    "YFinanceClient",
    "AnalystPriceTargets",
    "AnalystRatings",
    "EquityQuote",
    "FXRate",
    "OHLCBar",
    "OptionsChain",
    "OptionsContract",
    "OptionsGreeks",
    "OptionsSnapshot",
    "RecommendationTrend",
    "UpgradeDowngrade",
]
