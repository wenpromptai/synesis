"""Processing module - organized by flow type.

Submodules:
- news: Breaking news & analysis intelligence (Flow 1)
- common: Shared utilities (LLM factory, web search, watchlist)
"""

from synesis.processing.common.llm import create_model
from synesis.processing.common.watchlist import WatchlistManager
from synesis.processing.common.web_search import (
    Recency,
    format_search_results,
    search_market_impact,
)
from synesis.processing.news.analyzer import (
    AnalyzerDeps,
    SmartAnalyzer,
)
from synesis.processing.news.models import (
    ETFImpact,
    LightClassification,
    MarketEvaluation,
    NewsSignal,
    PrimaryTopic,
    SecondaryTopic,
    SmartAnalysis,
    SourcePlatform,
    UnifiedMessage,
    UrgencyLevel,
)

__all__ = [
    # News (Flow 1)
    "AnalyzerDeps",
    "SmartAnalyzer",
    "ETFImpact",
    "LightClassification",
    "MarketEvaluation",
    "NewsSignal",
    "PrimaryTopic",
    "SecondaryTopic",
    "SmartAnalysis",
    "SourcePlatform",
    "UnifiedMessage",
    "UrgencyLevel",
    "WatchlistManager",
    # Common utilities
    "Recency",
    "create_model",
    "format_search_results",
    "search_market_impact",
]
