"""Processing module - organized by flow type.

Submodules:
- news: Breaking news & analysis intelligence (Flow 1)
- common: Shared utilities (LLM factory, web search, watchlist)
"""

# Re-exports for backward compatibility (deprecated, use submodule imports)
from synesis.processing.common.llm import create_model
from synesis.processing.common.web_search import (
    Recency,
    SearchProvidersExhaustedError,
    format_search_results,
    search_market_impact,
)
from synesis.processing.news.analyzer import (
    AnalyzerDeps,
    SmartAnalyzer,
)
from synesis.processing.news.categorizer import categorize_news
from synesis.processing.news.models import (
    BeatMissStatus,
    Direction,
    LightClassification,
    MarketEvaluation,
    MetricReading,
    NewsCategory,
    NewsSignal,
    NumericExtraction,
    PrimaryTopic,
    ResearchQuality,
    SecondaryTopic,
    SmartAnalysis,
    SourcePlatform,
    TickerAnalysis,
    UnifiedMessage,
    UrgencyLevel,
)
from synesis.processing.common.watchlist import TickerMetadata, WatchlistManager

__all__ = [
    # News (Flow 1) - Analyzer
    "AnalyzerDeps",
    "SmartAnalyzer",
    "categorize_news",
    # News (Flow 1) - Models
    "BeatMissStatus",
    "Direction",
    "LightClassification",
    "MarketEvaluation",
    "MetricReading",
    "NewsCategory",
    "NewsSignal",
    "NumericExtraction",
    "PrimaryTopic",
    "ResearchQuality",
    "SecondaryTopic",
    "SmartAnalysis",
    "SourcePlatform",
    "TickerAnalysis",
    "UnifiedMessage",
    "UrgencyLevel",
    "TickerMetadata",
    "WatchlistManager",
    # Common utilities
    "Recency",
    "SearchProvidersExhaustedError",
    "create_model",
    "format_search_results",
    "search_market_impact",
]
