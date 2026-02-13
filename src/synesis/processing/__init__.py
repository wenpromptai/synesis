"""Processing module - organized by flow type.

Submodules:
- news: Breaking news & analysis intelligence (Flow 1)
- sentiment: Sentiment intelligence (Flow 2)
- market_intel: Prediction market intelligence (Flow 3, future)
- common: Cross-flow utilities (LLM factory, web search)
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
    analyze_with_context,
)
from synesis.processing.news.categorizer import categorize_news
from synesis.processing.news.models import (
    BeatMissStatus,
    Direction,
    EvaluatorOutput,
    EventType,
    NewsSignal,
    GICSSector,
    InvestmentAnalysis,
    LightClassification,
    MarketEvaluation,
    MarketOpportunity,
    MetricReading,
    NewsCategory,
    NumericExtraction,
    OddsEvaluation,
    ResearchQuality,
    SectorImplication,
    SmartAnalysis,
    SourcePlatform,
    SourceType,
    TickerAnalysis,
    UnifiedMessage,
    UrgencyLevel,
)
from synesis.processing.sentiment.models import (
    SentimentSignal,
    PostQualityAssessment,
    PostSentiment,
    SentimentRefinement,
    SentimentRefinementDeps,
    TickerSentimentSummary,
    ValidatedTicker,
)
from synesis.processing.sentiment.processor import SentimentProcessor
from synesis.processing.common.watchlist import TickerMetadata, WatchlistManager

__all__ = [
    # News (Flow 1) - Analyzer
    "AnalyzerDeps",
    "SmartAnalyzer",
    "analyze_with_context",
    "categorize_news",
    # News (Flow 1) - Models
    "BeatMissStatus",
    "Direction",
    "EvaluatorOutput",
    "EventType",
    "NewsSignal",
    "GICSSector",
    "InvestmentAnalysis",
    "LightClassification",
    "MarketEvaluation",
    "MarketOpportunity",
    "MetricReading",
    "NewsCategory",
    "NumericExtraction",
    "OddsEvaluation",
    "ResearchQuality",
    "SectorImplication",
    "SmartAnalysis",
    "SourcePlatform",
    "SourceType",
    "TickerAnalysis",
    "UnifiedMessage",
    "UrgencyLevel",
    # Sentiment (Flow 2)
    "SentimentProcessor",
    "SentimentSignal",
    "PostQualityAssessment",
    "PostSentiment",
    "SentimentRefinement",
    "SentimentRefinementDeps",
    "TickerMetadata",
    "TickerSentimentSummary",
    "ValidatedTicker",
    "WatchlistManager",
    # Common utilities
    "Recency",
    "SearchProvidersExhaustedError",
    "create_model",
    "format_search_results",
    "search_market_impact",
]
