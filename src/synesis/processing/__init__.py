"""Processing layer: LLM analysis, deduplication, entity extraction."""

from synesis.processing.categorizer import categorize_news
from synesis.processing.models import (
    BreakingClassification,
    Direction,
    EvaluatorOutput,
    EventType,
    Flow1Signal,
    ImpactLevel,
    InvestmentAnalysis,
    LightClassification,
    MarketEvaluation,
    MarketOpportunity,
    NewsCategory,
    ResearchQuality,
    SectorImplication,
    SmartAnalysis,
    SourcePlatform,
    SourceType,
    TickerAnalysis,
    UnifiedMessage,
)
from synesis.processing.smart_analyzer import (
    AnalyzerDeps,
    SmartAnalyzer,
    analyze_with_context,
)
from synesis.processing.web_search import (
    format_search_results,
    search_market_impact,
)

__all__ = [
    "AnalyzerDeps",
    "BreakingClassification",
    "Direction",
    "EvaluatorOutput",
    "EventType",
    "Flow1Signal",
    "ImpactLevel",
    "InvestmentAnalysis",
    "LightClassification",
    "MarketEvaluation",
    "MarketOpportunity",
    "NewsCategory",
    "ResearchQuality",
    "SectorImplication",
    "SmartAnalysis",
    "SmartAnalyzer",
    "SourcePlatform",
    "SourceType",
    "TickerAnalysis",
    "UnifiedMessage",
    "analyze_with_context",
    "categorize_news",
    "format_search_results",
    "search_market_impact",
]
