"""Flow 1: Breaking News & Analysis Intelligence.

This module contains:
- Unified message model and source types
- Stage 1: Fast entity extraction (classifier.py)
- Stage 2: Smart analysis with research context (analyzer.py)
- Message deduplication using semantic similarity
- News categorization (breaking, economic_calendar, other)
"""

from synesis.processing.news.analyzer import (
    AnalyzerDeps,
    SmartAnalyzer,
    analyze_with_context,
    get_smart_analyzer,
)
from synesis.processing.news.categorizer import (
    categorize_by_rules,
    categorize_news,
    classify_urgency_by_rules,
)
from synesis.processing.news.classifier import (
    NewsClassifier,
    classify_message,
    get_classifier,
)
from synesis.processing.news.deduplication import (
    DeduplicationResult,
    MessageDeduplicator,
    create_deduplicator,
)
from synesis.processing.news.models import (
    BeatMissStatus,
    BreakingClassification,
    Direction,
    EvaluatorOutput,
    EventType,
    NewsSignal,
    GICSSector,
    ImpactLevel,
    InvestmentAnalysis,
    LightClassification,
    MarketEvaluation,
    MarketOpportunity,
    MetricReading,
    NewsCategory,
    NumericExtraction,
    OddsEvaluation,
    ResearchAnalysis,
    ResearchQuality,
    SectorImplication,
    SmartAnalysis,
    SourcePlatform,
    SourceType,
    TickerAnalysis,
    UnifiedMessage,
    UrgencyLevel,
)

__all__ = [
    # Analyzer (Stage 2)
    "AnalyzerDeps",
    "SmartAnalyzer",
    "analyze_with_context",
    "get_smart_analyzer",
    # Categorizer
    "categorize_by_rules",
    "categorize_news",
    "classify_urgency_by_rules",
    # Classifier (Stage 1)
    "NewsClassifier",
    "classify_message",
    "get_classifier",
    # Deduplication
    "DeduplicationResult",
    "MessageDeduplicator",
    "create_deduplicator",
    # Models
    "BeatMissStatus",
    "BreakingClassification",
    "Direction",
    "EvaluatorOutput",
    "EventType",
    "NewsSignal",
    "GICSSector",
    "ImpactLevel",
    "InvestmentAnalysis",
    "LightClassification",
    "MarketEvaluation",
    "MarketOpportunity",
    "MetricReading",
    "NewsCategory",
    "NumericExtraction",
    "OddsEvaluation",
    "ResearchAnalysis",
    "ResearchQuality",
    "SectorImplication",
    "SmartAnalysis",
    "SourcePlatform",
    "SourceType",
    "TickerAnalysis",
    "UnifiedMessage",
    "UrgencyLevel",
]
