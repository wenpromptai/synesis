"""Flow 1: Breaking News Intelligence.

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

__all__ = [
    # Analyzer (Stage 2)
    "AnalyzerDeps",
    "SmartAnalyzer",
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
]
