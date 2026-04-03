"""Flow 1: Breaking News Intelligence.

- Stage 1: Instant classification — impact scoring + ticker matching (classifier.py)
- Stage 2: Smart analysis with LLM (analyzer.py)
- Message deduplication using semantic similarity
"""

from synesis.processing.news.analyzer import AnalyzerDeps, SmartAnalyzer
from synesis.processing.news.classifier import NewsClassifier
from synesis.processing.news.deduplication import (
    DeduplicationResult,
    MessageDeduplicator,
    create_deduplicator,
)
from synesis.processing.news.impact_scorer import (
    ImpactResult,
    compute_impact_score,
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
from synesis.processing.news.ticker_matcher import match_tickers

__all__ = [
    # Analyzer (Stage 2)
    "AnalyzerDeps",
    "SmartAnalyzer",
    # Classifier (Stage 1)
    "NewsClassifier",
    # Deduplication
    "DeduplicationResult",
    "MessageDeduplicator",
    "create_deduplicator",
    # Impact Scorer
    "ImpactResult",
    "compute_impact_score",
    # Ticker Matcher
    "match_tickers",
    # Models
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
]
