"""Flow 2: Sentiment Intelligence.

This module contains:
- Lexicon-based sentiment analyzer (Gate 1)
- LLM refinement processor (Gate 2)
- Ticker watchlist management with TTL-based expiration
- 6-hour sentiment signal generation
"""

from synesis.processing.sentiment.analyzer import SentimentAnalyzer
from synesis.processing.sentiment.models import (
    PostQualityAssessment,
    PostSentiment,
    SentimentRefinement,
    SentimentRefinementDeps,
    SentimentResult,
    SentimentSignal,
    StockEmotion,
    TickerSentimentSummary,
    ValidatedTicker,
)
from synesis.processing.sentiment.processor import (
    SentimentProcessor,
    create_sentiment_processor,
)
from synesis.processing.common.watchlist import (
    TickerMetadata,
    WatchlistManager,
)

__all__ = [
    # Analyzer
    "SentimentAnalyzer",
    "SentimentResult",
    # Models
    "PostQualityAssessment",
    "PostSentiment",
    "SentimentRefinement",
    "SentimentRefinementDeps",
    "SentimentSignal",
    "StockEmotion",
    "TickerSentimentSummary",
    "ValidatedTicker",
    # Processor
    "SentimentProcessor",
    "create_sentiment_processor",
    # Watchlist
    "TickerMetadata",
    "WatchlistManager",
]
