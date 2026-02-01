"""Flow 2: Sentiment Intelligence.

This module contains:
- Reddit sentiment processing (Gate 1: lexicon, Gate 2: LLM refinement)
- Ticker watchlist management with TTL-based expiration
- 6-hour sentiment signal generation
"""

from synesis.processing.sentiment.models import (
    SentimentSignal,
    PostQualityAssessment,
    PostSentiment,
    SentimentRefinement,
    SentimentRefinementDeps,
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
    # Models
    "SentimentSignal",
    "PostQualityAssessment",
    "PostSentiment",
    "SentimentRefinement",
    "SentimentRefinementDeps",
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
