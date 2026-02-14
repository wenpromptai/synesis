"""Data models for Flow 2: Sentiment Intelligence.

This module defines the schemas for:
- Gate 2 LLM refinement input/output
- Flow 2 signal output
- Sentiment aggregation structures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from synesis.ingestion.reddit import RedditPost


# =============================================================================
# Sentiment Result (lexicon analysis output)
# =============================================================================


@dataclass(frozen=True, slots=True)
class SentimentResult:
    """Result of sentiment analysis on text.

    Attributes:
        compound: Overall sentiment score from -1.0 (most negative) to 1.0 (most positive).
        positive: Proportion of text that is positive (0.0 to 1.0).
        negative: Proportion of text that is negative (0.0 to 1.0).
        neutral: Proportion of text that is neutral (0.0 to 1.0).
        tickers_mentioned: List of stock/crypto tickers found in the text.
        ticker_sentiments: Per-ticker sentiment scores.
        confidence: Confidence in the analysis based on lexicon coverage (0.0 to 1.0).
    """

    compound: float
    positive: float
    negative: float
    neutral: float
    tickers_mentioned: tuple[str, ...] = field(default_factory=tuple)
    ticker_sentiments: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0

    def __post_init__(self) -> None:
        """Validate score ranges."""
        if not -1.0 <= self.compound <= 1.0:
            raise ValueError(f"compound must be in [-1.0, 1.0], got {self.compound}")
        for attr in ("positive", "negative", "neutral", "confidence"):
            val = getattr(self, attr)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{attr} must be in [0.0, 1.0], got {val}")

    @property
    def is_bullish(self) -> bool:
        """Return True if sentiment is bullish (compound > 0.05)."""
        return self.compound > 0.05

    @property
    def is_bearish(self) -> bool:
        """Return True if sentiment is bearish (compound < -0.05)."""
        return self.compound < -0.05

    @property
    def is_neutral(self) -> bool:
        """Return True if sentiment is neutral (-0.05 <= compound <= 0.05)."""
        return -0.05 <= self.compound <= 0.05

    @property
    def strength(self) -> str:
        """Return sentiment strength category."""
        abs_compound = abs(self.compound)
        if abs_compound >= 0.6:
            return "strong"
        if abs_compound >= 0.3:
            return "moderate"
        if abs_compound >= 0.05:
            return "weak"
        return "neutral"


# =============================================================================
# Gate 2: LLM Refinement Models
# =============================================================================


@dataclass
class SentimentRefinementDeps:
    """Dependencies for Gate 2 LLM refinement.

    This is the input to the Gate 2 LLM agent, containing:
    - Raw Reddit posts
    - Gate 1 lexicon analysis results
    - Aggregated ticker mentions with sentiment scores
    - Pre-verified tickers (batch-checked before Gate 2)
    """

    posts: list[RedditPost]
    lexicon_results: list[tuple[RedditPost, SentimentResult]]
    raw_tickers: dict[str, list[float]]  # ticker -> [sentiment scores]
    subreddits: list[str] = field(default_factory=list)
    pre_verified: dict[str, str] = field(default_factory=dict)  # ticker -> company name
    unverified_tickers: list[str] = field(default_factory=list)  # couldn't check (provider down)


class ValidatedTicker(BaseModel):
    """A ticker that has been validated by Gate 2 LLM."""

    ticker: str = Field(description="Stock ticker symbol (e.g., AAPL)")
    company_name: str = Field(description="Full company name")
    is_valid_ticker: bool = Field(description="True if this is a real, tradable stock ticker")
    rejection_reason: str | None = Field(
        default=None,
        description="Why it's not a valid ticker (if rejected)",
    )
    avg_sentiment: float = Field(description="Average sentiment score from -1.0 to 1.0")
    sentiment_label: Literal["bullish", "bearish", "neutral"] = Field(
        description="Overall sentiment classification"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in validation (0.0 to 1.0)")
    key_catalysts: list[str] = Field(
        default_factory=list,
        description="Key catalysts mentioned (earnings, news, technical, etc.)",
    )


class PostQualityAssessment(BaseModel):
    """Quality assessment for a Reddit post."""

    post_id: str = Field(description="Reddit post ID")
    quality: Literal["high", "medium", "low", "spam"] = Field(description="Post quality tier")
    is_dd: bool = Field(default=False, description="True if this is a Due Diligence post")
    is_yolo: bool = Field(default=False, description="True if this is a YOLO/gain/loss porn post")
    has_thesis: bool = Field(default=False, description="True if post contains investment thesis")
    key_insight: str | None = Field(default=None, description="Key insight from the post (if any)")


class SentimentRefinement(BaseModel):
    """Gate 2 output: Refined sentiment analysis from LLM.

    This is the structured output from the Gate 2 LLM refinement,
    which validates tickers, assesses post quality, and generates
    a narrative summary.
    """

    # Validated tickers (false positives removed)
    validated_tickers: list[ValidatedTicker] = Field(
        default_factory=list,
        description="Tickers validated as real, tradable symbols",
    )
    rejected_tickers: list[str] = Field(
        default_factory=list,
        description="False positives like WEEK, WHAT, DD, etc.",
    )

    # Post quality assessments
    post_assessments: list[PostQualityAssessment] = Field(
        default_factory=list,
        description="Quality assessment for each post",
    )
    high_quality_posts: int = Field(default=0, description="Count of high quality posts")
    spam_posts: int = Field(default=0, description="Count of spam posts")

    # Aggregate sentiment
    overall_sentiment: Literal["bullish", "bearish", "neutral", "mixed"] = Field(
        default="neutral", description="Overall market sentiment"
    )
    sentiment_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence in overall sentiment"
    )
    # Narrative
    narrative_summary: str = Field(
        default="",
        description="2-3 sentence market narrative summary",
    )
    key_themes: list[str] = Field(
        default_factory=list,
        description='Key themes identified (e.g., "silver crash", "earnings season")',
    )


# =============================================================================
# Flow 2 Signal Output Models
# =============================================================================


class PostSentiment(BaseModel):
    """Sentiment data for a single post."""

    post_id: str
    subreddit: str
    title: str
    sentiment: float  # -1.0 to 1.0
    sentiment_label: Literal["bullish", "bearish", "neutral"]
    quality: Literal["high", "medium", "low", "spam"]
    url: str


class TickerSentimentSummary(BaseModel):
    """Sentiment summary for a single ticker over the signal period."""

    ticker: str = Field(description="Stock ticker symbol")
    company_name: str = Field(default="", description="Company name if known")
    mention_count: int = Field(default=0, description="Number of mentions")

    # Sentiment breakdown
    bullish_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Ratio of bullish mentions"
    )
    bearish_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Ratio of bearish mentions"
    )
    avg_sentiment: float = Field(default=0.0, description="Average sentiment score (-1.0 to 1.0)")

    # Derived fields
    volume_zscore: float = Field(
        default=0.0,
        description="Z-score of mention volume vs historical average",
    )

    # Flags
    is_volume_spike: bool = Field(default=False, description="True if volume z-score > 2")

    # Supporting posts
    top_posts: list[PostSentiment] = Field(
        default_factory=list,
        description="Top posts for this ticker (by quality)",
    )
    key_catalysts: list[str] = Field(
        default_factory=list,
        description="Key catalysts identified",
    )


class SentimentSignal(BaseModel):
    """Sentiment signal output: 6-hour sentiment intelligence summary.

    This is emitted every 6 hours with aggregated sentiment data
    from Reddit (and eventually Twitter).
    """

    # Timing
    timestamp: datetime = Field(description="Signal generation timestamp")
    signal_period: str = Field(default="6h", description="Signal period")
    period_start: datetime = Field(description="Start of analysis period")
    period_end: datetime = Field(description="End of analysis period")

    # Watchlist changes
    watchlist: list[str] = Field(default_factory=list, description="Current watchlist tickers")
    watchlist_added: list[str] = Field(
        default_factory=list, description="Tickers added this period"
    )
    watchlist_removed: list[str] = Field(
        default_factory=list, description="Tickers removed this period"
    )

    # Per-ticker sentiment
    ticker_sentiments: list[TickerSentimentSummary] = Field(
        default_factory=list,
        description="Sentiment summary for each ticker",
    )

    # Aggregates
    total_posts_analyzed: int = Field(default=0, description="Total posts analyzed")
    high_quality_posts: int = Field(default=0, description="High quality posts count")
    spam_posts: int = Field(default=0, description="Spam posts filtered")

    # Overall narrative
    overall_sentiment: Literal["bullish", "bearish", "neutral", "mixed"] = Field(
        default="neutral", description="Overall market sentiment"
    )
    narrative_summary: str = Field(
        default="",
        description="LLM-generated market narrative (2-3 sentences)",
    )
    key_themes: list[str] = Field(
        default_factory=list,
        description="Key themes identified this period",
    )

    # Source breakdown
    sources: dict[str, int] = Field(
        default_factory=dict,
        description="Post count by source (reddit, twitter, etc.)",
    )
    subreddits: dict[str, int] = Field(
        default_factory=dict,
        description="Post count by subreddit",
    )
