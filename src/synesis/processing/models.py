"""Processing models for Flow 1: Breaking News & Analysis.

These models define the data structures for the news processing pipeline:
- Unified message format from Twitter/Telegram
- LLM classification output schema
- Final signal output
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Source Types
# =============================================================================


class SourcePlatform(str, Enum):
    """Platform where the message originated."""

    twitter = "twitter"
    telegram = "telegram"


class SourceType(str, Enum):
    """Type of source - determines urgency."""

    news = "news"  # High urgency - act fast (e.g., @DeItaone, @marketfeed)
    analysis = "analysis"  # Normal urgency - consider (e.g., @elonmusk, analysts)


class NewsCategory(str, Enum):
    """Category of news content (detected per-message, not per-source)."""

    breaking = "breaking"  # Unexpected events: *BREAKING, JUST IN, sudden announcements
    economic_calendar = "economic_calendar"  # Scheduled releases: CPI, NFP, FOMC, GDP
    other = "other"  # Analysis, commentary, opinions, general news


# =============================================================================
# Unified Message
# =============================================================================


class UnifiedMessage(BaseModel):
    """Normalized message from any source (Twitter/Telegram).

    This is the common format after ingestion, before processing.
    """

    # Identity
    external_id: str  # Platform-specific ID (tweet_id, message_id)
    source_platform: SourcePlatform
    source_account: str  # @username or channel name

    # Content
    text: str
    timestamp: datetime

    # Classification (determined by config, not LLM)
    source_type: SourceType

    # Raw data for debugging
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def urgency(self) -> str:
        """Urgency level derived from source type."""
        return "high" if self.source_type == SourceType.news else "normal"


# =============================================================================
# LLM Classification (Output Schema)
# =============================================================================


class EventType(str, Enum):
    """Type of market event."""

    macro = "macro"  # Fed, CPI, GDP
    earnings = "earnings"  # Company results
    geopolitical = "geopolitical"  # Wars, sanctions
    corporate = "corporate"  # M&A, CEO changes
    regulatory = "regulatory"  # SEC, antitrust
    crypto = "crypto"  # ETF, exchange news
    political = "political"  # Elections, policy
    other = "other"


# =============================================================================
# Numeric Extraction (Economic/Earnings Data)
# =============================================================================


class BeatMissStatus(str, Enum):
    """Whether metric beat, missed, or met expectations."""

    beat = "beat"
    miss = "miss"
    inline = "inline"
    unknown = "unknown"  # No estimate available


class MetricReading(BaseModel):
    """A single numeric metric from economic/earnings data."""

    metric_name: str = Field(description="Name of metric (e.g., 'CPI Y/Y', 'EPS', 'Revenue')")
    actual: float = Field(description="Actual reported value")
    estimate: float | None = Field(default=None, description="Consensus estimate (expected)")
    previous: float | None = Field(default=None, description="Previous period value")
    unit: str = Field(default="%", description="Unit of measurement (%, bps, $, B, M, K)")
    period: str | None = Field(default=None, description="Time period (Q4, December, 2024)")
    beat_miss: BeatMissStatus = Field(
        default=BeatMissStatus.unknown, description="Beat/miss/inline status"
    )
    surprise_magnitude: float | None = Field(
        default=None, description="Surprise = actual - estimate (same units)"
    )


class NumericExtraction(BaseModel):
    """All numeric data extracted from economic/earnings news."""

    metrics: list[MetricReading] = Field(
        default_factory=list, description="All metrics extracted from the message"
    )
    headline_metric: str | None = Field(
        default=None, description="The primary/headline metric name"
    )
    overall_beat_miss: BeatMissStatus = Field(
        default=BeatMissStatus.unknown, description="Overall assessment: beat/miss/inline"
    )

    @property
    def has_surprise(self) -> bool:
        """Check if any metric has a surprise."""
        return any(m.estimate is not None for m in self.metrics)

    @property
    def beats(self) -> list[MetricReading]:
        """Get metrics that beat estimates."""
        return [m for m in self.metrics if m.beat_miss == BeatMissStatus.beat]

    @property
    def misses(self) -> list[MetricReading]:
        """Get metrics that missed estimates."""
        return [m for m in self.metrics if m.beat_miss == BeatMissStatus.miss]


# =============================================================================
# Urgency Classification
# =============================================================================


class UrgencyLevel(str, Enum):
    """Message-level urgency for trading prioritization."""

    critical = "critical"  # Act immediately: rate decision, surprise announcement, breaking M&A
    high = "high"  # Act fast: scheduled data release, earnings with beat/miss
    normal = "normal"  # Can wait: analysis, commentary, minor news
    low = "low"  # Background: opinions, old news, noise


class ImpactLevel(str, Enum):
    """Predicted market impact level."""

    high = "high"
    medium = "medium"
    low = "low"


class Direction(str, Enum):
    """Market direction prediction."""

    bullish = "bullish"
    bearish = "bearish"
    neutral = "neutral"


class ResearchAnalysis(BaseModel):
    """Insights from web research about historical patterns."""

    historical_precedent: str = Field(description="What happened last time similar news occurred")
    similar_events: list[str] = Field(
        default_factory=list,
        description="List of similar historical events",
    )
    typical_market_reaction: str = Field(
        description="How markets typically react to this type of news"
    )
    key_insights: str = Field(description="Key takeaways from research relevant to trading")


# =============================================================================
# Stage 2A: Investment Analysis (NEW)
# =============================================================================


class TickerAnalysis(BaseModel):
    """Analysis of a single ticker affected by news."""

    ticker: str = Field(description="Stock ticker symbol (e.g., AAPL, TSLA)")
    company_name: str | None = Field(default=None, description="Company name if known")
    bull_thesis: str = Field(description="Bull case: why this news is positive for the stock")
    bear_thesis: str = Field(description="Bear case: why this news is negative for the stock")
    net_direction: Direction = Field(description="Overall expected direction")
    conviction: float = Field(ge=0.0, le=1.0, description="Conviction level 0.0 to 1.0")
    time_horizon: str = Field(
        description="Expected impact timeframe: intraday | days | weeks | months"
    )
    catalysts: list[str] = Field(default_factory=list, description="Key catalysts to watch")
    risk_factors: list[str] = Field(default_factory=list, description="Key risks to the thesis")


class SectorImplication(BaseModel):
    """Sector-level implication from news."""

    sector: str = Field(description="Sector name (e.g., semiconductors, financials)")
    direction: Direction = Field(description="Expected direction for the sector")
    reasoning: str = Field(description="Explanation of sector impact")
    affected_subsectors: list[str] = Field(
        default_factory=list,
        description="Specific subsectors affected (e.g., AI chips, payment processors)",
    )


class ResearchQuality(str, Enum):
    """Quality of available research data."""

    high = "high"  # Strong analyst coverage, historical data
    medium = "medium"  # Some data but gaps
    low = "low"  # Limited or no external data


class InvestmentAnalysis(BaseModel):
    """Stage 2A: Investment analysis output.

    Comprehensive analysis of investment implications from news,
    including ticker-level analysis and sector impacts.
    """

    # Ticker-level analysis
    ticker_analyses: list[TickerAnalysis] = Field(
        default_factory=list,
        description="Analysis of individual tickers affected",
    )

    # Sector implications
    sector_implications: list[SectorImplication] = Field(
        default_factory=list,
        description="Sector-level impacts",
    )

    # Historical context
    historical_precedent: str = Field(
        default="",
        description="What happened historically with similar news",
    )
    similar_events: list[str] = Field(
        default_factory=list,
        description="List of similar historical events",
    )
    typical_market_reaction: str = Field(
        default="",
        description="Typical market reaction pattern",
    )

    # Actionable output
    actionable_insights: list[str] = Field(
        default_factory=list,
        description="Specific actionable trading insights",
    )

    # Primary thesis (for Stage 2B)
    primary_thesis: str = Field(
        default="",
        description="Primary investment thesis from this news",
    )
    thesis_confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.5,
        description="Confidence in the primary thesis",
    )

    # Research quality indicator
    research_quality: ResearchQuality = Field(
        default=ResearchQuality.medium,
        description="Quality of available research data",
    )

    @property
    def has_tradable_tickers(self) -> bool:
        """Check if any tickers have high conviction."""
        return any(t.conviction >= 0.7 for t in self.ticker_analyses)

    @property
    def top_ticker(self) -> TickerAnalysis | None:
        """Get the ticker with highest conviction."""
        if not self.ticker_analyses:
            return None
        return max(self.ticker_analyses, key=lambda t: t.conviction)


# =============================================================================
# Stage 1: Lightweight Classification (NEW)
# =============================================================================


class LightClassification(BaseModel):
    """Stage 1 lightweight entity extractor output.

    Fast, tool-free extraction focusing on entities and keywords.
    NO judgment calls (tickers, sectors, impact, direction) - those move to Stage 2.
    """

    # News category (rule-based mostly, LLM fallback)
    news_category: NewsCategory = Field(
        default=NewsCategory.other,
        description="Category: breaking (unexpected), economic_calendar (scheduled), other",
    )

    # Event classification
    event_type: EventType
    summary: str = Field(description="One-sentence summary of the event")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in classification")

    # Entity extraction (NO judgment calls)
    primary_entity: str = Field(
        description="The PRIMARY entity affected (company, person, institution)"
    )
    all_entities: list[str] = Field(
        default_factory=list,
        description="ALL entities mentioned (people, companies, institutions)",
    )

    # Search keywords (entity-focused, for Stage 2)
    search_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords for web search (entity-focused)",
    )
    polymarket_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords to search Polymarket (entity-focused, no noise)",
    )

    # Numeric extraction (for economic_calendar and earnings)
    numeric_data: NumericExtraction | None = Field(
        default=None,
        description="Extracted numeric data (actual/estimate/previous) for economic releases and earnings",
    )

    # Urgency (hybrid: rules + LLM)
    urgency: UrgencyLevel = Field(
        default=UrgencyLevel.normal,
        description="Message-level urgency for trading prioritization",
    )
    urgency_reasoning: str = Field(default="", description="LLM reasoning for urgency level")


class BreakingClassification(BaseModel):
    """LLM classification output for a news/analysis item.

    The LLM extracts structured information from raw text.
    NOTE: Urgency is NOT determined by LLM - it comes from source configuration.
    """

    # News category (breaking / economic_calendar / other)
    news_category: NewsCategory = Field(
        default=NewsCategory.other,
        description="Category: breaking (unexpected), economic_calendar (scheduled), other",
    )

    # Event classification
    event_type: EventType
    summary: str = Field(description="One-sentence summary of the event")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in classification")

    # Impact assessment
    predicted_impact: ImpactLevel
    market_direction: Direction

    # Extracted entities (for Flow 2 watchlist)
    tickers: list[str] = Field(
        default_factory=list, description="Stock tickers mentioned (e.g., AAPL, TSLA)"
    )
    sectors: list[str] = Field(
        default_factory=list, description="Sectors mentioned (e.g., semiconductors, energy)"
    )

    # Prediction market search
    search_keywords: list[str] = Field(
        default_factory=list, description="Keywords to search Polymarket/Kalshi"
    )
    related_markets: list[str] = Field(
        default_factory=list,
        description="Specific market topics (e.g., 'Fed rate cut', 'Trump tariffs')",
    )

    # Research-based analysis from web searches
    research_analysis: ResearchAnalysis | None = Field(
        default=None,
        description="Insights from web research about historical patterns",
    )


# =============================================================================
# Market Opportunity
# =============================================================================


class MarketOpportunity(BaseModel):
    """A potential prediction market trading opportunity."""

    # Market info
    market_id: str
    platform: str  # "polymarket" or "kalshi"
    question: str
    slug: str | None = None

    # Current state
    yes_price: float = Field(ge=0.0, le=1.0)
    no_price: float = Field(ge=0.0, le=1.0)
    volume_24h: float | None = None

    # Opportunity
    suggested_direction: str  # "yes" or "no"
    reason: str
    news_summary: str | None = None

    # Metadata
    end_date: datetime | None = None


class OddsEvaluation(BaseModel):
    """Evaluation of prediction market odds vs analyst estimates."""

    market_id: str
    market_question: str
    verdict: str = Field(description="undervalued | overvalued | fair")
    current_odds: float = Field(ge=0.0, le=1.0)
    estimated_fair_odds: float = Field(ge=0.0, le=1.0)
    edge: float = Field(description="Estimated edge (fair - current)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this estimate")
    reasoning: str = Field(description="Explanation of the evaluation")
    analyst_estimates: list[str] = Field(
        default_factory=list,
        description="Analyst probability estimates found",
    )
    historical_pm_accuracy: str | None = Field(
        default=None,
        description="Polymarket historical accuracy on similar events",
    )
    recommended_side: str = Field(description="yes | no | skip")


# =============================================================================
# Stage 2: Smart Evaluator Output (NEW)
# =============================================================================


class MarketEvaluation(BaseModel):
    """Individual market evaluation with relevance check.

    Used by Stage 2 Smart Evaluator to filter false positive market matches
    and evaluate relevant markets for mispricing.
    """

    market_id: str
    market_question: str

    # Relevance check (filter false positives)
    is_relevant: bool = Field(description="Whether this market is actually relevant to the news")
    relevance_reasoning: str = Field(description="Explanation of why this market is/isn't relevant")

    # Evaluation (only meaningful if relevant)
    current_price: float = Field(ge=0.0, le=1.0)
    estimated_fair_price: float | None = Field(
        default=None,
        description="Fair price estimate (null if not relevant or uncertain)",
    )
    edge: float | None = Field(
        default=None,
        description="Estimated edge: fair - current (null if not relevant)",
    )
    verdict: str = Field(
        description="undervalued | overvalued | fair | skip (skip if not relevant)"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in evaluation (0 if not relevant)",
    )
    reasoning: str = Field(description="Full reasoning for the evaluation")
    recommended_side: str = Field(description="yes | no | skip")


class EvaluatorOutput(BaseModel):
    """Stage 2 Smart Evaluator output.

    Contains all market evaluations, research summary, and best opportunity.
    """

    # Summary stats
    markets_found: int = Field(description="Total markets found via search")
    markets_relevant: int = Field(description="Markets that passed relevance filter")

    # All evaluations (including non-relevant for transparency)
    evaluations: list[MarketEvaluation] = Field(default_factory=list)

    # Research performed
    research_summary: str = Field(
        default="",
        description="Summary of web research performed",
    )

    # Quick access to best opportunity
    has_tradable_edge: bool = Field(
        default=False,
        description="Whether any market has tradable edge",
    )
    best_opportunity: MarketEvaluation | None = Field(
        default=None,
        description="Best market by edge (if any has edge)",
    )

    @property
    def relevant_evaluations(self) -> list[MarketEvaluation]:
        """Get only the relevant market evaluations."""
        return [e for e in self.evaluations if e.is_relevant]

    @property
    def opportunities_with_edge(self) -> list[MarketEvaluation]:
        """Get evaluations with positive edge."""
        return [
            e for e in self.evaluations if e.is_relevant and e.edge is not None and e.edge > 0.05
        ]


# =============================================================================
# Stage 2: Smart Analysis (Consolidated Output)
# =============================================================================


class SmartAnalysis(BaseModel):
    """Stage 2 consolidated output - ALL analysis happens here after research.

    This replaces both InvestmentAnalysis and EvaluatorOutput as the unified
    Stage 2 output. It contains all informed judgments made with research context.
    """

    # Informed judgments (made with research context, NOT Stage 1)
    tickers: list[str] = Field(
        default_factory=list,
        description="Stock tickers affected (informed by research)",
    )
    sectors: list[str] = Field(
        default_factory=list,
        description="Sectors affected (informed by research)",
    )
    predicted_impact: ImpactLevel = Field(
        default=ImpactLevel.medium,
        description="Impact level (informed by research)",
    )
    market_direction: Direction = Field(
        default=Direction.neutral,
        description="Market direction (informed by research)",
    )

    # Primary thesis (from investment analysis)
    primary_thesis: str = Field(
        default="",
        description="Primary investment thesis from this news",
    )
    thesis_confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.5,
        description="Confidence in the primary thesis",
    )

    # Ticker-level analysis
    ticker_analyses: list[TickerAnalysis] = Field(
        default_factory=list,
        description="Analysis of individual tickers affected",
    )

    # Sector implications
    sector_implications: list[SectorImplication] = Field(
        default_factory=list,
        description="Sector-level impacts",
    )

    # Historical context
    historical_context: str = Field(
        default="",
        description="Precedent events with dates and quantified market reactions (e.g., 'Similar to March 2020 Fed cut; SPY rallied 5% over 3 days')",
    )
    typical_market_reaction: str = Field(
        default="",
        description="Typical reaction pattern: immediate move, reversal probability, sector rotation (e.g., 'Initial 1% spike usually fades 50% within 24h; financials lag')",
    )

    # Market evaluations (prediction markets)
    market_evaluations: list[MarketEvaluation] = Field(
        default_factory=list,
        description="Prediction market evaluations",
    )

    # Research quality indicator
    research_quality: ResearchQuality = Field(
        default=ResearchQuality.medium,
        description="Quality of available research data",
    )

    @property
    def has_tradable_edge(self) -> bool:
        """Check if any market has tradable edge."""
        return any(
            e.is_relevant and e.edge is not None and abs(e.edge) > 0.05 and e.confidence > 0.5
            for e in self.market_evaluations
        )

    @property
    def best_opportunity(self) -> MarketEvaluation | None:
        """Get the evaluation with the highest edge."""
        with_edge = [
            e
            for e in self.market_evaluations
            if e.is_relevant and e.edge is not None and abs(e.edge) > 0.05 and e.confidence > 0.5
        ]
        if not with_edge:
            return None
        return max(with_edge, key=lambda e: abs(e.edge or 0))

    @property
    def relevant_evaluations(self) -> list[MarketEvaluation]:
        """Get only the relevant market evaluations."""
        return [e for e in self.market_evaluations if e.is_relevant]

    @property
    def has_tradable_tickers(self) -> bool:
        """Check if any tickers have high conviction."""
        return any(t.conviction >= 0.7 for t in self.ticker_analyses)

    @property
    def top_ticker(self) -> TickerAnalysis | None:
        """Get the ticker with highest conviction."""
        if not self.ticker_analyses:
            return None
        return max(self.ticker_analyses, key=lambda t: t.conviction)


# =============================================================================
# Flow 1 Signal (Final Output)
# =============================================================================


class Flow1Signal(BaseModel):
    """Real-time signal emitted for each news/analysis item.

    This is the final output of Flow 1, written to JSONL.
    Uses the 2-stage architecture: LightClassification (Stage 1) + SmartAnalysis (Stage 2).
    """

    timestamp: datetime

    # Source info
    source_platform: SourcePlatform
    source_account: str
    source_type: SourceType
    raw_text: str
    external_id: str

    # News category (from classification or rule-based)
    news_category: NewsCategory = Field(default=NewsCategory.other)

    # Derived urgency
    @property
    def urgency(self) -> str:
        """Urgency derived from source_type, not LLM."""
        return "high" if self.source_type == SourceType.news else "normal"

    # Stage 1: Entity extraction (fast, no judgment calls)
    extraction: LightClassification

    # Stage 2: Smart analysis (all judgment calls happen here with research context)
    analysis: SmartAnalysis | None = Field(
        default=None,
        description="Stage 2 smart analysis output (tickers, sectors, impact, markets)",
    )

    # Legacy fields for backwards compatibility
    classification: BreakingClassification | LightClassification | None = Field(
        default=None,
        description="Deprecated: Use extraction instead",
    )
    opportunities: list[MarketOpportunity] = Field(default_factory=list)
    evaluations: list[OddsEvaluation | MarketEvaluation] = Field(default_factory=list)
    evaluator_output: EvaluatorOutput | None = Field(default=None)

    # Watchlist additions (for Flow 2) - now comes from analysis
    watchlist_tickers: list[str] = Field(default_factory=list)
    watchlist_sectors: list[str] = Field(default_factory=list)

    # Processing metadata
    is_duplicate: bool = False
    duplicate_of: str | None = None
    processing_time_ms: float | None = None
    skipped_evaluation: bool = Field(
        default=False,
        description="True if Stage 2 was skipped",
    )

    # Convenience accessors for analysis data
    @property
    def tickers(self) -> list[str]:
        """Get tickers from analysis (Stage 2, informed by research)."""
        return self.analysis.tickers if self.analysis else []

    @property
    def sectors(self) -> list[str]:
        """Get sectors from analysis (Stage 2, informed by research)."""
        return self.analysis.sectors if self.analysis else []

    @property
    def entities(self) -> list[str]:
        """Get all entities from extraction (Stage 1)."""
        return self.extraction.all_entities

    @property
    def market_evaluations(self) -> list[MarketEvaluation]:
        """Get market evaluations from analysis."""
        return self.analysis.market_evaluations if self.analysis else []

    @property
    def has_edge(self) -> bool:
        """Check if any market has tradable edge."""
        return self.analysis.has_tradable_edge if self.analysis else False

    @property
    def best_opportunity(self) -> MarketEvaluation | None:
        """Get the best opportunity by edge."""
        return self.analysis.best_opportunity if self.analysis else None
