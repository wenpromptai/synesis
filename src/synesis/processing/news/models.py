"""Processing models for Flow 1: Breaking News Intelligence.

Models for the two-stage news processing pipeline:
- Stage 1 (instant): LightClassification — impact score + matched tickers
- Stage 2 (LLM): SmartAnalysis — entities, thesis, ETF impact, Polymarket
- NewsSignal: final output combining both stages
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

    telegram = "telegram"
    twitter = "twitter"


# =============================================================================
# Unified Message
# =============================================================================


class UnifiedMessage(BaseModel):
    """Normalized message from any source (Telegram/Twitter)."""

    external_id: str
    source_platform: SourcePlatform
    source_account: str  # channel name

    text: str
    timestamp: datetime

    raw: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Topic Enums (used in Stage 2 LLM prompt for classification)
# =============================================================================


class PrimaryTopic(str, Enum):
    """High-level event type, asset-class, and sector classification."""

    # Monetary/Economy
    monetary_policy = "monetary_policy"
    economic_data = "economic_data"
    trade_policy = "trade_policy"
    fiscal_policy = "fiscal_policy"

    # Corporate
    earnings = "earnings"
    corporate_actions = "corporate_actions"
    regulatory = "regulatory"

    # Geopolitical
    geopolitics = "geopolitics"
    political = "political"

    # Asset Classes
    crypto = "crypto"
    commodities = "commodities"
    fixed_income = "fixed_income"
    fx = "fx"
    equities_market = "equities_market"

    # Sectors (GICS)
    energy = "energy"
    materials = "materials"
    industrials = "industrials"
    utilities = "utilities"
    healthcare = "healthcare"
    financials = "financials"
    consumer_discretionary = "consumer_discretionary"
    consumer_staples = "consumer_staples"
    information_technology = "information_technology"
    communication_services = "communication_services"
    real_estate = "real_estate"

    other = "other"


class SecondaryTopic(str, Enum):
    """Granular industry / subsector classification."""

    # Technology & Hardware
    semiconductors = "semiconductors"
    optics_photonics = "optics_photonics"
    cloud_computing = "cloud_computing"
    software_saas = "software_saas"
    cybersecurity = "cybersecurity"
    consumer_tech = "consumer_tech"
    ev_autonomous = "ev_autonomous"

    # Healthcare & Life Sciences
    biotech = "biotech"
    pharma = "pharma"
    medical_devices = "medical_devices"
    health_insurance = "health_insurance"

    # Energy
    oil_gas = "oil_gas"
    renewables = "renewables"
    nuclear = "nuclear"
    utilities = "utilities"

    # Financials
    banks_lending = "banks_lending"
    insurance = "insurance"
    asset_management = "asset_management"
    private_equity = "private_equity"
    fintech_payments = "fintech_payments"

    # Industrials
    defense_weapons = "defense_weapons"
    aerospace_space = "aerospace_space"
    automation_robotics = "automation_robotics"
    chemicals_materials = "chemicals_materials"

    # Consumer
    retail_ecommerce = "retail_ecommerce"
    food_beverage = "food_beverage"
    media_entertainment = "media_entertainment"
    gaming = "gaming"
    luxury_fashion = "luxury_fashion"
    travel_hospitality = "travel_hospitality"

    # Materials & Resources
    metals_mining = "metals_mining"
    agriculture = "agriculture"

    # Transport & Logistics
    shipping_maritime = "shipping_maritime"
    automotive = "automotive"
    logistics_freight = "logistics_freight"

    # Real Estate
    residential_housing = "residential_housing"
    commercial_real_estate = "commercial_real_estate"

    # Telecom & Social
    telecom_5g = "telecom_5g"
    social_media_adtech = "social_media_adtech"


# =============================================================================
# Urgency
# =============================================================================


class UrgencyLevel(str, Enum):
    """Message-level urgency for trading prioritization."""

    critical = "critical"
    high = "high"
    normal = "normal"
    low = "low"


# =============================================================================
# Stage 1: Instant classification (NO LLM)
# =============================================================================


class LightClassification(BaseModel):
    """Stage 1 output — instant, no LLM.

    Produced by impact scoring + ticker matching.
    """

    matched_tickers: list[str] = Field(
        default_factory=list,
        description="Tickers matched from text (e.g. ['NVDA', 'MRVL', '~OPENAI'])",
    )

    impact_score: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Numeric impact score (0-100) from additive signal scoring",
    )
    impact_reasons: list[str] = Field(
        default_factory=list,
        description="Score component breakdown (e.g. 'wire_prefix:+15', 'mna:+15')",
    )

    urgency: UrgencyLevel = Field(
        default=UrgencyLevel.normal,
        description="Derived from impact_score thresholds or fast-track rules: >=55 critical, 32-54 high, 15-31 normal, <15 low",
    )


# =============================================================================
# Market Evaluation (Polymarket)
# =============================================================================


class MarketEvaluation(BaseModel):
    """Individual Polymarket evaluation with relevance check."""

    market_id: str
    market_question: str

    is_relevant: bool = Field(description="Whether this market is relevant to the news")
    relevance_reasoning: str = Field(description="Why this market is/isn't relevant")

    current_price: float = Field(ge=0.0, le=1.0)
    estimated_fair_price: float | None = Field(default=None)
    edge: float | None = Field(default=None, description="fair - current")
    verdict: str = Field(description="undervalued | overvalued | fair | skip")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Full reasoning for the evaluation")
    recommended_side: str = Field(description="yes | no | skip")


# =============================================================================
# Stage 2: Smart Analysis (LLM output)
# =============================================================================


class ETFImpact(BaseModel):
    """Sentiment assessment for a single ETF proxy."""

    ticker: str = Field(description="ETF ticker (e.g. SPY, GLD, XLK)")
    sentiment_score: float = Field(
        ge=-1.0,
        le=1.0,
        default=0.0,
        description="Sentiment score: -1.0 (max bearish) to 1.0 (max bullish)",
    )
    reason: str = Field(default="", description="One sentence explaining the impact (≤100 chars)")


class SmartAnalysis(BaseModel):
    """Stage 2 output — LLM analysis with research context.

    Entities, thesis, per-ETF sentiment, historical context, Polymarket.
    """

    all_entities: list[str] = Field(
        default_factory=list,
        description="All companies, people, institutions, and countries directly involved",
    )

    primary_thesis: str = Field(
        default="",
        description="Primary investment thesis (≤150 chars)",
    )

    macro_impact: list[ETFImpact] = Field(
        default_factory=list,
        description="Macro ETF proxies impacted (SPY, GLD, TLT, USO, VIXY, UUP, EEM)",
    )
    sector_impact: list[ETFImpact] = Field(
        default_factory=list,
        description="Sector ETF proxies impacted (XLK, XLF, XLE, XLV, etc.)",
    )

    historical_context: str = Field(
        default="",
        description="Precedent events with dates and quantified market reactions",
    )
    typical_market_reaction: str = Field(
        default="",
        description="Typical reaction pattern: initial move, reversal probability, sector rotation",
    )

    market_evaluations: list[MarketEvaluation] = Field(
        default_factory=list,
        description="Prediction market evaluations",
    )

    @property
    def has_tradable_edge(self) -> bool:
        """Check if any Polymarket has tradable edge."""
        return any(
            e.is_relevant and e.edge is not None and abs(e.edge) > 0.05 and e.confidence > 0.5
            for e in self.market_evaluations
        )

    @property
    def best_opportunity(self) -> MarketEvaluation | None:
        """Get the Polymarket evaluation with the highest edge."""
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


# =============================================================================
# News Signal (Final Output)
# =============================================================================


class NewsSignal(BaseModel):
    """Final output of Flow 1. Stored in DB and published to Redis."""

    timestamp: datetime

    source_platform: SourcePlatform
    source_account: str
    raw_text: str
    external_id: str

    extraction: LightClassification

    analysis: SmartAnalysis | None = Field(
        default=None,
        description="Stage 2 LLM analysis (None if Stage 2 was skipped)",
    )

    processing_time_ms: float = 0.0

    @property
    def matched_tickers(self) -> list[str]:
        return self.extraction.matched_tickers

    @property
    def market_evaluations(self) -> list[MarketEvaluation]:
        return self.analysis.market_evaluations if self.analysis else []

    @property
    def has_edge(self) -> bool:
        return self.analysis.has_tradable_edge if self.analysis else False

    @property
    def best_opportunity(self) -> MarketEvaluation | None:
        return self.analysis.best_opportunity if self.analysis else None
