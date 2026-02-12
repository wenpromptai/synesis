"""Unified market models for Flow 3: Market Intelligence.

These models provide a platform-agnostic representation of prediction
markets from Polymarket and Kalshi, along with alert types for odds
movements and insider activity.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class UnifiedMarket(BaseModel):
    """Platform-agnostic market representation."""

    platform: Literal["polymarket", "kalshi"]
    external_id: str = Field(min_length=1)
    condition_id: str | None = None  # Polymarket
    ticker: str | None = None  # Kalshi
    question: str
    yes_price: float = Field(ge=0.0, le=1.0)
    no_price: float = Field(ge=0.0, le=1.0)
    volume_24h: float = Field(default=0.0, ge=0.0)
    volume_total: float = Field(default=0.0, ge=0.0)  # All-time volume
    open_interest: float | None = None
    liquidity: float | None = None
    end_date: datetime | None = None
    is_active: bool = True
    url: str = ""
    category: str | None = None
    description: str | None = None
    outcome_label: str | None = None
    yes_outcome: str | None = None  # e.g. "Up", "Over" for non-Yes/No markets

    @model_validator(mode="after")
    def check_platform_identifier(self) -> UnifiedMarket:
        if self.platform == "polymarket" and not self.condition_id:
            raise ValueError("Polymarket markets require condition_id")
        if self.platform == "kalshi" and not self.ticker:
            raise ValueError("Kalshi markets require ticker")
        return self


class OddsMovement(BaseModel):
    """Significant price/odds change detection."""

    market: UnifiedMarket
    price_change_1h: float
    price_change_6h: float | None = None
    direction: Literal["up", "down"]


class InsiderAlert(BaseModel):
    """Insider/whale wallet activity detection."""

    market: UnifiedMarket
    wallet_address: str = Field(min_length=1)
    insider_score: float = Field(ge=0.0, le=1.0)
    trade_direction: Literal["yes", "no", "unknown"]
    trade_size: float = Field(ge=0.0)


class VolumeSpike(BaseModel):
    """Volume spike detection — current hour vs previous hour."""

    market: UnifiedMarket
    volume_current: float = Field(ge=0.0)
    volume_previous: float = Field(ge=0.0)
    pct_change: float


class ScanResult(BaseModel):
    """Result of a single market scan cycle."""

    timestamp: datetime
    trending_markets: list[UnifiedMarket] = Field(default_factory=list)
    expiring_markets: list[UnifiedMarket] = Field(default_factory=list)
    odds_movements: list[OddsMovement] = Field(default_factory=list)
    volume_spikes: list[VolumeSpike] = Field(default_factory=list)
    total_markets_scanned: int = 0
