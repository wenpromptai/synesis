"""Data models for Flow 3: Prediction Market Intelligence.

Signal and opportunity models for the mkt_intel flow.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from synesis.markets.models import (
    InsiderAlert,
    OddsMovement,
    UnifiedMarket,
)


class MarketIntelOpportunity(BaseModel):
    """A scored trading opportunity from market intelligence."""

    market: UnifiedMarket
    suggested_direction: Literal["yes", "no"]
    confidence: float = Field(ge=0.0, le=1.0)
    triggers: list[str] = Field(default_factory=list)
    insider_wallets_active: list[str] = Field(default_factory=list)
    hours_to_expiration: float | None = None
    reasoning: str = ""


class MarketIntelSignal(BaseModel):
    """Market intelligence signal emitted every 15 minutes.

    Contains aggregated data from REST scans + WebSocket streams.
    """

    # Timing
    timestamp: datetime
    signal_period: str = "15min"

    # Scan stats
    total_markets_scanned: int = 0

    # Market lists
    trending_markets: list[UnifiedMarket] = Field(default_factory=list)
    expiring_soon: list[UnifiedMarket] = Field(default_factory=list)

    # Alerts
    insider_activity: list[InsiderAlert] = Field(default_factory=list)
    odds_movements: list[OddsMovement] = Field(default_factory=list)

    # Opportunities
    opportunities: list[MarketIntelOpportunity] = Field(default_factory=list)

    # Aggregate metrics
    market_uncertainty_index: float = Field(default=0.0, ge=0.0, le=1.0)
    informed_activity_level: float = Field(default=0.0, ge=0.0, le=1.0)

    # WebSocket health
    ws_connected: bool = False
