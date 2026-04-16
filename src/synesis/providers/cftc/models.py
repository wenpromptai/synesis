"""Models for CFTC Commitment of Traders data."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class COTPositioning(BaseModel):
    """Net positioning for a single trader category on a report date."""

    report_date: date
    long_contracts: int = 0
    short_contracts: int = 0
    spread_contracts: int = 0
    net_contracts: int = 0
    change_long: int = 0
    change_short: int = 0
    pct_of_oi_long: float = 0.0
    pct_of_oi_short: float = 0.0


class COTReport(BaseModel):
    """COT report for a single futures contract with positioning + analytics."""

    contract_name: str
    contract_code: str
    ticker: str  # Our friendly ticker (ES, NQ, ZN, GC, CL, DX)
    report_date: date
    open_interest: int = 0

    # Positioning by trader category
    leveraged_funds: COTPositioning
    asset_managers: COTPositioning
    dealers: COTPositioning

    # Analytics (computed)
    lev_funds_net_pctl: float | None = Field(
        default=None, description="Leveraged funds net position percentile (0-100, 52-week)"
    )
    lev_funds_net_zscore: float | None = Field(
        default=None, description="Leveraged funds net position z-score (52-week)"
    )
