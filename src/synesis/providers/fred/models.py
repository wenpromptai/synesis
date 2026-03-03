"""Pydantic models for FRED economic data."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, computed_field


class FREDSeries(BaseModel):
    """Metadata for a FRED economic data series."""

    id: str
    title: str
    frequency: str = ""
    units: str = ""
    seasonal_adjustment: str = ""
    last_updated: str = ""
    popularity: int = 0
    notes: str = ""
    observation_start: date | None = None
    observation_end: date | None = None


class FREDObservation(BaseModel):
    """A single data point in a FRED time series."""

    date: date
    value: float | None = None


class FREDObservations(BaseModel):
    """Collection of observations for a FRED series."""

    series_id: str
    title: str = ""
    units: str = ""
    frequency: str = ""
    observations: list[FREDObservation] = []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        return len(self.observations)


class FREDRelease(BaseModel):
    """A FRED data release (e.g., 'Consumer Price Index')."""

    id: int
    name: str
    press_release: bool = False
    link: str = ""


class FREDReleaseDate(BaseModel):
    """A scheduled date for a FRED release."""

    release_id: int
    release_name: str = ""
    date: date
