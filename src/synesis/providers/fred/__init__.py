"""FRED provider for Federal Reserve Economic Data.

API docs: https://fred.stlouisfed.org/docs/api/fred/
Free API key required — register at https://fredaccount.stlouisfed.org
"""

from synesis.providers.fred.client import FREDClient
from synesis.providers.fred.models import (
    FREDObservation,
    FREDObservations,
    FREDRelease,
    FREDReleaseDate,
    FREDSeries,
)

__all__ = [
    "FREDClient",
    "FREDObservation",
    "FREDObservations",
    "FREDRelease",
    "FREDReleaseDate",
    "FREDSeries",
]
