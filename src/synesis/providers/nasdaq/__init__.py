"""NASDAQ provider for earnings calendar data.

Uses the free NASDAQ API (api.nasdaq.com) — no API key required.
"""

from synesis.providers.nasdaq.client import NasdaqClient
from synesis.providers.nasdaq.models import EarningsEvent

__all__ = [
    "NasdaqClient",
    "EarningsEvent",
]
