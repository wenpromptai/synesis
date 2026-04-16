"""CFTC Commitment of Traders (COT) data provider.

Uses the CFTC Socrata API — free, no API key required.
Data released every Friday at 3:30 PM ET, reflecting prior Tuesday's positions.
"""

from synesis.providers.cftc.client import CFTCClient
from synesis.providers.cftc.models import COTPositioning, COTReport

__all__ = [
    "CFTCClient",
    "COTPositioning",
    "COTReport",
]
