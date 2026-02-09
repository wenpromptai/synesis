"""Flow 3: Prediction Market Intelligence.

This module contains:
- Market scanner for volume spikes, odds movements, expiring markets
- Wallet tracker for insider/whale activity (Polymarket)
- Signal generation and opportunity scoring
- WebSocket integration for real-time data
"""

from synesis.processing.mkt_intel.models import (
    MarketIntelOpportunity,
    MarketIntelSignal,
)
from synesis.processing.mkt_intel.processor import MarketIntelProcessor
from synesis.processing.mkt_intel.scanner import MarketScanner
from synesis.processing.mkt_intel.wallets import WalletTracker

__all__ = [
    "MarketIntelOpportunity",
    "MarketIntelSignal",
    "MarketIntelProcessor",
    "MarketScanner",
    "WalletTracker",
]
