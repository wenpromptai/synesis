"""SEC EDGAR provider for SEC filings and insider transactions.

Uses the free SEC EDGAR API (data.sec.gov) — no API key required.
"""

from synesis.providers.sec_edgar.client import SECEdgarClient
from synesis.providers.sec_edgar.models import EarningsRelease, InsiderTransaction, SECFiling

__all__ = [
    "SECEdgarClient",
    "EarningsRelease",
    "SECFiling",
    "InsiderTransaction",
]
