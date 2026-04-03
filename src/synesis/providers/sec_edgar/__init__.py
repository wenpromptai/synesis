"""SEC EDGAR provider for SEC filings and insider transactions.

Uses the free SEC EDGAR API (data.sec.gov) — no API key required.
"""

from synesis.providers.sec_edgar.client import SECEdgarClient
from synesis.providers.sec_edgar.models import (
    ActivistFiling,
    CompanyFacts,
    CompanyInfo,
    DerivativeTransaction,
    EarningsRelease,
    EffectivenessNotice,
    EventFiling8K,
    Filing13F,
    FilingFeedEntry,
    ForeignFiling,
    Form144Filing,
    Holding13F,
    IPOFiling,
    InsiderTransaction,
    LateFilingAlert,
    ProxyFiling,
    SECFiling,
    TenderOfferFiling,
    XBRLFact,
    XBRLFrame,
    XBRLFrameEntry,
)

__all__ = [
    "ActivistFiling",
    "CompanyFacts",
    "CompanyInfo",
    "DerivativeTransaction",
    "EarningsRelease",
    "EffectivenessNotice",
    "EventFiling8K",
    "Filing13F",
    "FilingFeedEntry",
    "ForeignFiling",
    "Form144Filing",
    "Holding13F",
    "IPOFiling",
    "InsiderTransaction",
    "LateFilingAlert",
    "ProxyFiling",
    "SECEdgarClient",
    "SECFiling",
    "TenderOfferFiling",
    "XBRLFact",
    "XBRLFrame",
    "XBRLFrameEntry",
]
