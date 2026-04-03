"""Pydantic models for SEC EDGAR data."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, computed_field

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

# Human-readable labels for Form 4 transaction codes
TRANSACTION_CODE_LABELS: dict[str, str] = {
    "P": "Open Market Purchase",
    "S": "Open Market Sale",
    "A": "RSU/Award Grant",
    "F": "Tax Withholding Sale",
    "M": "Option Exercise",
    "G": "Gift",
    "J": "Other Acquisition/Disposition",
    "X": "Option Expiration",
    "C": "Conversion",
    "W": "Warrant Exercise",
    "D": "Sale Back to Issuer",
    "I": "Discretionary Transaction",
}

# Transaction codes that represent real open-market conviction signals
OPEN_MARKET_CODES: frozenset[str] = frozenset({"P", "S"})

# 8-K item code descriptions
ITEM_8K_DESCRIPTIONS: dict[str, str] = {
    "1.01": "Entry into Material Definitive Agreement",
    "1.02": "Termination of Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "2.01": "Completion of Acquisition or Disposition",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of Direct Financial Obligation",
    "2.04": "Triggering Events for Off-Balance Sheet Arrangements",
    "2.05": "Costs for Exit/Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Delisting Notice",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Change in Accountant",
    "4.02": "Non-Reliance on Prior Financial Statements",
    "5.01": "Change in Control",
    "5.02": "Departure/Appointment of Officers",
    "5.03": "Amendments to Articles/Bylaws",
    "5.05": "Amendments to Code of Ethics",
    "5.07": "Shareholder Vote Results",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}


# ─────────────────────────────────────────────────────────────
# Filing Models
# ─────────────────────────────────────────────────────────────


class CompanyInfo(BaseModel):
    """Company metadata from SEC EDGAR submissions."""

    ticker: str
    cik: str
    name: str
    entity_type: str  # "operating", "individual", etc.
    sic: str  # SIC industry code
    sic_description: str
    category: str  # "Large accelerated filer", "Non-accelerated filer", etc.
    fiscal_year_end: str  # Month-day, e.g., "0926" for Sept 26
    state_of_incorporation: str
    exchanges: list[str]  # ["Nasdaq", "NYSE"]
    tickers: list[str]  # All ticker symbols for the entity
    ein: str  # Employer Identification Number
    former_names: list[dict[str, str]]  # [{"name": "...", "from": "...", "to": "..."}]
    phone: str
    website: str


class SECFiling(BaseModel):
    """A single SEC filing."""

    ticker: str
    form: str  # "8-K", "10-K", "10-Q", "4", etc.
    filed_date: date
    accepted_datetime: datetime
    accession_number: str
    primary_document: str
    items: str  # e.g., "2.02" for earnings results
    url: str  # Full URL to filing document
    report_date: date | None = None  # Period end date (used by 13F, 10-K, 10-Q)


class EarningsRelease(BaseModel):
    """An 8-K earnings press release (Item 2.02) with optional full content."""

    ticker: str
    filed_date: date
    accepted_datetime: datetime
    accession_number: str
    url: str
    items: str
    content: str | None = None  # Markdown press release (from Crawl4AI)


class EventFiling8K(BaseModel):
    """An 8-K filing with parsed item codes and human-readable descriptions."""

    ticker: str
    filed_date: date
    accepted_datetime: datetime
    accession_number: str
    url: str
    items: list[str]  # e.g., ["2.02", "9.01"]
    item_descriptions: list[str]  # e.g., ["Results of Operations", "Financial Statements"]
    content: str | None = None


# ─────────────────────────────────────────────────────────────
# Insider Models
# ─────────────────────────────────────────────────────────────


class InsiderTransaction(BaseModel):
    """A single insider transaction parsed from Form 3/4/5 XML."""

    ticker: str
    owner_name: str
    owner_relationship: str  # "Director", "Officer", "10% Owner"
    owner_country: str = ""  # Country of reporting person (March 2026+ filings)
    transaction_date: date
    transaction_code: (
        str  # P=open-market buy, S=open-market sell, A=award, F=tax withholding, M=exercise
    )
    shares: float
    price_per_share: float | None
    shares_after: float
    acquired_or_disposed: str  # "A" or "D"
    filing_date: date
    filing_url: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def transaction_type_label(self) -> str:
        """Human-readable label for the transaction code."""
        return TRANSACTION_CODE_LABELS.get(
            self.transaction_code, f"Unknown ({self.transaction_code})"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_open_market(self) -> bool:
        """True if this is a discretionary open-market buy (P) or sell (S)."""
        return self.transaction_code in OPEN_MARKET_CODES


class DerivativeTransaction(BaseModel):
    """A derivative insider transaction (options, warrants) from Form 4 XML."""

    ticker: str
    owner_name: str
    owner_relationship: str
    owner_country: str = ""  # Country of reporting person (March 2026+ filings)
    transaction_date: date
    transaction_code: str
    security_title: str  # e.g., "Stock Option (Right to Buy)"
    exercise_price: float | None
    underlying_shares: float
    expiration_date: date | None
    acquired_or_disposed: str
    filing_date: date
    filing_url: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def transaction_type_label(self) -> str:
        """Human-readable label for the transaction code."""
        return TRANSACTION_CODE_LABELS.get(
            self.transaction_code, f"Unknown ({self.transaction_code})"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_open_market(self) -> bool:
        """True if this is a discretionary open-market transaction."""
        return self.transaction_code in OPEN_MARKET_CODES


# ─────────────────────────────────────────────────────────────
# XBRL Models
# ─────────────────────────────────────────────────────────────


class XBRLFact(BaseModel):
    """A single XBRL financial data point."""

    concept: str  # e.g., "NetIncomeLoss"
    label: str  # Human-readable label
    unit: str  # "USD", "USD/shares", "pure"
    period_end: date
    value: float
    form: str  # "10-K", "10-Q"
    frame: str  # "CY2024Q3"
    filed: date


class CompanyFacts(BaseModel):
    """All XBRL facts for a company from the companyfacts endpoint."""

    ticker: str
    cik: str
    entity_name: str
    facts: list[XBRLFact]
    concept_count: int


class XBRLFrameEntry(BaseModel):
    """One company's value in a cross-company XBRL frame."""

    cik: int
    entity_name: str
    value: float
    accession_number: str
    end: str  # Period end date (YYYY-MM-DD)


class XBRLFrame(BaseModel):
    """Cross-company screening result from the XBRL frames endpoint."""

    taxonomy: str  # "us-gaap"
    tag: str  # "NetIncomeLoss"
    unit: str  # "USD"
    period: str  # "CY2024Q3I"
    entries: list[XBRLFrameEntry]
    entry_count: int


# ─────────────────────────────────────────────────────────────
# 13F Models
# ─────────────────────────────────────────────────────────────


class Holding13F(BaseModel):
    """A single holding from a 13F-HR InfoTable."""

    name_of_issuer: str
    title_of_class: str
    cusip: str
    value_thousands: int
    shares: int
    investment_discretion: str


class Filing13F(BaseModel):
    """Parsed 13F-HR filing with holdings."""

    cik: str
    fund_name: str
    filed_date: date
    report_date: date
    accession_number: str
    url: str
    holdings: list[Holding13F]
    total_value_thousands: int


# ─────────────────────────────────────────────────────────────
# Ownership / Governance Models
# ─────────────────────────────────────────────────────────────


class ActivistFiling(BaseModel):
    """Schedule 13D/13G filing — activist/passive investor >5% ownership.

    Filer identity is in the filing document itself (accessible via URL),
    not in the submissions metadata.
    """

    ticker: str
    form_type: str  # "SC 13D", "SC 13G", etc.
    filed_date: date
    accession_number: str
    url: str
    is_amendment: bool


class Form144Filing(BaseModel):
    """Form 144 pre-sale notice for restricted stock."""

    ticker: str
    filed_date: date
    accession_number: str
    url: str


class LateFilingAlert(BaseModel):
    """NT 10-K / NT 10-Q late filing notification."""

    ticker: str
    form_type: str  # "NT 10-K" or "NT 10-Q"
    filed_date: date
    accession_number: str
    url: str
    original_form: str  # "10-K" or "10-Q"


class IPOFiling(BaseModel):
    """S-1 / S-1/A IPO registration statement."""

    entity_name: str
    form_type: str  # "S-1" or "S-1/A"
    filed_date: str  # String because EFTS returns varying formats
    accession_number: str
    url: str
    is_amendment: bool


class ProxyFiling(BaseModel):
    """DEF 14A proxy statement."""

    ticker: str
    filed_date: date
    accession_number: str
    url: str
    content: str | None = None


class TenderOfferFiling(BaseModel):
    """Tender offer filing (SC TO-T, SC TO-I, SC 14D-9)."""

    ticker: str
    form_type: str
    filed_date: date
    accession_number: str
    url: str
    is_amendment: bool


class ForeignFiling(BaseModel):
    """Foreign issuer filing (20-F, 6-K, 40-F)."""

    ticker: str
    form_type: str  # "20-F", "6-K", "40-F"
    filed_date: date
    accession_number: str
    url: str
    report_date: date | None = None


class EffectivenessNotice(BaseModel):
    """EFFECT notice — S-1 registration declared effective (IPO ready to price)."""

    entity_name: str
    form_type: str
    filed_date: str  # String — from EFTS search
    accession_number: str
    url: str


class FilingFeedEntry(BaseModel):
    """A single entry from SEC's RSS/Atom filing feed."""

    title: str
    link: str
    summary: str
    updated: str
    category: str  # Form type
