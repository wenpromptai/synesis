"""SEC EDGAR ownership mixin — 13D/13G, Form 144, NT filings, S-1, DEF 14A, tender offers, foreign filings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from synesis.core.logging import get_logger
from synesis.providers.sec_edgar.models import (
    ActivistFiling,
    EffectivenessNotice,
    ForeignFiling,
    Form144Filing,
    IPOFiling,
    LateFilingAlert,
    ProxyFiling,
    TenderOfferFiling,
)

if TYPE_CHECKING:
    from synesis.providers.crawler.crawl4ai import Crawl4AICrawlerProvider

logger = get_logger(__name__)

# Form types for 13D/13G filings (includes amendments)
_ACTIVIST_FORM_TYPES = ["SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"]

# Form types for late filing notifications
_LATE_FILING_FORM_TYPES = ["NT 10-K", "NT 10-Q"]

# Form types for IPO registrations
_IPO_FORM_TYPES = ["S-1", "S-1/A"]


class OwnershipMixin:
    """Ownership and governance filings — activist detection, pre-sale notices, etc."""

    # ─────────────────────────────────────────────────────────────
    # Schedule 13D/13G (Activist Investors)
    # ─────────────────────────────────────────────────────────────

    async def get_activist_filings(
        self,
        ticker: str,
        limit: int = 5,
    ) -> list[ActivistFiling]:
        """Get Schedule 13D/13G filings for activist investor detection.

        These filings are required when an investor acquires >5% beneficial
        ownership of a company. 13D indicates activist intent; 13G is passive.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum filings to return

        Returns:
            List of ActivistFiling objects, newest first
        """
        filings = await self.get_filings(  # type: ignore[attr-defined]
            ticker, form_types=_ACTIVIST_FORM_TYPES, limit=limit
        )

        return [
            ActivistFiling(
                ticker=f.ticker,
                form_type=f.form,
                filed_date=f.filed_date,
                accession_number=f.accession_number,
                url=f.url,
                is_amendment="/A" in f.form,
            )
            for f in filings
        ]

    # ─────────────────────────────────────────────────────────────
    # Form 144 (Pre-Sale Notices)
    # ─────────────────────────────────────────────────────────────

    async def get_form144_filings(
        self,
        ticker: str,
        limit: int = 10,
    ) -> list[Form144Filing]:
        """Get Form 144 pre-sale notices (restricted stock sale intent).

        Insiders must file Form 144 before selling restricted stock.
        This provides an early warning signal of upcoming insider sales.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum filings to return

        Returns:
            List of Form144Filing objects, newest first
        """
        filings = await self.get_filings(  # type: ignore[attr-defined]
            ticker, form_types=["144"], limit=limit
        )

        return [
            Form144Filing(
                ticker=f.ticker,
                filed_date=f.filed_date,
                accession_number=f.accession_number,
                url=f.url,
            )
            for f in filings
        ]

    # ─────────────────────────────────────────────────────────────
    # NT 10-K / NT 10-Q (Late Filing Alerts)
    # ─────────────────────────────────────────────────────────────

    async def get_late_filing_alerts(
        self,
        ticker: str,
    ) -> list[LateFilingAlert]:
        """Get late filing notifications (NT 10-K / NT 10-Q).

        These are often red flags indicating accounting issues, restatements,
        or other problems that prevent timely filing.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of LateFilingAlert objects, newest first
        """
        filings = await self.get_filings(  # type: ignore[attr-defined]
            ticker, form_types=_LATE_FILING_FORM_TYPES, limit=20
        )

        return [
            LateFilingAlert(
                ticker=f.ticker,
                form_type=f.form,
                filed_date=f.filed_date,
                accession_number=f.accession_number,
                url=f.url,
                original_form="10-K" if "10-K" in f.form else "10-Q",
            )
            for f in filings
        ]

    # ─────────────────────────────────────────────────────────────
    # S-1 (IPO Pipeline)
    # ─────────────────────────────────────────────────────────────

    async def get_ipo_filings(
        self,
        query: str | None = None,
        limit: int = 10,
    ) -> list[IPOFiling]:
        """Get S-1 IPO registration filings via full-text search.

        Pre-IPO companies may not have tickers, so this uses EFTS search.

        Args:
            query: Search query to filter (e.g., company name or sector keyword).
                   If None, returns recent S-1 filings.
            limit: Maximum filings to return

        Returns:
            List of IPOFiling objects, newest first
        """
        search_result = await self.search_filings(  # type: ignore[attr-defined]
            query=query or "*",
            forms=_IPO_FORM_TYPES,
            limit=limit,
        )

        ipo_filings: list[IPOFiling] = []
        for r in search_result["results"]:
            form = r.get("form") or "S-1"
            ipo_filings.append(
                IPOFiling(
                    entity_name=r.get("entity") or "",
                    form_type=form,
                    filed_date=r.get("filed") or "",
                    accession_number="",
                    url=r.get("url") or "",
                    is_amendment="/A" in form,
                )
            )
        return ipo_filings

    # ─────────────────────────────────────────────────────────────
    # DEF 14A (Proxy Statements)
    # ─────────────────────────────────────────────────────────────

    async def get_proxy_filings(
        self,
        ticker: str,
        limit: int = 5,
        crawler: Crawl4AICrawlerProvider | None = None,
    ) -> list[ProxyFiling]:
        """Get DEF 14A proxy statements.

        Proxy statements contain executive compensation, shareholder proposals,
        and board composition information.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum filings to return
            crawler: Optional Crawl4AI crawler for content extraction

        Returns:
            List of ProxyFiling objects, newest first
        """
        filings = await self.get_filings(  # type: ignore[attr-defined]
            ticker, form_types=["DEF 14A"], limit=limit
        )

        results: list[ProxyFiling] = []
        for f in filings:
            content: str | None = None
            if crawler and f.url:
                content = await self.get_filing_content(f.url, crawler=crawler)  # type: ignore[attr-defined]

            results.append(
                ProxyFiling(
                    ticker=f.ticker,
                    filed_date=f.filed_date,
                    accession_number=f.accession_number,
                    url=f.url,
                    content=content,
                )
            )

        return results

    # ─────────────────────────────────────────────────────────────
    # Tender Offers (SC TO-T, SC TO-I, SC 14D-9)
    # ─────────────────────────────────────────────────────────────

    async def get_tender_offers(
        self,
        ticker: str,
        limit: int = 5,
    ) -> list[TenderOfferFiling]:
        """Get tender offer filings for M&A detection.

        SC TO-T = third-party tender offer, SC TO-I = issuer tender offer (buyback),
        SC 14D-9 = target's recommendation on a tender offer.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum filings to return

        Returns:
            List of TenderOfferFiling objects, newest first
        """
        filings = await self.get_filings(  # type: ignore[attr-defined]
            ticker,
            form_types=["SC TO-T", "SC TO-T/A", "SC TO-I", "SC TO-I/A", "SC 14D9", "SC 14D9/A"],
            limit=limit,
        )

        return [
            TenderOfferFiling(
                ticker=f.ticker,
                form_type=f.form,
                filed_date=f.filed_date,
                accession_number=f.accession_number,
                url=f.url,
                is_amendment="/A" in f.form,
            )
            for f in filings
        ]

    # ─────────────────────────────────────────────────────────────
    # Foreign Issuer Filings (20-F, 6-K, 40-F)
    # ─────────────────────────────────────────────────────────────

    async def get_foreign_filings(
        self,
        ticker: str,
        limit: int = 10,
    ) -> list[ForeignFiling]:
        """Get foreign issuer filings (ADRs, international companies).

        20-F = annual report (equivalent to 10-K), 6-K = current report (equivalent to 8-K),
        40-F = Canadian issuer annual report.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum filings to return

        Returns:
            List of ForeignFiling objects, newest first
        """
        filings = await self.get_filings(  # type: ignore[attr-defined]
            ticker, form_types=["20-F", "20-F/A", "6-K", "40-F", "40-F/A"], limit=limit
        )

        return [
            ForeignFiling(
                ticker=f.ticker,
                form_type=f.form,
                filed_date=f.filed_date,
                accession_number=f.accession_number,
                url=f.url,
                report_date=f.report_date,
            )
            for f in filings
        ]

    # ─────────────────────────────────────────────────────────────
    # Effectiveness Notices (EFFECT — IPO approved)
    # ─────────────────────────────────────────────────────────────

    async def get_effectiveness_notices(
        self,
        query: str | None = None,
        limit: int = 10,
    ) -> list[EffectivenessNotice]:
        """Get EFFECT notices — S-1 registrations declared effective.

        When SEC declares an S-1 effective, the company can begin selling shares.
        This is the last step before an IPO prices.

        Args:
            query: Search query to filter (company name or keyword)
            limit: Maximum results to return

        Returns:
            List of EffectivenessNotice objects, newest first
        """
        search_result = await self.search_filings(  # type: ignore[attr-defined]
            query=query or "*",
            forms=["EFFECT"],
            limit=limit,
        )

        return [
            EffectivenessNotice(
                entity_name=r.get("entity") or "",
                form_type="EFFECT",
                filed_date=r.get("filed") or "",
                accession_number="",
                url=r.get("url") or "",
            )
            for r in search_result["results"]
        ]
