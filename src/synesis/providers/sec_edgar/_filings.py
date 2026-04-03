"""SEC EDGAR filings mixin — filing retrieval, search, content extraction."""

from __future__ import annotations

import hashlib
from datetime import date
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any

import orjson

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.providers.sec_edgar._base import (
    CACHE_PREFIX,
    SEC_EFTS_URL,
    SEC_WWW_URL,
    _build_filing_url,
    _parse_acceptance_datetime,
)
from synesis.providers.sec_edgar.models import (
    ITEM_8K_DESCRIPTIONS,
    EarningsRelease,
    EventFiling8K,
    SECFiling,
)

if TYPE_CHECKING:
    from synesis.providers.crawler.crawl4ai import Crawl4AICrawlerProvider

logger = get_logger(__name__)


class FilingsMixin:
    """Filings retrieval, search, and content extraction."""

    # ─────────────────────────────────────────────────────────────
    # Filings
    # ─────────────────────────────────────────────────────────────

    async def get_filings(
        self,
        ticker: str,
        form_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[SECFiling]:
        """Get recent SEC filings for a ticker.

        Args:
            ticker: Stock ticker symbol
            form_types: Filter by form types (e.g., ["8-K", "10-K"])
            limit: Maximum filings to return

        Returns:
            List of SECFiling objects, newest first
        """
        ticker = ticker.upper()
        cik = await self._get_cik(ticker)  # type: ignore[attr-defined]
        if not cik:
            logger.debug("No CIK found for ticker", ticker=ticker)
            return []

        data = await self._fetch_submissions(cik)  # type: ignore[attr-defined]
        if not data:
            return []

        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            return []

        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_documents = recent.get("primaryDocument", [])
        items_list = recent.get("items", [])
        acceptance_datetimes = recent.get("acceptanceDateTime", [])
        report_dates = recent.get("reportDate", [])

        filings: list[SECFiling] = []
        n = min(len(forms), len(filing_dates), len(accession_numbers))
        for i in range(n):
            doc = primary_documents[i] if i < len(primary_documents) else ""
            url = _build_filing_url(cik, accession_numbers[i], doc)
            acc_dt_str = acceptance_datetimes[i] if i < len(acceptance_datetimes) else ""
            accepted = _parse_acceptance_datetime(acc_dt_str, ticker)

            rpt_str = report_dates[i] if i < len(report_dates) else ""
            try:
                rpt_date = date.fromisoformat(rpt_str) if rpt_str else None
            except ValueError:
                rpt_date = None

            try:
                filed = date.fromisoformat(filing_dates[i])
            except ValueError:
                logger.warning("Bad filing date, skipping", ticker=ticker, date=filing_dates[i])
                continue

            filing = SECFiling(
                ticker=ticker,
                form=forms[i],
                filed_date=filed,
                accepted_datetime=accepted,
                accession_number=accession_numbers[i],
                primary_document=doc,
                items=items_list[i] if i < len(items_list) else "",
                url=url,
                report_date=rpt_date,
            )

            # Apply form type filter inline to avoid building full list
            if form_types and filing.form not in form_types:
                continue

            filings.append(filing)
            if len(filings) >= limit:
                break

        return filings

    # ─────────────────────────────────────────────────────────────
    # Full-Text Search
    # ─────────────────────────────────────────────────────────────

    async def search_filings(
        self,
        query: str,
        forms: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Full-text search across SEC filings (EFTS).

        Args:
            query: Search query (supports Boolean, exact phrases, wildcards)
            forms: Filter by form types
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            limit: Max results per page
            offset: Pagination offset (skip first N results, max 10000)

        Returns:
            Dict with: results (list), total_hits (int), offset (int)
        """
        params: dict[str, str | int] = {"q": query}
        if forms:
            params["forms"] = ",".join(forms)
        if date_from or date_to:
            params["dateRange"] = "custom"
            if date_from:
                params["startdt"] = date_from
            if date_to:
                params["enddt"] = date_to
        if offset > 0:
            params["from"] = offset

        try:
            resp = await self._fetch(  # type: ignore[attr-defined]
                f"{SEC_EFTS_URL}/search-index",
                params=params,
            )
            resp.raise_for_status()
            data: dict[str, Any] = orjson.loads(resp.content)
        except Exception as e:
            logger.warning("SEC full-text search failed", query=query, error=str(e))
            return {"results": [], "total_hits": 0, "offset": offset}

        total_hits = data.get("hits", {}).get("total", {}).get("value", 0)
        hits = data.get("hits", {}).get("hits", [])
        results: list[dict[str, Any]] = []
        for hit in hits[:limit]:
            source = hit.get("_source", {})
            results.append(
                {
                    "entity": source.get("display_names", [""])[0]
                    if source.get("display_names")
                    else "",
                    "filed": source.get("file_date"),
                    "form": source.get("form_type"),
                    "url": f"{SEC_WWW_URL}/Archives/edgar/data/{source.get('file_num', '')}"
                    if source.get("file_num")
                    else "",
                    "description": source.get("display_description", ""),
                }
            )

        return {"results": results, "total_hits": total_hits, "offset": offset}

    # ─────────────────────────────────────────────────────────────
    # Filing Content Extraction
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Strip HTML tags and collapse whitespace (stdlib fallback)."""

        class _TagStripper(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.pieces: list[str] = []

            def handle_data(self, data: str) -> None:
                self.pieces.append(data)

        stripper = _TagStripper()
        stripper.feed(html)
        raw = " ".join(stripper.pieces)
        # Collapse runs of whitespace into single spaces / newlines
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        return "\n".join(line for line in lines if line).strip()

    async def get_filing_content(
        self,
        filing_url: str,
        crawler: Crawl4AICrawlerProvider | None = None,
    ) -> str | None:
        """Fetch filing content as markdown (or plain text fallback).

        Primary path uses Crawl4AI for high-fidelity markdown with tables.
        Falls back to a simple HTML→text strip when crawler is unavailable.
        Results are cached in Redis (press releases don't change).

        Args:
            filing_url: Full URL to the SEC filing document
            crawler: Optional Crawl4AI crawler instance

        Returns:
            Markdown/text content, or None on failure
        """
        url_hash = hashlib.md5(filing_url.encode()).hexdigest()  # noqa: S324
        cache_key = f"{CACHE_PREFIX}:content:{url_hash}"

        # Check cache
        cached = await self._redis.get(cache_key)  # type: ignore[attr-defined]
        if cached:
            return cached.decode() if isinstance(cached, bytes) else str(cached)

        content: str | None = None

        # Primary: Crawl4AI
        if crawler is not None:
            try:
                result = await crawler.crawl_sec_filing(filing_url)
                if result.success and result.markdown:
                    content = result.markdown
            except Exception as e:
                logger.debug(
                    "Crawl4AI failed for filing, falling back to HTML", url=filing_url, error=str(e)
                )

        # Fallback: fetch raw HTML and strip tags
        if content is None:
            try:
                resp = await self._fetch(filing_url)  # type: ignore[attr-defined]
                resp.raise_for_status()
                content = self._html_to_text(resp.text)
            except Exception as e:
                logger.warning("Failed to fetch filing content", url=filing_url, error=str(e))
                return None

        if not content:
            return None

        # Cache filing content (filings don't change once published)
        settings = get_settings()
        await self._redis.set(  # type: ignore[attr-defined]
            cache_key, content, ex=settings.sec_edgar_cache_ttl_filing_content
        )
        return content

    # ─────────────────────────────────────────────────────────────
    # Earnings Releases (8-K Item 2.02)
    # ─────────────────────────────────────────────────────────────

    async def get_earnings_releases(
        self,
        ticker: str,
        limit: int = 5,
        crawler: Crawl4AICrawlerProvider | None = None,
    ) -> list[EarningsRelease]:
        """Get recent earnings press releases (8-K Item 2.02) with content.

        Fetches 8-K filings, filters to Item 2.02 (earnings results),
        and retrieves the full press release content for each.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum earnings releases to return
            crawler: Optional Crawl4AI crawler for markdown extraction

        Returns:
            List of EarningsRelease objects with populated content
        """
        # Fetch enough 8-Ks to find `limit` earnings filings
        filings = await self.get_filings(ticker, form_types=["8-K"], limit=50)

        earnings_filings = [f for f in filings if "2.02" in f.items][:limit]
        if not earnings_filings:
            return []

        releases: list[EarningsRelease] = []
        for filing in earnings_filings:
            content = (
                await self.get_filing_content(filing.url, crawler=crawler) if filing.url else None
            )
            releases.append(
                EarningsRelease(
                    ticker=filing.ticker,
                    filed_date=filing.filed_date,
                    accepted_datetime=filing.accepted_datetime,
                    accession_number=filing.accession_number,
                    url=filing.url,
                    items=filing.items,
                    content=content,
                )
            )

        return releases

    # ─────────────────────────────────────────────────────────────
    # 8-K Event Filtering
    # ─────────────────────────────────────────────────────────────

    async def get_8k_events(
        self,
        ticker: str,
        items: list[str] | None = None,
        limit: int = 10,
        crawler: Crawl4AICrawlerProvider | None = None,
    ) -> list[EventFiling8K]:
        """Get 8-K filings filtered by item codes with human-readable descriptions.

        Args:
            ticker: Stock ticker symbol
            items: Filter to specific 8-K item codes (e.g., ["1.01", "2.01"]).
                   None returns all 8-K filings.
            limit: Maximum events to return
            crawler: Optional Crawl4AI crawler for content extraction

        Returns:
            List of EventFiling8K objects, newest first
        """
        filings = await self.get_filings(ticker, form_types=["8-K"], limit=50)

        events: list[EventFiling8K] = []
        for filing in filings:
            # Parse multi-item strings like "2.02,9.01"
            filing_items = [i.strip() for i in filing.items.split(",") if i.strip()]
            if not filing_items:
                filing_items = [""]

            # Apply item filter
            if items:
                if not any(fi in items for fi in filing_items):
                    continue

            item_descriptions = [
                ITEM_8K_DESCRIPTIONS.get(i, f"Item {i}") for i in filing_items if i
            ]

            content: str | None = None
            if crawler and filing.url:
                content = await self.get_filing_content(filing.url, crawler=crawler)

            events.append(
                EventFiling8K(
                    ticker=filing.ticker,
                    filed_date=filing.filed_date,
                    accepted_datetime=filing.accepted_datetime,
                    accession_number=filing.accession_number,
                    url=filing.url,
                    items=filing_items,
                    item_descriptions=item_descriptions,
                    content=content,
                )
            )

            if len(events) >= limit:
                break

        return events
