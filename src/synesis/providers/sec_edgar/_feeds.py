"""SEC EDGAR feeds mixin — RSS/Atom real-time filing notifications."""

from __future__ import annotations

import defusedxml.ElementTree as ET
from defusedxml import DefusedXmlException

from synesis.core.logging import get_logger
from synesis.providers.sec_edgar._base import SEC_WWW_URL
from synesis.providers.sec_edgar.models import FilingFeedEntry

logger = get_logger(__name__)

# Atom namespace
_ATOM_NS = "http://www.w3.org/2005/Atom"


class FeedsMixin:
    """SEC EDGAR RSS/Atom filing feed access."""

    async def get_filing_feed(
        self,
        form_type: str | None = None,
        cik: str | None = None,
        count: int = 40,
    ) -> list[FilingFeedEntry]:
        """Get latest SEC filings from the EDGAR Atom feed.

        Near-real-time filing notifications — updated as filings are submitted.
        Filterable by form type and/or CIK.

        Args:
            form_type: Filter by form type (e.g., "8-K", "4", "SC 13D")
            cik: Filter by CIK number
            count: Number of entries to fetch (max 100)

        Returns:
            List of FilingFeedEntry objects, newest first
        """
        params: dict[str, str | int] = {
            "action": "getcurrent",
            "output": "atom",
            "count": min(count, 100),
            "owner": "include",
        }
        if form_type:
            params["type"] = form_type
        if cik:
            params["CIK"] = cik

        try:
            resp = await self._fetch(  # type: ignore[attr-defined]
                f"{SEC_WWW_URL}/cgi-bin/browse-edgar",
                params=params,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("SEC filing feed fetch failed", error=str(e))
            return []

        return self._parse_atom_feed(resp.text)

    @staticmethod
    def _parse_atom_feed(xml_text: str) -> list[FilingFeedEntry]:
        """Parse Atom XML feed into FilingFeedEntry objects."""
        try:
            root = ET.fromstring(xml_text)
        except (ET.ParseError, DefusedXmlException) as e:
            logger.warning("Atom feed parse failed", error=str(e))
            return []

        entries: list[FilingFeedEntry] = []
        for entry_el in root.findall(f"{{{_ATOM_NS}}}entry"):
            title_el = entry_el.find(f"{{{_ATOM_NS}}}title")
            link_el = entry_el.find(f"{{{_ATOM_NS}}}link")
            summary_el = entry_el.find(f"{{{_ATOM_NS}}}summary")
            updated_el = entry_el.find(f"{{{_ATOM_NS}}}updated")
            category_el = entry_el.find(f"{{{_ATOM_NS}}}category")

            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            link = link_el.get("href", "") if link_el is not None else ""
            summary = summary_el.text.strip() if summary_el is not None and summary_el.text else ""
            updated = updated_el.text.strip() if updated_el is not None and updated_el.text else ""
            category = category_el.get("term", "") if category_el is not None else ""

            if title or link:
                entries.append(
                    FilingFeedEntry(
                        title=title,
                        link=link,
                        summary=summary,
                        updated=updated,
                        category=category,
                    )
                )

        return entries
