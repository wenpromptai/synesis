"""SEC EDGAR XBRL mixin — Company Facts, Frames, and financial metrics."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any

import httpx
import orjson

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.providers.sec_edgar._base import CACHE_PREFIX, SEC_BASE_URL
from synesis.providers.sec_edgar.models import (
    CompanyFacts,
    XBRLFact,
    XBRLFrame,
    XBRLFrameEntry,
)

logger = get_logger(__name__)

# XBRL concept aliases — companies change tags over time (e.g., AAPL switched
# from "Revenues" to "RevenueFromContractWithCustomerExcludingAssessedTax" in 2018
# when ASC 606 was adopted). When a user asks for one, include both so they get
# the full history rather than a gap.
_CONCEPT_ALIASES: dict[str, list[str]] = {
    # Revenue — ASC 606 adoption (~2018) changed the tag for most companies
    "Revenues": ["RevenueFromContractWithCustomerExcludingAssessedTax"],
    "RevenueFromContractWithCustomerExcludingAssessedTax": ["Revenues"],
    # Cost of revenue — some filers use one, some the other
    "CostOfRevenue": ["CostOfGoodsAndServicesSold"],
    "CostOfGoodsAndServicesSold": ["CostOfRevenue"],
    # SGA — MSFT/GOOG/AMZN split into separate G&A and S&M lines
    "SellingGeneralAndAdministrativeExpense": ["GeneralAndAdministrativeExpense"],
    "GeneralAndAdministrativeExpense": ["SellingGeneralAndAdministrativeExpense"],
    # Long-term debt — some filers use "Noncurrent" variant (XOM, etc.)
    "LongTermDebt": ["LongTermDebtNoncurrent"],
    "LongTermDebtNoncurrent": ["LongTermDebt"],
    # Dividends per share — "Declared" vs "CashPaid" (XOM uses CashPaid)
    "CommonStockDividendsPerShareDeclared": ["CommonStockDividendsPerShareCashPaid"],
    "CommonStockDividendsPerShareCashPaid": ["CommonStockDividendsPerShareDeclared"],
}


def _filter_and_limit_facts(
    facts: list[XBRLFact],
    concepts: list[str] | None,
    limit: int,
) -> list[XBRLFact]:
    """Filter facts by concept names (with alias expansion) and limit per concept."""
    if concepts:
        # Expand aliases so e.g. "Revenues" also includes the ASC 606 tag
        expanded: set[str] = set()
        for c in concepts:
            expanded.add(c)
            for alias in _CONCEPT_ALIASES.get(c, []):
                expanded.add(alias)
        facts = [f for f in facts if f.concept in expanded]

    # Apply per-concept limit (most recent first)
    by_concept: dict[str, list[XBRLFact]] = defaultdict(list)
    for f in facts:
        by_concept[f.concept].append(f)

    result: list[XBRLFact] = []
    for concept_facts in by_concept.values():
        concept_facts.sort(key=lambda f: f.period_end, reverse=True)
        result.extend(concept_facts[:limit])

    # Sort final result by period descending
    result.sort(key=lambda f: f.period_end, reverse=True)
    return result


class XBRLMixin:
    """XBRL data access — company facts, frames, and individual concepts."""

    # ─────────────────────────────────────────────────────────────
    # Company Concept (single metric, single company)
    # ─────────────────────────────────────────────────────────────

    async def _get_xbrl_concept(
        self,
        ticker: str,
        concept: str,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        """Fetch quarterly data from the SEC XBRL companyconcept API.

        Args:
            ticker: Stock ticker symbol
            concept: US-GAAP concept name (e.g., EarningsPerShareBasic)
            limit: Maximum quarterly entries to return

        Returns:
            List of dicts with: period, actual, filed, form, frame
        """
        cache_key = f"{CACHE_PREFIX}:xbrl:{concept.lower()}:{ticker.upper()}"

        # Check cache
        cached = await self._redis.get(cache_key)  # type: ignore[attr-defined]
        if cached:
            try:
                entries: list[dict[str, Any]] = orjson.loads(cached)
                return entries[:limit]
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        cik = await self._get_cik(ticker)  # type: ignore[attr-defined]
        if not cik:
            logger.debug("No CIK for XBRL lookup", ticker=ticker, concept=concept)
            return []

        url = f"{SEC_BASE_URL}/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
        try:
            resp = await self._fetch(url)  # type: ignore[attr-defined]
            resp.raise_for_status()
            data: dict[str, Any] = orjson.loads(resp.content)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.debug("XBRL concept not found", ticker=ticker, concept=concept)
            else:
                logger.warning(
                    "XBRL fetch failed",
                    ticker=ticker,
                    concept=concept,
                    status=exc.response.status_code,
                )
            return []
        except Exception as e:
            logger.warning("XBRL fetch error", ticker=ticker, concept=concept, error=str(e))
            return []

        # Extract USD entries
        units = data.get("units", {})
        usd_entries = units.get("USD/shares", units.get("USD", []))
        if not usd_entries:
            return []

        # Keep only frame-bearing entries (deduplicated "best fit" from SEC)
        results: list[dict[str, Any]] = []
        for entry in usd_entries:
            frame = entry.get("frame", "")
            if frame:
                results.append(
                    {
                        "period": entry.get("end"),
                        "actual": entry.get("val"),
                        "filed": entry.get("filed"),
                        "form": entry.get("form"),
                        "frame": frame,
                    }
                )

        # Sort by period descending, take most recent
        results.sort(key=lambda x: x.get("period", ""), reverse=True)
        results = results[:limit]

        settings = get_settings()
        await self._redis.set(  # type: ignore[attr-defined]
            cache_key, orjson.dumps(results), ex=settings.sec_edgar_cache_ttl_company_facts
        )

        logger.debug(
            "Fetched XBRL data",
            ticker=ticker,
            concept=concept,
            count=len(results),
        )
        return results

    async def get_historical_eps(self, ticker: str, limit: int = 4) -> list[dict[str, Any]]:
        """Get historical quarterly EPS from SEC XBRL."""
        return await self._get_xbrl_concept(ticker, "EarningsPerShareBasic", limit)

    async def get_historical_revenue(self, ticker: str, limit: int = 4) -> list[dict[str, Any]]:
        """Get historical quarterly revenue from SEC XBRL.

        Tries RevenueFromContractWithCustomerExcludingAssessedTax first,
        falls back to Revenues if no data found.
        """
        results = await self._get_xbrl_concept(
            ticker, "RevenueFromContractWithCustomerExcludingAssessedTax", limit
        )
        if not results:
            results = await self._get_xbrl_concept(ticker, "Revenues", limit)
        return results

    # ─────────────────────────────────────────────────────────────
    # Company Facts (all XBRL in one call)
    # ─────────────────────────────────────────────────────────────

    async def get_company_facts(
        self,
        ticker: str,
        concepts: list[str] | None = None,
        limit: int = 20,
    ) -> CompanyFacts | None:
        """Get all XBRL financial facts for a company in one API call.

        Args:
            ticker: Stock ticker symbol
            concepts: Optional list of concept names to filter (e.g., ["NetIncomeLoss", "Assets"])
            limit: Maximum facts per concept to return (most recent first)

        Returns:
            CompanyFacts with all requested facts, or None if no data
        """
        ticker = ticker.upper()
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:companyfacts:{ticker}"

        cached = await self._redis.get(cache_key)  # type: ignore[attr-defined]
        if cached:
            try:
                result = CompanyFacts.model_validate(orjson.loads(cached))
                result.facts = _filter_and_limit_facts(result.facts, concepts, limit)
                result.concept_count = len({f.concept for f in result.facts})
                return result
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        cik = await self._get_cik(ticker)  # type: ignore[attr-defined]
        if not cik:
            logger.debug("No CIK for company facts", ticker=ticker)
            return None

        url = f"{SEC_BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            resp = await self._fetch(url)  # type: ignore[attr-defined]
            resp.raise_for_status()
            data: dict[str, Any] = orjson.loads(resp.content)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.debug("No company facts", ticker=ticker)
            else:
                logger.warning(
                    "Company facts fetch failed", ticker=ticker, status=exc.response.status_code
                )
            return None
        except Exception as e:
            logger.warning("Company facts fetch error", ticker=ticker, error=str(e))
            return None

        entity_name = data.get("entityName", "")
        all_facts: list[XBRLFact] = []

        # Iterate through taxonomies (us-gaap, dei, etc.)
        for taxonomy, tags in data.get("facts", {}).items():
            if not isinstance(tags, dict):
                continue
            for tag_name, tag_data in tags.items():
                if not isinstance(tag_data, dict):
                    continue

                label = tag_data.get("label", tag_name)
                for unit_name, entries in tag_data.get("units", {}).items():
                    if not isinstance(entries, list):
                        continue

                    # Only keep frame-bearing entries (deduplicated "best fit")
                    framed = [e for e in entries if e.get("frame")]
                    # Sort by end date descending
                    framed.sort(key=lambda e: e.get("end", ""), reverse=True)

                    for entry in framed:
                        try:
                            all_facts.append(
                                XBRLFact(
                                    concept=tag_name,
                                    label=label,
                                    unit=unit_name,
                                    period_end=date.fromisoformat(entry["end"]),
                                    value=float(entry["val"]),
                                    form=entry.get("form", ""),
                                    frame=entry.get("frame", ""),
                                    filed=date.fromisoformat(entry["filed"]),
                                )
                            )
                        except (ValueError, KeyError):
                            continue

        # Cache ALL facts (unfiltered, unlimited) so cache serves any query
        cache_result = CompanyFacts(
            ticker=ticker,
            cik=cik,
            entity_name=entity_name,
            facts=all_facts,
            concept_count=len({f.concept for f in all_facts}),
        )
        await self._redis.set(  # type: ignore[attr-defined]
            cache_key,
            orjson.dumps(cache_result.model_dump(mode="json")),
            ex=settings.sec_edgar_cache_ttl_company_facts,
        )

        # Apply filters for return
        filtered = _filter_and_limit_facts(all_facts, concepts, limit)
        return CompanyFacts(
            ticker=ticker,
            cik=cik,
            entity_name=entity_name,
            facts=filtered,
            concept_count=len({f.concept for f in filtered}),
        )

    # ─────────────────────────────────────────────────────────────
    # Frames (cross-company screening)
    # ─────────────────────────────────────────────────────────────

    async def get_xbrl_frame(
        self,
        taxonomy: str,
        tag: str,
        unit: str,
        period: str,
        limit: int = 100,
    ) -> XBRLFrame | None:
        """Get cross-company XBRL data for a single metric and period.

        Args:
            taxonomy: XBRL taxonomy (e.g., "us-gaap", "dei")
            tag: XBRL concept tag (e.g., "Revenues", "NetIncomeLoss")
            unit: Unit of measure (e.g., "USD", "USD/shares", "pure")
            period: Reporting period (e.g., "CY2024Q3", "CY2024", "CY2024Q3I")
            limit: Maximum entries to return (sorted by value descending)

        Returns:
            XBRLFrame with entries for all companies, or None if not found
        """
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:frame:{taxonomy}:{tag}:{unit}:{period}"

        cached = await self._redis.get(cache_key)  # type: ignore[attr-defined]
        if cached:
            try:
                result = XBRLFrame.model_validate(orjson.loads(cached))
                result.entries = result.entries[:limit]
                return result
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        url = f"{SEC_BASE_URL}/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json"
        try:
            resp = await self._fetch(url)  # type: ignore[attr-defined]
            resp.raise_for_status()
            data: dict[str, Any] = orjson.loads(resp.content)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.debug("XBRL frame not found", taxonomy=taxonomy, tag=tag, period=period)
            else:
                logger.warning("XBRL frame fetch failed", status=exc.response.status_code)
            return None
        except Exception as e:
            logger.warning("XBRL frame fetch error", error=str(e))
            return None

        entries: list[XBRLFrameEntry] = []
        for entry in data.get("data", []):
            try:
                entries.append(
                    XBRLFrameEntry(
                        cik=int(entry.get("cik", 0)),
                        entity_name=entry.get("entityName", ""),
                        value=float(entry.get("val", 0)),
                        accession_number=entry.get("accn", ""),
                        end=entry.get("end", ""),
                    )
                )
            except (ValueError, KeyError):
                continue

        # Sort by value descending
        entries.sort(key=lambda e: abs(e.value), reverse=True)

        result = XBRLFrame(
            taxonomy=taxonomy,
            tag=tag,
            unit=unit,
            period=period,
            entries=entries,
            entry_count=len(entries),
        )

        # Cache full result
        await self._redis.set(  # type: ignore[attr-defined]
            cache_key,
            orjson.dumps(result.model_dump(mode="json")),
            ex=settings.sec_edgar_cache_ttl_xbrl_frames,
        )

        # Apply limit for return
        result.entries = result.entries[:limit]
        return result
