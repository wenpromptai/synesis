"""SEC EDGAR 13F mixin — hedge fund quarterly holdings."""

from __future__ import annotations

from datetime import date
from typing import Any

import defusedxml.ElementTree as ET
from defusedxml import DefusedXmlException

import orjson

from synesis.core.constants import SEC_13F_DIFF_CACHE_TTL, SEC_13F_HOLDINGS_CACHE_TTL
from synesis.core.logging import get_logger
from synesis.providers.sec_edgar._base import (
    CACHE_PREFIX,
    SEC_BASE_URL,
    SEC_WWW_URL,
    _el_text,
    _parse_acceptance_datetime,
)
from synesis.providers.sec_edgar.models import Filing13F, Holding13F, SECFiling

logger = get_logger(__name__)


class ThirteenFMixin:
    """13F-HR quarterly holdings — filing retrieval, XML parsing, quarter comparison."""

    async def get_13f_filings(self, cik: str, limit: int = 2) -> list[SECFiling]:
        """Get recent 13F-HR filings for a CIK (hedge funds don't have tickers).

        Args:
            cik: SEC CIK number (unpadded or padded)
            limit: Maximum filings to return

        Returns:
            List of SECFiling objects filtered to 13F-HR, newest first
        """
        padded_cik = cik.zfill(10)
        cache_key = f"{CACHE_PREFIX}:13f_filings:{padded_cik}"

        cached = await self._redis.get(cache_key)  # type: ignore[attr-defined]
        if cached:
            try:
                return [SECFiling.model_validate(f) for f in orjson.loads(cached)][:limit]
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        try:
            resp = await self._fetch(f"{SEC_BASE_URL}/submissions/CIK{padded_cik}.json")  # type: ignore[attr-defined]
            resp.raise_for_status()
            data: dict[str, Any] = orjson.loads(resp.content)
        except Exception as e:
            logger.warning("Failed to fetch SEC submissions for CIK", cik=cik, error=str(e))
            return []

        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            return []

        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_documents = recent.get("primaryDocument", [])
        report_dates = recent.get("reportDate", [])
        acceptance_datetimes = recent.get("acceptanceDateTime", [])

        filings: list[SECFiling] = []
        n = min(len(forms), len(filing_dates), len(accession_numbers))
        for i in range(n):
            if forms[i] != "13F-HR":
                continue

            acc = accession_numbers[i].replace("-", "")
            doc = primary_documents[i] if i < len(primary_documents) else ""
            url = f"{SEC_WWW_URL}/Archives/edgar/data/{padded_cik}/{acc}/{doc}" if doc else ""

            acc_dt_str = acceptance_datetimes[i] if i < len(acceptance_datetimes) else ""
            accepted = _parse_acceptance_datetime(acc_dt_str)

            report_date_str = report_dates[i] if i < len(report_dates) else ""
            try:
                rpt_date = date.fromisoformat(report_date_str) if report_date_str else None
            except ValueError:
                rpt_date = None

            filings.append(
                SECFiling(
                    ticker="",
                    form="13F-HR",
                    filed_date=date.fromisoformat(filing_dates[i]),
                    accepted_datetime=accepted,
                    accession_number=accession_numbers[i],
                    primary_document=doc,
                    items="",
                    url=url,
                    report_date=rpt_date,
                )
            )

            if len(filings) >= limit:
                break

        # Cache for 1 hour
        await self._redis.set(  # type: ignore[attr-defined]
            cache_key,
            orjson.dumps([f.model_dump(mode="json") for f in filings]),
            ex=3600,
        )

        return filings

    async def get_13f_holdings(self, filing: SECFiling, cik: str) -> Filing13F | None:
        """Parse holdings from a 13F-HR filing's InfoTable XML.

        Args:
            filing: The 13F-HR filing metadata
            cik: CIK of the filer

        Returns:
            Filing13F with parsed holdings, or None on failure
        """
        cache_key = f"{CACHE_PREFIX}:13f_holdings:{filing.accession_number}"
        cached = await self._redis.get(cache_key)  # type: ignore[attr-defined]
        if cached:
            try:
                return Filing13F.model_validate(orjson.loads(cached))
            except Exception as e:
                logger.warning("13F holdings cache parse failed", error=str(e))

        cik_raw = cik.lstrip("0") or "0"
        acc_no_dashes = filing.accession_number.replace("-", "")
        index_url = f"{SEC_WWW_URL}/Archives/edgar/data/{cik_raw}/{acc_no_dashes}/index.json"

        try:
            resp = await self._fetch(index_url)  # type: ignore[attr-defined]
            resp.raise_for_status()
            index_data: dict[str, Any] = orjson.loads(resp.content)
        except Exception as e:
            logger.warning("Failed to fetch 13F index", cik=cik, error=str(e))
            return None

        # Find the InfoTable document
        infotable_name: str | None = None
        directory = index_data.get("directory", {})
        skip_prefixes = ("primary_doc", "r9999")
        for item in directory.get("item", []):
            name = item.get("name", "")
            name_lower = name.lower()
            if not name_lower.endswith(".xml"):
                continue
            if any(name_lower.startswith(p) for p in skip_prefixes):
                continue
            if "-index" in name_lower:
                continue
            infotable_name = name
            break

        if not infotable_name:
            logger.debug(
                "No InfoTable XML found in 13F filing", cik=cik, accession=filing.accession_number
            )
            return None

        xml_url = f"{SEC_WWW_URL}/Archives/edgar/data/{cik_raw}/{acc_no_dashes}/{infotable_name}"
        try:
            resp = await self._fetch(xml_url)  # type: ignore[attr-defined]
            resp.raise_for_status()
            holdings = self._parse_13f_xml(resp.text)
        except Exception as e:
            logger.warning("Failed to fetch/parse 13F InfoTable", cik=cik, error=str(e))
            return None

        # Sort by value descending
        holdings.sort(key=lambda h: h.value_thousands, reverse=True)
        total_value = sum(h.value_thousands for h in holdings)

        report_date = filing.report_date or filing.filed_date

        result = Filing13F(
            cik=cik,
            fund_name="",
            filed_date=filing.filed_date,
            report_date=report_date,
            accession_number=filing.accession_number,
            url=filing.url,
            holdings=holdings,
            total_value_thousands=total_value,
        )

        await self._redis.set(  # type: ignore[attr-defined]
            cache_key,
            orjson.dumps(result.model_dump(mode="json")),
            ex=SEC_13F_HOLDINGS_CACHE_TTL,
        )

        return result

    async def compare_13f_quarters(self, cik: str, fund_name: str) -> dict[str, Any] | None:
        """Compare two most recent 13F filings to find position changes.

        Returns dict with new_positions, exited_positions, increased, decreased,
        total_value_current, total_value_previous — or None if insufficient data.
        """
        cache_key = f"{CACHE_PREFIX}:13f_diff:{cik.zfill(10)}"
        cached = await self._redis.get(cache_key)  # type: ignore[attr-defined]
        if cached:
            try:
                return orjson.loads(cached)  # type: ignore[no-any-return]
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        filings = await self.get_13f_filings(cik, limit=2)
        if len(filings) < 2:
            logger.debug("Not enough 13F filings for comparison", cik=cik)
            return None

        current = await self.get_13f_holdings(filings[0], cik)
        previous = await self.get_13f_holdings(filings[1], cik)
        if not current or not previous:
            return None

        current.fund_name = fund_name
        previous.fund_name = fund_name

        # Build CUSIP-keyed dicts
        curr_map = {h.cusip: h for h in current.holdings}
        prev_map = {h.cusip: h for h in previous.holdings}

        new_positions = []
        exited_positions = []
        increased = []
        decreased = []

        for cusip, holding in curr_map.items():
            if cusip not in prev_map:
                new_positions.append(holding.model_dump())
            else:
                prev_h = prev_map[cusip]
                if prev_h.shares > 0:
                    change_pct = (holding.shares - prev_h.shares) / prev_h.shares
                    if change_pct > 0.10:
                        entry = holding.model_dump()
                        entry["change_pct"] = round(change_pct * 100, 1)
                        entry["prev_shares"] = prev_h.shares
                        increased.append(entry)
                    elif change_pct < -0.10:
                        entry = holding.model_dump()
                        entry["change_pct"] = round(change_pct * 100, 1)
                        entry["prev_shares"] = prev_h.shares
                        decreased.append(entry)

        for cusip, holding in prev_map.items():
            if cusip not in curr_map:
                exited_positions.append(holding.model_dump())

        # Sort by value and limit
        new_positions.sort(key=lambda x: x["value_thousands"], reverse=True)
        exited_positions.sort(key=lambda x: x["value_thousands"], reverse=True)
        increased.sort(key=lambda x: x["value_thousands"], reverse=True)
        decreased.sort(key=lambda x: x["value_thousands"], reverse=True)

        result: dict[str, Any] = {
            "fund_name": fund_name,
            "cik": cik,
            "current_report_date": current.report_date.isoformat(),
            "previous_report_date": previous.report_date.isoformat(),
            "total_value_current": current.total_value_thousands,
            "total_value_previous": previous.total_value_thousands,
            "new_positions": new_positions[:20],
            "exited_positions": exited_positions[:20],
            "increased": increased[:20],
            "decreased": decreased[:20],
        }

        await self._redis.set(cache_key, orjson.dumps(result), ex=SEC_13F_DIFF_CACHE_TTL)  # type: ignore[attr-defined]
        return result

    @staticmethod
    def _parse_13f_xml(xml_text: str) -> list[Holding13F]:
        """Parse a 13F InfoTable XML document into Holding13F objects."""
        try:
            root = ET.fromstring(xml_text)
        except (ET.ParseError, DefusedXmlException) as e:
            logger.warning("13F XML parse failed", error=str(e))
            return []

        # Namespace handling
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        holdings: list[Holding13F] = []
        for entry in root.findall(f".//{ns}infoTable"):
            name = _el_text(entry, f"{ns}nameOfIssuer")
            title = _el_text(entry, f"{ns}titleOfClass")
            cusip = _el_text(entry, f"{ns}cusip")
            value_str = _el_text(entry, f"{ns}value")
            discretion = _el_text(entry, f"{ns}investmentDiscretion")

            # Shares can be in shrsOrPrnAmt/sshPrnamt
            shares_parent = entry.find(f"{ns}shrsOrPrnAmt")
            shares_str = "0"
            if shares_parent is not None:
                shares_str = _el_text(shares_parent, f"{ns}sshPrnamt")

            try:
                value_k = int(value_str) if value_str else 0
            except ValueError:
                value_k = 0
            try:
                shares = int(shares_str) if shares_str else 0
            except ValueError:
                shares = 0

            if name and cusip:
                holdings.append(
                    Holding13F(
                        name_of_issuer=name,
                        title_of_class=title,
                        cusip=cusip,
                        value_thousands=value_k,
                        shares=shares,
                        investment_discretion=discretion or "SOLE",
                    )
                )

        return holdings
