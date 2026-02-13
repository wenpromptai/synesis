"""SEC EDGAR API client.

Free API — no key required, just a User-Agent header.
- Ticker→CIK mapping: https://www.sec.gov/files/company_tickers.json
- Company submissions: https://data.sec.gov/submissions/CIK{cik}.json
- Full-text search: https://efts.sec.gov/LATEST/search-index
- Form 4 XML: https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import date, datetime
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as ET
from defusedxml import DefusedXmlException

import httpx
import orjson

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.providers.sec_edgar.models import EarningsRelease, InsiderTransaction, SECFiling

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from synesis.providers.crawler.crawl4ai import Crawl4AICrawlerProvider

logger = get_logger(__name__)

SEC_BASE_URL = "https://data.sec.gov"
SEC_WWW_URL = "https://www.sec.gov"
SEC_EFTS_URL = "https://efts.sec.gov/LATEST"
CACHE_PREFIX = "synesis:sec_edgar"


class _SECRateLimiter:
    """Token bucket rate limiter for SEC EDGAR (10 req/sec)."""

    def __init__(self, calls_per_second: int = 10) -> None:
        self._max_calls = calls_per_second
        self._calls: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            self._calls = [t for t in self._calls if t > now - 1.0]
            if len(self._calls) >= self._max_calls:
                sleep_time = 1.0 - (now - self._calls[0]) + 0.05
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                now = asyncio.get_event_loop().time()
                self._calls = [t for t in self._calls if t > now - 1.0]
            self._calls.append(now)


_sec_rate_limiter = _SECRateLimiter()


class SECEdgarClient:
    """Client for the SEC EDGAR API.

    Provides access to SEC filings, insider transactions (Form 4),
    and full-text filing search. Uses Redis caching to minimize requests.

    Usage:
        client = SECEdgarClient(redis=redis_client)
        filings = await client.get_filings("AAPL", form_types=["8-K", "10-K"])
        insiders = await client.get_insider_transactions("AAPL")
        await client.close()
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        self._http_client: httpx.AsyncClient | None = None
        self._cik_map: dict[str, str] | None = None

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            settings = get_settings()
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={
                    "User-Agent": settings.sec_edgar_user_agent,
                    "Accept-Encoding": "gzip, deflate",
                },
            )
        return self._http_client

    async def _fetch(self, url: str, **kwargs: Any) -> httpx.Response:
        """Rate-limited HTTP GET."""
        await _sec_rate_limiter.acquire()
        client = self._get_http_client()
        return await client.get(url, **kwargs)

    # ─────────────────────────────────────────────────────────────
    # CIK Mapping
    # ─────────────────────────────────────────────────────────────

    async def _load_cik_mapping(self) -> dict[str, str]:
        """Load ticker→CIK mapping, with Redis caching."""
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:cik_map"

        # Check Redis cache
        cached = await self._redis.get(cache_key)
        if cached:
            self._cik_map = orjson.loads(cached)
            return self._cik_map

        # Fetch from SEC
        try:
            resp = await self._fetch(f"{SEC_WWW_URL}/files/company_tickers.json")
            resp.raise_for_status()
            data: dict[str, Any] = orjson.loads(resp.content)
        except Exception as e:
            logger.warning("Failed to fetch SEC CIK mapping", error=str(e))
            return {}

        # Build {TICKER: "CIK_padded_to_10"}
        cik_map: dict[str, str] = {}
        for entry in data.values():
            ticker = str(entry.get("ticker", "")).upper()
            cik = entry.get("cik_str")
            if ticker and cik is not None:
                cik_map[ticker] = str(cik).zfill(10)

        # Cache in Redis
        await self._redis.set(
            cache_key,
            orjson.dumps(cik_map),
            ex=settings.sec_edgar_cache_ttl_cik_map,
        )
        self._cik_map = cik_map
        logger.debug("Loaded SEC CIK mapping", count=len(cik_map))
        return cik_map

    async def _get_cik(self, ticker: str) -> str | None:
        """Get padded CIK for a ticker."""
        ticker = ticker.upper()
        if self._cik_map is None:
            await self._load_cik_mapping()
        if self._cik_map is None:
            return None
        return self._cik_map.get(ticker)

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
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:filings:{ticker}"

        # Check cache
        cached = await self._redis.get(cache_key)
        if cached:
            try:
                all_filings = [SECFiling.model_validate(f) for f in orjson.loads(cached)]
                if form_types:
                    all_filings = [f for f in all_filings if f.form in form_types]
                return all_filings[:limit]
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        cik = await self._get_cik(ticker)
        if not cik:
            logger.debug("No CIK found for ticker", ticker=ticker)
            return []

        try:
            resp = await self._fetch(f"{SEC_BASE_URL}/submissions/CIK{cik}.json")
            resp.raise_for_status()
            data: dict[str, Any] = orjson.loads(resp.content)
        except Exception as e:
            logger.warning("Failed to fetch SEC submissions", ticker=ticker, error=str(e))
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

        filings: list[SECFiling] = []
        n = min(len(forms), len(filing_dates), len(accession_numbers), 50)
        for i in range(n):
            acc = accession_numbers[i].replace("-", "")
            doc = primary_documents[i] if i < len(primary_documents) else ""
            url = f"{SEC_WWW_URL}/Archives/edgar/data/{cik}/{acc}/{doc}" if doc else ""

            # Parse acceptance datetime
            acc_dt_str = acceptance_datetimes[i] if i < len(acceptance_datetimes) else ""
            try:
                accepted = datetime.fromisoformat(acc_dt_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                logger.warning(
                    "Failed to parse acceptance datetime, using sentinel",
                    ticker=ticker,
                    acc_dt_str=acc_dt_str,
                )
                accepted = datetime.min

            filings.append(
                SECFiling(
                    ticker=ticker,
                    form=forms[i],
                    filed_date=date.fromisoformat(filing_dates[i]),
                    accepted_datetime=accepted,
                    accession_number=accession_numbers[i],
                    primary_document=doc,
                    items=items_list[i] if i < len(items_list) else "",
                    url=url,
                )
            )

        # Cache all filings (unfiltered)
        await self._redis.set(
            cache_key,
            orjson.dumps([f.model_dump(mode="json") for f in filings]),
            ex=settings.sec_edgar_cache_ttl_submissions,
        )

        # Apply filters for return
        if form_types:
            filings = [f for f in filings if f.form in form_types]
        return filings[:limit]

    # ─────────────────────────────────────────────────────────────
    # Insider Transactions (Form 4 XML)
    # ─────────────────────────────────────────────────────────────

    async def get_insider_transactions(
        self,
        ticker: str,
        limit: int = 10,
    ) -> list[InsiderTransaction]:
        """Get insider transactions by parsing Form 4 filings.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum transactions to return

        Returns:
            List of InsiderTransaction objects, newest first
        """
        ticker = ticker.upper()
        cache_key = f"{CACHE_PREFIX}:insiders:{ticker}"

        # Check cache
        cached = await self._redis.get(cache_key)
        if cached:
            try:
                txns = [InsiderTransaction.model_validate(t) for t in orjson.loads(cached)]
                return txns[:limit]
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        # Get recent Form 4 filings
        form4s = await self.get_filings(ticker, form_types=["4"], limit=limit)
        if not form4s:
            return []

        transactions: list[InsiderTransaction] = []

        for filing in form4s:
            if not filing.url:
                continue
            try:
                # Strip XSLT prefix (e.g. "xslF345X05/") to get raw XML
                fetch_url = filing.url
                doc = filing.primary_document
                if "/" in doc and doc.split("/")[0].startswith("xsl"):
                    raw_doc = doc.split("/", 1)[1]
                    fetch_url = filing.url.replace(doc, raw_doc)

                resp = await self._fetch(fetch_url)
                resp.raise_for_status()
                parsed = self._parse_form4_xml(resp.text, ticker, filing.filed_date, filing.url)
                transactions.extend(parsed)
            except Exception as e:
                logger.debug(
                    "Failed to parse Form 4",
                    ticker=ticker,
                    accession=filing.accession_number,
                    error=str(e),
                )
                continue

        # Sort by transaction date descending
        transactions.sort(key=lambda t: t.transaction_date, reverse=True)
        transactions = transactions[:limit]

        # Cache
        settings = get_settings()
        await self._redis.set(
            cache_key,
            orjson.dumps([t.model_dump(mode="json") for t in transactions]),
            ex=settings.sec_edgar_cache_ttl_submissions,
        )

        return transactions

    @staticmethod
    def _parse_form4_xml(
        xml_text: str,
        ticker: str,
        filing_date: date,
        filing_url: str,
    ) -> list[InsiderTransaction]:
        """Parse a Form 4 XML document into InsiderTransaction objects."""
        transactions: list[InsiderTransaction] = []

        try:
            root = ET.fromstring(xml_text)
        except (ET.ParseError, DefusedXmlException) as e:
            logger.warning("Form 4 XML parse failed", ticker=ticker, error=str(e))
            return []

        # Namespace handling — SEC Form 4 XML uses default namespace
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        # Owner info
        owner_el = root.find(f".//{ns}reportingOwner")
        if owner_el is None:
            return []

        owner_id = owner_el.find(f"{ns}reportingOwnerId")
        owner_name = ""
        if owner_id is not None:
            name_el = owner_id.find(f"{ns}rptOwnerName")
            owner_name = name_el.text.strip() if name_el is not None and name_el.text else ""

        owner_rel_el = owner_el.find(f"{ns}reportingOwnerRelationship")
        relationship = "Unknown"
        if owner_rel_el is not None:
            if _el_text(owner_rel_el, f"{ns}isDirector") == "1":
                relationship = "Director"
            elif _el_text(owner_rel_el, f"{ns}isOfficer") == "1":
                title_el = owner_rel_el.find(f"{ns}officerTitle")
                title = (
                    title_el.text.strip() if title_el is not None and title_el.text else "Officer"
                )
                relationship = f"Officer ({title})"
            elif _el_text(owner_rel_el, f"{ns}isTenPercentOwner") == "1":
                relationship = "10% Owner"

        # Non-derivative transactions
        for txn_el in root.findall(f".//{ns}nonDerivativeTransaction"):
            txn = _parse_transaction_element(
                txn_el, ns, ticker, owner_name, relationship, filing_date, filing_url
            )
            if txn:
                transactions.append(txn)

        return transactions

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
    ) -> list[dict[str, Any]]:
        """Full-text search across SEC filings.

        Args:
            query: Search query
            forms: Filter by form types
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            limit: Max results

        Returns:
            List of search result dicts with: entity, filed, form, url
        """
        params: dict[str, str] = {"q": query}
        if forms:
            params["forms"] = ",".join(forms)
        if date_from or date_to:
            params["dateRange"] = "custom"
            if date_from:
                params["startdt"] = date_from
            if date_to:
                params["enddt"] = date_to

        try:
            resp = await self._fetch(
                f"{SEC_EFTS_URL}/search-index",
                params=params,
            )
            resp.raise_for_status()
            data: dict[str, Any] = orjson.loads(resp.content)
        except Exception as e:
            logger.warning("SEC full-text search failed", query=query, error=str(e))
            return []

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

        return results

    # ─────────────────────────────────────────────────────────────
    # Insider Sentiment (computed from Form 4 data)
    # ─────────────────────────────────────────────────────────────

    async def get_insider_sentiment(self, ticker: str) -> dict[str, Any] | None:
        """Compute insider sentiment from recent Form 4 transactions.

        Returns a simplified sentiment dict with net buy/sell ratio,
        compatible with the FundamentalsProvider protocol.
        """
        transactions = await self.get_insider_transactions(ticker, limit=20)
        if not transactions:
            return None

        total_buys = 0.0
        total_sells = 0.0
        buy_count = 0
        sell_count = 0

        for txn in transactions:
            value = txn.shares * (txn.price_per_share or 0)
            if txn.transaction_code == "P":
                total_buys += value
                buy_count += 1
            elif txn.transaction_code == "S":
                total_sells += value
                sell_count += 1

        total = total_buys + total_sells
        mspr = (total_buys - total_sells) / total if total > 0 else 0.0

        return {
            "ticker": ticker.upper(),
            "mspr": round(mspr, 4),
            "change": buy_count - sell_count,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "total_buy_value": round(total_buys, 2),
            "total_sell_value": round(total_sells, 2),
        }

    # ─────────────────────────────────────────────────────────────
    # XBRL Company Concept (Historical EPS / Revenue)
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
        cached = await self._redis.get(cache_key)
        if cached:
            try:
                entries: list[dict[str, Any]] = orjson.loads(cached)
                return entries[:limit]
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        cik = await self._get_cik(ticker)
        if not cik:
            logger.debug("No CIK for XBRL lookup", ticker=ticker, concept=concept)
            return []

        url = f"{SEC_BASE_URL}/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
        try:
            resp = await self._fetch(url)
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

        # Extract USD entries (prefer 10-Q filings for quarterly data)
        units = data.get("units", {})
        usd_entries = units.get("USD/shares", units.get("USD", []))
        if not usd_entries:
            return []

        # Filter to quarterly entries (frame contains "Q")
        quarterly: list[dict[str, Any]] = []
        for entry in usd_entries:
            frame = entry.get("frame", "")
            if "Q" in str(frame):
                quarterly.append(
                    {
                        "period": entry.get("end"),
                        "actual": entry.get("val"),
                        "filed": entry.get("filed"),
                        "form": entry.get("form"),
                        "frame": frame,
                    }
                )

        # Sort by period descending, take most recent
        quarterly.sort(key=lambda x: x.get("period", ""), reverse=True)
        quarterly = quarterly[:limit]

        # Cache for 6 hours
        await self._redis.set(cache_key, orjson.dumps(quarterly), ex=21600)

        logger.debug(
            "Fetched XBRL data",
            ticker=ticker,
            concept=concept,
            count=len(quarterly),
        )
        return quarterly

    async def get_historical_eps(self, ticker: str, limit: int = 4) -> list[dict[str, Any]]:
        """Get historical quarterly EPS from SEC XBRL.

        Uses the EarningsPerShareBasic concept from the SEC XBRL API.

        Args:
            ticker: Stock ticker symbol
            limit: Number of recent quarters to return

        Returns:
            List of dicts with: period, actual, filed, form, frame
        """
        return await self._get_xbrl_concept(ticker, "EarningsPerShareBasic", limit)

    async def get_historical_revenue(self, ticker: str, limit: int = 4) -> list[dict[str, Any]]:
        """Get historical quarterly revenue from SEC XBRL.

        Tries RevenueFromContractWithCustomerExcludingAssessedTax first,
        falls back to Revenues if no data found.

        Args:
            ticker: Stock ticker symbol
            limit: Number of recent quarters to return

        Returns:
            List of dicts with: period, actual, filed, form, frame
        """
        results = await self._get_xbrl_concept(
            ticker, "RevenueFromContractWithCustomerExcludingAssessedTax", limit
        )
        if not results:
            results = await self._get_xbrl_concept(ticker, "Revenues", limit)
        return results

    # ─────────────────────────────────────────────────────────────
    # Earnings Press Releases
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
        cached = await self._redis.get(cache_key)
        if cached:
            return cached.decode() if isinstance(cached, bytes) else str(cached)

        content: str | None = None

        # Primary: Crawl4AI
        if crawler is not None:
            try:
                result = await crawler.crawl_sec_filing(filing_url)
                if result.success and result.markdown:
                    content = result.markdown
            except Exception:
                logger.debug("Crawl4AI failed for filing, falling back to HTML", url=filing_url)

        # Fallback: fetch raw HTML and strip tags
        if content is None:
            try:
                resp = await self._fetch(filing_url)
                resp.raise_for_status()
                content = self._html_to_text(resp.text)
            except Exception:
                logger.warning("Failed to fetch filing content", url=filing_url)
                return None

        if not content:
            return None

        # Cache for 7 days (press releases don't change)
        await self._redis.set(cache_key, content, ex=604800)
        return content

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
    # Lifecycle
    # ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Clean up resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.debug("SECEdgarClient closed")


# ─────────────────────────────────────────────────────────────
# XML Helpers
# ─────────────────────────────────────────────────────────────


def _el_text(parent: Element, tag: str) -> str:
    """Get text from a child element, or empty string."""
    el = parent.find(tag)
    return el.text.strip() if el is not None and el.text else ""


def _parse_transaction_element(
    txn_el: Element,
    ns: str,
    ticker: str,
    owner_name: str,
    relationship: str,
    filing_date: date,
    filing_url: str,
) -> InsiderTransaction | None:
    """Parse a single <nonDerivativeTransaction> element."""
    amounts = txn_el.find(f"{ns}transactionAmounts")
    if amounts is None:
        return None

    # Transaction code
    coding = txn_el.find(f"{ns}transactionCoding")
    code = ""
    if coding is not None:
        code = _el_text(coding, f"{ns}transactionCode")

    # Transaction date
    date_el = txn_el.find(f"{ns}transactionDate")
    txn_date_str = _el_text(date_el, f"{ns}value") if date_el is not None else ""
    if not txn_date_str:
        return None
    try:
        txn_date = date.fromisoformat(txn_date_str)
    except ValueError:
        return None

    # Shares
    shares_el = amounts.find(f"{ns}transactionShares")
    shares_str = _el_text(shares_el, f"{ns}value") if shares_el is not None else "0"
    try:
        shares = float(shares_str)
    except ValueError:
        shares = 0.0

    # Price per share
    price_el = amounts.find(f"{ns}transactionPricePerShare")
    price_str = _el_text(price_el, f"{ns}value") if price_el is not None else ""
    price: float | None = None
    if price_str:
        try:
            price = float(price_str)
        except ValueError:
            pass

    # Acquired or disposed
    ad_el = amounts.find(f"{ns}transactionAcquiredDisposedCode")
    ad_code = _el_text(ad_el, f"{ns}value") if ad_el is not None else ""

    # Post-transaction holdings
    post_el = txn_el.find(f"{ns}postTransactionAmounts")
    post_shares_str = "0"
    if post_el is not None:
        ownership_el = post_el.find(f"{ns}sharesOwnedFollowingTransaction")
        if ownership_el is not None:
            post_shares_str = _el_text(ownership_el, f"{ns}value")
    try:
        shares_after = float(post_shares_str)
    except ValueError:
        shares_after = 0.0

    return InsiderTransaction(
        ticker=ticker,
        owner_name=owner_name,
        owner_relationship=relationship,
        transaction_date=txn_date,
        transaction_code=code,
        shares=shares,
        price_per_share=price,
        shares_after=shares_after,
        acquired_or_disposed=ad_code,
        filing_date=filing_date,
        filing_url=filing_url,
    )
