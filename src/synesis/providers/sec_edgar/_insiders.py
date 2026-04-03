"""SEC EDGAR insiders mixin — Form 3/4/5 parsing for insider transactions."""

from __future__ import annotations

from datetime import date
from typing import Any

import defusedxml.ElementTree as ET
from defusedxml import DefusedXmlException

import orjson

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.providers.sec_edgar._base import CACHE_PREFIX, _el_text
from synesis.providers.sec_edgar.models import DerivativeTransaction, InsiderTransaction

logger = get_logger(__name__)


def _strip_xslt_prefix(filing_url: str, primary_document: str) -> str:
    """Strip XSLT prefix (e.g. 'xslF345X05/') from SEC filing URLs to get raw XML."""
    if "/" in primary_document and primary_document.split("/")[0].startswith("xsl"):
        raw_doc = primary_document.split("/", 1)[1]
        return filing_url.replace(primary_document, raw_doc)
    return filing_url


class InsidersMixin:
    """Insider transaction parsing from Form 3/4/5 XML."""

    # ─────────────────────────────────────────────────────────────
    # Insider Transactions (Form 3/4/5 XML)
    # ─────────────────────────────────────────────────────────────

    async def get_insider_transactions(
        self,
        ticker: str,
        limit: int = 10,
        codes: list[str] | None = None,
    ) -> list[InsiderTransaction]:
        """Get insider transactions by parsing Form 3/4/5 filings.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum transactions to return
            codes: Filter to specific transaction codes (e.g. ["P", "S"] for open-market only).
                   None means return all codes. Use ["P", "S"] for conviction signals only.

        Returns:
            List of InsiderTransaction objects, newest first
        """
        ticker = ticker.upper()
        # Cache key includes codes filter so different filters don't collide
        codes_key = "_".join(sorted(codes)) if codes else "all"
        cache_key = f"{CACHE_PREFIX}:insiders:{ticker}:{codes_key}"

        # Check cache
        cached = await self._redis.get(cache_key)  # type: ignore[attr-defined]
        if cached:
            try:
                txns = [InsiderTransaction.model_validate(t) for t in orjson.loads(cached)]
                return txns[:limit]
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        # Fetch Form 3/4/5 filings
        fetch_limit = limit * 5 if codes else limit
        form4s = await self.get_filings(  # type: ignore[attr-defined]
            ticker, form_types=["3", "4", "5"], limit=fetch_limit
        )
        if not form4s:
            return []

        transactions: list[InsiderTransaction] = []

        for filing in form4s:
            if not filing.url:
                continue
            try:
                fetch_url = _strip_xslt_prefix(filing.url, filing.primary_document)
                resp = await self._fetch(fetch_url)  # type: ignore[attr-defined]
                resp.raise_for_status()
                parsed = self._parse_form4_xml(resp.text, ticker, filing.filed_date, filing.url)
                transactions.extend(parsed)
            except Exception as e:
                logger.debug(
                    "Failed to parse Form 3/4/5",
                    ticker=ticker,
                    accession=filing.accession_number,
                    error=str(e),
                )
                continue

        # Sort by transaction date descending
        transactions.sort(key=lambda t: t.transaction_date, reverse=True)

        # Apply code filter
        if codes:
            code_set = set(codes)
            transactions = [t for t in transactions if t.transaction_code in code_set]

        transactions = transactions[:limit]

        # Cache
        settings = get_settings()
        await self._redis.set(  # type: ignore[attr-defined]
            cache_key,
            orjson.dumps([t.model_dump(mode="json") for t in transactions]),
            ex=settings.sec_edgar_cache_ttl_submissions,
        )

        return transactions

    # ─────────────────────────────────────────────────────────────
    # Derivative Transactions
    # ─────────────────────────────────────────────────────────────

    async def get_derivative_transactions(
        self,
        ticker: str,
        limit: int = 10,
    ) -> list[DerivativeTransaction]:
        """Get derivative insider transactions (options, warrants) from Form 4 XML.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum transactions to return

        Returns:
            List of DerivativeTransaction objects, newest first
        """
        ticker = ticker.upper()
        cache_key = f"{CACHE_PREFIX}:derivatives:{ticker}"

        cached = await self._redis.get(cache_key)  # type: ignore[attr-defined]
        if cached:
            try:
                txns = [DerivativeTransaction.model_validate(t) for t in orjson.loads(cached)]
                return txns[:limit]
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        form4s = await self.get_filings(  # type: ignore[attr-defined]
            ticker, form_types=["4"], limit=limit * 3
        )
        if not form4s:
            return []

        transactions: list[DerivativeTransaction] = []

        for filing in form4s:
            if not filing.url:
                continue
            try:
                fetch_url = _strip_xslt_prefix(filing.url, filing.primary_document)
                resp = await self._fetch(fetch_url)  # type: ignore[attr-defined]
                resp.raise_for_status()
                parsed = self._parse_derivative_xml(
                    resp.text, ticker, filing.filed_date, filing.url
                )
                transactions.extend(parsed)
            except Exception as e:
                logger.debug(
                    "Failed to parse Form 4 derivatives",
                    ticker=ticker,
                    accession=filing.accession_number,
                    error=str(e),
                )
                continue

        transactions.sort(key=lambda t: t.transaction_date, reverse=True)
        transactions = transactions[:limit]

        settings = get_settings()
        await self._redis.set(  # type: ignore[attr-defined]
            cache_key,
            orjson.dumps([t.model_dump(mode="json") for t in transactions]),
            ex=settings.sec_edgar_cache_ttl_submissions,
        )

        return transactions

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
    # XML Parsers
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_owner_info(root: Any, ns: str) -> tuple[str, str, str]:
        """Extract owner name, relationship, and country from Form 3/4/5 XML.

        Returns:
            Tuple of (owner_name, relationship, country).
            Country field available on filings from March 2026+.
        """
        owner_el = root.find(f".//{ns}reportingOwner")
        if owner_el is None:
            return "", "Unknown", ""

        owner_id = owner_el.find(f"{ns}reportingOwnerId")
        owner_name = ""
        country = ""
        if owner_id is not None:
            name_el = owner_id.find(f"{ns}rptOwnerName")
            owner_name = name_el.text.strip() if name_el is not None and name_el.text else ""
            # Country field added March 2026
            country = _el_text(owner_id, f"{ns}rptOwnerCountry")

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

        return owner_name, relationship, country

    @staticmethod
    def _parse_form4_xml(
        xml_text: str,
        ticker: str,
        filing_date: date,
        filing_url: str,
    ) -> list[InsiderTransaction]:
        """Parse a Form 3/4/5 XML document into InsiderTransaction objects."""
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

        owner_name, relationship, country = InsidersMixin._parse_owner_info(root, ns)
        if not owner_name:
            return []

        # Non-derivative transactions
        for txn_el in root.findall(f".//{ns}nonDerivativeTransaction"):
            txn = _parse_transaction_element(
                txn_el, ns, ticker, owner_name, relationship, country, filing_date, filing_url
            )
            if txn:
                transactions.append(txn)

        # Form 3 also has nonDerivativeHolding (initial positions, no transaction)
        for hold_el in root.findall(f".//{ns}nonDerivativeHolding"):
            holding = _parse_holding_element(
                hold_el, ns, ticker, owner_name, relationship, country, filing_date, filing_url
            )
            if holding:
                transactions.append(holding)

        return transactions

    @staticmethod
    def _parse_derivative_xml(
        xml_text: str,
        ticker: str,
        filing_date: date,
        filing_url: str,
    ) -> list[DerivativeTransaction]:
        """Parse derivative transactions from Form 4 XML."""
        transactions: list[DerivativeTransaction] = []

        try:
            root = ET.fromstring(xml_text)
        except (ET.ParseError, DefusedXmlException) as e:
            logger.warning("Form 4 derivative XML parse failed", ticker=ticker, error=str(e))
            return []

        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        owner_name, relationship, country = InsidersMixin._parse_owner_info(root, ns)
        if not owner_name:
            return []

        for txn_el in root.findall(f".//{ns}derivativeTransaction"):
            txn = _parse_derivative_element(
                txn_el, ns, ticker, owner_name, relationship, country, filing_date, filing_url
            )
            if txn:
                transactions.append(txn)

        return transactions


# ─────────────────────────────────────────────────────────────
# Element Parsers (module-level)
# ─────────────────────────────────────────────────────────────


def _parse_transaction_element(
    txn_el: Any,
    ns: str,
    ticker: str,
    owner_name: str,
    relationship: str,
    country: str,
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
        owner_country=country,
        transaction_date=txn_date,
        transaction_code=code,
        shares=shares,
        price_per_share=price,
        shares_after=shares_after,
        acquired_or_disposed=ad_code,
        filing_date=filing_date,
        filing_url=filing_url,
    )


def _parse_holding_element(
    hold_el: Any,
    ns: str,
    ticker: str,
    owner_name: str,
    relationship: str,
    country: str,
    filing_date: date,
    filing_url: str,
) -> InsiderTransaction | None:
    """Parse a <nonDerivativeHolding> element (Form 3 initial positions)."""
    # Holdings have post-transaction amounts but no transaction amounts
    post_el = hold_el.find(f"{ns}postTransactionAmounts")
    post_shares_str = "0"
    if post_el is not None:
        ownership_el = post_el.find(f"{ns}sharesOwnedFollowingTransaction")
        if ownership_el is not None:
            post_shares_str = _el_text(ownership_el, f"{ns}value")
    try:
        shares_after = float(post_shares_str)
    except ValueError:
        shares_after = 0.0

    if shares_after == 0.0:
        return None

    # Use filing date as transaction date for initial holdings
    return InsiderTransaction(
        ticker=ticker,
        owner_name=owner_name,
        owner_relationship=relationship,
        owner_country=country,
        transaction_date=filing_date,
        transaction_code="J",  # Other — initial holding report
        shares=shares_after,
        price_per_share=None,
        shares_after=shares_after,
        acquired_or_disposed="A",
        filing_date=filing_date,
        filing_url=filing_url,
    )


def _parse_derivative_element(
    txn_el: Any,
    ns: str,
    ticker: str,
    owner_name: str,
    relationship: str,
    country: str,
    filing_date: date,
    filing_url: str,
) -> DerivativeTransaction | None:
    """Parse a single <derivativeTransaction> element."""
    # Security title
    security_el = txn_el.find(f"{ns}securityTitle")
    security_title = _el_text(security_el, f"{ns}value") if security_el is not None else ""

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

    # Exercise price
    exercise_el = txn_el.find(f"{ns}conversionOrExercisePrice")
    exercise_str = _el_text(exercise_el, f"{ns}value") if exercise_el is not None else ""
    exercise_price: float | None = None
    if exercise_str:
        try:
            exercise_price = float(exercise_str)
        except ValueError:
            pass

    # Underlying shares
    underlying_el = txn_el.find(f"{ns}underlyingSecurity")
    underlying_shares_str = "0"
    if underlying_el is not None:
        underlying_shares_str = _el_text(underlying_el, f"{ns}underlyingSecurityShares")
        if not underlying_shares_str:
            val_el = underlying_el.find(f"{ns}underlyingSecurityShares")
            if val_el is not None:
                underlying_shares_str = _el_text(val_el, f"{ns}value")
    try:
        underlying_shares = float(underlying_shares_str) if underlying_shares_str else 0.0
    except ValueError:
        underlying_shares = 0.0

    # Expiration date
    expiration_el = txn_el.find(f"{ns}expirationDate")
    exp_str = _el_text(expiration_el, f"{ns}value") if expiration_el is not None else ""
    expiration_date: date | None = None
    if exp_str:
        try:
            expiration_date = date.fromisoformat(exp_str)
        except ValueError:
            pass

    # Acquired or disposed
    amounts = txn_el.find(f"{ns}transactionAmounts")
    ad_code = ""
    if amounts is not None:
        ad_el = amounts.find(f"{ns}transactionAcquiredDisposedCode")
        ad_code = _el_text(ad_el, f"{ns}value") if ad_el is not None else ""

    return DerivativeTransaction(
        ticker=ticker,
        owner_name=owner_name,
        owner_relationship=relationship,
        owner_country=country,
        transaction_date=txn_date,
        transaction_code=code,
        security_title=security_title,
        exercise_price=exercise_price,
        underlying_shares=underlying_shares,
        expiration_date=expiration_date,
        acquired_or_disposed=ad_code,
        filing_date=filing_date,
        filing_url=filing_url,
    )
