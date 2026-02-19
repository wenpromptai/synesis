"""FactSet data provider for historical prices, fundamentals, and corporate actions."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

from synesis.providers.factset.client import (
    FactSetClient,
    get_cached_max_price_date,
)
from synesis.providers.base import CompanyInfo, FundamentalsSnapshot, PriceSnapshot
from synesis.providers.factset.models import (
    FactSetCorporateAction,
    FactSetFundamentals,
    FactSetPrice,
    FactSetSecurity,
    FactSetSharesOutstanding,
)
from synesis.providers.factset.queries import (
    ADJUSTMENT_FACTORS,
    ADJUSTMENT_FACTORS_FOR_HISTORY,
    COMPANY_PROFILE,
    CORPORATE_ACTIONS,
    CORPORATE_ACTIONS_MAX_DATE,
    DIVIDENDS,
    ENTITY_INFO,
    FUNDAMENTALS_ANNUAL,
    FUNDAMENTALS_LTM,
    FUNDAMENTALS_MAX_DATE_ANNUAL,
    FUNDAMENTALS_MAX_DATE_LTM,
    FUNDAMENTALS_MAX_DATE_QUARTERLY,
    FUNDAMENTALS_QUARTERLY,
    GET_SECURITY_ID,
    LATEST_PRICE_FOR_SECURITY,
    PRICE_BY_DATE,
    PRICE_HISTORY,
    SEARCH_SECURITIES,
    SECURITY_BY_TICKER,
    SHARES_OUTSTANDING_AS_OF_DATE,
    SHARES_OUTSTANDING_CURRENT,
    SPLITS,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Event type mappings
EVENT_TYPE_MAP = {
    "DVC": "dividend",
    "FSP": "split",  # Forward split
    "RSP": "split",  # Reverse split
    "BNS": "split",  # Bonus issue (stock split)
    "RGT": "rights",  # Rights issue
}


class FactSetProvider:
    """FactSet data provider for historical prices, fundamentals, and corporate actions.

    Provides access to FactSet's SQL Server database containing:
    - Global prices (OHLCV, pre-calculated returns)
    - Security master data (names, exchanges, types)
    - Fundamentals (EPS, BPS, margins, ratios)
    - Corporate actions (dividends, splits)
    - Shares outstanding

    Performance Notes:
    - Ticker resolution is cached (TTL: 24 hours via LRU cache)
    - Global max price date is cached (TTL: 4 hours)
    - Avoid ORDER BY DESC on large tables - use date filters instead
    """

    def __init__(self, client: FactSetClient | None = None) -> None:
        """Initialize FactSet provider.

        Args:
            client: FactSet database client. If None, creates new client.
        """
        self._client = client or FactSetClient()
        # In-memory cache for fsym_id lookups
        self._ticker_cache: dict[str, str] = {}

    # =========================================================================
    # TICKER / SECURITY RESOLUTION
    # =========================================================================

    def _normalize_ticker(self, ticker: str) -> str:
        """Normalize ticker to region format (e.g., 'NVDA' -> 'NVDA-US')."""
        ticker = ticker.upper().strip()
        if "-" not in ticker:
            # Default to US region
            return f"{ticker}-US"
        return ticker

    async def resolve_ticker(self, ticker: str) -> FactSetSecurity | None:
        """Resolve ticker (e.g., 'NVDA', 'NVDA-US') to FactSet security.

        Args:
            ticker: Ticker symbol, optionally with region suffix

        Returns:
            FactSetSecurity if found, None otherwise
        """
        normalized = self._normalize_ticker(ticker)

        rows = await self._client.execute_query(SECURITY_BY_TICKER, {"ticker": normalized})

        if not rows:
            logger.debug(f"Ticker not found: {normalized}")
            return None

        row = rows[0]
        fsym_id = row["fsym_id"]
        fsym_security_id = row.get("fsym_security_id")

        # Get entity info (country, sector, industry)
        # Note: Uses fsym_security_id to join to ff_sec_entity (not fsym_id)
        entity = {}
        if fsym_security_id:
            entity_rows = await self._client.execute_query(
                ENTITY_INFO, {"fsym_security_id": fsym_security_id}
            )
            entity = entity_rows[0] if entity_rows else {}

        security = FactSetSecurity(
            fsym_id=fsym_id,
            fsym_security_id=row.get("fsym_security_id"),
            ticker=row.get("ticker_region"),
            name=row["proper_name"],
            exchange_code=row["fref_exchange_code"],
            security_type=row["fref_security_type"],
            currency=row["currency"],
            country=entity.get("iso_country"),
            sector=entity.get("sector_code"),
            industry=entity.get("industry_code"),
        )

        # Cache the fsym_id
        self._ticker_cache[normalized] = fsym_id

        return security

    async def get_fsym_id(self, ticker: str) -> str | None:
        """Get fsym_id for a ticker (cached).

        Args:
            ticker: Ticker symbol

        Returns:
            fsym_id if found, None otherwise
        """
        normalized = self._normalize_ticker(ticker)

        if normalized in self._ticker_cache:
            return self._ticker_cache[normalized]

        security = await self.resolve_ticker(ticker)
        return security.fsym_id if security else None

    async def search_securities(self, query: str, limit: int = 20) -> list[FactSetSecurity]:
        """Search securities by name or ticker.

        Args:
            query: Search query (company name or ticker)
            limit: Maximum results to return

        Returns:
            List of matching securities
        """
        # Use LIKE pattern for search
        search_pattern = f"%{query}%"

        rows = await self._client.execute_query(
            SEARCH_SECURITIES, {"query": search_pattern, "limit": limit}
        )

        securities = []
        for row in rows:
            securities.append(
                FactSetSecurity(
                    fsym_id=row["fsym_id"],
                    fsym_security_id=row.get("fsym_security_id"),
                    ticker=row.get("ticker_region"),
                    name=row["proper_name"],
                    exchange_code=row["fref_exchange_code"],
                    security_type=row["fref_security_type"],
                    currency=row["currency"],
                    country=None,
                    sector=None,
                    industry=None,
                )
            )

        return securities

    # =========================================================================
    # PRICE DATA
    # =========================================================================

    async def get_price(self, ticker: str, price_date: date | None = None) -> FactSetPrice | None:
        """Get price for a single date (default: latest).

        Args:
            ticker: Ticker symbol
            price_date: Specific date, or None for latest

        Returns:
            FactSetPrice if found, None otherwise
        """
        fsym_id = await self.get_fsym_id(ticker)
        if not fsym_id:
            return None

        # For latest price (no date specified), use security-specific query
        # This handles cases where global max date doesn't have data for this security
        if price_date is None:
            rows = await self._client.execute_query(LATEST_PRICE_FOR_SECURITY, {"fsym_id": fsym_id})
            if not rows:
                return None
            return self._row_to_price(rows[0])

        # For specific date, use date-based query
        rows = await self._client.execute_query(
            PRICE_BY_DATE, {"fsym_id": fsym_id, "price_date": price_date}
        )

        if not rows:
            return None

        row = rows[0]
        return self._row_to_price(row)

    async def get_price_history(
        self, ticker: str, start_date: date, end_date: date | None = None
    ) -> list[FactSetPrice]:
        """Get historical prices for date range.

        Args:
            ticker: Ticker symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive), defaults to today

        Returns:
            List of FactSetPrice objects, ordered by date ascending
        """
        fsym_id = await self.get_fsym_id(ticker)
        if not fsym_id:
            return []

        if end_date is None:
            end_date = await get_cached_max_price_date(self._client)
            if not end_date:
                return []

        rows = await self._client.execute_query(
            PRICE_HISTORY,
            {"fsym_id": fsym_id, "start_date": start_date, "end_date": end_date},
        )

        return [self._row_to_price(row) for row in rows]

    async def get_latest_prices(self, tickers: list[str]) -> dict[str, FactSetPrice]:
        """Get latest prices for multiple tickers.

        Uses security-specific latest price queries to handle cases where
        the global max date doesn't have data for all securities.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dict mapping ticker to FactSetPrice (only includes found tickers)
        """
        if not tickers:
            return {}

        results: dict[str, FactSetPrice] = {}
        for ticker in tickers:
            price = await self.get_price(ticker)  # Uses LATEST_PRICE_FOR_SECURITY
            if price:
                results[ticker] = price

        return results

    def _row_to_price(self, row: dict[str, Any]) -> FactSetPrice:
        """Convert database row to FactSetPrice model."""
        return FactSetPrice(
            fsym_id=row["fsym_id"],
            price_date=row["price_date"],
            open=row.get("price_open"),
            high=row.get("price_high"),
            low=row.get("price_low"),
            close=row["price_close"],
            volume=row.get("volume"),
            one_day_pct=row.get("one_day_pct"),
            wtd_pct=row.get("wtd_pct"),
            mtd_pct=row.get("mtd_pct"),
            qtd_pct=row.get("qtd_pct"),
            ytd_pct=row.get("ytd_pct"),
            one_mth_pct=row.get("one_mth_pct"),
            three_mth_pct=row.get("three_mth_pct"),
            six_mth_pct=row.get("six_mth_pct"),
            one_yr_pct=row.get("one_yr_pct"),
            two_yr_pct=row.get("two_yr_pct"),
            three_yr_pct=row.get("three_yr_pct"),
            five_yr_pct=row.get("five_yr_pct"),
            ten_yr_pct=row.get("ten_yr_pct"),
        )

    # =========================================================================
    # FUNDAMENTALS
    # =========================================================================

    async def get_fundamentals(
        self, ticker: str, period_type: str = "annual", limit: int = 4
    ) -> list[FactSetFundamentals]:
        """Get fundamental data (annual, quarterly, or ltm).

        Args:
            ticker: Ticker symbol
            period_type: 'annual', 'quarterly', or 'ltm'
            limit: Number of periods to return

        Returns:
            List of FactSetFundamentals, most recent first
        """
        fsym_id = await self.get_fsym_id(ticker)
        if not fsym_id:
            return []

        # Get appropriate query and max date query based on period type
        if period_type == "quarterly":
            query = FUNDAMENTALS_QUARTERLY
            max_date_query = FUNDAMENTALS_MAX_DATE_QUARTERLY
            years_back = 2  # ~8 quarters
        elif period_type == "ltm":
            query = FUNDAMENTALS_LTM
            max_date_query = FUNDAMENTALS_MAX_DATE_LTM
            years_back = 2
        else:  # annual
            query = FUNDAMENTALS_ANNUAL
            max_date_query = FUNDAMENTALS_MAX_DATE_ANNUAL
            years_back = limit + 1

        # Get max date for this security
        max_date = await self._client.execute_scalar(max_date_query, {"fsym_id": fsym_id})
        if not max_date:
            return []

        # Calculate start date for filtering
        start_date = date(max_date.year - years_back, 1, 1)

        rows = await self._client.execute_query(
            query, {"fsym_id": fsym_id, "start_date": start_date}
        )

        fundamentals = []
        for row in rows[:limit]:
            fundamentals.append(
                FactSetFundamentals(
                    fsym_id=row["fsym_id"],
                    period_end=row["period_end"],
                    fiscal_year=row.get("fiscal_year"),
                    period_type=row["period_type"],
                    eps_diluted=row.get("eps_diluted"),
                    bps=row.get("bps"),
                    dps=row.get("dps"),
                    roe=row.get("roe"),
                    roa=row.get("roa"),
                    net_margin=row.get("net_margin"),
                    gross_margin=row.get("gross_margin"),
                    operating_margin=row.get("operating_margin"),
                    debt_to_equity=row.get("debt_to_equity"),
                    debt_to_assets=row.get("debt_to_assets"),
                    ev_to_ebitda=row.get("ev_to_ebitda"),
                    price_to_book=row.get("price_to_book"),
                    price_to_sales=row.get("price_to_sales"),
                )
            )

        return fundamentals

    async def get_company_profile(self, ticker: str) -> str | None:
        """Get company business description.

        Args:
            ticker: Ticker symbol

        Returns:
            Company description text, or None if not found
        """
        # Get security info to get fsym_security_id
        security = await self.resolve_ticker(ticker)
        if not security or not security.fsym_security_id:
            return None

        # Use fsym_security_id to find entity (not fsym_id)
        result = await self._client.execute_scalar(
            COMPANY_PROFILE, {"fsym_security_id": security.fsym_security_id}
        )
        return str(result) if result else None

    # =========================================================================
    # CORPORATE ACTIONS
    # =========================================================================

    async def get_corporate_actions(
        self, ticker: str, start_date: date | None = None, limit: int = 20
    ) -> list[FactSetCorporateAction]:
        """Get dividends, splits, and other corporate actions.

        Args:
            ticker: Ticker symbol
            start_date: Start date for filtering, defaults to 2 years ago
            limit: Maximum actions to return

        Returns:
            List of FactSetCorporateAction, most recent first
        """
        fsym_id = await self.get_fsym_id(ticker)
        if not fsym_id:
            return []

        # Get max date and calculate start date
        if start_date is None:
            max_date = await self._client.execute_scalar(
                CORPORATE_ACTIONS_MAX_DATE, {"fsym_id": fsym_id}
            )
            if max_date:
                start_date = date(max_date.year - 2, max_date.month, max_date.day)
            else:
                start_date = date.today() - timedelta(days=730)

        rows = await self._client.execute_query(
            CORPORATE_ACTIONS, {"fsym_id": fsym_id, "start_date": start_date}
        )

        actions = []
        for row in rows[:limit]:
            actions.append(self._row_to_corporate_action(row))

        return actions

    async def get_dividends(self, ticker: str, limit: int = 10) -> list[FactSetCorporateAction]:
        """Get dividend history.

        Args:
            ticker: Ticker symbol
            limit: Maximum dividends to return

        Returns:
            List of dividend events, most recent first
        """
        fsym_id = await self.get_fsym_id(ticker)
        if not fsym_id:
            return []

        # Get start date (5 years back for dividend history)
        start_date = date.today() - timedelta(days=1825)

        rows = await self._client.execute_query(
            DIVIDENDS, {"fsym_id": fsym_id, "start_date": start_date}
        )

        actions = []
        for row in rows[:limit]:
            actions.append(self._row_to_corporate_action(row))

        return actions

    async def get_splits(self, ticker: str, limit: int = 10) -> list[FactSetCorporateAction]:
        """Get stock split history.

        Args:
            ticker: Ticker symbol
            limit: Maximum splits to return

        Returns:
            List of split events, most recent first
        """
        fsym_id = await self.get_fsym_id(ticker)
        if not fsym_id:
            return []

        # Get start date (20 years back for split history to capture older splits)
        start_date = date.today() - timedelta(days=7300)

        rows = await self._client.execute_query(
            SPLITS, {"fsym_id": fsym_id, "start_date": start_date}
        )

        actions = []
        for row in rows[:limit]:
            actions.append(self._row_to_corporate_action(row))

        return actions

    def _row_to_corporate_action(self, row: dict[str, Any]) -> FactSetCorporateAction:
        """Convert database row to FactSetCorporateAction model."""
        event_code = row["ca_event_type"]
        event_type = EVENT_TYPE_MAP.get(event_code, "other")

        # Calculate split factor from adj_shares_to / adj_shares_from
        # For FSP/RSP (Forward/Reverse Split): split_factor = new_term / old_term
        # For BNS (Bonus Issue): split_factor = (old_term + new_term) / old_term
        #   because "6 new for 1 existing" means you end up with 7 total
        split_factor = None
        split_to = row.get("adj_shares_to")
        split_from = row.get("adj_shares_from")
        if split_to and split_from and split_from > 0:
            if event_code == "BNS":
                # Bonus issue: you get new_term NEW shares for every old_term existing
                # Total = old_term + new_term
                split_factor = (split_from + split_to) / split_from
            else:
                # Forward/Reverse split: direct ratio
                split_factor = split_to / split_from

        # Convert div_type_code (int) to string if present
        dividend_type = row.get("dvd_type_desc")
        if dividend_type is not None:
            dividend_type = str(dividend_type)

        return FactSetCorporateAction(
            fsym_id=row["fsym_id"],
            event_type=event_type,
            event_code=event_code,
            effective_date=row["effective_date"],
            record_date=row.get("record_date"),
            pay_date=row.get("pay_date"),
            ex_date=row.get("ex_date"),
            dividend_amount=row.get("gross_dvd_cash"),
            dividend_currency=row.get("gross_dvd_cash_currency"),
            dividend_type=dividend_type,
            split_factor=split_factor or row.get("adj_factor"),
            split_from=split_from,
            split_to=split_to,
        )

    # =========================================================================
    # ADJUSTMENT FACTORS
    # =========================================================================

    async def get_adjustment_factors(
        self, ticker: str, start_date: date, end_date: date
    ) -> dict[date, float]:
        """Get price adjustment factors for date range.

        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            Dict mapping date to adjustment factor
        """
        fsym_id = await self.get_fsym_id(ticker)
        if not fsym_id:
            return {}

        rows = await self._client.execute_query(
            ADJUSTMENT_FACTORS,
            {"fsym_id": fsym_id, "start_date": start_date, "end_date": end_date},
        )

        # Use adj_factor_combined when available (split dates), otherwise div_spl_spin_adj_factor
        result: dict[date, float] = {}
        for row in rows:
            factor = row["adj_factor_combined"]
            if factor is None:
                factor = row["div_spl_spin_adj_factor"]
            if factor is not None:
                result[row["effective_date"]] = factor
        return result

    async def _get_split_period_factors(
        self, fsym_id: str
    ) -> tuple[list[tuple[date, float]], float]:
        """Get cumulative split adjustment factors organized by period.

        Queries all adjustment factors up to today and builds a list of
        (split_date, cumulative_factor) pairs. For each period between splits,
        the factor represents the product of all splits AFTER that period.

        Args:
            fsym_id: FactSet security ID

        Returns:
            Tuple of (period_factors, earliest_factor) where:
            - period_factors: list of (split_date, factor) sorted ascending by date.
              Factor is the cumulative product of splits on or after that date.
            - earliest_factor: factor for dates before the earliest split
              (product of ALL split ratios).
        """
        rows = await self._client.execute_query(
            ADJUSTMENT_FACTORS_FOR_HISTORY,
            {"fsym_id": fsym_id, "end_date": date.today()},
        )

        # Collect split events (only adj_factor_combined entries)
        split_events: list[tuple[date, float]] = []
        for row in rows:
            combined = row["adj_factor_combined"]
            if combined is not None:
                split_events.append((row["effective_date"], combined))

        # Sort by date descending (most recent first)
        split_events.sort(reverse=True)

        # Build cumulative factors going backwards from today
        # For each period, calculate product of all splits AFTER that period
        cumulative = 1.0
        period_factors: list[tuple[date, float]] = []
        for split_date, split_ratio in split_events:
            # Prices ON or AFTER split_date use the current cumulative
            # Prices BEFORE split_date need this split applied
            period_factors.append((split_date, cumulative))
            cumulative *= split_ratio

        # The cumulative now contains product of ALL splits (for pre-earliest-split prices)
        earliest_factor = cumulative

        # Sort by date ascending for lookup
        period_factors.sort()

        return period_factors, earliest_factor

    @staticmethod
    def _adjustment_factor_for_date(
        period_factors: list[tuple[date, float]], earliest_factor: float, target_date: date
    ) -> float:
        """Get the split adjustment factor for a specific date.

        Given precomputed period factors (from _get_split_period_factors),
        returns the cumulative factor that should be applied to an unadjusted
        price on target_date to get the split-adjusted price.

        Args:
            period_factors: list of (split_date, factor) sorted ascending by date
            earliest_factor: factor for dates before all splits
            target_date: the date to get the factor for

        Returns:
            Adjustment factor to multiply with unadjusted price.
        """
        factor = earliest_factor  # Default for prices before all splits
        for split_date, period_factor in period_factors:
            if target_date >= split_date:
                factor = period_factor
        return factor

    async def get_adjusted_price_history(
        self, ticker: str, start_date: date, end_date: date | None = None
    ) -> list[FactSetPrice]:
        """Get historical prices adjusted for splits and dividends.

        FactSet stores UNADJUSTED prices (actual traded prices). The adj_factor_combined
        field contains per-split ratios (NOT cumulative). For example:
        - 2014-06-09: 0.1428 = 1/7 (7:1 split)
        - 2020-08-31: 0.25 = 1/4 (4:1 split)

        To adjust historical prices, we multiply by the product of ALL split ratios
        where split_date > price_date:
        - Post-2020 prices: factor = 1.0 (no splits after)
        - 2014-2020 prices: factor = 0.25 (4:1 split after)
        - Pre-2014 prices: factor = 0.25 × 0.1428 = 0.0357 (both splits after)

        Args:
            ticker: Ticker symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive), defaults to latest available

        Returns:
            List of FactSetPrice objects with adjusted OHLC prices, ordered by date ascending
        """
        fsym_id = await self.get_fsym_id(ticker)
        if not fsym_id:
            return []

        raw_prices = await self.get_price_history(ticker, start_date, end_date)
        if not raw_prices:
            return []

        period_factors, earliest_factor = await self._get_split_period_factors(fsym_id)

        # Apply factors to prices
        adjusted_prices: list[FactSetPrice] = []
        for price in raw_prices:
            factor = self._adjustment_factor_for_date(
                period_factors, earliest_factor, price.price_date
            )

            adjusted_prices.append(
                FactSetPrice(
                    fsym_id=price.fsym_id,
                    price_date=price.price_date,
                    open=price.open * factor if price.open is not None else None,
                    high=price.high * factor if price.high is not None else None,
                    low=price.low * factor if price.low is not None else None,
                    close=price.close * factor,
                    volume=price.volume,
                    is_adjusted=True,
                    one_day_pct=price.one_day_pct,
                    wtd_pct=price.wtd_pct,
                    mtd_pct=price.mtd_pct,
                    qtd_pct=price.qtd_pct,
                    ytd_pct=price.ytd_pct,
                    one_mth_pct=price.one_mth_pct,
                    three_mth_pct=price.three_mth_pct,
                    six_mth_pct=price.six_mth_pct,
                    one_yr_pct=price.one_yr_pct,
                    two_yr_pct=price.two_yr_pct,
                    three_yr_pct=price.three_yr_pct,
                    five_yr_pct=price.five_yr_pct,
                    ten_yr_pct=price.ten_yr_pct,
                )
            )

        return adjusted_prices

    # =========================================================================
    # SHARES OUTSTANDING
    # =========================================================================

    async def get_shares_outstanding(self, ticker: str) -> FactSetSharesOutstanding | None:
        """Get current shares outstanding (split-adjusted).

        Note: adj_shares_outstanding is retroactively adjusted for ALL subsequent
        stock splits. The returned value reflects the current split-adjusted count,
        not the actual share count at any historical point. For historical market
        cap, use get_market_cap() with a price_date parameter instead.

        Args:
            ticker: Ticker symbol

        Returns:
            FactSetSharesOutstanding if found, None otherwise
        """
        # First, get the security to find fsym_security_id
        security = await self.resolve_ticker(ticker)
        if not security or not security.fsym_security_id:
            # Try to get fsym_security_id directly
            fsym_id = await self.get_fsym_id(ticker)
            if not fsym_id:
                return None

            sec_id_result = await self._client.execute_scalar(GET_SECURITY_ID, {"fsym_id": fsym_id})
            if not sec_id_result:
                return None
            fsym_security_id = sec_id_result
        else:
            fsym_security_id = security.fsym_security_id

        rows = await self._client.execute_query(
            SHARES_OUTSTANDING_CURRENT, {"fsym_security_id": fsym_security_id}
        )

        if not rows:
            return None

        row = rows[0]
        # Shares are stored in millions - convert to actual count
        shares_in_millions = row["adj_shares_outstanding"]
        shares_actual = shares_in_millions * 1_000_000

        return FactSetSharesOutstanding(
            fsym_id=security.fsym_id if security else "",
            fsym_security_id=fsym_security_id,
            report_date=row["report_date"],
            shares_outstanding=shares_actual,
            shares_outstanding_raw=shares_in_millions,
            adr_ratio=row.get("adr_share_ratio"),
            has_adr=row.get("hasadr_flag") == "Y",
        )

    async def _get_shares_as_of(self, fsym_security_id: str, target_date: date) -> float | None:
        """Get adj_shares_outstanding (in millions) for most recent report on or before target_date."""
        rows = await self._client.execute_query(
            SHARES_OUTSTANDING_AS_OF_DATE,
            {"fsym_security_id": fsym_security_id, "target_date": target_date},
        )
        if not rows:
            return None
        value: float = rows[0]["adj_shares_outstanding"]
        return value

    async def get_market_cap(self, ticker: str, price_date: date | None = None) -> float | None:
        """Calculate market capitalization.

        For the current date (price_date=None): uses unadjusted price × current
        adj_shares_outstanding. This is correct because current adj_shares equals
        the actual share count (no future splits to inflate it).

        For historical dates: uses adjusted_price × adj_shares_outstanding. The
        split factors cancel out — adj_shares is inflated by 1/F for each
        subsequent split, while adjusted_price is deflated by F, so
        (price × F) × (shares / F) = price × shares = true market cap.

        Args:
            ticker: Ticker symbol
            price_date: Date for historical market cap (None = current)

        Returns:
            Market cap in dollars, or None if data unavailable
        """
        if price_date is None:
            # Current market cap: unadjusted price × current adj_shares (correct
            # because no future splits exist to inflate adj_shares)
            shares = await self.get_shares_outstanding(ticker)
            if not shares:
                return None

            price = await self.get_price(ticker)
            if not price:
                return None

            return shares.shares_outstanding * price.close

        # Historical market cap: adjusted_price × adj_shares
        # Split factors cancel: (price × F) × (shares / F) = price × shares
        security = await self.resolve_ticker(ticker)
        if not security or not security.fsym_security_id:
            return None

        unadj_price = await self.get_price(ticker, price_date)
        if not unadj_price:
            return None

        period_factors, earliest_factor = await self._get_split_period_factors(security.fsym_id)
        adj_factor = self._adjustment_factor_for_date(period_factors, earliest_factor, price_date)
        adjusted_price = unadj_price.close * adj_factor

        adj_shares_millions = await self._get_shares_as_of(security.fsym_security_id, price_date)
        if adj_shares_millions is None:
            return None

        return adjusted_price * adj_shares_millions * 1_000_000

    # =========================================================================
    # CLEANUP
    # =========================================================================

    async def close(self) -> None:
        """Close database connection."""
        await self._client.close()


class FactSetWatchlistAdapter:
    """Adapts FactSetProvider to the WatchlistDataProvider protocol."""

    def __init__(self, provider: FactSetProvider) -> None:
        self._provider = provider

    async def resolve_company(self, ticker: str) -> CompanyInfo | None:
        security = await self._provider.resolve_ticker(ticker)
        return CompanyInfo(name=security.name) if security else None

    async def get_market_cap(self, ticker: str) -> float | None:
        return await self._provider.get_market_cap(ticker)

    async def get_fundamentals(self, ticker: str) -> FundamentalsSnapshot | None:
        results = await self._provider.get_fundamentals(ticker, "ltm", 1)
        if not results:
            return None
        f = results[0]
        return FundamentalsSnapshot(
            eps_diluted=f.eps_diluted,
            price_to_book=f.price_to_book,
            price_to_sales=f.price_to_sales,
            ev_to_ebitda=f.ev_to_ebitda,
            roe=f.roe,
            net_margin=f.net_margin,
            gross_margin=f.gross_margin,
            debt_to_equity=f.debt_to_equity,
            period_type=f.period_type,
            period_end=f.period_end,
        )

    async def get_price(self, ticker: str) -> PriceSnapshot | None:
        price = await self._provider.get_price(ticker)
        if not price:
            return None
        return PriceSnapshot(
            one_day_pct=price.one_day_pct,
            one_mth_pct=price.one_mth_pct,
            price_date=price.price_date,
        )

    async def close(self) -> None:
        await self._provider.close()
