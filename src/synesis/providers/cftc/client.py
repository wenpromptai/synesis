"""CFTC COT data client using the Socrata SODA API.

API: https://publicreporting.cftc.gov
No API key required. Data released Fridays at 3:30 PM ET (Tuesday's positions).
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import httpx

from synesis.core.logging import get_logger
from synesis.providers.cftc.models import COTPositioning, COTReport

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)

SOCRATA_BASE = "https://publicreporting.cftc.gov/resource"

# Traders in Financial Futures (TFF) — financial contracts
TFF_DATASET = "gpe5-46if"
# Disaggregated Futures Only — commodity contracts
DISAGG_DATASET = "kh3c-gbw2"

CACHE_PREFIX = "synesis:cftc"
CACHE_TTL = 3600 * 6  # 6 hours — data updates weekly

# Contract code mapping: our ticker → CFTC code + which report
_CONTRACTS: dict[str, tuple[str, str]] = {
    # TFF report (financial)
    "ES": ("13874A", TFF_DATASET),  # E-mini S&P 500
    "NQ": ("209742", TFF_DATASET),  # E-mini Nasdaq-100
    "ZN": ("043602", TFF_DATASET),  # 10-Year Treasury Note
    "ZB": ("020601", TFF_DATASET),  # 30-Year Treasury Bond
    "DX": ("098662", TFF_DATASET),  # US Dollar Index
    # Disaggregated report (commodities)
    "GC": ("088691", DISAGG_DATASET),  # Gold
    "CL": ("067651", DISAGG_DATASET),  # WTI Crude Oil
}

# Default contracts to fetch for macro analysis
DEFAULT_TICKERS = ["ES", "NQ", "ZN", "GC", "CL", "DX"]


class CFTCClient:
    """Client for CFTC Commitment of Traders data."""

    def __init__(self, redis: Redis | None = None) -> None:
        self._redis = redis

    async def get_positioning(
        self,
        ticker: str,
        weeks: int = 104,
    ) -> list[COTReport]:
        """Fetch COT positioning history for a futures contract.

        Args:
            ticker: Our ticker symbol (ES, NQ, ZN, ZB, DX, GC, CL).
            weeks: Number of weeks of history.

        Returns:
            List of COTReport sorted by date ascending, with percentile/z-score computed.
        """
        contract = _CONTRACTS.get(ticker)
        if not contract:
            raise ValueError(f"Unknown CFTC contract ticker: {ticker}")

        code, dataset = contract
        is_tff = dataset == TFF_DATASET

        # Check cache
        cache_key = f"{CACHE_PREFIX}:{ticker}:{weeks}"
        if self._redis:
            cached = await self._redis.get(cache_key)
            if cached:
                import orjson

                rows = orjson.loads(cached)
                return [COTReport.model_validate(r) for r in rows]

        # Fetch from Socrata API
        reports = await self._fetch_from_socrata(ticker, code, dataset, weeks, is_tff)

        # Compute percentiles and z-scores
        _add_analytics(reports)

        # Cache
        if self._redis and reports:
            import orjson

            data = [r.model_dump(mode="json") for r in reports]
            await self._redis.set(cache_key, orjson.dumps(data), ex=CACHE_TTL)

        return reports

    async def get_latest(
        self,
        tickers: list[str] | None = None,
    ) -> dict[str, COTReport]:
        """Fetch latest COT report for multiple contracts.

        Args:
            tickers: List of ticker symbols. Defaults to DEFAULT_TICKERS.

        Returns:
            Dict of ticker → latest COTReport (with 52-week percentiles).
        """
        tickers = tickers or DEFAULT_TICKERS
        result: dict[str, COTReport] = {}

        for ticker in tickers:
            try:
                reports = await self.get_positioning(ticker, weeks=53)
                if reports:
                    result[ticker] = reports[-1]
            except Exception:
                logger.warning("CFTC fetch failed", ticker=ticker, exc_info=True)

        return result

    async def _fetch_from_socrata(
        self,
        ticker: str,
        code: str,
        dataset: str,
        weeks: int,
        is_tff: bool,
    ) -> list[COTReport]:
        """Fetch raw data from the Socrata SODA API."""
        url = f"{SOCRATA_BASE}/{dataset}.json"

        params = {
            "$where": f"cftc_contract_market_code='{code}'",
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$limit": str(weeks),
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            rows = resp.json()

        if not rows:
            logger.warning("No CFTC data returned", ticker=ticker, code=code)
            return []

        reports = []
        for row in reversed(rows):  # oldest first
            try:
                report = _parse_row(row, ticker, code, is_tff)
                reports.append(report)
            except Exception:
                logger.warning("Failed to parse COT row", ticker=ticker, exc_info=True)

        logger.debug("CFTC data fetched", ticker=ticker, rows=len(reports))
        return reports


def _safe_int(val: Any) -> int:
    """Convert a value to int, defaulting to 0."""
    if val is None:
        return 0
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def _safe_float(val: Any) -> float:
    """Convert a value to float, defaulting to 0.0."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _parse_row(row: dict[str, Any], ticker: str, code: str, is_tff: bool) -> COTReport:
    """Parse a Socrata API row into a COTReport."""
    report_date = date.fromisoformat(row["report_date_as_yyyy_mm_dd"][:10])
    contract_name = row.get("market_and_exchange_names", "")

    if is_tff:
        # TFF column names (no _all suffix for lev/asset_mgr)
        lev = COTPositioning(
            report_date=report_date,
            long_contracts=_safe_int(row.get("lev_money_positions_long")),
            short_contracts=_safe_int(row.get("lev_money_positions_short")),
            spread_contracts=_safe_int(row.get("lev_money_positions_spread")),
            change_long=_safe_int(row.get("change_in_lev_money_long")),
            change_short=_safe_int(row.get("change_in_lev_money_short")),
            pct_of_oi_long=_safe_float(row.get("pct_of_oi_lev_money_long")),
            pct_of_oi_short=_safe_float(row.get("pct_of_oi_lev_money_short")),
        )
        lev.net_contracts = lev.long_contracts - lev.short_contracts

        asset_mgr = COTPositioning(
            report_date=report_date,
            long_contracts=_safe_int(row.get("asset_mgr_positions_long")),
            short_contracts=_safe_int(row.get("asset_mgr_positions_short")),
        )
        asset_mgr.net_contracts = asset_mgr.long_contracts - asset_mgr.short_contracts

        dealer = COTPositioning(
            report_date=report_date,
            long_contracts=_safe_int(row.get("dealer_positions_long_all")),
            short_contracts=_safe_int(row.get("dealer_positions_short_all")),
        )
        dealer.net_contracts = dealer.long_contracts - dealer.short_contracts
    else:
        # Disaggregated: m_money (Managed Money) ≈ Leveraged Funds, Swap ≈ Dealer
        lev = COTPositioning(
            report_date=report_date,
            long_contracts=_safe_int(row.get("m_money_positions_long_all")),
            short_contracts=_safe_int(row.get("m_money_positions_short_all")),
            spread_contracts=_safe_int(row.get("m_money_positions_spread")),
            change_long=_safe_int(row.get("change_in_m_money_long_all")),
            change_short=_safe_int(row.get("change_in_m_money_short_all")),
            pct_of_oi_long=_safe_float(row.get("pct_of_oi_m_money_long_all")),
            pct_of_oi_short=_safe_float(row.get("pct_of_oi_m_money_short_all")),
        )
        lev.net_contracts = lev.long_contracts - lev.short_contracts

        asset_mgr = COTPositioning(report_date=report_date)
        dealer = COTPositioning(
            report_date=report_date,
            long_contracts=_safe_int(row.get("swap_positions_long_all")),
            short_contracts=_safe_int(row.get("swap_positions_short_all")),
        )
        dealer.net_contracts = dealer.long_contracts - dealer.short_contracts

    return COTReport(
        contract_name=contract_name,
        contract_code=code,
        ticker=ticker,
        report_date=report_date,
        open_interest=_safe_int(row.get("open_interest_all")),
        leveraged_funds=lev,
        asset_managers=asset_mgr,
        dealers=dealer,
    )


def _add_analytics(reports: list[COTReport], lookback: int = 52) -> None:
    """Add 52-week percentile and z-score to each report in-place."""
    if len(reports) < lookback:
        return

    nets = [r.leveraged_funds.net_contracts for r in reports]

    for i in range(lookback, len(reports)):
        window = nets[i - lookback : i + 1]
        current = nets[i]
        w_min = min(window)
        w_max = max(window)
        rng = w_max - w_min

        if rng > 0:
            reports[i].lev_funds_net_pctl = round((current - w_min) / rng * 100, 1)

        mean = sum(window) / len(window)
        std = (sum((x - mean) ** 2 for x in window) / len(window)) ** 0.5
        if std > 0:
            reports[i].lev_funds_net_zscore = round((current - mean) / std, 2)
