"""Event source fetchers — pulls structured calendar events from external APIs.

Structured sources (no LLM needed):
  - FRED release dates → US macro events (CPI, PCE, NFP, etc.)
  - NASDAQ earnings → upcoming earnings (>$10B market cap)
  - FOMC calendar → Fed meeting and minutes release dates
  - SEC EDGAR → 13F-HR filings from tracked hedge funds
"""

from __future__ import annotations

import asyncio
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import yaml

from synesis.core.logging import get_logger
from synesis.processing.events.models import CalendarEvent

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from synesis.providers.fred import FREDClient
    from synesis.providers.nasdaq import NasdaqClient
    from synesis.providers.sec_edgar.client import SECEdgarClient

logger = get_logger(__name__)

# FRED release IDs for priority US macro data
# Maps FRED release_id -> event name
FRED_MACRO_RELEASES: dict[int, str] = {
    10: "CPI",  # Consumer Price Index
    46: "GDP",  # GDP Advance Estimate
    47: "GDP (2nd)",  # GDP Second Estimate
    48: "GDP (3rd)",  # GDP Third Estimate
    50: "Employment Situation (NFP)",
    21: "PCE / Personal Income",
    53: "PPI",
    # FOMC is handled by fetch_fomc_events() — a dedicated structured source
}

_SOURCES_PATH = Path(__file__).parent / "sources.yaml"

MIN_EARNINGS_MARKET_CAP = 10_000_000_000  # $10B

_TIME_LABEL: dict[str, str] = {
    "pre-market": "PM",
    "after-hours": "AH",
    "during-market": "DM",
}


def load_sources_config() -> dict[str, Any]:
    """Load the sources.yaml configuration."""
    with open(_SOURCES_PATH) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


# Cached registry (loaded once per process)
_hedge_fund_registry: dict[str, str] | None = None
_hedge_fund_top_tier: set[str] | None = None


def load_hedge_fund_registry() -> tuple[dict[str, str], set[str]]:
    """Load the 13F hedge fund registry from sources.yaml.

    Returns:
        (cik_to_name dict, top_tier_cik set)
    """
    global _hedge_fund_registry, _hedge_fund_top_tier
    if _hedge_fund_registry is not None and _hedge_fund_top_tier is not None:
        return _hedge_fund_registry, _hedge_fund_top_tier

    config = load_sources_config()
    funds = config.get("hedge_fund_13f", [])

    registry: dict[str, str] = {}
    top_tier: set[str] = set()
    for fund in funds:
        cik = fund["cik"]
        registry[cik] = fund["name"]
        if fund.get("tier") == "top":
            top_tier.add(cik)

    _hedge_fund_registry = registry
    _hedge_fund_top_tier = top_tier
    return registry, top_tier


# ---------------------------------------------------------------------------
# FRED release dates -> CalendarEvent
# ---------------------------------------------------------------------------


async def fetch_fred_macro_events(
    fred: FREDClient,
    days_ahead: int = 30,
) -> list[CalendarEvent]:
    """Fetch upcoming US macro data release dates from FRED."""
    today = date.today()
    cutoff = today + timedelta(days=days_ahead)
    events: list[CalendarEvent] = []

    for release_id, name in FRED_MACRO_RELEASES.items():
        try:
            dates = await fred.get_release_dates(release_id, include_future=True, limit=10)
            for rd in dates:
                if today <= rd.date <= cutoff:
                    events.append(
                        CalendarEvent(
                            title=f"{name} Release",
                            description=f"US {name} data release ({rd.release_name})",
                            event_date=rd.date,
                            category="economic_data",
                            region=["US"],
                            source_urls=[f"https://fred.stlouisfed.org/release?rid={release_id}"],
                        )
                    )
        except Exception:
            logger.exception("Failed to fetch FRED release dates", release_id=release_id)

    logger.info("FRED macro events fetched", count=len(events))
    return events


# ---------------------------------------------------------------------------
# NASDAQ earnings -> CalendarEvent
# ---------------------------------------------------------------------------


async def fetch_nasdaq_earnings_events(
    nasdaq: NasdaqClient,
    days_ahead: int = 14,
) -> list[CalendarEvent]:
    """Fetch upcoming earnings from NASDAQ, filtered to >$10B market cap."""
    today = date.today()
    targets = [today + timedelta(days=i) for i in range(days_ahead)]
    sem = asyncio.Semaphore(5)

    async def _fetch(target_date: date) -> list[CalendarEvent]:
        async with sem:
            try:
                earnings = await nasdaq.get_earnings_by_date(target_date)
            except Exception:
                logger.exception("NASDAQ earnings fetch failed", date=target_date.isoformat())
                return []

            result: list[CalendarEvent] = []
            for e in earnings:
                if e.market_cap and e.market_cap >= MIN_EARNINGS_MARKET_CAP:
                    time_str = f" ({e.time})" if e.time else ""
                    eps_str = (
                        f", EPS est: ${e.eps_forecast:.2f}" if e.eps_forecast is not None else ""
                    )
                    try:
                        result.append(
                            CalendarEvent(
                                title=f"Earnings: {e.company_name} ({e.ticker})",
                                description=f"{e.ticker} Q{e.fiscal_quarter} earnings{time_str}{eps_str}",
                                event_date=e.earnings_date,
                                category="earnings",
                                region=["US"],
                                tickers=[e.ticker],
                                source_urls=["https://www.nasdaq.com/market-activity/earnings"],
                                time_label=_TIME_LABEL.get(e.time or "") or None,
                            )
                        )
                    except Exception:
                        logger.warning(
                            "Skipping malformed NASDAQ earnings entry",
                            ticker=e.ticker,
                            exc_info=True,
                        )
            return result

    batches = await asyncio.gather(*(_fetch(d) for d in targets))
    events = [ev for batch in batches for ev in batch]
    logger.info("NASDAQ earnings events fetched", count=len(events))
    return events


# ---------------------------------------------------------------------------
# FOMC calendar -> CalendarEvent
# ---------------------------------------------------------------------------

_MONTH_NAMES = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


async def fetch_fomc_events(days_ahead: int = 180) -> list[CalendarEvent]:
    """Fetch FOMC meeting dates directly from the Fed calendar (no LLM)."""
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    today = date.today()
    cutoff = today + timedelta(days=days_ahead)

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        html = resp.text

    # Split HTML into per-year blocks on "YYYY FOMC Meetings"
    year_chunks = re.split(r"(\d{4})\s+FOMC\s+Meetings", html)
    # [prefix, "2026", block2026, "2025", block2025, ...]

    if len(year_chunks) < 3:
        logger.error(
            "FOMC HTML parse yielded no year blocks — Fed page structure may have changed",
            url=url,
            chunk_count=len(year_chunks),
        )

    events: list[CalendarEvent] = []
    for i in range(1, len(year_chunks), 2):
        year = int(year_chunks[i])
        block = year_chunks[i + 1] if i + 1 < len(year_chunks) else ""

        triples = re.findall(
            r"fomc-meeting__month[^>]+><strong>(\w+)</strong>"
            r".*?fomc-meeting__date[^>]+>([\d\-]+)\*?<"
            r".*?fomc-meeting__minutes[^>]+>(.*?)</div>",
            block,
            re.DOTALL,
        )
        for month_name, date_range, minutes_content in triples:
            month_num = _MONTH_NAMES.get(month_name)
            if not month_num:
                continue
            parts = date_range.split("-")
            try:
                last_day = int(parts[-1])
                first_day = int(parts[0])
            except (ValueError, IndexError):
                logger.warning(
                    "FOMC date_range parse failed", month=month_name, date_range=date_range
                )
                continue

            # Handle month straddle (e.g. "31-1" → decision day in next month)
            if len(parts) == 2 and last_day < first_day:
                next_month = month_num % 12 + 1
                next_year = year + 1 if month_num == 12 else year
                try:
                    event_date = date(next_year, next_month, last_day)
                except ValueError:
                    logger.warning(
                        "FOMC date parse failed", month=month_name, date_range=date_range
                    )
                    continue
            else:
                try:
                    event_date = date(year, month_num, last_day)
                except ValueError:
                    logger.warning(
                        "FOMC date parse failed", month=month_name, date_range=date_range
                    )
                    continue

            if today <= event_date <= cutoff:
                events.append(
                    CalendarEvent(
                        title="FOMC Rate Decision",
                        description=f"Federal Reserve FOMC meeting — rate decision ({month_name} {date_range})",
                        event_date=event_date,
                        category="fed",
                        region=["US"],
                        source_urls=[url],
                    )
                )

            # Parse minutes release date if officially scheduled
            minutes_match = re.search(r"Released (\w+ \d+, \d{4})", minutes_content)
            if minutes_match:
                try:
                    minutes_date = datetime.strptime(minutes_match.group(1), "%B %d, %Y").date()
                    if today <= minutes_date <= cutoff:
                        events.append(
                            CalendarEvent(
                                title="FOMC Minutes Release",
                                description=f"Federal Reserve FOMC minutes released — {month_name} {date_range} meeting",
                                event_date=minutes_date,
                                category="fed",
                                region=["US"],
                                source_urls=[url],
                            )
                        )
                except ValueError:
                    logger.warning("FOMC minutes date parse failed", raw=minutes_match.group(1))

    logger.info("FOMC events fetched", count=len(events))
    return events


# ---------------------------------------------------------------------------
# SEC 13F hedge fund filings -> CalendarEvent
# ---------------------------------------------------------------------------


async def fetch_13f_events(
    sec_edgar: SECEdgarClient,
    redis: Redis,
) -> list[CalendarEvent]:
    """Detect new 13F-HR filings from tracked hedge funds and build events."""
    from synesis.core.constants import SEC_13F_SEEN_TTL

    HEDGE_FUND_13F, HEDGE_FUND_13F_TOP_TIER = load_hedge_fund_registry()

    events: list[CalendarEvent] = []
    sem = asyncio.Semaphore(2)

    async def _check_fund(cik: str, fund_name: str) -> CalendarEvent | None:
        async with sem:
            try:
                filings = await sec_edgar.get_13f_filings(cik, limit=2)
            except Exception:
                logger.exception("13F filing fetch failed", cik=cik, fund=fund_name)
                return None

            if not filings:
                return None

            filing = filings[0]
            seen_key = f"synesis:event_radar:13f_seen:{filing.accession_number}"
            try:
                if await redis.get(seen_key):
                    return None
            except Exception:
                logger.error(
                    "Redis unavailable for 13F seen-key check — fund skipped to prevent duplicate",
                    cik=cik,
                    fund=fund_name,
                )
                return None

            # New filing — get QoQ diff
            diff = None
            try:
                diff = await sec_edgar.compare_13f_quarters(cik, fund_name)
            except Exception:
                logger.exception("13F QoQ comparison failed", cik=cik, fund=fund_name)

            # Build description
            desc_parts = [f"New 13F-HR filing by {fund_name}."]
            tickers: list[str] = []

            if diff:
                total_curr = diff.get("total_value_current", 0)
                total_prev = diff.get("total_value_previous", 0)
                if total_curr and total_prev:
                    change_pct = (total_curr - total_prev) / total_prev * 100
                    desc_parts.append(
                        f"Portfolio: ${total_curr / 1000:.1f}M "
                        f"({'+' if change_pct >= 0 else ''}{change_pct:.1f}% QoQ)"
                    )

                new_pos = diff.get("new_positions", [])
                if new_pos:
                    names = [p["name_of_issuer"] for p in new_pos[:5]]
                    desc_parts.append(f"New: {', '.join(names)}")

                exited = diff.get("exited_positions", [])
                if exited:
                    names = [p["name_of_issuer"] for p in exited[:5]]
                    desc_parts.append(f"Exited: {', '.join(names)}")

                increased = diff.get("increased", [])
                if increased:
                    names = [
                        f"{p['name_of_issuer']} (+{p.get('change_pct', 0):.0f}%)"
                        for p in increased[:3]
                    ]
                    desc_parts.append(f"Increased: {', '.join(names)}")

                decreased = diff.get("decreased", [])
                if decreased:
                    names = [
                        f"{p['name_of_issuer']} ({p.get('change_pct', 0):.0f}%)"
                        for p in decreased[:3]
                    ]
                    desc_parts.append(f"Decreased: {', '.join(names)}")

            # Determine quarter from report date
            report_date_str = filing.items  # We stored reportDate here
            try:
                rd = date.fromisoformat(report_date_str)
                quarter = (rd.month - 1) // 3 + 1
                year = rd.year
                title = f"13F Filing: {fund_name} (Q{quarter} {year})"
            except (ValueError, TypeError):
                title = f"13F Filing: {fund_name}"

            # Mark as seen (best-effort; DB ON CONFLICT acts as last-resort dedup on Redis failure)
            try:
                await redis.set(seen_key, "1", ex=SEC_13F_SEEN_TTL)
            except Exception:
                logger.warning(
                    "Failed to mark 13F filing as seen in Redis — may reprocess on next run",
                    cik=cik,
                    accession=filing.accession_number,
                )

            return CalendarEvent(
                title=title,
                description="\n".join(desc_parts),
                event_date=filing.filed_date,
                category="13f_filing",
                region=["US"],
                tickers=tickers[:10],
                source_urls=[filing.url] if filing.url else [],
            )

    tasks = [_check_fund(cik, name) for cik, name in HEDGE_FUND_13F.items()]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in raw_results:
        if isinstance(result, BaseException):
            logger.error(
                "13F fund check failed",
                error=str(result),
                error_type=type(result).__name__,
            )
        elif result is not None:
            events.append(result)

    logger.info("13F hedge fund events fetched", count=len(events))
    return events
