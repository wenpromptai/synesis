"""Event source crawler — fetches events from structured APIs and curated web sources.

Structured sources (no LLM needed):
  - FRED release dates → US macro events (CPI, PCE, NFP, FOMC, etc.)
  - NASDAQ earnings → upcoming earnings (>$10B market cap)

Curated sources (need LLM extraction):
  - Crawl4AI crawls curated URLs from sources.yaml
  - Frequency tracking in Redis to avoid redundant crawls
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from synesis.core.logging import get_logger
from synesis.processing.events.models import CalendarEvent

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from synesis.providers.crawler.crawl4ai import Crawl4AICrawlerProvider, CrawlResult
    from synesis.providers.fred import FREDClient
    from synesis.providers.nasdaq import NasdaqClient
    from synesis.providers.sec_edgar.client import SECEdgarClient

logger = get_logger(__name__)

# Redis key prefix for crawl frequency tracking
CRAWL_PREFIX = "synesis:event_radar:last_crawled"

# FRED release IDs for priority US macro data
# Maps FRED release_id -> (event name, importance)
FRED_MACRO_RELEASES: dict[int, tuple[str, int]] = {
    10: ("CPI", 9),  # Consumer Price Index
    46: ("GDP", 9),  # GDP Advance Estimate
    47: ("GDP (2nd)", 7),  # GDP Second Estimate
    48: ("GDP (3rd)", 7),  # GDP Third Estimate
    50: ("Employment Situation (NFP)", 9),
    21: ("PCE / Personal Income", 8),
    53: ("PPI", 7),
    # FOMC is not a FRED release — it's on the Fed calendar curated source
}

_SOURCES_PATH = Path(__file__).parent / "sources.yaml"

# Frequency -> minimum hours between crawls
_FREQUENCY_HOURS: dict[str, int] = {
    "daily": 20,
    "weekly": 144,  # 6 days
    "monthly": 600,  # 25 days
}

MIN_EARNINGS_MARKET_CAP = 10_000_000_000  # $10B


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
# Structured: FRED release dates -> CalendarEvent
# ---------------------------------------------------------------------------


async def fetch_fred_macro_events(
    fred: FREDClient,
    days_ahead: int = 30,
) -> list[CalendarEvent]:
    """Fetch upcoming US macro data release dates from FRED."""
    today = date.today()
    cutoff = today + timedelta(days=days_ahead)
    events: list[CalendarEvent] = []

    for release_id, (name, importance) in FRED_MACRO_RELEASES.items():
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
                            importance=importance,
                            confidence=0.99,
                            source_urls=[f"https://fred.stlouisfed.org/release?rid={release_id}"],
                        )
                    )
        except Exception:
            logger.exception("Failed to fetch FRED release dates", release_id=release_id)

    logger.info("FRED macro events fetched", count=len(events))
    return events


# ---------------------------------------------------------------------------
# Structured: NASDAQ earnings -> CalendarEvent
# ---------------------------------------------------------------------------


async def fetch_nasdaq_earnings_events(
    nasdaq: NasdaqClient,
    days_ahead: int = 14,
) -> list[CalendarEvent]:
    """Fetch upcoming earnings from NASDAQ, filtered to >$500M market cap."""
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
                    result.append(
                        CalendarEvent(
                            title=f"Earnings: {e.company_name} ({e.ticker})",
                            description=f"{e.ticker} Q{e.fiscal_quarter} earnings{time_str}{eps_str}",
                            event_date=e.earnings_date,
                            category="earnings",
                            region=["US"],
                            tickers=[e.ticker],
                            importance=_earnings_importance(e.market_cap),
                            confidence=0.95,
                            source_urls=["https://www.nasdaq.com/market-activity/earnings"],
                        )
                    )
            return result

    batches = await asyncio.gather(*(_fetch(d) for d in targets))
    events = [ev for batch in batches for ev in batch]
    logger.info("NASDAQ earnings events fetched", count=len(events))
    return events


def _earnings_importance(market_cap: float) -> int:
    """Score earnings importance by market cap."""
    if market_cap >= 200_000_000_000:  # $200B+ mega-cap
        return 8
    if market_cap >= 50_000_000_000:  # $50B+
        return 7
    return 6


# ---------------------------------------------------------------------------
# Curated: Crawl4AI web crawling with frequency tracking
# ---------------------------------------------------------------------------


async def crawl_curated_sources(
    crawler: Crawl4AICrawlerProvider,
    redis: Redis,
    force: bool = False,
) -> list[tuple[dict[str, Any], CrawlResult]]:
    """Crawl curated sources that are due based on their frequency.

    Returns list of (source_config, crawl_result) tuples for sources that
    were successfully crawled and contain content.
    """
    config = load_sources_config()
    sources = config.get("curated_sources", [])

    due_sources: list[dict[str, Any]] = []
    for source in sources:
        if force or await _is_due(redis, source):
            due_sources.append(source)

    if not due_sources:
        logger.debug("No curated sources due for crawling")
        return []

    logger.info("Crawling curated sources", due=len(due_sources), total=len(sources))

    results: list[tuple[dict[str, Any], CrawlResult]] = []
    # Crawl in batches of 3 to avoid overwhelming Crawl4AI
    for i in range(0, len(due_sources), 3):
        batch = due_sources[i : i + 3]
        urls = [s["url"] for s in batch]
        crawl_results = await crawler.crawl_many(urls)

        for source, result in zip(batch, crawl_results):
            # Mark as crawled regardless of success
            await _mark_crawled(redis, source)

            if result.success and result.markdown.strip():
                results.append((source, result))
            else:
                logger.debug(
                    "Curated crawl failed or empty",
                    source=source["name"],
                    error=result.error,
                )

    logger.info("Curated sources crawled", success=len(results), attempted=len(due_sources))
    return results


async def _is_due(redis: Redis, source: dict[str, Any]) -> bool:
    """Check if a curated source is due for crawling based on its frequency."""
    key = f"{CRAWL_PREFIX}:{source['name']}"
    last = await redis.get(key)
    if last is None:
        return True

    import time

    last_ts = float(last)
    freq = source.get("frequency", "daily")
    min_hours = _FREQUENCY_HOURS.get(freq, 20)
    return (time.time() - last_ts) >= (min_hours * 3600)


async def _mark_crawled(redis: Redis, source: dict[str, Any]) -> None:
    """Mark a source as crawled with current timestamp."""
    import time

    key = f"{CRAWL_PREFIX}:{source['name']}"
    await redis.set(key, str(time.time()), ex=86400 * 30)  # Expire after 30 days


# ---------------------------------------------------------------------------
# Structured: SEC 13F hedge fund filings -> CalendarEvent
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
            if await redis.get(seen_key):
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

                # No CUSIP→ticker map — leave tickers empty for 13F events
                # Issuer names stay in the filing brief description only

            # Determine quarter from report date
            report_date_str = filing.items  # We stored reportDate here
            try:
                rd = date.fromisoformat(report_date_str)
                quarter = (rd.month - 1) // 3 + 1
                year = rd.year
                title = f"13F Filing: {fund_name} (Q{quarter} {year})"
            except (ValueError, TypeError):
                title = f"13F Filing: {fund_name}"

            importance = 8 if cik in HEDGE_FUND_13F_TOP_TIER else 7

            # Mark as seen
            await redis.set(seen_key, "1", ex=SEC_13F_SEEN_TTL)

            return CalendarEvent(
                title=title,
                description="\n".join(desc_parts),
                event_date=filing.filed_date,
                category="13f_filing",
                region=["US"],
                tickers=tickers[:10],
                importance=importance,
                confidence=0.99,
                source_urls=[filing.url] if filing.url else [],
            )

    tasks = [_check_fund(cik, name) for cik, name in HEDGE_FUND_13F.items()]
    results = await asyncio.gather(*tasks)

    for result in results:
        if result is not None:
            events.append(result)

    logger.info("13F hedge fund events fetched", count=len(events))
    return events
