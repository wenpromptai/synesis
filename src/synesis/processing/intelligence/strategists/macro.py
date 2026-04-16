"""MacroStrategist — assesses market regime from multi-source context.

Pre-fetches (all in parallel):
  1. FRED economic indicators (VIX, yields, credit spreads, NFCI, Sahm Rule, claims)
  2. Benchmark quotes via yfinance (indices, bonds, commodities incl. copper, dollar, credit, sectors)
  3. Yesterday's calendar events enriched with outcomes (earnings 8-K, FRED actuals,
     Fed minutes/statements, 13F position changes)
  4. Upcoming catalysts from DB (next 7 days)
  5. CFTC COT futures positioning (hedge fund net positioning on ES, NQ, ZN, GC, CL, DX)
  6. HMM regime model (statistical regime probabilities from 10yr weekly cross-asset data)

Computes derived signals:
  - Cross-asset signals (HYG/LQD ratio, copper/gold, credit spread direction, NFCI reading)
  - Deterministic regime prior (threshold-based: VIX, NFCI, HY OAS, SPY vs 200d, yield curve, Sahm)

The LLM receives all context + both regime priors (threshold + HMM) and synthesizes
into a regime assessment, thematic tilts, positioning signals, and risks.

Web search (budget: 5) is reserved for what data can't tell you: Fed rhetoric,
geopolitical shifts, thematic validation, and breaking developments.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent, RunContext

from synesis.core.constants import FRED_OUTCOME_SERIES
from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model, web_search_config
from synesis.config import get_settings
from synesis.processing.common.web_search import (
    Recency,
    format_search_results,
    read_web_page,
    search_market_impact,
)
from synesis.processing.events.fetchers import load_hedge_fund_registry
from synesis.processing.intelligence.models import MacroView
from synesis.processing.regime.detector import RegimeDetector
from synesis.providers.cftc.client import CFTCClient

if TYPE_CHECKING:
    from synesis.providers.cftc.models import COTReport
    from synesis.providers.crawler.crawl4ai import Crawl4AICrawlerProvider
    from synesis.providers.fred.client import FREDClient
    from synesis.providers.sec_edgar.client import SECEdgarClient
    from synesis.providers.yfinance.client import YFinanceClient
    from synesis.storage.database import Database

logger = get_logger(__name__)

_WEB_SEARCH_CAP = 5


_SEARCH_DESC = (
    "look up Fed rhetoric, geopolitical developments, trade policy shifts, "
    "validate thematic theses (supply chain, subsector trends, earnings backing), "
    "or breaking events not yet reflected in the data"
)

# Key FRED series for macro regime assessment
_FRED_SERIES = {
    "VIXCLS": "VIX (volatility index)",
    "DGS10": "10-Year Treasury Yield",
    "DGS2": "2-Year Treasury Yield",
    "FEDFUNDS": "Federal Funds Rate",
    "UNRATE": "Unemployment Rate",
    # Credit & financial stress
    "BAMLH0A0HYM2": "HY Credit Spread (OAS)",
    "BAMLC0A0CM": "IG Credit Spread (OAS)",
    "NFCI": "Chicago Fed Financial Conditions Index",
    "T10Y3M": "Yield Curve Spread (10Y - 3M)",
    # Recession indicators
    "SAHMREALTIME": "Sahm Rule Recession Indicator",
    "ICSA": "Initial Jobless Claims",
    # Financial stress
    "STLFSI4": "St. Louis Fed Financial Stress Index",
    # Macro
    "DTWEXBGS": "Trade-Weighted Dollar Index",
}


_BENCHMARKS: dict[str, tuple[str, str]] = {
    # ticker: (label, category)
    "SPY": ("S&P 500", "equities"),
    "QQQ": ("Nasdaq 100", "equities"),
    "IWM": ("Russell 2000", "equities"),
    "TLT": ("20+ Year Treasury", "treasuries"),
    "SHY": ("1-3 Year Treasury", "treasuries"),
    "GLD": ("Gold", "commodities"),
    "USO": ("Crude Oil", "commodities"),
    "HG=F": ("Copper", "commodities"),
    "UUP": ("US Dollar", "dollar"),
    "HYG": ("High Yield Corp", "credit"),
    "LQD": ("Investment Grade Corp", "credit"),
    "XLE": ("Energy", "sectors"),
    "XLF": ("Financials", "sectors"),
    "XLK": ("Technology", "sectors"),
    "XLU": ("Utilities", "sectors"),
    # Sub-sector ETFs
    "SMH": ("Semiconductors", "sub_sectors"),
    "IGV": ("Software/Cloud", "sub_sectors"),
    "XBI": ("Biotech", "sub_sectors"),
    "KWEB": ("China Tech", "sub_sectors"),
    "KRE": ("Regional Banks", "sub_sectors"),
    "XHB": ("Homebuilders", "sub_sectors"),
    "ITA": ("Defense/Aerospace", "sub_sectors"),
}


@dataclass
class MacroStrategistDeps:
    """Dependencies for MacroStrategist."""

    fred: FREDClient
    db: Database
    yfinance: YFinanceClient
    sec_edgar: SECEdgarClient
    crawler: Crawl4AICrawlerProvider | None = None
    current_date: date = field(default_factory=lambda: datetime.now(UTC).date())
    web_search_calls: int = 0


async def _fetch_fred_data(fred: FREDClient, lookback: int = 10) -> dict[str, Any]:
    """Fetch recent observations for key FRED series (latest + trend).

    Args:
        fred: FRED client.
        lookback: Number of recent observations to fetch per series.
    """
    data: dict[str, Any] = {}
    for series_id, label in _FRED_SERIES.items():
        try:
            obs = await fred.get_observations(series_id, limit=lookback, sort_order="desc")
            if obs and obs.observations:
                # Most recent first (desc order)
                history = [{"value": o.value, "date": str(o.date)} for o in obs.observations]
                latest = history[0]
                data[series_id] = {
                    "label": label,
                    "value": latest["value"],
                    "date": latest["date"],
                    "history": history,
                }
        except Exception:
            logger.warning("FRED fetch failed", series_id=series_id, exc_info=True)
            data[series_id] = {"label": label, "value": None, "date": None, "history": []}

    # Compute yield curve spread (latest)
    y10 = data.get("DGS10", {}).get("value")
    y2 = data.get("DGS2", {}).get("value")
    if y10 is not None and y2 is not None:
        try:
            data["yield_curve_spread"] = {
                "label": "Yield Curve Spread (10Y - 2Y)",
                "value": round(float(y10) - float(y2), 3),
            }
        except (ValueError, TypeError):
            logger.warning("Yield curve spread computation failed", y10=y10, y2=y2)

    return data


def _format_fred_context(fred_data: dict[str, Any]) -> str:
    """Format FRED data with full history for the LLM prompt."""
    lines = ["## Current Economic Indicators (FRED)"]
    for series_id, info in fred_data.items():
        val = info.get("value")
        label = info.get("label", series_id)
        if val is None:
            lines.append(f"- **{label}**: unavailable")
            continue

        # Show full history so LLM can see the shape of the trend
        history = info.get("history", [])
        if len(history) >= 2:
            # Reverse to chronological (oldest first) for readability
            points = " → ".join(
                f"{h['value']} ({h['date']})" for h in reversed(history) if h.get("value")
            )
            lines.append(f"- **{label}**: {points}")
        else:
            date_str = info.get("date", "")
            lines.append(f"- **{label}**: {val} (as of {date_str})")

    return "\n".join(lines)


# ── Benchmark Data (yfinance) ───────────────────────────────────


async def _fetch_benchmark_data(yf: YFinanceClient) -> dict[str, dict[str, Any]]:
    """Fetch quotes for macro benchmarks. Returns {ticker: quote_data}."""
    tickers = list(_BENCHMARKS.keys())

    async def _safe_quote(ticker: str) -> tuple[str, dict[str, Any] | None]:
        try:
            q = await yf.get_quote(ticker)
            return ticker, {
                "last": q.last,
                "prev_close": q.prev_close,
                "avg_50d": q.avg_50d,
                "avg_200d": q.avg_200d,
            }
        except Exception:
            logger.warning("Benchmark quote failed", ticker=ticker, exc_info=True)
            return ticker, None

    results = await asyncio.gather(*[_safe_quote(t) for t in tickers])
    return {t: data for t, data in results if data is not None}


def _format_benchmark_context(data: dict[str, dict[str, Any]]) -> str:
    """Format benchmark quotes as markdown for the LLM prompt."""
    if not data:
        return "## Market Benchmarks\nNo benchmark data available."

    categories: dict[str, list[str]] = {}
    for ticker, quote in data.items():
        label, category = _BENCHMARKS.get(ticker, (ticker, "other"))
        last = quote.get("last")
        prev = quote.get("prev_close")
        avg50 = quote.get("avg_50d")
        avg200 = quote.get("avg_200d")

        if last is None:
            continue

        parts = [f"**{label}** ({ticker}): ${last:.2f}"]
        if prev and prev > 0:
            chg_pct = (last - prev) / prev * 100
            parts.append(f"1d: {chg_pct:+.1f}%")
        if avg50:
            rel = "above" if last > avg50 else "below"
            parts.append(f"{rel} 50d MA (${avg50:.2f})")
        if avg200:
            rel = "above" if last > avg200 else "below"
            parts.append(f"{rel} 200d MA (${avg200:.2f})")

        categories.setdefault(category, []).append(" | ".join(parts))

    lines = ["## Market Benchmarks"]
    category_order = [
        "equities",
        "treasuries",
        "commodities",
        "dollar",
        "credit",
        "sectors",
        "sub_sectors",
    ]
    category_labels = {
        "equities": "Equities",
        "treasuries": "Treasuries",
        "commodities": "Commodities",
        "dollar": "Dollar",
        "credit": "Credit",
        "sectors": "Sectors",
        "sub_sectors": "Sub-Sectors",
    }
    for cat in category_order:
        items = categories.get(cat, [])
        if items:
            lines.append(f"\n### {category_labels.get(cat, cat)}")
            for item in items:
                lines.append(f"- {item}")

    return "\n".join(lines)


# ── Event Outcome Enrichment ───────────────────────────────────


async def _enrich_events_with_outcomes(
    events: list[dict[str, Any]],
    sec_edgar: SECEdgarClient,
    fred: FREDClient,
    crawler: Crawl4AICrawlerProvider | None,
    db: Database,
) -> list[dict[str, Any]]:
    """Enrich calendar events with actual outcomes from SEC, FRED, Fed."""
    sem = asyncio.Semaphore(5)

    async def _fetch_one(ev: dict[str, Any]) -> None:
        category = ev.get("category", "")
        if category == "13f_filing":
            return  # 13F handled separately via _fetch_13f_briefs

        async with sem:
            try:
                if category == "earnings":
                    outcome = await _get_earnings_outcome(ev, sec_edgar)
                elif category == "economic_data":
                    outcome = await _get_economic_data_outcome(ev, fred)
                elif category == "fed" and "minute" not in ev.get("title", "").lower():
                    outcome = await _get_crawled_outcome(ev, crawler)
                elif category == "fed":
                    outcome = await _get_fomc_minutes_outcome(ev, db, crawler)
                else:
                    outcome = ""
            except Exception:
                logger.warning(
                    "Outcome fetch failed",
                    title=ev.get("title"),
                    category=category,
                    exc_info=True,
                )
                outcome = ""

        if outcome:
            ev["outcome"] = outcome

    await asyncio.gather(*[_fetch_one(ev) for ev in events])
    return events


async def _get_earnings_outcome(ev: dict[str, Any], sec_edgar: SECEdgarClient) -> str:
    """Get earnings outcome from SEC 8-K Item 2.02 press release."""
    tickers = ev.get("tickers") or []
    if not tickers:
        return ""
    try:
        releases = await sec_edgar.get_earnings_releases(tickers[0], limit=1)
        if releases and releases[0].content:
            return releases[0].content[:3000]
    except Exception:
        logger.warning("SEC earnings release fetch failed", ticker=tickers[0], exc_info=True)
    return ""


async def _get_economic_data_outcome(ev: dict[str, Any], fred: FREDClient) -> str:
    """Get economic data outcome from FRED API observations."""
    title = ev.get("title", "")
    matched_key = ""
    for key in sorted(FRED_OUTCOME_SERIES.keys(), key=len, reverse=True):
        if key.lower() in title.lower():
            matched_key = key
            break
    if not matched_key:
        return ""

    series_id, units = FRED_OUTCOME_SERIES[matched_key]
    try:
        obs = await fred.get_observations(series_id, sort_order="desc", limit=2, units=units)
        if not obs.observations:
            return ""
        latest = obs.observations[0]
        if latest.value is None:
            return ""
        suffix = "%" if units != "lin" else ""
        if len(obs.observations) >= 2 and obs.observations[1].value is not None:
            prev = obs.observations[1].value
            return f"{matched_key}: {latest.value:.1f}{suffix} (prev {prev:.1f}{suffix})"
        return f"{matched_key}: {latest.value:.1f}{suffix}"
    except Exception:
        logger.warning("FRED outcome fetch failed", series_id=series_id, exc_info=True)
    return ""


async def _get_fomc_minutes_outcome(
    ev: dict[str, Any],
    db: Database,
    crawler: Crawl4AICrawlerProvider | None,
) -> str:
    """Get FOMC minutes by looking up meeting date and crawling Fed URL."""
    if not crawler:
        logger.debug("Crawler unavailable — skipping FOMC minutes outcome", title=ev.get("title"))
        return ""
    release_date = ev.get("event_date")
    if not release_date:
        return ""
    meeting_date = await db.get_last_fomc_meeting_date(release_date)
    if not meeting_date:
        return ""
    ds = (
        meeting_date.strftime("%Y%m%d")
        if isinstance(meeting_date, date)
        else str(meeting_date).replace("-", "")[:8]
    )
    url = f"https://www.federalreserve.gov/monetarypolicy/fomcminutes{ds}.htm"
    try:
        result = await crawler.crawl(url)
        if result.success and result.markdown.strip():
            # FOMC minutes can be 15-30K+ chars; truncate at source so the
            # format-layer 2000-char trim keeps the most relevant opening content.
            return result.markdown[:6000]
    except Exception:
        logger.warning("FOMC minutes crawl failed", url=url, exc_info=True)
    return ""


async def _get_crawled_outcome(ev: dict[str, Any], crawler: Crawl4AICrawlerProvider | None) -> str:
    """Get outcome by crawling event source URL or Fed press release."""
    if not crawler:
        logger.debug("Crawler unavailable — skipping crawled outcome", title=ev.get("title"))
        return ""
    source_urls = ev.get("source_urls") or []
    url = source_urls[0] if source_urls else None

    if ev.get("category") == "fed":
        event_date = ev.get("event_date")
        if event_date is not None:
            ds = (
                event_date.strftime("%Y%m%d")
                if isinstance(event_date, date)
                else str(event_date).replace("-", "")[:8]
            )
            url = f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{ds}a.htm"
        elif not url:
            logger.debug("Fed event has no date or source URL — skipping crawl")
            return ""

    if not url:
        return ""
    try:
        result = await crawler.crawl(url)
        if result.success and result.markdown.strip():
            return result.markdown[:6000]
    except Exception:
        logger.warning("Outcome crawl failed", url=url, exc_info=True)
    return ""


async def _fetch_13f_briefs(
    sec_edgar: SECEdgarClient, event_rows: list[Any]
) -> list[dict[str, Any]]:
    """Fetch 13F quarter-over-quarter position changes for filing events."""
    briefs: list[dict[str, Any]] = []
    try:
        fund_registry, _ = load_hedge_fund_registry()
    except Exception:
        logger.warning("Failed to load hedge fund registry — skipping 13F briefs", exc_info=True)
        return []
    for row in event_rows:
        cat = row.get("category") if isinstance(row, dict) else getattr(row, "category", "")
        if cat != "13f_filing":
            continue
        title = row.get("title", "") if isinstance(row, dict) else getattr(row, "title", "")
        for cik, fund_name in fund_registry.items():
            if fund_name.lower() in title.lower():
                try:
                    diff = await sec_edgar.compare_13f_quarters(cik, fund_name)
                    if diff:
                        diff["fund_name"] = fund_name
                        briefs.append(diff)
                except Exception:
                    logger.warning("13F comparison failed", fund=fund_name, cik=cik, exc_info=True)
                break
    return briefs


# ── Event Context Assembly ─────────────────────────────────────


async def _fetch_event_context(
    deps: MacroStrategistDeps,
) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch upcoming events + enrich yesterday's events with outcomes + 13F briefs.

    Returns (upcoming, enriched_recent, filing_briefs).
    """
    yesterday = deps.current_date - timedelta(days=1)

    # Parallel: upcoming + recent events from DB
    try:
        upcoming, recent_rows = await asyncio.gather(
            deps.db.get_upcoming_events(days=7),
            deps.db.get_events_by_date_range(yesterday, deps.current_date),
        )
    except Exception:
        logger.warning("Event DB fetch failed — proceeding without event context", exc_info=True)
        return [], [], []

    recent_events = [dict(r) for r in recent_rows]

    # Wrap 13F in its own guard so a failure doesn't discard enriched events
    async def _safe_13f() -> list[dict[str, Any]]:
        try:
            return await _fetch_13f_briefs(deps.sec_edgar, recent_rows)
        except Exception:
            logger.warning("13F brief fetch failed", exc_info=True)
            return []

    enriched, filing_briefs = await asyncio.gather(
        _enrich_events_with_outcomes(
            recent_events, deps.sec_edgar, deps.fred, deps.crawler, deps.db
        ),
        _safe_13f(),
    )

    return upcoming, enriched, filing_briefs


def _format_event_context(
    upcoming: list[Any],
    recent: list[dict[str, Any]],
    filing_briefs: list[dict[str, Any]],
) -> str:
    """Format enriched events, 13F briefs, and upcoming catalysts as markdown."""
    lines: list[str] = []

    # ── Yesterday's events with outcomes ──
    if recent:
        lines.append("## Yesterday's Events (with outcomes)")
        for ev in recent:
            cat = ev.get("category", "other")
            title = ev.get("title", "?")
            tickers = ev.get("tickers", [])
            ticker_str = f" [{', '.join(tickers)}]" if tickers else ""
            lines.append(f"\n### [{cat}]{ticker_str} {title}")

            desc = ev.get("description", "")
            if desc:
                lines.append(desc[:300])

            outcome = ev.get("outcome", "")
            if outcome:
                # Truncate very long outcomes (Fed minutes, earnings PRs)
                if len(outcome) > 2000:
                    outcome = outcome[:2000] + "\n[...truncated]"
                lines.append(f"\n**Outcome:**\n{outcome}")
        lines.append("")

    # ── 13F hedge fund position changes ──
    if filing_briefs:
        lines.append("## 13F Hedge Fund Position Changes")
        for brief in filing_briefs:
            fund = brief.get("fund_name", "Unknown Fund")
            curr_date = brief.get("current_report_date", "?")
            prev_date = brief.get("previous_report_date", "?")
            lines.append(f"\n### {fund} (filing: {curr_date} vs {prev_date})")

            total_curr = brief.get("total_value_current")
            total_prev = brief.get("total_value_previous")
            if total_curr and total_prev:
                chg = (total_curr - total_prev) / total_prev * 100 if total_prev else 0
                lines.append(
                    f"Portfolio: ${total_curr / 1e6:.0f}M "
                    f"(prev ${total_prev / 1e6:.0f}M, {chg:+.1f}%)"
                )

            for label, key in [
                ("New positions", "new_positions"),
                ("Exited", "exited_positions"),
                ("Increased", "increased"),
                ("Decreased", "decreased"),
            ]:
                items = brief.get(key, [])
                if items:
                    names = [p.get("name_of_issuer", p.get("symbol", "?")) for p in items[:8]]
                    lines.append(f"- **{label}:** {', '.join(names)}")
        lines.append("")

    # ── Upcoming catalysts ──
    if upcoming:
        lines.append("## Upcoming Catalysts (Next 7 Days)")
        by_cat: dict[str, list[str]] = {}
        for ev in upcoming:
            cat = (
                ev.get("category", "other")
                if isinstance(ev, dict)
                else getattr(ev, "category", "other")
            )
            title = ev.get("title", "?") if isinstance(ev, dict) else getattr(ev, "title", "?")
            ev_date = (
                ev.get("event_date", "?")
                if isinstance(ev, dict)
                else getattr(ev, "event_date", "?")
            )
            by_cat.setdefault(cat, []).append(f"{ev_date}: {title}")
        for cat, items in sorted(by_cat.items()):
            lines.append(f"\n**{cat}:**")
            for item in items[:8]:
                lines.append(f"- {item}")
        lines.append("")

    if not lines:
        return "## Events & Calendar\nNo event data available."

    return "\n".join(lines)


def _format_l1_intelligence(state: dict[str, Any]) -> str:
    """Format full Layer 1 intelligence for the MacroStrategist.

    Includes ticker mentions, macro themes, thematic research, discovered
    themes, and news clusters — everything the social and news analysts
    surfaced. The strategist uses this alongside FRED/benchmarks/events to
    form regime assessments and thematic tilts.
    """
    lines = ["## Layer 1 Intelligence (Social + News)"]

    social = state.get("social_analysis", {})
    news = state.get("news_analysis", {})

    # ── Social ticker mentions with context ────────────────────
    social_mentions = social.get("ticker_mentions", [])
    if social_mentions:
        lines.append("\n### Social Ticker Signals")
        for mention in social_mentions:
            ticker = mention.get("ticker", "")
            context = mention.get("context", "")
            accounts = ", ".join(mention.get("source_accounts", []))
            acct_str = f" [from: {accounts}]" if accounts else ""
            lines.append(f"- **{ticker}**: {context}{acct_str}")

    # ── Social macro themes ────────────────────────────────────
    social_themes = social.get("macro_themes", [])
    if social_themes:
        lines.append("\n### Social Macro Themes")
        for theme in social_themes:
            accounts = ", ".join(theme.get("source_accounts", []))
            acct_str = f" [from: {accounts}]" if accounts else ""
            lines.append(f"- {theme.get('theme', '?')} — {theme.get('context', '')}{acct_str}")

    # ── Thematic research (web-verified findings) ──────────────
    research = social.get("research_context", [])
    if research:
        lines.append("\n### Thematic Research (verified via web search)")
        for item in research:
            lines.append(f"- {item}")

    # ── Discovered themes (found through research) ─────────────
    discovered = social.get("discovered_themes", [])
    if discovered:
        lines.append("\n### Discovered Themes")
        for item in discovered:
            lines.append(f"- {item}")

    # ── News macro themes ──────────────────────────────────────
    news_themes = news.get("macro_themes", [])
    if news_themes:
        lines.append("\n### News Macro Themes")
        for theme in news_themes:
            lines.append(f"- {theme.get('theme', '?')} — {theme.get('context', '')}")

    # ── News story clusters ────────────────────────────────────
    clusters = news.get("story_clusters", [])
    if clusters:
        lines.append("\n### News Clusters")
        for cluster in clusters:
            urgency = cluster.get("urgency", "normal")
            event_type = cluster.get("event_type", "other")
            headline = cluster.get("headline", "?")
            cluster_tickers = [t.get("ticker", "") for t in cluster.get("tickers", [])]
            ticker_str = f" [{', '.join(cluster_tickers)}]" if cluster_tickers else ""
            lines.append(f"- **[{event_type}, {urgency}]** {headline}{ticker_str}")
            for fact in cluster.get("key_facts", [])[:4]:
                lines.append(f"  - {fact}")

    if len(lines) == 1:
        lines.append("- No signals from Layer 1 analysts.")

    return "\n".join(lines)


def _compute_cross_asset_signals(
    fred_data: dict[str, Any], benchmark_data: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Compute cross-asset relationship signals from existing data."""
    signals: dict[str, Any] = {}

    # HYG/LQD price ratio (credit risk appetite)
    hyg = benchmark_data.get("HYG", {})
    lqd = benchmark_data.get("LQD", {})
    if hyg.get("last") and lqd.get("last") and lqd["last"] > 0:
        signals["hyg_lqd_ratio"] = round(hyg["last"] / lqd["last"], 4)

    # HY credit spread direction (from FRED history)
    hy_oas = fred_data.get("BAMLH0A0HYM2", {})
    if hy_oas.get("history") and len(hy_oas["history"]) >= 2:
        latest_oas = hy_oas["history"][0]["value"]
        prev_oas = hy_oas["history"][1]["value"]
        if latest_oas is not None and prev_oas is not None:
            try:
                signals["hy_oas_level"] = float(latest_oas)
                signals["hy_oas_direction"] = "tightening" if float(latest_oas) < float(prev_oas) else "widening"
            except (ValueError, TypeError):
                pass

    # NFCI reading
    nfci = fred_data.get("NFCI", {})
    if nfci.get("value") is not None:
        try:
            val = float(nfci["value"])
            signals["nfci_level"] = val
            signals["nfci_reading"] = "loose" if val < 0 else "tight"
        except (ValueError, TypeError):
            pass

    # Yield curve signals
    t10y3m = fred_data.get("T10Y3M", {})
    if t10y3m.get("value") is not None:
        try:
            signals["yield_curve_10y3m"] = float(t10y3m["value"])
        except (ValueError, TypeError):
            pass

    # Copper/Gold ratio (risk appetite proxy)
    copper = benchmark_data.get("HG=F", {})
    gold = benchmark_data.get("GLD", {})
    if copper.get("last") and gold.get("last") and gold["last"] > 0:
        signals["copper_gold_ratio"] = round(copper["last"] / gold["last"], 4)

    # SPY vs 200d MA
    spy = benchmark_data.get("SPY", {})
    if spy.get("last") and spy.get("avg_200d"):
        signals["spy_above_200d"] = spy["last"] > spy["avg_200d"]
        signals["spy_200d_pct"] = round((spy["last"] / spy["avg_200d"] - 1) * 100, 2)

    return signals


def _format_cross_asset_signals(signals: dict[str, Any]) -> str:
    """Format cross-asset signals as markdown context for the LLM."""
    if not signals:
        return ""

    lines = ["## Cross-Asset Signals"]

    if "hy_oas_level" in signals:
        direction = signals.get("hy_oas_direction", "?")
        lines.append(f"- **HY Credit Spread (OAS):** {signals['hy_oas_level']:.0f}bp ({direction})")

    if "nfci_level" in signals:
        reading = signals.get("nfci_reading", "?")
        lines.append(f"- **Financial Conditions (NFCI):** {signals['nfci_level']:.2f} ({reading})")

    if "yield_curve_10y3m" in signals:
        val = signals["yield_curve_10y3m"]
        status = "inverted" if val < 0 else "positive"
        lines.append(f"- **Yield Curve (10Y-3M):** {val:+.2f}% ({status})")

    if "hyg_lqd_ratio" in signals:
        lines.append(f"- **HYG/LQD Ratio:** {signals['hyg_lqd_ratio']:.4f}")

    if "copper_gold_ratio" in signals:
        lines.append(f"- **Copper/Gold Ratio:** {signals['copper_gold_ratio']:.4f}")

    if "spy_above_200d" in signals:
        above = "above" if signals["spy_above_200d"] else "below"
        lines.append(f"- **SPY vs 200d MA:** {above} ({signals.get('spy_200d_pct', 0):+.1f}%)")

    return "\n".join(lines)


def _compute_regime_prior(
    fred_data: dict[str, Any],
    benchmark_data: dict[str, dict[str, Any]],
    cross_asset: dict[str, Any],
) -> dict[str, Any]:
    """Compute a deterministic regime prior from quantitative signals.

    Each signal scored as: +1 (risk_on), 0 (neutral), -1 (risk_off).
    Average score maps to a regime suggestion.
    """
    scores: dict[str, tuple[int, str]] = {}

    # VIX
    vix = fred_data.get("VIXCLS", {}).get("value")
    if vix is not None:
        try:
            v = float(vix)
            if v < 18:
                scores["VIX"] = (+1, f"{v:.1f} (low)")
            elif v <= 25:
                scores["VIX"] = (0, f"{v:.1f} (normal)")
            else:
                scores["VIX"] = (-1, f"{v:.1f} (elevated)")
        except (ValueError, TypeError):
            pass

    # NFCI
    nfci = cross_asset.get("nfci_level")
    if nfci is not None:
        if nfci < -0.2:
            scores["NFCI"] = (+1, f"{nfci:.2f} (loose)")
        elif nfci <= 0.2:
            scores["NFCI"] = (0, f"{nfci:.2f} (neutral)")
        else:
            scores["NFCI"] = (-1, f"{nfci:.2f} (tight)")

    # HY OAS
    hy_oas = cross_asset.get("hy_oas_level")
    if hy_oas is not None:
        if hy_oas < 400:
            scores["HY Spread"] = (+1, f"{hy_oas:.0f}bp (tight)")
        elif hy_oas <= 500:
            scores["HY Spread"] = (0, f"{hy_oas:.0f}bp (normal)")
        else:
            scores["HY Spread"] = (-1, f"{hy_oas:.0f}bp (stressed)")

    # SPY vs 200d MA
    spy_above = cross_asset.get("spy_above_200d")
    spy_pct = cross_asset.get("spy_200d_pct", 0)
    if spy_above is not None:
        scores["SPY vs 200d"] = (+1 if spy_above else -1, f"{'above' if spy_above else 'below'} ({spy_pct:+.1f}%)")

    # Yield curve (10Y-3M)
    yc = cross_asset.get("yield_curve_10y3m")
    if yc is not None:
        if yc > 0.5:
            scores["Yield Curve"] = (+1, f"{yc:+.2f}% (steep)")
        elif yc >= 0:
            scores["Yield Curve"] = (0, f"{yc:+.2f}% (flat)")
        else:
            scores["Yield Curve"] = (-1, f"{yc:+.2f}% (inverted)")

    # Sahm Rule
    sahm = fred_data.get("SAHMREALTIME", {}).get("value")
    if sahm is not None:
        try:
            s = float(sahm)
            if s < 0.3:
                scores["Sahm Rule"] = (+1, f"{s:.2f} (no recession)")
            elif s <= 0.5:
                scores["Sahm Rule"] = (0, f"{s:.2f} (warning)")
            else:
                scores["Sahm Rule"] = (-1, f"{s:.2f} (recession)")
        except (ValueError, TypeError):
            pass

    if not scores:
        return {"regime": "uncertain", "confidence": "no data", "signals": {}}

    # Average score → regime
    avg = sum(s for s, _ in scores.values()) / len(scores)
    if avg > 0.3:
        regime = "risk_on"
    elif avg < -0.3:
        regime = "risk_off"
    elif abs(avg) <= 0.3 and any(s != 0 for s, _ in scores.values()):
        regime = "transitioning"
    else:
        regime = "uncertain"

    agree = sum(1 for s, _ in scores.values() if s != 0 and (s > 0) == (avg > 0))
    total = sum(1 for s, _ in scores.values() if s != 0)
    confidence = f"{agree}/{total} signals agree" if total > 0 else "no directional signals"

    return {
        "regime": regime,
        "score": round(avg, 2),
        "confidence": confidence,
        "signals": {k: {"score": s, "detail": d} for k, (s, d) in scores.items()},
    }


def _format_regime_prior(prior: dict[str, Any]) -> str:
    """Format the quantitative regime prior as markdown."""
    if not prior.get("signals"):
        return ""

    regime = prior["regime"]
    score = prior.get("score", 0)
    confidence = prior.get("confidence", "")

    lines = ["## Quantitative Regime Assessment"]
    lines.append(f"**Suggested regime:** {regime} (score: {score:+.2f}, {confidence})")
    lines.append("")
    lines.append("| Signal | Score | Detail |")
    lines.append("|--------|-------|--------|")
    for name, info in prior["signals"].items():
        s = info["score"]
        label = "risk_on" if s > 0 else "risk_off" if s < 0 else "neutral"
        lines.append(f"| {name} | {s:+d} ({label}) | {info['detail']} |")
    lines.append("")
    lines.append("*This is a data-grounded starting point. Override with reasoning if you disagree.*")

    return "\n".join(lines)


async def _fetch_cftc_positioning() -> dict[str, COTReport]:
    """Fetch latest CFTC COT positioning for key futures contracts."""
    try:
        client = CFTCClient()
        return await client.get_latest(["ES", "NQ", "ZN", "GC", "CL", "DX"])
    except Exception:
        logger.warning("CFTC COT fetch failed", exc_info=True)
        return {}


def _format_cftc_positioning(cot_data: dict[str, COTReport]) -> str:
    """Format CFTC COT positioning as markdown context for the LLM."""
    if not cot_data:
        return ""

    lines = ["## Futures Positioning (CFTC COT)"]
    lines.append(
        "Weekly hedge fund (leveraged funds) net positioning on major futures. "
        "Percentile = position vs 52-week range (>90% = crowded long, <10% = crowded short)."
    )
    lines.append("")
    lines.append("| Contract | Net Contracts | 52wk Pctl | Z-Score | Wk Change (L/S) | Date |")
    lines.append("|----------|--------------|-----------|---------|-----------------|------|")

    for ticker, report in sorted(cot_data.items()):
        lev = report.leveraged_funds
        pctl = f"{report.lev_funds_net_pctl:.0f}%" if report.lev_funds_net_pctl is not None else "N/A"
        zs = f"{report.lev_funds_net_zscore:+.2f}" if report.lev_funds_net_zscore is not None else "N/A"
        chg = f"{lev.change_long:+,} / {lev.change_short:+,}"
        lines.append(
            f"| {ticker} ({report.contract_name[:30]}) "
            f"| {lev.net_contracts:+,} | {pctl} | {zs} | {chg} | {report.report_date} |"
        )

    # Flag extremes
    extremes = []
    for ticker, report in cot_data.items():
        if report.lev_funds_net_pctl is not None:
            if report.lev_funds_net_pctl >= 90:
                extremes.append(f"**{ticker}**: crowded LONG ({report.lev_funds_net_pctl:.0f}th pctl) — contrarian bearish")
            elif report.lev_funds_net_pctl <= 10:
                extremes.append(f"**{ticker}**: crowded SHORT ({report.lev_funds_net_pctl:.0f}th pctl) — contrarian bullish")

    if extremes:
        lines.append("")
        lines.append("**Extreme positioning (contrarian signals):**")
        for e in extremes:
            lines.append(f"- {e}")

    return "\n".join(lines)


async def _fetch_hmm_regime(fred_api_key: str) -> dict[str, Any]:
    """Run the HMM regime detector and return the current assessment."""
    if not fred_api_key:
        logger.info("HMM regime skipped — no FRED API key")
        return {}

    try:
        detector = RegimeDetector(fred_api_key=fred_api_key)
        await detector.fit(lookback_years=10)
        return detector.predict_current()
    except Exception:
        logger.warning("HMM regime detection failed", exc_info=True)
        return {}


def _format_hmm_regime(hmm_result: dict[str, Any]) -> str:
    """Format HMM regime assessment as markdown."""
    if not hmm_result:
        return ""

    regime = hmm_result.get("regime", "unknown")
    confidence = hmm_result.get("confidence", 0)
    duration = hmm_result.get("duration_weeks", 0)
    probs = hmm_result.get("probabilities", {})

    lines = ["## HMM Regime Model"]
    lines.append(
        f"**Assessment:** {regime} ({confidence:.0%} confidence, {duration} weeks in this state)"
    )

    if probs:
        prob_parts = [f"{r}: {p:.0%}" for r, p in sorted(probs.items(), key=lambda x: -x[1])]
        lines.append(f"**Probabilities:** {' | '.join(prob_parts)}")

    lines.append(
        "\n*Statistical model trained on 10 years of weekly cross-asset data "
        "(equity returns, volatility, credit spreads, yield curve, dollar). "
        "Complements the threshold-based regime prior above.*"
    )

    return "\n".join(lines)


_SCREENING_INSTRUCTIONS = """\
7. **Ticker Screening** (you have a ticker pool from today's signals — see below):
   You are the CIO tagging which tickers from today's signals are worth watching and why. \
List ALL tickers that have a meaningful signal — do NOT cull or limit the list. Your job \
is to categorize each ticker by its thematic angle and direction lean so the team knows \
what to watch. Only drop tickers that are genuinely noise (single weak mention, no context).

   For each ticker, fill `watchlist_picks` with: ticker, thematic_angle (be specific — \
"custom silicon displacing GPU for inference" not just "AI"), direction_lean \
(your initial read — bullish or bearish), signal_strength (why this stands out \
from the noise), and research_note (anything you found via web search).

   **Wildcards:** If your research reveals tickers NOT in the signal pool \
that better express an identified theme, include them with is_wildcard=True. They must \
be directly connected to a theme from today's signals.

   Drop only genuine noise into `tickers_dropped` with a one-liner reason each in \
`drop_reasons`. Be honest — "single weak mention with no context" is a valid reason, \
but "too many tickers" is NOT."""

_NO_SCREENING_INSTRUCTIONS = ""


def _format_ticker_pool(state: dict[str, Any]) -> str:
    """Format Layer 1 tickers with their context for screening."""
    tickers = state.get("target_tickers", [])
    if not tickers:
        return ""

    lines = ["## Ticker Pool for Screening"]
    lines.append(
        f"Layer 1 surfaced {len(tickers)} tickers from social + news signals. "
        "Tag all meaningful tickers for the watchlist with thematic context."
    )

    # Social mentions with context
    social = state.get("social_analysis", {})
    social_mentions = social.get("ticker_mentions", [])
    if social_mentions:
        lines.append("\n### Social Signals")
        for mention in social_mentions:
            ticker = mention.get("ticker", "")
            if ticker not in tickers:
                continue
            context = mention.get("context", "")
            accounts = ", ".join(mention.get("source_accounts", []))
            lines.append(f"- **{ticker}**: {context} [from: {accounts}]")

    # News clusters with tickers
    news = state.get("news_analysis", {})
    clusters = news.get("story_clusters", [])
    if clusters:
        lines.append("\n### News Clusters")
        for cluster in clusters:
            cluster_tickers = [t.get("ticker", "") for t in cluster.get("tickers", [])]
            relevant = [t for t in cluster_tickers if t in tickers]
            if not relevant:
                continue
            urgency = cluster.get("urgency", "normal")
            event_type = cluster.get("event_type", "other")
            headline = cluster.get("headline", "?")
            lines.append(f"\n**[{event_type}, urgency={urgency}]** {headline}")
            for fact in cluster.get("key_facts", [])[:4]:
                lines.append(f"  - {fact}")
            for t_info in cluster.get("tickers", []):
                t = t_info.get("ticker", "")
                if t in tickers:
                    lines.append(f"  - **{t}**: {t_info.get('context', '')}")

    lines.append(f"\n**Full ticker pool:** {', '.join(tickers)}")
    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are a senior macro strategist at a multi-strategy fund.
You assess the current market regime, analyze yesterday's events, and produce thematic tilts.

Today's date: {current_date}

## Context Provided

You receive pre-fetched data from multiple sources:
- **Quantitative regime prior** — a deterministic regime suggestion computed from VIX, NFCI, \
HY credit spreads, yield curve, SPY vs 200d MA, and Sahm Rule. This is your starting point — \
agree or override with reasoning.
- **HMM regime model** — a statistical model (Hidden Markov Model) trained on 10 years of \
weekly cross-asset data. It outputs regime probabilities (e.g., 85% risk_on, 12% transitioning). \
Cross-reference with the threshold-based prior — if both agree, high confidence. If they \
diverge, investigate why.
- **FRED indicators** — VIX, yields, fed funds, unemployment, credit spreads (HY/IG OAS), \
financial conditions (NFCI), Sahm Rule, initial claims, yield curves, dollar index — with trend history.
- **Cross-asset signals** — computed relationships: HYG/LQD ratio, copper/gold ratio, \
credit spread direction, financial conditions reading, SPY distance from 200d MA.
- **CFTC futures positioning** — weekly Commitment of Traders data showing hedge fund \
(leveraged funds) net positioning on equity indices, treasuries, gold, crude, and dollar \
futures. Includes 52-week percentile rankings — extreme positioning (>90th or <10th pctl) \
is a contrarian signal.
- **Market benchmarks** — equity indices, treasuries, commodities (incl. copper), dollar, \
credit, and sector ETFs with spot prices, daily changes, and moving average positioning.
- **Yesterday's events with actual outcomes** — earnings results (SEC 8-K press releases), \
economic data releases (FRED actuals vs previous), Fed statements/minutes (full text), \
and other market-moving events. This is YOUR primary source for analyzing what happened.
- **13F hedge fund position changes** — quarterly position disclosures showing new, exited, \
increased, and decreased positions from tracked hedge funds.
- **Upcoming catalysts** — next 7 days of scheduled events (FOMC, CPI, earnings, 13F).
- **Layer 1 intelligence** — the full output from today's social sentiment and news analysts: \
ticker mentions with context and source attribution, macro themes, thematic research (web-verified \
findings on key theses), discovered themes (related themes found through research), and news \
story clusters with key facts. This is your richest signal source — ticker mentions often tie \
directly into thematic tilts, and research findings can validate or contradict price action.

## How to Use Your Two Data Sources

You have **structured data** (FRED, benchmarks, events, 13F) and **Layer 1 intelligence** \
(social/news signals). These serve different purposes:

- **Structured data** tells you WHAT happened — prices, yields, economic prints, earnings \
outcomes, fund positioning. This is objective and timestamped. Use it to anchor your regime \
assessment and validate (or invalidate) narratives.
- **Layer 1 intelligence** tells you WHAT THE MARKET IS TALKING ABOUT — which tickers are \
in play, which themes are forming, what the smart accounts are flagging, and what the social \
analysts verified through web research. This is your edge for identifying emerging themes \
before they show up in price.

**Synthesize, don't silo.** A social ticker mention is more interesting when price confirms \
the thesis. A benchmark move is more meaningful when you know the narrative driving it. \
Thematic research findings can validate a tilt ("CoWoS capacity constraint confirmed by \
TSMC capex guidance") or flag a risk ("social hype on POET but no institutional flow"). \
Discovered themes point to adjacent opportunities the structured data alone won't surface.

## Your Job

1. **Regime Assessment**: Classify the current market as risk_on, risk_off, transitioning, or uncertain.
   - sentiment_score: -1.0 (strongly bearish) to 1.0 (strongly bullish) for the broad market.
   - Start from the **quantitative regime prior** — it scores 6 data-driven signals. You may \
agree or disagree, but if you override it, explain which signals you weight differently and why.
   - Cross-reference structured data with Layer 1 signals. Divergences between data and \
narrative are the most informative signal.

2. **Key Drivers**: List 3-5 factors driving the current regime.
   - Blend structured data (event outcomes, price moves) with Layer 1 signals (narrative shifts, \
social convergence on a theme). A driver is strongest when both sources point the same way.

3. **Thematic Tilts**: Think thematically, not by ETF. The goal is to capture every \
meaningful investment theme with conviction — most themes do NOT map to a single ETF.
   - **Granularity is the point.** "AI" is too broad. Break it into the real sub-themes: \
"HBM supply chain", "CoWoS/advanced packaging", "AI inference silicon", "optical networking \
(CPO)", "data center power/cooling", "AI infrastructure services", "custom silicon (TPUs/ASICs)" \
are all distinct themes with different drivers and different winners.
   - **ETF is optional context, not the organizing principle.** If an ETF exists that tracks \
the theme, include it in the `etf` field for reference. But most actionable themes have no ETF \
— and that's fine. Never skip a theme just because there's no ETF for it.
   - **Produce at least 8-12 tilts** spanning the signal set. Include macro (rates, credit, \
dollar), sector (energy, financials), AND granular sub-themes (specific technology verticals, \
supply chain segments, geopolitical exposure). The signal set is rich — your tilts should be too.
   - **`key_evidence` is required for every tilt** — list 1-3 specific data points that ground \
the tilt. Draw from BOTH sources: price levels and benchmark moves (structured), social ticker \
convergence and thematic research findings (Layer 1), earnings outcomes and 13F positions \
(events). The strongest tilts have evidence from multiple sources.
   - **`persistence`** — classify each tilt as structural (multi-year, e.g. AI infrastructure \
buildout), cyclical (quarter-to-quarter, e.g. seasonal energy demand), or event_driven \
(resolves on a specific catalyst, e.g. FOMC decision). Default is cyclical.
   - **`catalyst_date`** — for event_driven and cyclical tilts, when does it resolve? Reference \
upcoming catalysts from the event calendar (e.g. "Q2 earnings July 24", "FOMC June 12", \
"CPI May 14"). Leave empty for structural tilts that have no near-term resolution date.
   - **`related_tickers`** — for each tilt, list 2-5 tickers that best express it. These should \
be specific companies, not ETFs (the ETF goes in the `etf` field). This helps downstream \
agents connect tilts to watchlist picks.
   - **Reasoning must explain WHY, not restate WHAT.** Don't say "XLE above 50d/200d MA — \
positive." Say "Energy bid is pricing a sustained Hormuz risk premium; XLE above trend with \
USO +2.9% and 3 social accounts flagging supply route risk. Upstream capex beneficiaries \
(services, E&P) have cleaner risk/reward than integrated majors."
   - sentiment_score per tilt: -1.0 (strongly underweight) to 1.0 (strongly overweight).
   - Note divergences: if credit (HYG) is weakening while equities are flat, flag it.

4. **Event Analysis**: Synthesize yesterday's events into a brief narrative.
   - What happened (earnings, economic data, Fed, filings) and what it means for the regime.
   - Which sectors/themes were affected and how.
   - For geopolitical and policy events (tariffs, sanctions, elections, central bank pivots):
     - Identify key actors and their incentives.
     - Define 2-3 scenarios with rough probability weights.
     - For each scenario, trace the transmission: event → first-order effect → second-order → asset impact.
     - Flag where consensus positioning is most vulnerable to a scenario shift.

5. **Positioning Signals**: What are hedge funds and smart money doing?
   - **CFTC COT data** — leveraged funds net positioning on ES, NQ, ZN, GC, CL, DX futures. \
Flag extreme percentiles (>90th = crowded long, <10th = crowded short) as contrarian signals. \
Week-over-week changes reveal positioning momentum.
   - **13F position changes** — quarterly filings showing new/exited/increased/decreased positions.
   - Cross-reference: do futures positioning and 13F filings tell the same story? Divergence matters.

6. **Risks**: What could shift the regime? List 2-4 scenarios.
   - Reference upcoming catalysts — an FOMC meeting or CPI release this week is a concrete risk.
   - Reference 13F positioning — crowded trades that could unwind.

{screening_instructions}

## Tools

{search_docs}\
- `web_read(url)` — Read a web page for full article content (~4000 chars). Unlimited calls.
- `get_fred_data(series_id)` — Fetch a FRED economic data series (last 5 observations).

## When to web_search (budget is tight — save for what data can't tell you)
- **Thematic validation** — verify supply chain theses, subsector trends, or earnings data \
backing a thematic tilt. "Is the CoWoS capacity constraint real?", "What are the latest HBM4 \
yield reports?", "Are data center power bookings accelerating?" are high-value searches that \
turn a shallow tilt into a grounded one.
- **Fed rhetoric** — what did Powell/Waller/Bostic say? Any shift in forward guidance or dot plot?
- **Geopolitical / trade policy** — tariff announcements, sanctions, trade negotiations, military escalation.
- **Breaking developments** — anything in the last 24h not yet reflected in the pre-fetched data.
- Do NOT search for data you already have (prices, FRED indicators, event calendar, earnings outcomes).

## Rules
- Ground regime assessment in the pre-fetched data, not speculation. Cite specific numbers.
- The quantitative regime prior is a data-grounded starting point. If you override it, explain \
which signals you weight differently and why.
- Cross-reference sources: benchmark prices vs FRED vs credit spreads vs events vs themes vs 13F. \
Flag contradictions — especially divergences between credit and equity signals.
- Thematic tilts should be probabilistic, not prescriptive.
- sentiment_score calibration: ±0.8-1.0 = high conviction, ±0.4-0.7 = moderate, ±0.1-0.3 = weak.
"""


# ── Tool Functions ────────────────────────────────────────────────


async def _tool_web_search(
    ctx: RunContext[MacroStrategistDeps],
    query: str,
    recency: str = "day",
) -> str:
    """Search the web for macro context."""
    if ctx.deps.web_search_calls >= _WEB_SEARCH_CAP:
        return f"Web search budget exhausted ({_WEB_SEARCH_CAP} calls used)."
    ctx.deps.web_search_calls += 1

    valid_recencies = ("day", "week", "month", "year", "none")
    recency_val: Recency = recency if recency in valid_recencies else "day"  # type: ignore[assignment]
    results = await search_market_impact(query, recency=recency_val)
    if not results:
        return "No search results found."
    return format_search_results(results)


async def _tool_web_read(ctx: RunContext[MacroStrategistDeps], url: str) -> str:
    """Read a web page for full article content."""
    return await read_web_page(url)


async def _tool_get_fred_data(
    ctx: RunContext[MacroStrategistDeps],
    series_id: str,
) -> str:
    """Fetch a FRED economic data series."""
    try:
        obs = await ctx.deps.fred.get_observations(series_id, limit=5, sort_order="desc")
        if not obs or not obs.observations:
            return f"No data found for FRED series '{series_id}'."
        lines = [f"FRED {series_id} — last {len(obs.observations)} observations:"]
        for o in obs.observations:
            lines.append(f"  {o.date}: {o.value}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning("FRED tool fetch failed", series_id=series_id, error=str(e), exc_info=True)
        return f"FRED fetch failed for '{series_id}': {e}"


# ── Public API ───────────────────────────────────────────────────


async def analyze_macro(
    state: dict[str, Any],
    deps: MacroStrategistDeps,
) -> MacroView:
    """Run the MacroStrategist on pipeline state."""
    logger.info("Starting MacroStrategist")

    # Pre-fetch all data sources in parallel (graceful degradation per source)
    _settings = get_settings()
    fred_api_key = _settings.fred_api_key.get_secret_value() if _settings.fred_api_key else ""
    fred_res, bench_res, event_res, cftc_res, hmm_res = await asyncio.gather(
        _fetch_fred_data(deps.fred),
        _fetch_benchmark_data(deps.yfinance),
        _fetch_event_context(deps),
        _fetch_cftc_positioning(),
        _fetch_hmm_regime(fred_api_key),
        return_exceptions=True,
    )

    if isinstance(fred_res, BaseException):
        logger.warning("FRED data fetch failed — proceeding without", exc_info=fred_res)
        fred_data: dict[str, Any] = {}
    else:
        fred_data = fred_res

    if isinstance(bench_res, BaseException):
        logger.warning("Benchmark data fetch failed — proceeding without", exc_info=bench_res)
        benchmark_data: dict[str, dict[str, Any]] = {}
    else:
        benchmark_data = bench_res

    if isinstance(event_res, BaseException):
        logger.warning("Event context fetch failed — proceeding without", exc_info=event_res)
        upcoming: list[Any] = []
        enriched_events: list[dict[str, Any]] = []
        filing_briefs: list[dict[str, Any]] = []
    else:
        upcoming, enriched_events, filing_briefs = event_res

    if isinstance(cftc_res, BaseException):
        logger.warning("CFTC COT fetch failed — proceeding without", exc_info=cftc_res)
        cftc_data: dict[str, COTReport] = {}
    else:
        cftc_data = cftc_res

    if isinstance(hmm_res, BaseException):
        logger.warning("HMM regime fetch failed — proceeding without", exc_info=hmm_res)
        hmm_result: dict[str, Any] = {}
    else:
        hmm_result = hmm_res

    logger.info(
        "MacroStrategist context fetched",
        benchmarks=len(benchmark_data),
        recent_events=len(enriched_events),
        enriched_with_outcome=sum(1 for e in enriched_events if e.get("outcome")),
        filing_briefs=len(filing_briefs),
        upcoming_events=len(upcoming),
        cftc_contracts=len(cftc_data),
        hmm_regime=hmm_result.get("regime"),
    )

    # Compute derived signals
    cross_asset = _compute_cross_asset_signals(fred_data, benchmark_data)
    regime_prior = _compute_regime_prior(fred_data, benchmark_data, cross_asset)

    logger.info(
        "Regime prior computed",
        regime=regime_prior.get("regime"),
        score=regime_prior.get("score"),
        confidence=regime_prior.get("confidence"),
    )

    # Format all context sections
    regime_prior_context = _format_regime_prior(regime_prior)
    hmm_context = _format_hmm_regime(hmm_result)
    fred_context = _format_fred_context(fred_data)
    cross_asset_context = _format_cross_asset_signals(cross_asset)
    benchmark_context = _format_benchmark_context(benchmark_data)
    cftc_context = _format_cftc_positioning(cftc_data)
    event_context = _format_event_context(upcoming, enriched_events, filing_briefs)
    l1_context = _format_l1_intelligence(state)
    ticker_context = _format_ticker_pool(state)

    has_tickers = bool(state.get("target_tickers"))

    # Build prompt (regime assessments first, then data, then signals)
    user_prompt = "\n\n".join(
        section
        for section in [
            regime_prior_context,
            hmm_context,
            fred_context,
            cross_asset_context,
            benchmark_context,
            cftc_context,
            event_context,
            l1_context,
            ticker_context,
        ]
        if section
    )

    # Construct agent
    search = web_search_config(_WEB_SEARCH_CAP, _SEARCH_DESC)
    tools: list[Any] = [_tool_web_read, _tool_get_fred_data]
    if not search.native:
        tools.append(_tool_web_search)

    agent: Agent[MacroStrategistDeps, MacroView] = Agent(
        model=create_model(tier="vsmart"),
        deps_type=MacroStrategistDeps,
        output_type=MacroView,
        system_prompt=SYSTEM_PROMPT.format(
            current_date=deps.current_date,
            search_docs=search.prompt_docs,
            screening_instructions=(
                _SCREENING_INSTRUCTIONS if has_tickers else _NO_SCREENING_INSTRUCTIONS
            ),
        ),
        tools=tools,
        builtin_tools=search.builtin_tools,
    )

    try:
        result = await agent.run(user_prompt, deps=deps)
        output: MacroView = result.output
    except Exception:
        logger.exception("MacroStrategist LLM call failed")
        return MacroView(
            regime="uncertain",
            sentiment_score=0.0,
            key_drivers=["[LLM synthesis failed]"],
            analysis_date=deps.current_date,
        )

    logger.info(
        "MacroStrategist complete",
        regime=output.regime,
        sentiment=output.sentiment_score,
        tilts=len(output.thematic_tilts),
        watchlist_picks=len(output.watchlist_picks),
        tickers_dropped=len(output.tickers_dropped),
    )

    if output.analysis_date != deps.current_date:
        output = output.model_copy(update={"analysis_date": deps.current_date})

    return output
