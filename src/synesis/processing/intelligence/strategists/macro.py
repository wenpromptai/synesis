"""MacroStrategist — assesses market regime from multi-source context.

Pre-fetches:
  1. FRED economic indicators (VIX, yields, fed funds, unemployment)
  2. Benchmark quotes via yfinance (indices, bonds, commodities, dollar, credit, sectors)
  3. Yesterday's calendar events enriched with actual outcomes (earnings 8-K,
     FRED actuals, Fed minutes/statements, 13F position changes)
  4. Upcoming catalysts from DB (next 7 days)
  5. Layer 1 macro themes (social + news)

The LLM synthesizes all context into a regime assessment, sector tilts, and risks.
This replaces the separate yesterday brief pipeline — the MacroStrategist is the
single source of truth for "what happened + what it means + what's the regime."

Web search (budget: 3) is reserved for what data can't tell you: Fed rhetoric,
geopolitical shifts, and breaking policy changes.
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
from synesis.processing.common.web_search import (
    Recency,
    format_search_results,
    read_web_page,
    search_market_impact,
)
from synesis.processing.events.fetchers import load_hedge_fund_registry
from synesis.processing.intelligence.models import MacroView

if TYPE_CHECKING:
    from synesis.providers.crawler.crawl4ai import Crawl4AICrawlerProvider
    from synesis.providers.fred.client import FREDClient
    from synesis.providers.sec_edgar.client import SECEdgarClient
    from synesis.providers.yfinance.client import YFinanceClient
    from synesis.storage.database import Database

logger = get_logger(__name__)

_WEB_SEARCH_CAP = 5


_SEARCH_DESC = (
    "look up Fed rhetoric, geopolitical developments, trade policy shifts, "
    "or breaking events not yet reflected in the data"
)

# Key FRED series for macro regime assessment
_FRED_SERIES = {
    "VIXCLS": "VIX (volatility index)",
    "DGS10": "10-Year Treasury Yield",
    "DGS2": "2-Year Treasury Yield",
    "FEDFUNDS": "Federal Funds Rate",
    "UNRATE": "Unemployment Rate",
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


def _format_macro_themes(state: dict[str, Any]) -> str:
    """Format Layer 1 macro themes for the LLM prompt."""
    lines = ["## Macro Themes from Layer 1 Analysts"]

    social = state.get("social_analysis", {})
    for theme in social.get("macro_themes", []):
        lines.append(f"- [Social] {theme.get('theme', '?')} — {theme.get('context', '')}")

    news = state.get("news_analysis", {})
    for theme in news.get("macro_themes", []):
        lines.append(f"- [News] {theme.get('theme', '?')} — {theme.get('context', '')}")

    # Include high-urgency macro news clusters
    for cluster in news.get("story_clusters", []):
        if cluster.get("event_type") == "macro":
            lines.append(
                f"- [News cluster] {cluster.get('headline', '?')} "
                f"(urgency: {cluster.get('urgency', '?')})"
            )

    if len(lines) == 1:
        lines.append("- No macro themes identified by Layer 1 analysts.")

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
You assess the current market regime, analyze yesterday's events, and produce sector tilts.

Today's date: {current_date}

## Context Provided

You receive pre-fetched data from multiple sources:
- **FRED indicators** — VIX, yields, fed funds, unemployment, yield curve spread with trend history.
- **Market benchmarks** — equity indices, treasuries, commodities, dollar, credit, and sector ETFs \
with spot prices, daily changes, and moving average positioning.
- **Yesterday's events with actual outcomes** — earnings results (SEC 8-K press releases), \
economic data releases (FRED actuals vs previous), Fed statements/minutes (full text), \
and other market-moving events. This is YOUR primary source for analyzing what happened.
- **13F hedge fund position changes** — quarterly position disclosures showing new, exited, \
increased, and decreased positions from tracked hedge funds.
- **Upcoming catalysts** — next 7 days of scheduled events (FOMC, CPI, earnings, 13F).
- **Layer 1 themes** — today's social/news macro themes from curated financial accounts.

## Your Job

1. **Regime Assessment**: Classify the current market as risk_on, risk_off, transitioning, or uncertain.
   - sentiment_score: -1.0 (strongly bearish) to 1.0 (strongly bullish) for the broad market.
   - Ground your assessment in the data provided. Cross-reference: do benchmark prices confirm \
what FRED data and events suggest? Are equities and credit aligned or diverging?

2. **Key Drivers**: List 3-5 factors driving the current regime.
   - Incorporate yesterday's event outcomes: did earnings beat/miss change sector narratives? \
Did economic data shift rate expectations? Did Fed language signal a policy pivot?

3. **Thematic Tilts**: Produce BOTH broad sector tilts AND granular thematic tilts.
   - **ETF-backed tilts**: Use sector/sub-sector ETF data to validate. Set the `etf` field. \
Examples: "Semiconductors" (etf=SMH), "Energy" (etf=XLE), "Software/Cloud" (etf=IGV).
   - **Pure thematic tilts**: Themes with no ETF, identified from earnings, events, social, \
news, or 13F signals. Leave `etf` null. Examples: "CPO/Optical Networking", \
"AI Infrastructure", "HBM/Memory", "Data Center Power", "Nuclear/SMR".
   - sentiment_score per tilt: -1.0 (strongly underweight) to 1.0 (strongly overweight).
   - Be granular: "Semiconductors: +0.5" AND "CPO/Optical: +0.8" are both valid — the second \
captures a sub-theme within the first. Not all of tech is the same.
   - Note divergences: if credit (HYG) is weakening while equities are flat, flag it.

4. **Event Analysis**: Synthesize yesterday's events into a brief narrative.
   - What happened (earnings, economic data, Fed, filings) and what it means for the regime.
   - Which sectors/themes were affected and how.

5. **Positioning Signals**: What are hedge funds and smart money doing?
   - 13F position changes — new themes, exits, conviction bets.
   - Any notable convergence across funds.

6. **Risks**: What could shift the regime? List 2-4 scenarios.
   - Reference upcoming catalysts — an FOMC meeting or CPI release this week is a concrete risk.
   - Reference 13F positioning — crowded trades that could unwind.

{screening_instructions}

## Tools

{search_docs}\
- `web_read(url)` — Read a web page for full article content (~4000 chars). Unlimited calls.
- `get_fred_data(series_id)` — Fetch a FRED economic data series (last 5 observations).

## When to web_search (budget is tight — save for what data can't tell you)
- **Fed rhetoric** — what did Powell/Waller/Bostic say? Any shift in forward guidance or dot plot?
- **Geopolitical / trade policy** — tariff announcements, sanctions, trade negotiations, military escalation.
- **Breaking developments** — anything in the last 24h not yet reflected in the pre-fetched data.
- Do NOT search for data you already have (prices, FRED indicators, event calendar, earnings outcomes).

## Rules
- Ground regime assessment in the pre-fetched data, not speculation. Cite specific numbers.
- Cross-reference sources: benchmark prices vs FRED vs events vs themes vs 13F. Flag contradictions.
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
    results = await asyncio.gather(
        _fetch_fred_data(deps.fred),
        _fetch_benchmark_data(deps.yfinance),
        _fetch_event_context(deps),
        return_exceptions=True,
    )

    if isinstance(results[0], BaseException):
        logger.warning("FRED data fetch failed — proceeding without", exc_info=results[0])
        fred_data: dict[str, Any] = {}
    else:
        fred_data = results[0]

    if isinstance(results[1], BaseException):
        logger.warning("Benchmark data fetch failed — proceeding without", exc_info=results[1])
        benchmark_data: dict[str, dict[str, Any]] = {}
    else:
        benchmark_data = results[1]

    if isinstance(results[2], BaseException):
        logger.warning("Event context fetch failed — proceeding without", exc_info=results[2])
        upcoming: list[Any] = []
        enriched_events: list[dict[str, Any]] = []
        filing_briefs: list[dict[str, Any]] = []
    else:
        upcoming, enriched_events, filing_briefs = results[2]

    logger.info(
        "MacroStrategist context fetched",
        benchmarks=len(benchmark_data),
        recent_events=len(enriched_events),
        enriched_with_outcome=sum(1 for e in enriched_events if e.get("outcome")),
        filing_briefs=len(filing_briefs),
        upcoming_events=len(upcoming),
    )

    # Format all context sections
    fred_context = _format_fred_context(fred_data)
    benchmark_context = _format_benchmark_context(benchmark_data)
    event_context = _format_event_context(upcoming, enriched_events, filing_briefs)
    themes_context = _format_macro_themes(state)
    ticker_context = _format_ticker_pool(state)

    has_tickers = bool(state.get("target_tickers"))

    # Build prompt
    user_prompt = "\n\n".join(
        section
        for section in [
            fred_context,
            benchmark_context,
            event_context,
            themes_context,
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
