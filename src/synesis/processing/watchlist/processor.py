"""Flow 4: Watchlist Intelligence Processor.

Periodically sweeps watchlist tickers with fundamental providers
(FactSet, SEC EDGAR, NASDAQ) and produces LLM-synthesized reports.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.output import PromptedOutput

from synesis.core.constants import WATCHLIST_INTEL_DEFAULT_BATCH_SIZE
from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model
from synesis.processing.watchlist.models import (
    CatalystAlert,
    TickerIntelligence,
    TickerReport,
    WatchlistSignal,
)

if TYPE_CHECKING:
    from synesis.processing.common.watchlist import WatchlistManager
    from synesis.providers.base import TickerProvider
    from synesis.providers.factset.provider import FactSetProvider
    from synesis.providers.nasdaq.client import NasdaqClient
    from synesis.providers.sec_edgar.client import SECEdgarClient
    from synesis.storage.database import Database

logger = get_logger(__name__)


# =============================================================================
# LLM Output Model
# =============================================================================


class WatchlistSynthesis(BaseModel):
    """LLM output for watchlist synthesis."""

    ticker_reports: list[TickerReport] = Field(default_factory=list)
    summary: str = ""


# =============================================================================
# PydanticAI Deps
# =============================================================================


@dataclass
class WatchlistDeps:
    """Dependencies for the watchlist LLM agent."""

    intelligence: list[TickerIntelligence]
    alerts: list[CatalystAlert]


# =============================================================================
# System Prompt
# =============================================================================

WATCHLIST_SYSTEM_PROMPT = """You are a senior equity research analyst producing a concise watchlist intelligence report.

You receive fundamental data gathered from FactSet, SEC EDGAR, and NASDAQ for a set of tickers on a shared watchlist.

## Your Tasks

### 1. Per-Ticker Reports
For each ticker, produce a TickerReport:
- **fundamental_summary**: 2-3 sentence overview of financial health and valuation
- **catalyst_flags**: upcoming events that could move the stock (earnings, filings, insider activity)
- **risk_flags**: concerns or red flags
- **overall_outlook**: bullish / bearish / neutral
- **confidence**: 0.0-1.0

### 2. Overall Summary
A 2-3 sentence narrative covering the watchlist as a whole:
- Which tickers have the most compelling setups?
- Any sector-wide themes?
- Key dates to watch

## Guidelines
- Be concise — this is a monitoring report, not a deep dive
- Flag actionable catalysts (earnings within 7 days, material insider activity)
- If data is missing for a ticker, note it and skip that section
- Confidence should reflect data completeness — low data = low confidence"""


# =============================================================================
# Processor
# =============================================================================


class WatchlistProcessor:
    """Watchlist intelligence processor.

    Gathers fundamental data for watchlist tickers, detects rule-based alerts,
    and produces LLM-synthesized reports.
    """

    def __init__(
        self,
        watchlist: WatchlistManager,
        factset: FactSetProvider | None = None,
        sec_edgar: SECEdgarClient | None = None,
        nasdaq: NasdaqClient | None = None,
        db: Database | None = None,
        ticker_provider: TickerProvider | None = None,
    ) -> None:
        self._watchlist = watchlist
        self._factset = factset
        self._sec_edgar = sec_edgar
        self._nasdaq = nasdaq
        self._db = db
        self._ticker_provider = ticker_provider
        self._agent: Agent[WatchlistDeps, WatchlistSynthesis] | None = None

    @property
    def agent(self) -> Agent[WatchlistDeps, WatchlistSynthesis]:
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent

    def _create_agent(self) -> Agent[WatchlistDeps, WatchlistSynthesis]:
        model = create_model(smart=True)

        agent: Agent[WatchlistDeps, WatchlistSynthesis] = Agent(
            model,
            deps_type=WatchlistDeps,
            output_type=PromptedOutput(WatchlistSynthesis),
            system_prompt=WATCHLIST_SYSTEM_PROMPT,
        )

        @agent.system_prompt
        def inject_data(ctx: RunContext[WatchlistDeps]) -> str:
            sections: list[str] = []

            # Ticker intelligence
            for intel in ctx.deps.intelligence:
                lines = [f"## {intel.ticker}"]
                if intel.company_name:
                    lines.append(f"Company: {intel.company_name}")
                if intel.market_cap is not None:
                    lines.append(f"Market Cap: ${intel.market_cap / 1e9:,.1f}B")
                if intel.fundamentals:
                    lines.append(f"Fundamentals: {intel.fundamentals}")
                if intel.price_change_1d is not None:
                    lines.append(f"Price Change 1D: {intel.price_change_1d:+.2f}%")
                if intel.price_change_1w is not None:
                    lines.append(f"Price Change 1W: {intel.price_change_1w:+.2f}%")
                if intel.price_change_1m is not None:
                    lines.append(f"Price Change 1M: {intel.price_change_1m:+.2f}%")
                if intel.recent_filings:
                    lines.append(f"Recent Filings: {intel.recent_filings}")
                if intel.insider_sentiment:
                    lines.append(f"Insider Sentiment: {intel.insider_sentiment}")
                if intel.recent_insider_txns:
                    lines.append(f"Recent Insider Txns: {intel.recent_insider_txns}")
                if intel.eps_history:
                    lines.append(f"EPS History: {intel.eps_history}")
                if intel.next_earnings:
                    lines.append(f"Next Earnings: {intel.next_earnings}")
                sections.append("\n".join(lines))

            # Alerts
            if ctx.deps.alerts:
                alert_lines = ["## Alerts"]
                for alert in ctx.deps.alerts:
                    alert_lines.append(
                        f"- [{alert.severity.upper()}] {alert.ticker}: "
                        f"{alert.alert_type} — {alert.summary}"
                    )
                sections.append("\n".join(alert_lines))

            return "\n\n".join(sections)

        return agent

    async def gather_intelligence(self, ticker: str) -> TickerIntelligence:
        """Fetch all provider data for one ticker (parallel).

        Watchlist stores bare tickers (e.g. ``"AAPL"``).  FactSet needs
        ``ticker-region`` format (e.g. ``"AAPL-US"``), so we resolve at
        runtime via ``ticker_provider.verify_ticker()``.  SEC EDGAR and
        NASDAQ work with bare tickers directly.
        """
        intel = TickerIntelligence(ticker=ticker)

        # Resolve bare ticker → ticker_region for FactSet
        ticker_region: str | None = None
        if self._ticker_provider:
            try:
                is_valid, region, _company = await self._ticker_provider.verify_ticker(ticker)
                if is_valid and region:
                    ticker_region = region
            except Exception as e:
                logger.warning("ticker_provider resolution failed, using bare ticker for FactSet", ticker=ticker, error=str(e))

        async def fetch_factset() -> None:
            if self._factset is None:
                return
            factset_ticker = ticker_region or ticker
            try:
                security = await self._factset.resolve_ticker(factset_ticker)
                if security:
                    intel.company_name = security.name

                market_cap = await self._factset.get_market_cap(factset_ticker)
                intel.market_cap = market_cap

                fundamentals = await self._factset.get_fundamentals(factset_ticker, "ltm", 1)
                if fundamentals:
                    f = fundamentals[0]
                    intel.fundamentals = {
                        "eps_diluted": f.eps_diluted,
                        "price_to_book": f.price_to_book,
                        "price_to_sales": f.price_to_sales,
                        "ev_to_ebitda": f.ev_to_ebitda,
                        "roe": f.roe,
                        "net_margin": f.net_margin,
                        "gross_margin": f.gross_margin,
                        "debt_to_equity": f.debt_to_equity,
                    }

                price = await self._factset.get_price(factset_ticker)
                if price:
                    intel.price_change_1d = price.one_day_pct
                    intel.price_change_1m = price.one_mth_pct
            except Exception as e:
                logger.warning("FactSet fetch failed", ticker=ticker, error=str(e))

        async def fetch_sec_edgar() -> None:
            if self._sec_edgar is None:
                return
            try:
                filings = await self._sec_edgar.get_filings(ticker, limit=5)
                intel.recent_filings = [
                    {"form": f.form, "filed_date": f.filed_date.isoformat()} for f in filings
                ]

                txns = await self._sec_edgar.get_insider_transactions(ticker, limit=5)
                intel.recent_insider_txns = [
                    {
                        "owner": t.owner_name,
                        "code": t.transaction_code,
                        "shares": t.shares,
                        "price": t.price_per_share,
                        "date": t.filing_date.isoformat(),
                    }
                    for t in txns
                ]

                sentiment = await self._sec_edgar.get_insider_sentiment(ticker)
                intel.insider_sentiment = sentiment

                eps = await self._sec_edgar.get_historical_eps(ticker, limit=4)
                intel.eps_history = eps
            except Exception as e:
                logger.warning("SEC EDGAR fetch failed", ticker=ticker, error=str(e))

        async def fetch_nasdaq() -> None:
            if self._nasdaq is None:
                return
            try:
                upcoming = await self._nasdaq.get_upcoming_earnings([ticker])
                if upcoming:
                    ev = upcoming[0]
                    intel.next_earnings = {
                        "date": ev.earnings_date.isoformat(),
                        "time": ev.time,
                        "eps_forecast": ev.eps_forecast,
                    }
            except Exception as e:
                logger.warning("NASDAQ fetch failed", ticker=ticker, error=str(e))

        await asyncio.gather(fetch_factset(), fetch_sec_edgar(), fetch_nasdaq())
        return intel

    def detect_alerts(
        self,
        intel: TickerIntelligence,
        earnings_alert_days: int = 7,
    ) -> list[CatalystAlert]:
        """Rule-based alert detection (no LLM needed)."""
        alerts: list[CatalystAlert] = []

        # Earnings imminent
        if intel.next_earnings and intel.next_earnings.get("date"):
            try:
                earnings_date = datetime.fromisoformat(intel.next_earnings["date"]).replace(
                    tzinfo=UTC
                )
                days_until = (earnings_date - datetime.now(UTC)).days
                if 0 <= days_until <= earnings_alert_days:
                    severity = "high" if days_until <= 2 else "medium"
                    alerts.append(
                        CatalystAlert(
                            ticker=intel.ticker,
                            alert_type="earnings_imminent",
                            severity=severity,
                            summary=f"Earnings in {days_until} day(s) on {intel.next_earnings['date']}",
                            data=intel.next_earnings,
                        )
                    )
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Failed to parse earnings date",
                    ticker=intel.ticker,
                    error=str(e),
                )

        # Insider buying/selling
        if intel.insider_sentiment and intel.insider_sentiment.get("mspr") is not None:
            mspr = intel.insider_sentiment["mspr"]
            if mspr > 0.25:
                alerts.append(
                    CatalystAlert(
                        ticker=intel.ticker,
                        alert_type="insider_buying",
                        severity="high" if mspr > 0.50 else "medium",
                        summary=f"Net insider buying (MSPR: {mspr:.1f})",
                        data=intel.insider_sentiment,
                    )
                )
            elif mspr < -0.25:
                alerts.append(
                    CatalystAlert(
                        ticker=intel.ticker,
                        alert_type="insider_selling",
                        severity="high" if mspr < -0.50 else "medium",
                        summary=f"Net insider selling (MSPR: {mspr:.1f})",
                        data=intel.insider_sentiment,
                    )
                )

        # New material SEC filing (8-K within 24h)
        if intel.recent_filings:
            now = datetime.now(UTC)
            for filing in intel.recent_filings:
                if filing.get("form") == "8-K" and filing.get("filed_date"):
                    try:
                        filed = datetime.fromisoformat(filing["filed_date"]).replace(tzinfo=UTC)
                        if (now - filed) < timedelta(hours=24):
                            alerts.append(
                                CatalystAlert(
                                    ticker=intel.ticker,
                                    alert_type="new_sec_filing",
                                    severity="high",
                                    summary=f"8-K filed {filing['filed_date']}",
                                    data=filing,
                                )
                            )
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            "Failed to parse filing date",
                            ticker=intel.ticker,
                            error=str(e),
                        )

        # Valuation extreme
        if intel.fundamentals:
            ev_ebitda = intel.fundamentals.get("ev_to_ebitda")
            if ev_ebitda is not None and ev_ebitda > 50:
                alerts.append(
                    CatalystAlert(
                        ticker=intel.ticker,
                        alert_type="valuation_extreme",
                        severity="low",
                        summary=f"EV/EBITDA {ev_ebitda:.1f}x — elevated valuation",
                        data={"ev_to_ebitda": ev_ebitda},
                    )
                )

        return alerts

    async def run_analysis(self) -> WatchlistSignal:
        """Full cycle: gather -> detect alerts -> LLM synthesis -> signal."""
        from synesis.config import get_settings

        settings = get_settings()
        now = datetime.now(UTC)

        # 1. Get watchlist tickers
        tickers = await self._watchlist.get_all()
        max_tickers = settings.watchlist_intel_max_tickers
        tickers = tickers[:max_tickers]

        if not tickers:
            logger.info("Watchlist empty, nothing to analyze")
            return WatchlistSignal(timestamp=now, summary="Watchlist is empty.")

        logger.debug("Watchlist intel starting", ticker_count=len(tickers))

        # 2. Gather intelligence (parallel, batched)
        batch_size = WATCHLIST_INTEL_DEFAULT_BATCH_SIZE
        all_intel: list[TickerIntelligence] = []
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            results = await asyncio.gather(
                *[self.gather_intelligence(t) for t in batch],
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, TickerIntelligence):
                    all_intel.append(result)
                else:
                    logger.warning("Ticker intelligence failed", error=str(result))

        # 3. Detect rule-based alerts
        earnings_days = settings.watchlist_intel_earnings_alert_days
        all_alerts: list[CatalystAlert] = []
        for intel in all_intel:
            all_alerts.extend(self.detect_alerts(intel, earnings_alert_days=earnings_days))

        # 4. LLM synthesis
        ticker_reports: list[TickerReport] = []
        summary = ""
        if all_intel:
            try:
                deps = WatchlistDeps(intelligence=all_intel, alerts=all_alerts)
                user_prompt = (
                    "Analyze the watchlist data above. "
                    "Produce a TickerReport for each ticker and an overall summary."
                )
                agent_result = await self.agent.run(user_prompt, deps=deps)
                ticker_reports = agent_result.output.ticker_reports
                summary = agent_result.output.summary
            except Exception as e:
                logger.exception("Watchlist LLM synthesis failed", error=str(e))
                summary = f"LLM synthesis failed: {e}"

        # 5. Build signal
        signal = WatchlistSignal(
            timestamp=now,
            tickers_analyzed=len(all_intel),
            ticker_reports=ticker_reports,
            alerts=all_alerts,
            summary=summary,
        )

        # 6. Store in DB (generic signal insert)
        if self._db:
            try:
                import orjson

                payload = signal.model_dump(mode="json")
                tickers_list = [r.ticker for r in ticker_reports]
                query = """
                    INSERT INTO signals (time, flow_id, signal_type, payload, tickers)
                    VALUES ($1, $2, $3, $4, $5)
                """
                await self._db.execute(
                    query,
                    now,
                    "watchlist",
                    "watchlist_intel",
                    orjson.dumps(payload).decode("utf-8"),
                    tickers_list or None,
                )
                logger.debug("Watchlist signal stored to DB")
            except Exception as e:
                logger.error("Failed to store watchlist signal", error=str(e))

        logger.info(
            "Watchlist intel complete",
            tickers_analyzed=signal.tickers_analyzed,
            alerts=len(signal.alerts),
            reports=len(signal.ticker_reports),
        )

        return signal
