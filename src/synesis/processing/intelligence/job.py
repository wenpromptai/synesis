"""Intelligence pipeline jobs.

Two entry points:

1. ``run_scan_brief`` — daily scheduled scan. Discovers signals, runs
   MacroStrategist for regime + watchlist, saves brief to KG, sends to Discord.

2. ``run_ticker_analysis`` — on-demand deep analysis. Takes specific tickers,
   runs company/price/debate/trader, saves brief to KG as
   ``{date}-tradeideas.md``, and returns the full result.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.notifications.discord import send_discord
from synesis.processing.intelligence.compiler import (
    format_brief_as_markdown,
    format_scan_brief_as_markdown,
)
from synesis.processing.intelligence.discord_format import format_scan_brief
from synesis.processing.intelligence.graph import build_analyze_graph, build_scan_graph

if TYPE_CHECKING:
    from synesis.providers.crawler.crawl4ai import Crawl4AICrawlerProvider
    from synesis.providers.fred.client import FREDClient
    from synesis.providers.massive.client import MassiveClient
    from synesis.providers.sec_edgar.client import SECEdgarClient
    from synesis.providers.yfinance.client import YFinanceClient
    from synesis.storage.database import Database

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Scan Brief — daily scheduled pipeline
# ═══════════════════════════════════════════════════════════════════


async def run_scan_brief(
    db: Database,
    sec_edgar: SECEdgarClient,
    yfinance: YFinanceClient,
    fred: FREDClient | None = None,
    crawler: Crawl4AICrawlerProvider | None = None,
) -> dict[str, Any]:
    """Run the daily scan pipeline and send brief to Discord.

    Returns the compiled scan brief dict.
    """
    start = time.monotonic()
    current_date = datetime.now(UTC).date()

    logger.info("Scan pipeline starting", date=current_date.isoformat())

    graph = build_scan_graph(
        db=db,
        sec_edgar=sec_edgar,
        yfinance=yfinance,
        fred=fred,
        crawler=crawler,
    )

    result = await graph.ainvoke(
        {"current_date": current_date.isoformat()},
        config={"recursion_limit": 20},
    )

    brief: dict[str, Any] = result.get("brief", {})
    elapsed = time.monotonic() - start

    if not brief.get("date"):
        logger.error(
            "Scan pipeline produced no brief",
            date=current_date.isoformat(),
            elapsed_s=round(elapsed, 1),
        )
        return brief

    errors = brief.get("errors", {})
    logger.info(
        "Scan pipeline complete",
        date=current_date.isoformat(),
        regime=brief.get("macro", {}).get("regime"),
        watchlist_tickers=len(brief.get("watchlist", {}).get("selected", [])),
        elapsed_s=round(elapsed, 1),
        had_errors=any(errors.get(k) for k in errors),
    )

    # Save brief to KG
    _save_brief_to_kg(brief, format_scan_brief_as_markdown)

    # Send to Discord
    settings = get_settings()
    webhook = settings.discord_brief_webhook_url or settings.discord_webhook_url
    if not webhook:
        logger.error("No Discord webhook configured — scan brief will not be delivered")
        return brief

    batches = format_scan_brief(brief)
    for i, batch in enumerate(batches):
        if batch:
            try:
                await send_discord(batch, webhook_url_override=webhook)
            except Exception:
                logger.exception(
                    "Failed to send Discord batch", batch_index=i, batch_embeds=len(batch)
                )
            if i < len(batches) - 1:
                await asyncio.sleep(1.0)

    return brief


# ═══════════════════════════════════════════════════════════════════
#  Ticker Analysis — on-demand deep analysis
# ═══════════════════════════════════════════════════════════════════


async def run_ticker_analysis(
    tickers: list[str],
    sec_edgar: SECEdgarClient,
    yfinance: YFinanceClient,
    massive: MassiveClient | None = None,
    crawler: Crawl4AICrawlerProvider | None = None,
    twitter_api_key: str | None = None,
) -> dict[str, Any]:
    """Run deep ticker analysis and return the compiled brief.

    Saves brief to KG as
    ``{date}-tradeideas.md``.
    """
    start = time.monotonic()
    current_date = datetime.now(UTC).date()

    logger.info(
        "Ticker analysis starting",
        date=current_date.isoformat(),
        tickers=tickers,
    )

    graph = build_analyze_graph(
        sec_edgar=sec_edgar,
        yfinance=yfinance,
        massive=massive,
        crawler=crawler,
        twitter_api_key=twitter_api_key,
    )

    result = await graph.ainvoke(
        {"current_date": current_date.isoformat(), "target_tickers": tickers},
        config={"recursion_limit": 50},
    )

    brief: dict[str, Any] = result.get("brief", {})
    elapsed = time.monotonic() - start

    if not brief.get("date"):
        logger.error(
            "Ticker analysis produced no brief",
            date=current_date.isoformat(),
            tickers=tickers,
            elapsed_s=round(elapsed, 1),
        )
        return brief

    trade_ideas = brief.get("trade_ideas", [])
    logger.info(
        "Ticker analysis complete",
        date=current_date.isoformat(),
        tickers_analyzed=len(brief.get("tickers_analyzed", [])),
        trade_ideas=len(trade_ideas),
        elapsed_s=round(elapsed, 1),
    )

    # Save brief to KG as {date}-tradeideas.md
    _save_brief_to_kg(brief, format_brief_as_markdown, suffix="-tradeideas")

    return brief


# ═══════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════


def _save_brief_to_kg(
    brief: dict[str, Any],
    formatter: Callable[[dict[str, Any]], str],
    suffix: str = "",
) -> None:
    """Save compiled brief as markdown to the knowledge graph raw directory.

    Args:
        brief: Compiled brief dict.
        formatter: Converts brief dict to markdown string.
        suffix: Optional filename suffix (e.g. "-tradeideas").

    Output: ``docs/kg/raw/synesis_briefs/{date}{suffix}.md``
    """
    brief_date = brief.get("date", "")
    if not brief_date:
        logger.warning("Brief has no date — skipping KG save")
        return

    try:
        settings = get_settings()
        kg_dir = Path(settings.kg_briefs_dir)
        if not kg_dir.is_absolute():
            project_root = Path.cwd()
            kg_dir = project_root / kg_dir
        kg_dir.mkdir(parents=True, exist_ok=True)
        brief_path = kg_dir / f"{brief_date}{suffix}.md"
        brief_path.write_text(formatter(brief), encoding="utf-8")
        logger.info("Brief saved to KG", path=str(brief_path))
    except Exception:
        logger.error("Failed to save brief to KG", brief_date=brief_date, exc_info=True)
