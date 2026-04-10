"""Intelligence pipeline job — run the LangGraph pipeline and send to Discord.

Also saves each brief as markdown to the knowledge graph at
``docs/kg/raw/synesis_briefs/`` for future LLM compilation.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.notifications.discord import send_discord
from synesis.processing.intelligence.compiler import format_brief_as_markdown
from synesis.processing.intelligence.discord_format import format_intelligence_brief
from synesis.processing.intelligence.graph import build_intelligence_graph

if TYPE_CHECKING:
    from synesis.providers.crawler.crawl4ai import Crawl4AICrawlerProvider
    from synesis.providers.fred.client import FREDClient
    from synesis.providers.massive.client import MassiveClient
    from synesis.providers.sec_edgar.client import SECEdgarClient
    from synesis.providers.yfinance.client import YFinanceClient
    from synesis.storage.database import Database

logger = get_logger(__name__)


async def run_intelligence_brief(
    db: Database,
    sec_edgar: SECEdgarClient,
    yfinance: YFinanceClient,
    fred: FREDClient | None = None,
    massive: MassiveClient | None = None,
    crawler: Crawl4AICrawlerProvider | None = None,
) -> dict[str, Any]:
    """Run the daily intelligence pipeline and send brief to Discord.

    Returns the compiled brief dict.
    """
    start = time.monotonic()
    current_date = datetime.now(UTC).date()

    logger.info("Intelligence pipeline starting", date=current_date.isoformat())

    graph = build_intelligence_graph(
        db=db,
        sec_edgar=sec_edgar,
        yfinance=yfinance,
        fred=fred,
        massive=massive,
        crawler=crawler,
    )

    result = await graph.ainvoke(
        {"current_date": current_date.isoformat()},
        config={"recursion_limit": 50},
    )

    brief: dict[str, Any] = result.get("brief", {})
    elapsed = time.monotonic() - start

    if not brief.get("date"):
        logger.error(
            "Intelligence pipeline produced no brief",
            date=current_date.isoformat(),
            elapsed_s=round(elapsed, 1),
        )
        return brief

    tickers = brief.get("tickers_analyzed", [])
    trade_ideas = brief.get("trade_ideas", [])
    errors = brief.get("errors", {})

    logger.info(
        "Intelligence pipeline complete",
        date=current_date.isoformat(),
        tickers_analyzed=len(tickers),
        trade_ideas=len(trade_ideas),
        elapsed_s=round(elapsed, 1),
        had_errors=any(errors.get(k) for k in ("social_failed", "news_failed", "macro_failed"))
        or any(
            errors.get(k)
            for k in (
                "company_failures",
                "price_failures",
                "bull_failures",
                "bear_failures",
                "trader_failures",
            )
        ),
    )

    # Save brief to KG for future compilation (never raises)
    _save_brief_to_kg(brief)

    # Send to Discord (isolate batch failures so remaining batches still send)
    settings = get_settings()
    webhook = settings.discord_events_webhook_url or settings.discord_webhook_url
    if not webhook:
        logger.error("No Discord webhook configured — intelligence brief will not be delivered")
        return brief

    batches = format_intelligence_brief(brief)
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


def _save_brief_to_kg(brief: dict[str, Any]) -> None:
    """Save compiled brief as markdown to the knowledge graph raw directory.

    Synchronous — logs error on failure, never raises.
    Output: ``docs/kg/raw/synesis_briefs/YYYY-MM-DD.md``
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
        brief_path = kg_dir / f"{brief_date}.md"
        brief_path.write_text(format_brief_as_markdown(brief), encoding="utf-8")
        logger.info("Brief saved to KG", path=str(brief_path))
    except Exception:
        logger.error("Failed to save brief to KG", brief_date=brief_date, exc_info=True)
