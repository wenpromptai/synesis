"""LLM event extractor — extracts CalendarEvent list from crawled markdown.

Uses PydanticAI agent with structured output to identify dated events
from crawled web page content.
"""

from __future__ import annotations

from datetime import date

from pydantic_ai import Agent

from synesis.core.constants import EXTRACTOR_MAX_CONTENT_CHARS
from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model
from synesis.processing.events.models import CalendarEvent, ExtractedEvents

logger = get_logger(__name__)

SYSTEM_PROMPT = """\
You are an event extraction agent. Given crawled web page content, extract
market-relevant events that have SPECIFIC DATES.

Rules:
- Only extract events with concrete dates (not "coming soon" or "TBD")
- Each event needs: title, date, category, region, importance score
- Score importance 1-10:
  - 8-10: Central bank decisions, major product launches from key companies,
          sector-defining events (e.g., NVIDIA GTC keynote, Fed rate decision)
  - 5-7: Conferences, significant earnings, regulatory proposals,
          important industry reports
  - 1-4: Minor conferences, small product updates, routine meetings
- Map to tickers: direct (NVDA for GTC) AND indirect effects
  (e.g., "NVIDIA $4B fiber optics deal" -> NVDA, COHR, LITE)
- Assign sector: ai, semiconductors, ai_infrastructure, power, energy, precious_metals, or leave null for macro/general
- Confidence scoring:
  - 0.9+ for official announcements with confirmed dates
  - 0.5-0.8 for "sources say" or unconfirmed reports
  - <0.5 for rumors or speculation
- Region: where the event happens + which markets it impacts
  Use: US, JP, SG, HK, global
- Categories (pick the best fit):
  - earnings: Quarterly reports, guidance, EPS
  - economic_data: CPI, NFP, GDP, PPI, PCE, PMI releases
  - fed: FOMC rate decisions, minutes releases, speeches, testimony, press conferences.
            IMPORTANT: Extract FOMC meeting dates AND minutes release dates as SEPARATE events.
            Meeting → "FOMC Rate Decision", Minutes → "FOMC Minutes Released".
  - 13f_filing: Hedge fund 13F position disclosures
  - conference: GTC, WWDC, CES, OPEC meetings, investor days
  - release: AI model releases (ANY new model from ANY lab worldwide), chip launches,
            major product launches, new technology breakthroughs, major investments
            (e.g. NVIDIA investing $2B in Coherent), strategic partnerships,
            new business ventures, significant corporate announcements
  - regulatory: Legislation, SEC enforcement, antitrust, tariffs, FDA, M&A, IPOs, direct listings, splits, buybacks, spinoffs, management changes
  - other: Fallback for uncategorizable events

Today's date is {today}. Only extract events from today onward.
If the page has no extractable future-dated events, return an empty list.
"""


def _build_extractor() -> Agent[None, ExtractedEvents]:
    """Build the PydanticAI event extractor agent."""
    return Agent(
        model=create_model(smart=True),
        system_prompt=SYSTEM_PROMPT.format(today=date.today().isoformat()),
        output_type=ExtractedEvents,
        retries=1,
    )


async def extract_events_from_markdown(
    markdown: str,
    source_url: str,
    source_name: str = "",
    default_region: str = "US",
    default_tickers: list[str] | None = None,
) -> list[CalendarEvent]:
    """Extract calendar events from crawled markdown content.

    Args:
        markdown: Crawled page content in markdown format
        source_url: URL the content was crawled from
        source_name: Human-readable source name
        default_region: Default region if not clear from content
        default_tickers: Default tickers to associate (from curated source config)

    Returns:
        List of extracted CalendarEvent objects
    """
    if not markdown or len(markdown.strip()) < 50:
        logger.debug("Content too short for extraction", source=source_name)
        return []

    # Truncate very long content to avoid token limits
    if len(markdown) > EXTRACTOR_MAX_CONTENT_CHARS:
        markdown = markdown[:EXTRACTOR_MAX_CONTENT_CHARS] + "\n\n[Content truncated]"

    agent = _build_extractor()

    prompt = f"Source: {source_name} ({source_url})\nDefault region: {default_region}\n\n{markdown}"

    try:
        result = await agent.run(prompt)
        events = result.output.events
    except Exception:
        logger.exception("LLM event extraction failed", source=source_name)
        return []

    # Post-process: add source URL and default tickers
    for event in events:
        if source_url not in event.source_urls:
            event.source_urls.append(source_url)
        if default_tickers:
            for ticker in default_tickers:
                if ticker not in event.tickers:
                    event.tickers.append(ticker)

    logger.info(
        "Events extracted",
        source=source_name,
        count=len(events),
    )
    return events
