"""Stage 1: Fast Entity Extractor for breaking news.

This module provides fast, tool-free entity extraction that:
- Extracts primary entity and all mentioned entities
- Generates search keywords for web and Polymarket research
- Determines news category and event type

Stage 1 makes NO judgment calls (no tickers, sectors, impact, direction).
All informed judgments are deferred to Stage 2 Smart Analyzer.
"""

from functools import lru_cache

from pydantic_ai import Agent

from synesis.core.logging import get_logger
from synesis.processing.llm_factory import create_model
from synesis.processing.models import (
    LightClassification,
    UnifiedMessage,
)

logger = get_logger(__name__)

# System prompt for Stage 1: Entity Extraction (NO judgment calls)
CLASSIFIER_SYSTEM_PROMPT = """You are a fast entity extractor. Extract entities, keywords, numeric data, AND urgency - NO judgment calls on tickers/sectors/impact/direction.

## What to Extract

1. **Primary Entity**: The MAIN entity affected by this news.
   - "Trump praises Visa for rewards" → "Visa" (not Trump - he's commenting)
   - "Fed announces rate decision" → "Federal Reserve"
   - "Apple reports earnings" → "Apple"

2. **All Entities**: ALL companies, people, and institutions mentioned.
   - "Trump praises Visa, MasterCard could follow" → ["Visa", "MasterCard", "Trump"]
   - Include everyone mentioned, even peripherally
   - **EXCLUDE the news source/account** (e.g., @DeItaone, @FirstSquawk, @marketfeed are sources, not entities)
   - **EXCLUDE entities from URLs** (e.g., "reuters.com", "bloomberg.com", "zerohedge" in links are not entities)

3. **News Category**:
   - breaking: Unexpected events (*BREAKING, JUST IN, surprise announcements)
   - economic_calendar: Scheduled releases (CPI, FOMC, NFP, earnings)
   - other: Analysis, commentary, opinions

4. **Event Type**: macro | earnings | geopolitical | corporate | regulatory | crypto | political | other

5. **Summary**: One clear sentence describing what happened.

6. **Search Keywords** (for web research):
   Generate 2-3 specific search queries for research:
   - "{primary_entity} {event} analyst forecast"
   - "{topic} market impact historical"
   - "{primary_entity} news latest"

7. **Polymarket Keywords** (for prediction market search):
   Generate 3-5 keywords to search prediction markets.
   RULES:
   - Use PRIMARY ENTITY and specific topic
   - NO peripheral mentions (person commenting ≠ entity affected)
   - Examples:
     * "Trump announces Visa will change rewards" → ["Visa", "credit card", "rewards"]
     * "Fed cuts rates by 25bps" → ["Federal Reserve", "interest rate", "rate cut"]
     * "NVDA beats earnings by 20%" → ["Nvidia", "earnings", "AI chips"]

8. **Numeric Data** (for economic_calendar and earnings only):
   Extract ALL numeric metrics when present.

   Example input:
   "*AUSTRALIA CPI Q4: 2.4% Y/Y (vs 2.5% est, prev 2.8%)
    *AUSTRALIA TRIMMED MEAN CPI Q4: 3.2% Y/Y (vs 3.3% est)"

   Extract:
   - metric_name: "CPI Y/Y", actual: 2.4, estimate: 2.5, previous: 2.8, unit: "%", period: "Q4"
   - metric_name: "Trimmed Mean CPI Y/Y", actual: 3.2, estimate: 3.3, previous: null, unit: "%", period: "Q4"

   For each metric:
   - actual: The reported value (REQUIRED)
   - estimate: Expected/consensus value (if "vs X est" or "expected X")
   - previous: Prior period value (if "prev X" or "prior X")
   - unit: %, bps, $, B (billion), M (million), K (thousand)
   - period: Q1-Q4, month name, year

   Calculate beat_miss:
   - For inflation metrics (CPI, PPI, PCE): lower than estimate = beat, higher = miss
   - For growth metrics (GDP, earnings, revenue): higher than estimate = beat, lower = miss
   - inline: actual == estimate (within 0.1)
   - unknown: no estimate provided

   Calculate surprise_magnitude: actual - estimate (same units)
   Set headline_metric to the primary metric (usually the first or most important one).
   Set overall_beat_miss based on headline metric's status.

9. **Urgency Level** (for trading prioritization):
   - critical: Act IMMEDIATELY - surprise Fed decision, breaking M&A, unexpected rate cut/hike
   - high: Act FAST - scheduled economic data release, earnings with beat/miss, policy announcements
   - normal: Can wait - general news, updates, routine announcements
   - low: Background noise with NO market impact - opinions, analysis, commentary, retweets, threads, AND promotional/spam content

   **IMPORTANT: Mark as LOW urgency if the message is:**
   - Promotional content (subscribe, follow, boost, join, sign up)
   - Giveaways, airdrops, referral codes, affiliate links
   - Engagement bait (don't miss, last chance, act now, limited time)
   - Channel/account self-promotion
   - Content with no financial/market relevance

   Provide brief reasoning (1 sentence) for your urgency assessment.

   Examples:
   - "*FED CUTS RATES BY 50BPS VS 25BPS EXPECTED" → critical (surprise magnitude)
   - "CPI comes in at 2.4% vs 2.5% expected" → high (scheduled data, small beat)
   - "Analysts expect Fed to cut in March" → low (opinion/forecast)
   - "Apple reports Q4 earnings beat" → high (earnings with outcome)
   - "Boost this channel ♥️" → low (promotional content, no market impact)
   - "Subscribe for exclusive signals!" → low (promotional spam)
   - "GIVEAWAY: Free crypto for followers" → low (promotional spam)

## What NOT to Extract (moved to Stage 2)

DO NOT include in your output:
- Stock tickers (Stage 2 determines these with research)
- Sectors (Stage 2 determines these with research)
- Impact level (Stage 2 determines this with research)
- Market direction (Stage 2 determines this with research)

Your job is FAST entity extraction. Stage 2 makes all informed judgments."""


def create_classifier_agent() -> Agent[None, LightClassification]:
    """Create a PydanticAI agent for lightweight classification.

    Uses fast model (gpt-4o-mini) with NO tools for speed and cost.

    Returns:
        Agent configured for LightClassification output
    """
    # Use fast model for classification (smart=False)
    model = create_model(smart=False)

    agent: Agent[None, LightClassification] = Agent(
        model,
        output_type=LightClassification,
        system_prompt=CLASSIFIER_SYSTEM_PROMPT,
    )

    # NO tools - this is lightweight classification only

    return agent


class NewsClassifier:
    """Stage 1: Fast entity extractor for breaking news and analysis.

    Fast, tool-free extraction that:
    - Extracts entities and keywords (no judgment calls)
    - All judgment calls (tickers, sectors, impact, direction) happen in Stage 2
    """

    def __init__(self) -> None:
        self._agent: Agent[None, LightClassification] | None = None

    @property
    def agent(self) -> Agent[None, LightClassification]:
        """Get or create the classifier agent."""
        if self._agent is None:
            self._agent = create_classifier_agent()
        return self._agent

    async def classify(self, message: UnifiedMessage) -> LightClassification:
        """Classify a unified message.

        Uses hybrid categorization:
        1. Rule-based categorization first (fast, free)
        2. Rule-based urgency classification (fast, free)
        3. LLM classification for full analysis
        4. Rule-based category/urgency overrides LLM if rules matched

        Args:
            message: The message to classify

        Returns:
            LightClassification with extracted information
        """
        from synesis.processing.categorizer import categorize_by_rules, classify_urgency_by_rules

        # Try rule-based categorization first
        rule_category = categorize_by_rules(message.text)
        rule_urgency = classify_urgency_by_rules(message.text)

        # Build the prompt
        prompt = self._build_prompt(message)

        logger.debug(
            "Extracting entities (Stage 1)",
            message_id=message.external_id,
            source=message.source_account,
            text_preview=message.text[:100],
            rule_category=rule_category.value if rule_category else "ambiguous",
            rule_urgency=rule_urgency.value if rule_urgency else "ambiguous",
        )

        # Run extraction (NO tool calls - pure entity extraction)
        result = await self.agent.run(prompt)
        extraction = result.output

        # Apply rule-based category if rules matched (override LLM)
        if rule_category is not None:
            extraction.news_category = rule_category

        # Apply rule-based urgency if rules matched (override LLM)
        if rule_urgency is not None:
            extraction.urgency = rule_urgency
            extraction.urgency_reasoning = "rule-based match"

        logger.info(
            "Stage 1 extraction complete",
            message_id=message.external_id,
            news_category=extraction.news_category.value,
            event_type=extraction.event_type.value,
            primary_entity=extraction.primary_entity,
            all_entities=extraction.all_entities,
            polymarket_keywords=extraction.polymarket_keywords,
            urgency=extraction.urgency.value,
            has_numeric_data=extraction.numeric_data is not None,
        )

        return extraction

    def _build_prompt(self, message: UnifiedMessage) -> str:
        """Build the extraction prompt."""
        return f"""Extract entities and keywords from this financial message:

Source: {message.source_account} ({message.source_platform.value})
Timestamp: {message.timestamp.isoformat()}
Type: {message.source_type.value}

Message:
{message.text}

Focus on the PRIMARY ENTITY affected (not peripheral mentions).
Extract ALL entities mentioned for the all_entities list.
Generate search keywords for web research and Polymarket."""


@lru_cache(maxsize=1)
def get_classifier() -> NewsClassifier:
    """Get the singleton classifier instance (thread-safe via lru_cache)."""
    return NewsClassifier()


async def classify_message(message: UnifiedMessage) -> LightClassification:
    """Convenience function to classify a message.

    Args:
        message: The message to classify

    Returns:
        LightClassification with extracted information
    """
    classifier = get_classifier()
    return await classifier.classify(message)
