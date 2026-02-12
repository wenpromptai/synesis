"""Stage 1: Fast Entity Extractor for breaking news.

This module provides fast, tool-free entity extraction that:
- Extracts primary entity and all mentioned entities
- Generates search keywords for web and Polymarket research
- Determines news category and event type
- Estimates urgency and impact for Stage 2 gating

Stage 1 makes minimal judgment calls (impact estimate only).
Tickers, sectors, direction deferred to Stage 2 Smart Analyzer.
"""

from functools import lru_cache

from pydantic_ai import Agent
from pydantic_ai.output import PromptedOutput

from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model
from synesis.processing.news.models import (
    LightClassification,
    UnifiedMessage,
)

logger = get_logger(__name__)

# System prompt for Stage 1: Entity Extraction (minimal judgment calls)
CLASSIFIER_SYSTEM_PROMPT = """Fast entity extractor. Extract entities, keywords, numeric data, urgency, impact. NO tickers/sectors/direction.

## Extract

1. **primary_entity**: Main entity AFFECTED (not who's commenting)
   "Trump praises Visa" → "Visa" | "Fed cuts rates" → "Federal Reserve"

2. **all_entities**: All companies/people/institutions mentioned. EXCLUDE news sources (@DeItaone) and URLs.

3. **news_category**: breaking | economic_calendar | other

4. **event_type**: macro | earnings | geopolitical | corporate | regulatory | crypto | political | other

5. **summary**: One sentence.

6. **search_keywords**: Generate 3 web search queries for CURRENT context only:

   - Query 1: "{primary_entity} {event}"
     → Core query about what happened
   - Query 2: "{primary_entity} {event} market reaction" or "{primary_entity} {event} stocks"
     → Immediate market impact
   - Query 3: "{primary_entity} {event} forecast" or "{primary_entity} {event} expectations"
     → Forward-looking analyst views

   IMPORTANT:
   - Do NOT search for historical data here (Stage 2 will do that with tools)
   - Focus on CURRENT event and immediate reaction
   - Always include year (2026) for recency

   Examples:
   - "Fed cuts rates 25bps" →
     ["Fed rate cut 25bps 2026", "Fed rate cut market reaction", "Fed rate forecast"]
   - "Apple Q4 earnings beat" →
     ["Apple Q4 earnings 2026", "Apple earnings market reaction", "Apple stock forecast"]
   - "Trump announces China tariffs" →
     ["Trump China tariff 2026", "China tariff stocks affected", "trade war market reaction"]

7. **polymarket_keywords**: 3-5 keywords for prediction markets:

   Pattern: [primary phrase, 2-3 variations, related terms]

   Examples:
   - "Fed rate cut" → ["Fed rate cut", "interest rate", "FOMC", "Federal Reserve"]
   - "Trump China tariff" → ["Trump tariff", "China tariff", "trade war", "US China trade"]
   - "Apple earnings" → ["Apple earnings", "AAPL earnings", "Apple Q4 results"]

8. **numeric_data** (economic/earnings only):
   Extract metrics: actual (required), estimate ("vs X est"), previous ("prev X"), unit (%, bps, $, B, M), period
   beat_miss: inflation lower=beat | growth higher=beat | inline if within 0.1 | unknown if no estimate
   surprise_magnitude: actual - estimate

9. **urgency** + urgency_reasoning (1 sentence):
   - critical: Surprise Fed, breaking M&A, unexpected policy
   - high: Scheduled data release, earnings beat/miss
   - normal: General news, routine announcements
   - low: Opinions, commentary, promotional/spam content
   Mark LOW if: promotional, giveaways, engagement bait, self-promotion, no market relevance

10. **impact** + impact_reasoning (1 sentence):
    - high: Major policy, earnings surprise >10%, significant M&A, breaking geopolitical
    - medium: Sector news, guidance, regulatory, expected data
    - low: Minor updates, commentary, routine, no market relevance
    Consider: New info or priced in? Major indices or niche? Historical reaction?

## DO NOT Extract (Stage 2)
Tickers, sectors, market direction - Stage 2 determines with research."""


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
        output_type=PromptedOutput(LightClassification),
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
        from synesis.processing.news.categorizer import (
            categorize_by_rules,
            classify_urgency_by_rules,
        )

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
