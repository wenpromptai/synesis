"""Two-stage news processor with unified analysis.

Architecture:
  Message → [Stage 1: Entity Extractor] → [Pre-fetch Context] → [Stage 2: Smart Analyzer]
                                                                        ↓
                                                                  Signal Output

Stage 1 (Entity Extractor):
- Fast, tool-free entity extraction
- Extracts entities and keywords for search
- NO judgment calls (tickers, sectors, impact, direction)

Stage 2 (Smart Analyzer):
- Takes message + extraction + web results + markets
- Makes ALL informed judgments with research context
- Returns unified SmartAnalysis output
"""

import asyncio
from dataclasses import dataclass

from redis.asyncio import Redis

from synesis.config import get_settings

from synesis.core.logging import get_logger
from synesis.markets.polymarket import PolymarketClient
from synesis.processing.classifier import NewsClassifier
from synesis.processing.deduplication import MessageDeduplicator, create_deduplicator
from synesis.processing.models import (
    Flow1Signal,
    LightClassification,
    MarketEvaluation,
    SmartAnalysis,
    UnifiedMessage,
)
from synesis.processing.smart_analyzer import SmartAnalyzer
from synesis.processing.web_search import (
    SearchProvidersExhaustedError,
    format_search_results,
    search_market_impact,
)

logger = get_logger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a single message through the two-stage pipeline."""

    # Original message
    message: UnifiedMessage

    # Processing status
    skipped: bool = False
    skip_reason: str | None = None

    # Deduplication
    is_duplicate: bool = False
    duplicate_of: str | None = None

    # Stage 1: Entity Extraction
    extraction: LightClassification | None = None

    # Stage 2: Smart Analysis (all informed judgments)
    analysis: SmartAnalysis | None = None

    # Timing
    processing_time_ms: float = 0.0

    def to_signal(self) -> Flow1Signal | None:
        """Convert to Flow1Signal for storage/notification.

        Returns None if the message was skipped or has no extraction.
        """
        if self.skipped or self.extraction is None:
            return None

        return Flow1Signal(
            timestamp=self.message.timestamp,
            source_platform=self.message.source_platform,
            source_account=self.message.source_account,
            source_type=self.message.source_type,
            raw_text=self.message.text,
            external_id=self.message.external_id,
            news_category=self.extraction.news_category,
            extraction=self.extraction,
            analysis=self.analysis,
            # Legacy fields for backwards compatibility
            classification=self.extraction,
            watchlist_tickers=self.analysis.tickers if self.analysis else [],
            watchlist_sectors=self.analysis.sectors if self.analysis else [],
            is_duplicate=self.is_duplicate,
            duplicate_of=self.duplicate_of,
            processing_time_ms=self.processing_time_ms,
            skipped_evaluation=self.analysis is None,
        )

    @property
    def has_edge(self) -> bool:
        """Check if any evaluation shows tradable edge."""
        if not self.analysis:
            return False
        return self.analysis.has_tradable_edge

    @property
    def best_opportunity(self) -> MarketEvaluation | None:
        """Get the evaluation with the highest edge."""
        if not self.analysis:
            return None
        return self.analysis.best_opportunity


class NewsProcessor:
    """Two-stage news processor with unified analysis.

    This class encapsulates the full news processing pipeline:
    1. Deduplication - skip already-seen messages
    2. Stage 1: Entity extraction (fast, no judgment calls)
    3. Pre-fetch context (web search + polymarket in parallel)
    4. Stage 2: Smart analysis (all informed judgments)

    Usage:
        processor = NewsProcessor(redis)
        await processor.initialize()
        result = await processor.process_message(message)
    """

    def __init__(self, redis: Redis) -> None:
        """Initialize the processor.

        Args:
            redis: Redis client for deduplication storage
        """
        self._redis = redis
        self._deduplicator: MessageDeduplicator | None = None
        self._classifier: NewsClassifier | None = None
        self._analyzer: SmartAnalyzer | None = None
        self._polymarket: PolymarketClient | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        logger.info("Initializing NewsProcessor (two-stage architecture)")

        # Create deduplicator (loads Model2Vec)
        self._deduplicator = await create_deduplicator(self._redis)

        # Create Stage 1 classifier (entity extraction, no judgment calls)
        self._classifier = NewsClassifier()

        # Create Polymarket client (shared with Stage 2)
        self._polymarket = PolymarketClient()

        # Create Stage 2 smart analyzer (all informed judgments)
        self._analyzer = SmartAnalyzer(polymarket_client=self._polymarket)

        self._initialized = True
        logger.info("NewsProcessor initialized (two-stage architecture)")

    async def close(self) -> None:
        """Clean up resources."""
        if self._analyzer:
            await self._analyzer.close()
        if self._polymarket:
            await self._polymarket.close()
        if self._deduplicator:
            await self._deduplicator.cleanup()

    async def __aenter__(self) -> "NewsProcessor":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""
        await self.close()

    @property
    def deduplicator(self) -> MessageDeduplicator:
        """Get the deduplicator (must be initialized)."""
        if self._deduplicator is None:
            raise RuntimeError("NewsProcessor not initialized. Call initialize() first.")
        return self._deduplicator

    @property
    def classifier(self) -> NewsClassifier:
        """Get the classifier (must be initialized)."""
        if self._classifier is None:
            raise RuntimeError("NewsProcessor not initialized. Call initialize() first.")
        return self._classifier

    @property
    def analyzer(self) -> SmartAnalyzer:
        """Get the smart analyzer (must be initialized)."""
        if self._analyzer is None:
            raise RuntimeError("NewsProcessor not initialized. Call initialize() first.")
        return self._analyzer

    @property
    def polymarket(self) -> PolymarketClient:
        """Get the Polymarket client (must be initialized)."""
        if self._polymarket is None:
            raise RuntimeError("NewsProcessor not initialized. Call initialize() first.")
        return self._polymarket

    async def _fetch_web_results(self, queries: list[str]) -> list[str]:
        """Fetch web search results for given queries.

        Args:
            queries: List of search queries

        Returns:
            List of formatted search result strings
        """
        settings = get_settings()
        results: list[str] = []
        for query in queries[: settings.web_search_max_queries]:
            try:
                search_results = await search_market_impact(query, count=5)
                results.append(format_search_results(search_results))
            except SearchProvidersExhaustedError:
                # All search providers failed - this is an infrastructure issue
                logger.error(
                    "All search providers exhausted",
                    query=query,
                )
                results.append("Search unavailable: all providers failed or not configured")
            except Exception as e:
                logger.warning("Web search failed", query=query, error=str(e))
                results.append(f"Search failed: {e}")
        return results

    async def process_message(self, message: UnifiedMessage) -> ProcessingResult:
        """Process a single message through the two-stage pipeline.

        Flow:
        1. Check for duplicates
        2. Stage 1: Entity extraction (fast, no judgment calls)
        3. Pre-fetch context: Web search + Polymarket (parallel)
        4. Stage 2: Smart analysis (all informed judgments with context)

        Args:
            message: The unified message to process

        Returns:
            ProcessingResult with extraction and analysis
        """
        import time

        start_time = time.perf_counter()

        log = logger.bind(
            message_id=message.external_id,
            source=message.source_account,
            platform=message.source_platform.value,
        )

        log.info("Processing message (two-stage)", text_preview=message.text[:100])

        # 1. Check for duplicates
        dedup_result = await self.deduplicator.process_message(message)
        if dedup_result.is_duplicate:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            log.info(
                "Message is duplicate, skipping",
                duplicate_of=dedup_result.duplicate_of,
            )
            return ProcessingResult(
                message=message,
                skipped=True,
                skip_reason="duplicate",
                is_duplicate=True,
                duplicate_of=dedup_result.duplicate_of,
                processing_time_ms=elapsed_ms,
            )

        # 2. Stage 1: Entity extraction (fast, no judgment calls)
        log.debug("Stage 1: Extracting entities")
        extraction = await self.classifier.classify(message)

        log.info(
            "Stage 1 complete",
            event_type=extraction.event_type.value,
            primary_entity=extraction.primary_entity,
            all_entities=extraction.all_entities,
        )

        # 3. Pre-fetch context (parallel)
        log.debug(
            "Pre-fetching context",
            search_keywords=extraction.search_keywords[:2],
            polymarket_keywords=extraction.polymarket_keywords,
        )

        # Fetch web results and Polymarket markets in parallel
        web_results, markets_text = await asyncio.gather(
            self._fetch_web_results(extraction.search_keywords),
            self.analyzer.search_polymarket(extraction.polymarket_keywords),
        )

        # 4. Stage 2: Smart analysis (all informed judgments)
        log.debug("Stage 2: Smart analysis with context")

        analysis = await self.analyzer.analyze(message, extraction, web_results, markets_text)

        if analysis is None:
            # Stage 2 failed - log error and continue with partial results
            log.error(
                "Stage 2 failed - analysis unavailable",
                message_id=message.external_id,
            )
        else:
            log.info(
                "Stage 2 complete",
                tickers=analysis.tickers,
                sectors=analysis.sectors,
                impact=analysis.predicted_impact.value,
                direction=analysis.market_direction.value,
                thesis=analysis.primary_thesis[:100] if analysis.primary_thesis else "None",
                thesis_confidence=f"{analysis.thesis_confidence:.0%}",
                markets_evaluated=len(analysis.market_evaluations),
                has_edge=analysis.has_tradable_edge,
            )

            # Log if edge found
            if analysis.has_tradable_edge and analysis.best_opportunity:
                best = analysis.best_opportunity
                log.info(
                    "Found opportunity with edge",
                    market_id=best.market_id,
                    edge=f"{best.edge:.2%}" if best.edge else "N/A",
                    verdict=best.verdict,
                    recommended_side=best.recommended_side,
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        result = ProcessingResult(
            message=message,
            skipped=False,
            extraction=extraction,
            analysis=analysis,
            processing_time_ms=elapsed_ms,
        )

        log.info(
            "Message processing complete",
            processing_time_ms=f"{elapsed_ms:.1f}",
            has_edge=result.has_edge,
        )

        return result

    async def check_duplicate(self, message: UnifiedMessage) -> bool:
        """Quick check if a message is a duplicate.

        This is useful for pre-filtering without full processing.

        Args:
            message: The message to check

        Returns:
            True if the message is a duplicate
        """
        result = await self.deduplicator.check_duplicate(message)
        return result.is_duplicate


# Module-level instance for convenience
_processor: NewsProcessor | None = None


async def get_processor(redis: Redis) -> NewsProcessor:
    """Get or create the shared processor instance.

    Args:
        redis: Redis client

    Returns:
        Initialized NewsProcessor
    """
    global _processor
    if _processor is None:
        _processor = NewsProcessor(redis)
        await _processor.initialize()
    return _processor
