"""Two-stage news processor.

Architecture:
  Message → [Stage 1: Impact scoring + ticker matching] → [Stage 2: LLM analysis]
                                                                    ↓
                                                              Signal Output

Stage 1 (Instant, no LLM):
- Impact scoring (rule-based urgency)
- Ticker matching (regex + curated names)

Stage 2 (LLM with tools):
- Entity extraction, sentiment, ETF impact
- Web search, page reading, Polymarket search via tools
"""

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from redis.asyncio import Redis

from synesis.config import get_settings

from synesis.core.logging import get_logger
from synesis.markets.polymarket import PolymarketClient
from synesis.processing.news import (
    NewsSignal,
    LightClassification,
    MarketEvaluation,
    MessageDeduplicator,
    NewsClassifier,
    SmartAnalysis,
    SmartAnalyzer,
    UnifiedMessage,
    UrgencyLevel,
    create_deduplicator,
)

logger = get_logger(__name__)

Stage1Callback = Callable[[UnifiedMessage, LightClassification], Awaitable[None]]


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

    def to_signal(self) -> NewsSignal | None:
        """Convert to NewsSignal for storage/notification.

        Returns None if the message was skipped or has no extraction.
        """
        if self.skipped or self.extraction is None:
            return None

        return NewsSignal(
            timestamp=self.message.timestamp,
            source_platform=self.message.source_platform,
            source_account=self.message.source_account,
            raw_text=self.message.text,
            external_id=self.message.external_id,
            extraction=self.extraction,
            analysis=self.analysis,
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
    3. Early exit if low/normal urgency (no Stage 2)
    4. Fire on_stage1_complete callback (e.g. Telegram notification)
    5. Pre-fetch context (web search + polymarket in parallel)
    6. Stage 2: Smart analysis (all informed judgments)

    Usage:
        processor = NewsProcessor(redis)
        await processor.initialize()
        result = await processor.process_message(message)
    """

    def __init__(
        self,
        redis: Redis,
    ) -> None:
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
            try:
                await self._analyzer.close()
            except Exception:
                logger.error("Error closing analyzer", exc_info=True)
        if self._polymarket:
            try:
                await self._polymarket.close()
            except Exception:
                logger.error("Error closing Polymarket client", exc_info=True)
        if self._deduplicator:
            try:
                await self._deduplicator.cleanup()
            except Exception:
                logger.error("Error cleaning up deduplicator", exc_info=True)

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

    async def process_message(
        self,
        message: UnifiedMessage,
        on_stage1_complete: Stage1Callback | None = None,
    ) -> ProcessingResult:
        """Process a single message through the two-stage pipeline.

        Flow:
        1. Check for duplicates
        2. Stage 1: Entity extraction (fast, no judgment calls)
        3. Early exit if low/normal urgency (no Stage 2)
        4. Fire on_stage1_complete callback (e.g. notification)
        5. Early exit if Stage 2 disabled by config
        6. Pre-fetch context: Web search + Polymarket (parallel)
        7. Stage 2: Smart analysis (all informed judgments with context)

        Args:
            message: The unified message to process
            on_stage1_complete: Optional async callback invoked after Stage 1
                extraction, only when urgency is high/critical. Skipped for
                low/normal urgency (which exit before reaching this point).

        Returns:
            ProcessingResult with extraction and analysis
        """
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

        # 2. Stage 1: Instant classification (no LLM)
        extraction = await self.classifier.classify(message)

        # 3. Early exit for low/normal urgency (skip notification + Stage 2)
        low_urgency = extraction.urgency in (UrgencyLevel.low, UrgencyLevel.normal)

        if low_urgency:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            log.info(
                "Skipping Stage 2",
                skip_reason="filtered by impact",
                impact_score=extraction.impact_score,
                urgency=extraction.urgency.value,
                processing_time_ms=f"{elapsed_ms:.1f}",
            )
            return ProcessingResult(
                message=message,
                skipped=False,  # Not skipped - just minimal processing
                skip_reason=None,
                extraction=extraction,
                analysis=None,  # No Stage 2 analysis
                is_duplicate=False,
                duplicate_of=None,
                processing_time_ms=elapsed_ms,
            )

        # 4. Fire Stage 1 callback (sends notification regardless of stage2_enabled)
        if on_stage1_complete:
            try:
                await on_stage1_complete(message, extraction)
            except Exception:
                log.error(
                    "Stage 1 callback failed — early notification not sent",
                    exc_info=True,
                )
        else:
            log.warning("No Stage 1 callback provided for high-urgency message")

        # 5. Early exit if Stage 2 is disabled by config
        stage2_disabled = not get_settings().stage2_enabled

        if stage2_disabled:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            log.info(
                "Skipping Stage 2",
                skip_reason="disabled by config",
                urgency=extraction.urgency.value,
                processing_time_ms=f"{elapsed_ms:.1f}",
            )
            return ProcessingResult(
                message=message,
                skipped=False,
                skip_reason=None,
                extraction=extraction,
                analysis=None,
                is_duplicate=False,
                duplicate_of=None,
                processing_time_ms=elapsed_ms,
            )

        # 6. Stage 2: Smart analysis (entities, sentiment, ETF impact, Polymarket)
        # Polymarket search is done inside the analyzer via LLM tool calls
        log.debug("Stage 2: Smart analysis starting")

        analysis = await self.analyzer.analyze(
            message,
            extraction,
            http_client=self.polymarket._get_client(),
        )

        if analysis is None:
            # Stage 2 failed - log error and continue with partial results
            log.error(
                "Stage 2 failed - analysis unavailable",
                message_id=message.external_id,
            )
        else:
            log.info(
                "Stage 2 complete",
                macro_impact=[e.ticker for e in analysis.macro_impact],
                sector_impact=[e.ticker for e in analysis.sector_impact],
                thesis=analysis.primary_thesis[:100] if analysis.primary_thesis else "None",
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
