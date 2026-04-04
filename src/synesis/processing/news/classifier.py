"""Stage 1: Instant news classifier — NO LLM.

Produces LightClassification from:
  1. Impact scoring (impact_scorer.py) — urgency + score
  2. Ticker matching (ticker_matcher.py) — matched tickers from text

All LLM-dependent work (entity analysis, sentiment, ETF impact)
is deferred to Stage 2.
"""

from synesis.core.logging import get_logger
from synesis.processing.news.impact_scorer import compute_impact_score
from synesis.processing.news.models import (
    LightClassification,
    UnifiedMessage,
)
from synesis.processing.news.ticker_matcher import match_tickers

logger = get_logger(__name__)


class NewsClassifier:
    """Stage 1: Instant classifier — impact scoring + ticker matching.

    No LLM calls. Produces LightClassification in ~2ms.
    """

    async def classify(self, message: UnifiedMessage) -> LightClassification:
        """Classify a message instantly (no LLM).

        Returns LightClassification with impact score and matched tickers.
        """
        # 1. Impact scoring (~1ms)
        impact = compute_impact_score(message.text, message.source_account)

        # 2. Ticker matching (~1ms)
        tickers = match_tickers(message.text)

        extraction = LightClassification(
            matched_tickers=tickers,
            impact_score=impact.score,
            impact_reasons=impact.reasons,
            urgency=impact.urgency,
        )

        logger.info(
            "Stage 1 complete",
            message_id=message.external_id,
            impact_score=impact.score,
            urgency=impact.urgency.value,
            matched_tickers=tickers,
            impact_reasons=impact.reasons[:5],
        )

        return extraction
