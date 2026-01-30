"""Data models for sentiment analysis results."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SentimentResult:
    """Result of sentiment analysis on text.

    Attributes:
        compound: Overall sentiment score from -1.0 (most negative) to 1.0 (most positive).
        positive: Proportion of text that is positive (0.0 to 1.0).
        negative: Proportion of text that is negative (0.0 to 1.0).
        neutral: Proportion of text that is neutral (0.0 to 1.0).
        tickers_mentioned: List of stock/crypto tickers found in the text.
        ticker_sentiments: Per-ticker sentiment scores.
        confidence: Confidence in the analysis based on lexicon coverage (0.0 to 1.0).
    """

    compound: float
    positive: float
    negative: float
    neutral: float
    tickers_mentioned: tuple[str, ...] = field(default_factory=tuple)
    ticker_sentiments: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0

    def __post_init__(self) -> None:
        """Validate score ranges."""
        if not -1.0 <= self.compound <= 1.0:
            raise ValueError(f"compound must be in [-1.0, 1.0], got {self.compound}")
        for attr in ("positive", "negative", "neutral", "confidence"):
            val = getattr(self, attr)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{attr} must be in [0.0, 1.0], got {val}")

    @property
    def is_bullish(self) -> bool:
        """Return True if sentiment is bullish (compound > 0.05)."""
        return self.compound > 0.05

    @property
    def is_bearish(self) -> bool:
        """Return True if sentiment is bearish (compound < -0.05)."""
        return self.compound < -0.05

    @property
    def is_neutral(self) -> bool:
        """Return True if sentiment is neutral (-0.05 <= compound <= 0.05)."""
        return -0.05 <= self.compound <= 0.05

    @property
    def strength(self) -> str:
        """Return sentiment strength category."""
        abs_compound = abs(self.compound)
        if abs_compound >= 0.6:
            return "strong"
        if abs_compound >= 0.3:
            return "moderate"
        if abs_compound >= 0.05:
            return "weak"
        return "neutral"
