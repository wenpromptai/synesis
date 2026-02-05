"""Custom sentiment analyzer optimized for financial/trading content.

Inspired by VADER (Valence Aware Dictionary and sEntiment Reasoner) but built
from scratch with finance-specific lexicons and async support.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from synesis.processing.sentiment.lexicon import (
    BASE_LEXICON,
    BOOSTERS,
    CAPS_EMPHASIS,
    DAMPENERS,
    EMOJI_LEXICON,
    EXCLAMATION_BONUS,
    FINANCE_LEXICON,
    MAX_EXCLAMATION,
    NEGATION_SCALAR,
    NEGATIONS,
    NORMALIZATION_ALPHA,
    QUESTION_BONUS,
    SLANG_PHRASES,
    SPECIAL_CONJUNCTIONS,
    TICKER_BLACKLIST,
)
from synesis.processing.sentiment.models import SentimentResult


@dataclass(slots=True)
class TokenInfo:
    """Information about a single token."""

    text: str
    lower: str
    is_caps: bool
    index: int


class SentimentLexicon:
    """Manages sentiment lexicon data from multiple sources."""

    def __init__(
        self,
        base_lexicon: dict[str, float] | None = None,
        finance_lexicon: dict[str, float] | None = None,
        emoji_lexicon: dict[str, float] | None = None,
        slang_phrases: dict[str, float] | None = None,
    ) -> None:
        """Initialize lexicon with optional custom dictionaries.

        Args:
            base_lexicon: Override default base lexicon.
            finance_lexicon: Override default finance lexicon.
            emoji_lexicon: Override default emoji lexicon.
            slang_phrases: Override default slang phrases.
        """
        self._base = base_lexicon if base_lexicon is not None else BASE_LEXICON
        self._finance = finance_lexicon if finance_lexicon is not None else FINANCE_LEXICON
        self._emoji = emoji_lexicon if emoji_lexicon is not None else EMOJI_LEXICON
        self._slang = slang_phrases if slang_phrases is not None else SLANG_PHRASES

        # Build combined lexicon (finance takes precedence over base)
        self._combined: dict[str, float] = {}
        self._combined.update(self._base)
        self._combined.update(self._finance)

        # Precompute sorted slang phrases for efficient matching (longest first)
        self._sorted_slang = sorted(self._slang.keys(), key=len, reverse=True)

    def get_word_valence(self, word: str) -> float | None:
        """Get sentiment valence for a word.

        Args:
            word: Word to look up (case-insensitive for words, exact for emoji).

        Returns:
            Valence score or None if not in lexicon.
        """
        # Check emoji first (exact match)
        if word in self._emoji:
            return self._emoji[word]

        # Check word lexicon (case-insensitive)
        lower = word.lower()
        return self._combined.get(lower)

    def get_phrase_valence(self, text: str) -> list[tuple[str, float, int, int]]:
        """Find all slang phrases in text with their positions.

        Args:
            text: Text to search for phrases.

        Returns:
            List of (phrase, valence, start, end) tuples.
        """
        text_lower = text.lower()
        matches: list[tuple[str, float, int, int]] = []
        used_positions: set[int] = set()

        for phrase in self._sorted_slang:
            start = 0
            while True:
                pos = text_lower.find(phrase, start)
                if pos == -1:
                    break

                # Check if any character in this range is already used
                phrase_range = set(range(pos, pos + len(phrase)))
                if not phrase_range & used_positions:
                    matches.append((phrase, self._slang[phrase], pos, pos + len(phrase)))
                    used_positions.update(phrase_range)

                start = pos + 1

        return sorted(matches, key=lambda x: x[2])

    @property
    def size(self) -> int:
        """Total number of entries in combined lexicon."""
        return len(self._combined) + len(self._emoji) + len(self._slang)

    @classmethod
    def from_json_files(
        cls,
        base_path: Path | None = None,
        finance_path: Path | None = None,
    ) -> SentimentLexicon:
        """Load lexicons from JSON files.

        Args:
            base_path: Path to base lexicon JSON.
            finance_path: Path to finance lexicon JSON.

        Returns:
            SentimentLexicon instance.
        """
        import json

        base_lexicon = None
        finance_lexicon = None

        if base_path and base_path.exists():
            with base_path.open() as f:
                base_lexicon = json.load(f)

        if finance_path and finance_path.exists():
            with finance_path.open() as f:
                finance_lexicon = json.load(f)

        return cls(base_lexicon=base_lexicon, finance_lexicon=finance_lexicon)


class SentimentAnalyzer:
    """Lexicon-based sentiment analyzer optimized for financial content.

    Implements VADER-inspired algorithms with finance-specific enhancements:
    - Custom financial lexicon (bullish/bearish trading terms)
    - Emoji support for social media content
    - WSB/crypto slang recognition
    - Ticker extraction and per-ticker sentiment
    - Async-native for high throughput
    """

    # Regex patterns
    _TICKER_PATTERN = re.compile(r"\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b")
    _WORD_PATTERN = re.compile(r"[\w']+|[^\w\s]")
    _EMOJI_PATTERN = re.compile(
        r"[\U0001F300-\U0001F9FF]|[\u2600-\u26FF]|[\u2700-\u27BF]",
        re.UNICODE,
    )

    def __init__(self, lexicon: SentimentLexicon | None = None) -> None:
        """Initialize analyzer with lexicon.

        Args:
            lexicon: Custom lexicon or None for defaults.
        """
        self._lexicon = lexicon if lexicon is not None else SentimentLexicon()

    async def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of text.

        This is the main entry point. Despite being async, the core computation
        is CPU-bound and fast. The async signature enables easy integration
        with async pipelines.

        Args:
            text: Text to analyze.

        Returns:
            SentimentResult with compound score and breakdowns.
        """
        if not text or not text.strip():
            return SentimentResult(
                compound=0.0,
                positive=0.0,
                negative=0.0,
                neutral=1.0,
                confidence=0.0,
            )

        # Extract tickers before processing
        tickers = self._extract_tickers(text)

        # Tokenize
        tokens = self._tokenize(text)

        if not tokens:
            return SentimentResult(
                compound=0.0,
                positive=0.0,
                negative=0.0,
                neutral=1.0,
                tickers_mentioned=tuple(tickers),
                confidence=0.0,
            )

        # Score slang phrases first (they override individual word scores)
        phrase_matches = self._lexicon.get_phrase_valence(text)
        phrase_positions: set[int] = set()
        phrase_valences: list[float] = []

        for phrase, valence, start, end in phrase_matches:
            phrase_positions.update(range(start, end))
            phrase_valences.append(valence)

        # Score individual tokens
        sentiments: list[float] = []
        lexicon_hits = 0

        for i, token in enumerate(tokens):
            # Skip if token is part of a phrase match
            # (rough check based on character position in original text)
            text_pos = text.lower().find(token.lower)
            if text_pos != -1 and text_pos in phrase_positions:
                continue

            valence = self._score_token(tokens, i)
            if valence != 0.0:
                sentiments.append(valence)
                lexicon_hits += 1

        # Add phrase valences
        sentiments.extend(phrase_valences)

        # Apply punctuation modifiers
        punct_modifier = self._punctuation_modifier(text)

        # Apply but-clause weighting
        sentiments = self._apply_but_clause(text, tokens, sentiments)

        # Calculate pos/neg/neu proportions and compound
        if not sentiments:
            return SentimentResult(
                compound=0.0,
                positive=0.0,
                negative=0.0,
                neutral=1.0,
                tickers_mentioned=tuple(tickers),
                confidence=0.0,
            )

        pos_sum = sum(s for s in sentiments if s > 0)
        neg_sum = sum(s for s in sentiments if s < 0)
        neu_count = len(tokens) - len(sentiments)

        total = pos_sum + abs(neg_sum) + neu_count
        if total == 0:
            total = 1  # Avoid division by zero

        positive = pos_sum / total
        negative = abs(neg_sum) / total
        neutral = neu_count / total

        # Compound score
        raw_sum = sum(sentiments) + punct_modifier
        compound = self._normalize(raw_sum)

        # Confidence based on lexicon coverage
        confidence = min(lexicon_hits / max(len(tokens), 1), 1.0)

        # Per-ticker sentiment (simplified: use overall compound)
        ticker_sentiments = {ticker: compound for ticker in tickers}

        return SentimentResult(
            compound=round(compound, 4),
            positive=round(positive, 4),
            negative=round(negative, 4),
            neutral=round(neutral, 4),
            tickers_mentioned=tuple(tickers),
            ticker_sentiments=ticker_sentiments,
            confidence=round(confidence, 4),
        )

    def _tokenize(self, text: str) -> list[TokenInfo]:
        """Tokenize text into words and emoji.

        Args:
            text: Text to tokenize.

        Returns:
            List of TokenInfo objects.
        """
        # Find all words and punctuation
        word_matches = self._WORD_PATTERN.findall(text)

        # Find all emoji
        emoji_matches = self._EMOJI_PATTERN.findall(text)

        # Combine and maintain order
        tokens: list[TokenInfo] = []

        for i, match in enumerate(word_matches):
            is_caps = match.isupper() and len(match) > 1 and match.isalpha()
            tokens.append(
                TokenInfo(
                    text=match,
                    lower=match.lower(),
                    is_caps=is_caps,
                    index=i,
                )
            )

        # Add emoji as separate tokens
        for emoji in emoji_matches:
            tokens.append(
                TokenInfo(
                    text=emoji,
                    lower=emoji,
                    is_caps=False,
                    index=len(tokens),
                )
            )

        return tokens

    def _score_token(self, tokens: list[TokenInfo], index: int) -> float:
        """Score a single token with context modifiers.

        Args:
            tokens: All tokens in text.
            index: Index of token to score.

        Returns:
            Sentiment valence (may be 0.0 if not in lexicon).
        """
        token = tokens[index]
        valence = self._lexicon.get_word_valence(token.text)

        if valence is None:
            return 0.0

        # Apply ALL CAPS emphasis
        if token.is_caps:
            if valence > 0:
                valence += CAPS_EMPHASIS
            elif valence < 0:
                valence -= CAPS_EMPHASIS

        # Apply boosters/dampeners from preceding words
        valence = self._apply_boosters(tokens, index, valence)

        # Apply negation
        valence = self._apply_negation(tokens, index, valence)

        return valence

    def _apply_boosters(
        self,
        tokens: list[TokenInfo],
        index: int,
        valence: float,
    ) -> float:
        """Apply booster/dampener modifiers from preceding words.

        Checks up to 3 words before the target for modifiers.

        Args:
            tokens: All tokens.
            index: Index of target token.
            valence: Current valence.

        Returns:
            Modified valence.
        """
        # Check up to 3 preceding tokens
        for offset in range(1, 4):
            if index - offset < 0:
                break

            preceding = tokens[index - offset].lower

            # Check boosters
            if preceding in BOOSTERS:
                modifier = BOOSTERS[preceding]
                # Decay modifier by distance
                modifier *= 0.95 ** (offset - 1)
                if valence > 0:
                    valence += modifier
                elif valence < 0:
                    valence -= modifier
                break

            # Check dampeners
            if preceding in DAMPENERS:
                modifier = DAMPENERS[preceding]
                modifier *= 0.95 ** (offset - 1)
                if valence > 0:
                    valence += modifier  # Dampeners have negative values
                elif valence < 0:
                    valence -= modifier
                break

        return valence

    def _apply_negation(
        self,
        tokens: list[TokenInfo],
        index: int,
        valence: float,
    ) -> float:
        """Apply negation from preceding words.

        Checks up to 3 words before for negation terms.

        Args:
            tokens: All tokens.
            index: Index of target token.
            valence: Current valence.

        Returns:
            Modified valence (flipped if negated).
        """
        # Check up to 3 preceding tokens for negation
        for offset in range(1, 4):
            if index - offset < 0:
                break

            preceding = tokens[index - offset].lower

            # Remove apostrophes for contraction matching
            clean_preceding = preceding.replace("'", "")

            if clean_preceding in NEGATIONS or preceding in NEGATIONS:
                # Apply negation scalar (VADER uses -0.74)
                valence *= NEGATION_SCALAR
                break

            # Check for "never so" or "never this" pattern
            if preceding == "never" and index - offset - 1 >= 0:
                next_word = tokens[index - offset + 1].lower if offset > 1 else ""
                if next_word in {"so", "this", "really"}:
                    valence *= NEGATION_SCALAR * 1.25
                    break

        return valence

    def _apply_but_clause(
        self,
        text: str,
        tokens: list[TokenInfo],
        sentiments: list[float],
    ) -> list[float]:
        """Apply but-clause weighting.

        Content after "but" is weighted higher (1.5x).

        Args:
            text: Original text.
            tokens: Tokenized text.
            sentiments: Current sentiment scores.

        Returns:
            Modified sentiment scores.
        """
        text_lower = text.lower()

        for conj, weight in SPECIAL_CONJUNCTIONS.items():
            if conj in text_lower:
                conj_pos = text_lower.find(conj)
                # Find which tokens come after the conjunction
                for i, sent in enumerate(sentiments):
                    if i >= len(tokens):
                        break
                    # Rough heuristic: tokens in second half get weighted
                    token_pos = text_lower.find(tokens[i].lower)
                    if token_pos > conj_pos:
                        sentiments[i] = sent * weight

        return sentiments

    def _punctuation_modifier(self, text: str) -> float:
        """Calculate punctuation-based sentiment modifier.

        Args:
            text: Original text.

        Returns:
            Modifier value to add to sum.
        """
        modifier = 0.0

        # Count exclamation marks (cap at MAX_EXCLAMATION)
        excl_count = min(text.count("!"), MAX_EXCLAMATION)
        modifier += excl_count * EXCLAMATION_BONUS

        # Question marks add slight positive (curiosity/engagement)
        if "?" in text:
            modifier += QUESTION_BONUS

        return modifier

    def _normalize(self, score: float) -> float:
        """Normalize raw score to [-1, 1] range.

        Uses VADER's normalization formula: score / sqrt(score^2 + alpha)

        Args:
            score: Raw sentiment sum.

        Returns:
            Normalized compound score.
        """
        norm = score / math.sqrt(score * score + NORMALIZATION_ALPHA)
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, norm))

    def _extract_tickers(self, text: str) -> list[str]:
        """Extract stock/crypto tickers from text.

        Handles:
        - Cashtags: $TSLA, $BTC
        - Bare tickers: AAPL, NVDA

        Args:
            text: Text to search.

        Returns:
            List of unique tickers found.
        """
        tickers: set[str] = set()

        for match in self._TICKER_PATTERN.finditer(text):
            # match.group(1) is cashtag, match.group(2) is bare ticker
            ticker = match.group(1) or match.group(2)
            if ticker and ticker.upper() not in TICKER_BLACKLIST:
                tickers.add(ticker.upper())

        return sorted(tickers)

    @property
    def lexicon(self) -> SentimentLexicon:
        """Get the lexicon used by this analyzer."""
        return self._lexicon


# Convenience function for quick analysis
async def analyze_sentiment(text: str) -> SentimentResult:
    """Analyze sentiment of text using default analyzer.

    Args:
        text: Text to analyze.

    Returns:
        SentimentResult.
    """
    analyzer = SentimentAnalyzer()
    return await analyzer.analyze(text)
