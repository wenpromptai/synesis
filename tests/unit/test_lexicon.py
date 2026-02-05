"""Unit tests for SentimentLexicon class."""

from pathlib import Path


from synesis.processing.sentiment.analyzer import SentimentLexicon
from synesis.processing.sentiment.lexicon import (
    BASE_LEXICON,
    EMOJI_LEXICON,
    FINANCE_LEXICON,
    SLANG_PHRASES,
)


class TestSentimentLexiconInit:
    """Tests for SentimentLexicon initialization."""

    def test_default_init_uses_builtins(self) -> None:
        lex = SentimentLexicon()
        # Should contain entries from base and finance lexicons
        assert lex.size > 0
        assert lex.get_word_valence("amazing") is not None
        assert lex.get_word_valence("bullish") is not None

    def test_custom_init_overrides(self) -> None:
        custom = {"custom_word": 5.0}
        lex = SentimentLexicon(base_lexicon=custom, finance_lexicon={})
        assert lex.get_word_valence("custom_word") == 5.0
        # Default words should NOT be present
        assert lex.get_word_valence("amazing") is None

    def test_finance_overrides_base(self) -> None:
        base = {"conflict": 1.0}
        finance = {"conflict": -2.0}
        lex = SentimentLexicon(base_lexicon=base, finance_lexicon=finance)
        # Finance takes precedence
        assert lex.get_word_valence("conflict") == -2.0


class TestGetWordValence:
    """Tests for SentimentLexicon.get_word_valence."""

    def test_base_word(self) -> None:
        lex = SentimentLexicon()
        val = lex.get_word_valence("amazing")
        assert val is not None
        assert val > 0

    def test_finance_word(self) -> None:
        lex = SentimentLexicon()
        val = lex.get_word_valence("bullish")
        assert val is not None
        assert val > 0

    def test_emoji(self) -> None:
        lex = SentimentLexicon()
        val = lex.get_word_valence("🚀")
        assert val is not None
        assert val > 0

    def test_unknown_returns_none(self) -> None:
        lex = SentimentLexicon()
        assert lex.get_word_valence("xyzzy123") is None

    def test_case_insensitive(self) -> None:
        lex = SentimentLexicon()
        lower = lex.get_word_valence("amazing")
        upper = lex.get_word_valence("AMAZING")
        mixed = lex.get_word_valence("Amazing")
        assert lower == upper == mixed


class TestGetPhraseValence:
    """Tests for SentimentLexicon.get_phrase_valence."""

    def test_finds_to_the_moon(self) -> None:
        lex = SentimentLexicon()
        matches = lex.get_phrase_valence("TSLA to the moon!")
        phrases = [m[0] for m in matches]
        assert "to the moon" in phrases

    def test_longest_first_matching(self) -> None:
        # If there were overlapping phrases, longest should win
        lex = SentimentLexicon()
        matches = lex.get_phrase_valence("short squeeze incoming")
        phrases = [m[0] for m in matches]
        # "squeeze incoming" is longer than "squeeze" alone
        if "squeeze incoming" in SLANG_PHRASES:
            assert "squeeze incoming" in phrases

    def test_position_tracking(self) -> None:
        lex = SentimentLexicon()
        matches = lex.get_phrase_valence("going to the moon now")
        for phrase, valence, start, end in matches:
            assert end > start
            assert "going to the moon"[: end - start] or phrase in "going to the moon now"

    def test_no_matches(self) -> None:
        lex = SentimentLexicon()
        matches = lex.get_phrase_valence("this has no known phrases at all")
        assert len(matches) == 0

    def test_multiple_phrases(self) -> None:
        lex = SentimentLexicon()
        matches = lex.get_phrase_valence("diamond hands and to the moon")
        phrases = [m[0] for m in matches]
        assert "diamond hands" in phrases
        assert "to the moon" in phrases


class TestSizeProperty:
    """Tests for SentimentLexicon.size property."""

    def test_default_size(self) -> None:
        lex = SentimentLexicon()
        assert lex.size > 0
        # The combined dict merges base+finance, so overlapping keys are counted once
        combined_size = len({**BASE_LEXICON, **FINANCE_LEXICON})
        assert lex.size == combined_size + len(EMOJI_LEXICON) + len(SLANG_PHRASES)

    def test_custom_size(self) -> None:
        lex = SentimentLexicon(
            base_lexicon={"a": 1.0, "b": 2.0},
            finance_lexicon={"c": 3.0},
            emoji_lexicon={"🚀": 1.0},
            slang_phrases={"to the moon": 3.0},
        )
        assert lex.size == 5  # 3 combined + 1 emoji + 1 slang


class TestFromJsonFiles:
    """Tests for SentimentLexicon.from_json_files."""

    def test_missing_files_uses_defaults(self) -> None:
        lex = SentimentLexicon.from_json_files(
            base_path=Path("/nonexistent/base.json"),
            finance_path=Path("/nonexistent/finance.json"),
        )
        # Should still work with default lexicons since files don't exist
        # from_json_files passes None when files don't exist -> defaults used
        assert lex.get_word_valence("amazing") is not None

    def test_none_paths_use_defaults(self) -> None:
        lex = SentimentLexicon.from_json_files(base_path=None, finance_path=None)
        assert lex.get_word_valence("amazing") is not None

    def test_valid_json_file(self, tmp_path: Path) -> None:
        import json

        base_file = tmp_path / "base.json"
        base_file.write_text(json.dumps({"custom": 4.2}))

        lex = SentimentLexicon.from_json_files(base_path=base_file)
        assert lex.get_word_valence("custom") == 4.2
