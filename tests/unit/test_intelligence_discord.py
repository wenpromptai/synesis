"""Tests for intelligence brief Discord formatter."""

from __future__ import annotations

from typing import Any


from synesis.processing.intelligence.discord_format import (
    _MAX_CHARS_PER_MSG,
    _format_debate_side_with_variant,
    _split_field,
    _split_oversized_embed,
    format_intelligence_brief,
)


def _make_brief(**overrides: Any) -> dict[str, Any]:
    """Create a minimal valid brief dict with overrides."""
    base: dict[str, Any] = {
        "date": "2026-04-08",
        "macro": {
            "regime": "risk_on",
            "sentiment_score": 0.65,
            "key_drivers": ["Strong earnings season"],
            "thematic_tilts": [{"theme": "Technology", "sentiment_score": 0.8}],
            "risks": ["Tariff escalation"],
        },
        "debates": [],
        "l1_summary": {"social": "Bullish sentiment on tech", "news": "Earnings beat expectations"},
        "tickers_analyzed": [],
        "company_analyses": [],
        "price_analyses": [],
        "macro_themes": [],
        "ticker_mentions": {"social": [], "news_clusters": []},
        "messages_analyzed": 42,
        "trade_ideas": [],
        "errors": {
            "social_failed": False,
            "news_failed": False,
            "company_failures": [],
            "price_failures": [],
            "bull_failures": [],
            "bear_failures": [],
            "macro_failed": False,
            "trader_failures": [],
        },
    }
    base.update(overrides)
    return base


class TestFormatIntelligenceBrief:
    """Tests for format_intelligence_brief."""

    def test_returns_batches(self) -> None:
        brief = _make_brief()
        batches = format_intelligence_brief(brief)
        assert isinstance(batches, list)
        assert len(batches) >= 1
        assert isinstance(batches[0], list)

    def test_header_embed_always_present(self) -> None:
        brief = _make_brief()
        batches = format_intelligence_brief(brief)
        header = batches[0][0]
        assert "Daily Brief" in header["title"]
        assert "Apr 8, 2026" in header["title"]
        assert "timestamp" in header

    def test_macro_fields(self) -> None:
        brief = _make_brief()
        batches = format_intelligence_brief(brief)
        header = batches[0][0]
        field_names = [f["name"] for f in header["fields"]]
        assert "Regime" in field_names
        assert "Sentiment" in field_names
        assert "Key Drivers" in field_names
        assert "Thematic Tilts" in field_names
        assert "Risks" in field_names

    def test_regime_colors(self) -> None:
        from synesis.core.constants import COLOR_BEARISH, COLOR_BULLISH, COLOR_NEUTRAL

        for regime, expected_color in [
            ("risk_on", COLOR_BULLISH),
            ("risk_off", COLOR_BEARISH),
            ("transitioning", COLOR_NEUTRAL),
        ]:
            brief = _make_brief(macro={"regime": regime})
            batches = format_intelligence_brief(brief)
            assert batches[0][0]["color"] == expected_color

    def test_l1_summary_embed(self) -> None:
        brief = _make_brief()
        batches = format_intelligence_brief(brief)
        all_embeds = [e for batch in batches for e in batch]
        summary_embeds = [e for e in all_embeds if e.get("title") == "\U0001f4e1 Signal Summary"]
        assert len(summary_embeds) == 1
        field_names = [f["name"] for f in summary_embeds[0]["fields"]]
        assert "Social" in field_names
        assert "News" in field_names

    def test_no_l1_summary_when_empty(self) -> None:
        brief = _make_brief(l1_summary={"social": "", "news": ""})
        batches = format_intelligence_brief(brief)
        all_embeds = [e for batch in batches for e in batch]
        summary_embeds = [e for e in all_embeds if e.get("title") == "\U0001f4e1 Signal Summary"]
        assert len(summary_embeds) == 0

    def test_debate_embed_per_ticker(self) -> None:
        brief = _make_brief(
            debates=[
                {
                    "ticker": "NVDA",
                    "bull": {
                        "ticker": "NVDA",
                        "role": "bull",
                        "argument": "Strong AI demand",
                        "key_evidence": ["Data center revenue up 200%"],
                        "round": 1,
                    },
                    "bear": {
                        "ticker": "NVDA",
                        "role": "bear",
                        "argument": "Valuation stretched",
                        "key_evidence": ["P/E at 60x", "Competition from AMD"],
                        "round": 1,
                    },
                },
            ]
        )
        batches = format_intelligence_brief(brief)
        all_embeds = [e for batch in batches for e in batch]
        debate_embeds = [e for e in all_embeds if "NVDA" in e.get("title", "")]
        assert len(debate_embeds) == 1

        fields = debate_embeds[0]["fields"]
        field_names = [f["name"] for f in fields]
        assert "\U0001f7e2 Bull Case" in field_names
        assert "\U0001f534 Bear Case" in field_names

        bull_field = next(f for f in fields if "Bull" in f["name"])
        assert "Strong AI demand" in bull_field["value"]
        assert "Data center revenue up 200%" in bull_field["value"]

        bear_field = next(f for f in fields if "Bear" in f["name"])
        assert "Valuation stretched" in bear_field["value"]
        assert "P/E at 60x" in bear_field["value"]

    def test_trade_ideas_in_unified_section(self) -> None:
        """All trade ideas appear in the Trade Ideas section, not in debate embeds."""
        brief = _make_brief(
            debates=[{"ticker": "AAPL", "bull": {"argument": "Good", "key_evidence": []}}],
            trade_ideas=[
                {
                    "tickers": ["AAPL"],
                    "trade_structure": "long AAPL",
                    "thesis": "iPhone cycle",
                    "catalyst": "Q2 earnings",
                    "timeframe": "3 months",
                    "key_risk": "Trade war",
                    "entry_price": 180.0,
                    "target_price": 220.0,
                    "stop_price": 165.0,
                    "risk_reward_ratio": 2.7,
                    "conviction_tier": 2,
                    "conviction_rationale": "Strong thesis, catalyst in 3 months",
                }
            ],
        )
        batches = format_intelligence_brief(brief)
        all_embeds = [e for batch in batches for e in batch]

        # Trade ideas should NOT be in the debate embed
        aapl_debate = next(e for e in all_embeds if "AAPL" in e.get("title", ""))
        idea_fields = [f for f in aapl_debate["fields"] if "AAPL" in f["name"]]
        assert len(idea_fields) == 0

        # Should be in the unified Trade Ideas section
        trade_embeds = [e for e in all_embeds if e.get("title") == "\U0001f4bc Trade Ideas"]
        assert len(trade_embeds) == 1
        idea_fields = trade_embeds[0]["fields"]
        assert len(idea_fields) == 1
        value = idea_fields[0]["value"]
        assert "long AAPL" in value
        assert "iPhone cycle" in value
        assert "Q2 earnings" in value
        assert "3 months" in value
        assert "Trade war" in value
        assert "Entry $180.00" in value
        assert "Target $220.00" in value
        assert "Stop $165.00" in value
        assert "R/R 2.7:1" in value
        assert "Tier 2" in value

    def test_trade_idea_timeframe_without_catalyst(self) -> None:
        """Timeframe is rendered even when catalyst is empty."""
        brief = _make_brief(
            trade_ideas=[
                {
                    "tickers": ["NVDA"],
                    "trade_structure": "long NVDA",
                    "thesis": "Momentum",
                    "catalyst": "",
                    "timeframe": "6 weeks",
                    "key_risk": "",
                }
            ],
        )
        batches = format_intelligence_brief(brief)
        all_embeds = [e for batch in batches for e in batch]
        trade_embeds = [e for e in all_embeds if e.get("title") == "\U0001f4bc Trade Ideas"]
        assert len(trade_embeds) == 1
        value = trade_embeds[0]["fields"][0]["value"]
        assert "**Timeframe:** 6 weeks" in value

    def test_portfolio_note_in_trade_ideas_section(self) -> None:
        """portfolio_note renders as description in Trade Ideas embed."""
        brief = _make_brief(
            trade_ideas=[
                {
                    "tickers": ["NVDA"],
                    "trade_structure": "buy NVDA",
                    "thesis": "",
                    "catalyst": "",
                    "timeframe": "",
                    "key_risk": "",
                }
            ],
            portfolio_note="Concentrated in semis — sizing conservatively given correlation.",
        )
        batches = format_intelligence_brief(brief)
        all_embeds = [e for batch in batches for e in batch]
        trade_embeds = [e for e in all_embeds if e.get("title") == "\U0001f4bc Trade Ideas"]
        assert len(trade_embeds) == 1
        assert "sizing conservatively" in trade_embeds[0].get("description", "")

    def test_no_debates_no_trade_ideas(self) -> None:
        """Brief with no tickers should still produce header + l1 summary."""
        brief = _make_brief()
        batches = format_intelligence_brief(brief)
        all_embeds = [e for batch in batches for e in batch]
        # Header + signal summary
        assert len(all_embeds) == 2

    def test_batch_splitting_by_count(self) -> None:
        """More than 10 embeds should be split into multiple batches."""
        # Use minimal content so char limit isn't hit first
        debates = [
            {"ticker": f"T{i}", "bull": {"argument": "x", "key_evidence": []}} for i in range(12)
        ]
        brief = _make_brief(debates=debates, l1_summary={"social": "", "news": ""})
        batches = format_intelligence_brief(brief)
        total_embeds = sum(len(b) for b in batches)
        assert total_embeds == 13  # header + 12 debates
        assert all(len(b) <= 10 for b in batches)

    def test_batch_splitting_by_char_limit(self) -> None:
        """Batches should split when total chars exceed 6000."""
        # Each debate with ~800 chars of argument → 6 debates ≈ 4800+ chars
        # + header ≈ ~200 chars → should split before all 6 fit in one batch
        debates = [
            {
                "ticker": f"T{i}",
                "bull": {"argument": "a" * 800, "key_evidence": ["e" * 200]},
                "bear": {"argument": "b" * 800, "key_evidence": ["f" * 200]},
            }
            for i in range(6)
        ]
        brief = _make_brief(debates=debates, l1_summary={"social": "", "news": ""})
        batches = format_intelligence_brief(brief)
        # Should be split into multiple batches
        assert len(batches) > 1
        # Each batch should respect char limit
        for batch in batches:
            from synesis.processing.intelligence.discord_format import _embed_char_count

            total_chars = sum(_embed_char_count(e) for e in batch)
            assert total_chars <= _MAX_CHARS_PER_MSG

    def test_long_field_values_split(self) -> None:
        """Long field values should be split into multiple fields, not truncated."""
        long_argument = "x" * 2000
        brief = _make_brief(
            debates=[
                {
                    "ticker": "LONG",
                    "bull": {
                        "argument": long_argument,
                        "key_evidence": [],
                    },
                }
            ]
        )
        batches = format_intelligence_brief(brief)
        all_embeds = [e for batch in batches for e in batch]
        long_embeds = [e for e in all_embeds if "LONG" in e.get("title", "")]
        bull_fields = [f for e in long_embeds for f in e.get("fields", []) if "Bull" in f["name"]]
        assert len(bull_fields) >= 2
        assert all(len(f["value"]) <= 1024 for f in bull_fields)
        # All content preserved across splits
        total_chars = sum(len(f["value"]) for f in bull_fields)
        assert total_chars == 2000

    def test_invalid_date_fallback(self) -> None:
        brief = _make_brief(date="not-a-date")
        batches = format_intelligence_brief(brief)
        header = batches[0][0]
        assert "not-a-date" in header["title"]


class TestSplitField:
    """Tests for _split_field helper."""

    def test_short_value_returns_single_field(self) -> None:
        result = _split_field("Name", "short text")
        assert len(result) == 1
        assert result[0] == {"name": "Name", "value": "short text", "inline": False}

    def test_exactly_1024_returns_single_field(self) -> None:
        result = _split_field("F", "x" * 1024)
        assert len(result) == 1
        assert len(result[0]["value"]) == 1024

    def test_1025_chars_splits_into_two(self) -> None:
        result = _split_field("F", "x" * 1025)
        assert len(result) == 2
        assert all(len(f["value"]) <= 1024 for f in result)
        assert sum(len(f["value"]) for f in result) == 1025

    def test_continuation_naming(self) -> None:
        result = _split_field("Bull Case", "x" * 2000)
        assert result[0]["name"] == "Bull Case"
        assert result[1]["name"] == "Bull Case (cont.)"

    def test_newline_break_above_512(self) -> None:
        """Splits at newline when it is past position 512."""
        value = "a" * 600 + "\n" + "b" * 500
        result = _split_field("F", value)
        assert len(result) == 2
        # First chunk should end at the newline (600 chars of 'a')
        assert result[0]["value"] == "a" * 600
        # Second chunk starts with the newline
        assert result[1]["value"].startswith("\n")

    def test_newline_before_512_ignored(self) -> None:
        """Newlines before position 512 are ignored — splits at 1024."""
        value = "a" * 100 + "\n" + "b" * 1200
        result = _split_field("F", value)
        assert len(result) == 2
        assert len(result[0]["value"]) == 1024

    def test_three_way_split(self) -> None:
        result = _split_field("F", "x" * 3000)
        assert len(result) == 3
        assert all(len(f["value"]) <= 1024 for f in result)
        assert sum(len(f["value"]) for f in result) == 3000
        assert result[0]["name"] == "F"
        assert result[1]["name"] == "F (cont.)"
        assert result[2]["name"] == "F (cont.)"

    def test_inline_propagated(self) -> None:
        result = _split_field("F", "x" * 2000, inline=True)
        assert all(f["inline"] is True for f in result)


class TestSplitOversizedEmbed:
    """Tests for _split_oversized_embed helper."""

    def test_small_embed_unchanged(self) -> None:
        embed = {"title": "T", "color": 0x00FF00, "fields": [{"name": "F", "value": "v"}]}
        result = _split_oversized_embed(embed)
        assert len(result) == 1
        assert result[0] == embed

    def test_no_fields_truncates_description(self) -> None:
        embed = {"title": "T", "description": "d" * 5000}
        result = _split_oversized_embed(embed)
        assert len(result) == 1
        assert len(result[0]["description"]) == 4096

    def test_oversized_embed_splits_by_fields(self) -> None:
        # Create fields totaling well over 6000 chars
        fields = [{"name": f"F{i}", "value": "x" * 900} for i in range(8)]
        embed = {"title": "Big", "color": 0xFF0000, "fields": fields}
        result = _split_oversized_embed(embed)
        assert len(result) >= 2
        # Each part should be within the char limit
        for part in result:
            total = (
                len(part.get("title", ""))
                + len(part.get("description", ""))
                + sum(len(f["name"]) + len(f["value"]) for f in part.get("fields", []))
            )
            assert total <= _MAX_CHARS_PER_MSG

    def test_continuation_title_and_no_description(self) -> None:
        fields = [{"name": f"F{i}", "value": "x" * 900} for i in range(8)]
        embed = {
            "title": "Original",
            "color": 0xFF0000,
            "description": "Intro text",
            "footer": {"text": "Footer"},
            "fields": fields,
        }
        result = _split_oversized_embed(embed)
        assert len(result) >= 2
        # First part keeps original structure
        assert result[0]["title"] == "Original"
        assert result[0]["description"] == "Intro text"
        # Continuation parts have (cont.) title and no description/footer
        for part in result[1:]:
            assert part["title"] == "Original (cont.)"
            assert "description" not in part
            assert "footer" not in part

    def test_all_fields_preserved(self) -> None:
        fields = [{"name": f"F{i}", "value": "x" * 900} for i in range(8)]
        embed = {"title": "T", "fields": fields}
        result = _split_oversized_embed(embed)
        all_fields = [f for part in result for f in part.get("fields", [])]
        assert len(all_fields) == 8
        assert [f["name"] for f in all_fields] == [f"F{i}" for i in range(8)]


class TestFormatDebateSideWithVariant:
    """Tests for _format_debate_side_with_variant helper."""

    def test_argument_only(self) -> None:
        result = _format_debate_side_with_variant({"argument": "Strong thesis"})
        assert "Strong thesis" in result

    def test_with_evidence(self) -> None:
        side = {
            "argument": "Bull case",
            "key_evidence": ["Evidence 1", "Evidence 2"],
        }
        result = _format_debate_side_with_variant(side)
        assert "Bull case" in result
        assert "\u203a Evidence 1" in result
        assert "\u203a Evidence 2" in result

    def test_evidence_capped_at_4(self) -> None:
        side = {
            "argument": "Argument",
            "key_evidence": [f"Point {i}" for i in range(10)],
        }
        result = _format_debate_side_with_variant(side)
        assert result.count("\u203a") == 4

    def test_variant_fields_included(self) -> None:
        side = {
            "argument": "Bull thesis",
            "key_evidence": ["EV1"],
            "variant_vs_consensus": "Consensus expects 12%; I expect 18%",
            "estimated_upside_downside": "+25% to $180",
            "catalyst": "Q2 earnings",
            "catalyst_timeline": "July 24",
            "what_would_change_my_mind": "Margin miss below 20%",
        }
        result = _format_debate_side_with_variant(side)
        assert "**Variant:** Consensus expects 12%; I expect 18%" in result
        assert "**Target:** +25% to $180" in result
        assert "**Catalyst:** Q2 earnings (July 24)" in result
        assert "**Invalidation:** Margin miss below 20%" in result

    def test_missing_variant_fields_skipped(self) -> None:
        side = {"argument": "Simple argument", "key_evidence": []}
        result = _format_debate_side_with_variant(side)
        assert "**Variant:**" not in result
        assert "**Catalyst:**" not in result
        assert "Simple argument" in result
