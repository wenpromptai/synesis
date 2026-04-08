"""Tests for intelligence brief Discord formatter."""

from __future__ import annotations

from typing import Any


from synesis.processing.intelligence.discord_format import (
    _format_debate_side,
    _format_errors,
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
            "sector_tilts": [{"sector": "Technology", "sentiment_score": 0.8}],
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
        assert "Sector Tilts" in field_names
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

    def test_single_ticker_trade_idea_in_debate_embed(self) -> None:
        brief = _make_brief(
            debates=[{"ticker": "AAPL", "bull": {"argument": "Good", "key_evidence": []}}],
            trade_ideas=[
                {
                    "tickers": ["AAPL"],
                    "trade_structure": "buy 100 shares AAPL",
                    "thesis": "iPhone cycle",
                    "catalyst": "Q2 earnings",
                    "timeframe": "3 months",
                    "key_risk": "Trade war",
                }
            ],
        )
        batches = format_intelligence_brief(brief)
        all_embeds = [e for batch in batches for e in batch]
        aapl_embed = next(e for e in all_embeds if "AAPL" in e.get("title", ""))
        idea_fields = [f for f in aapl_embed["fields"] if "Trade Idea" in f["name"]]
        assert len(idea_fields) == 1
        assert "buy 100 shares AAPL" in idea_fields[0]["value"]
        assert "iPhone cycle" in idea_fields[0]["value"]
        assert "Q2 earnings" in idea_fields[0]["value"]
        assert "3 months" in idea_fields[0]["value"]
        assert "Trade war" in idea_fields[0]["value"]

    def test_multi_ticker_idea_only_in_portfolio_section(self) -> None:
        """Multi-ticker ideas should NOT appear in per-ticker embeds."""
        brief = _make_brief(
            debates=[
                {"ticker": "NVDA", "bull": {"argument": "Bull NVDA", "key_evidence": []}},
                {"ticker": "AMD", "bull": {"argument": "Bull AMD", "key_evidence": []}},
            ],
            trade_ideas=[
                {
                    "tickers": ["NVDA", "AMD"],
                    "trade_structure": "equity L/S: long NVDA / short AMD",
                    "thesis": "Relative value",
                    "catalyst": "",
                    "timeframe": "6 weeks",
                    "key_risk": "",
                }
            ],
        )
        batches = format_intelligence_brief(brief)
        all_embeds = [e for batch in batches for e in batch]

        # Should NOT appear in per-ticker embeds
        nvda_embed = next(e for e in all_embeds if "NVDA" in e.get("title", ""))
        idea_fields = [f for f in nvda_embed["fields"] if "Trade Idea" in f["name"]]
        assert len(idea_fields) == 0

        # Should appear in portfolio section
        portfolio_embeds = [e for e in all_embeds if e.get("title") == "\U0001f4bc Portfolio Ideas"]
        assert len(portfolio_embeds) == 1
        assert "equity L/S" in portfolio_embeds[0]["fields"][0]["value"]

    def test_no_debates_no_trade_ideas(self) -> None:
        """Brief with no tickers should still produce header + l1 summary."""
        brief = _make_brief()
        batches = format_intelligence_brief(brief)
        all_embeds = [e for batch in batches for e in batch]
        # Header + signal summary
        assert len(all_embeds) == 2

    def test_error_embed_shown(self) -> None:
        brief = _make_brief(
            errors={
                "social_failed": True,
                "news_failed": False,
                "company_failures": ["TSLA"],
                "price_failures": [],
                "bull_failures": [],
                "bear_failures": ["NVDA"],
                "macro_failed": False,
                "trader_failures": [],
            }
        )
        batches = format_intelligence_brief(brief)
        all_embeds = [e for batch in batches for e in batch]
        error_embeds = [e for e in all_embeds if "Pipeline Errors" in e.get("description", "")]
        assert len(error_embeds) == 1
        desc = error_embeds[0]["description"]
        assert "Social analysis failed" in desc
        assert "Company failed: TSLA" in desc
        assert "Bear failed: NVDA" in desc

    def test_no_error_embed_when_clean(self) -> None:
        brief = _make_brief()
        batches = format_intelligence_brief(brief)
        all_embeds = [e for batch in batches for e in batch]
        error_embeds = [e for e in all_embeds if "Pipeline Errors" in e.get("description", "")]
        assert len(error_embeds) == 0

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
        from synesis.processing.intelligence.discord_format import _MAX_CHARS_PER_MSG

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

    def test_field_values_truncated(self) -> None:
        """Long field values should be truncated to Discord's 1024 char limit."""
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
        debate_embed = next(e for e in all_embeds if "LONG" in e.get("title", ""))
        bull_field = debate_embed["fields"][0]
        assert len(bull_field["value"]) <= 1024

    def test_invalid_date_fallback(self) -> None:
        brief = _make_brief(date="not-a-date")
        batches = format_intelligence_brief(brief)
        header = batches[0][0]
        assert "not-a-date" in header["title"]


class TestFormatDebateSide:
    """Tests for _format_debate_side helper."""

    def test_argument_only(self) -> None:
        result = _format_debate_side("Strong thesis", [])
        assert result == "Strong thesis"

    def test_with_evidence(self) -> None:
        result = _format_debate_side("Bull case", ["Evidence 1", "Evidence 2"])
        assert "Bull case" in result
        assert "\u203a Evidence 1" in result
        assert "\u203a Evidence 2" in result

    def test_evidence_capped_at_4(self) -> None:
        evidence = [f"Point {i}" for i in range(10)]
        result = _format_debate_side("Argument", evidence)
        assert result.count("\u203a") == 4


class TestFormatErrors:
    """Tests for _format_errors helper."""

    def test_empty_errors(self) -> None:
        errors: dict[str, Any] = {
            "social_failed": False,
            "news_failed": False,
            "company_failures": [],
            "price_failures": [],
            "bull_failures": [],
            "bear_failures": [],
            "macro_failed": False,
            "trader_failures": [],
        }
        assert _format_errors(errors) == []

    def test_boolean_failures(self) -> None:
        errors: dict[str, Any] = {"social_failed": True, "macro_failed": True}
        lines = _format_errors(errors)
        assert any("Social" in line for line in lines)
        assert any("Macro" in line for line in lines)

    def test_ticker_list_failures(self) -> None:
        errors: dict[str, Any] = {
            "company_failures": ["AAPL", "TSLA"],
            "trader_failures": ["NVDA"],
        }
        lines = _format_errors(errors)
        assert any("Company failed: AAPL, TSLA" in line for line in lines)
        assert any("Trader failed: NVDA" in line for line in lines)
