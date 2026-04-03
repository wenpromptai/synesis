"""Tests for ticker matching from financial news text.

Tests match_tickers() against:
  1. Curated name matching (NVIDIA→NVDA, APPLE→AAPL)
  2. Explicit $TICKER format
  3. Private company detection (~OPENAI, ~SPACEX)
  4. False positive prevention (macro/geopolitical headlines should return [])
  5. Real headlines from data/marketfeed.jsonl
"""

from synesis.processing.news.ticker_matcher import match_tickers


# =============================================================================
# Explicit $TICKER matching
# =============================================================================


class TestDollarTickerMatching:
    """$TICKER is the highest-confidence match — always works."""

    def test_dollar_prefix_tickers(self) -> None:
        assert match_tickers("$TSLA $AAPL $MSFT ALL DOWN TODAY") == ["TSLA", "AAPL", "MSFT"]

    def test_preserves_position_order(self) -> None:
        assert match_tickers("$MSFT beats, $AAPL misses, $TSLA flat") == ["MSFT", "AAPL", "TSLA"]

    def test_dollar_amount_not_matched(self) -> None:
        """$2B, $122B should not match as tickers (not uppercase letters)."""
        result = match_tickers("NVIDIA INVESTS $2B IN NEW PROJECT")
        assert "NVDA" in result
        assert len([t for t in result if not t.startswith("~")]) == 1  # Only NVDA


# =============================================================================
# Curated company name matching
# =============================================================================


class TestCuratedNameMatching:
    """Curated single-word and multi-word company name resolution."""

    def test_nvidia(self) -> None:
        assert match_tickers("*NVIDIA INVESTS $2B IN MARVELL TECHNOLOGY") == ["NVDA", "MRVL"]

    def test_apple_not_aple(self) -> None:
        """APPLE → AAPL (Apple Inc), NOT APLE (Apple Hospitality REIT)."""
        assert match_tickers("*APPLE Q4 EARNINGS BEAT, RAISES GUIDANCE") == ["AAPL"]

    def test_amazon(self) -> None:
        assert "AMZN" in match_tickers("AMAZON IN TALKS TO INVEST UP TO $50B IN OPENAI")

    def test_jpmorgan(self) -> None:
        """JPMORGAN → JPM, NOT JPIE or other JPM ETFs."""
        assert match_tickers("JPMORGAN BEATS Q4 EARNINGS ON STRONG TRADING") == ["JPM"]

    def test_oracle(self) -> None:
        """ORACLE → ORCL, NOT OHAQ."""
        assert match_tickers("ORACLE CUTS 18% OF ITS WORKFORCE") == ["ORCL"]

    def test_boeing(self) -> None:
        assert match_tickers("BOEING 737 MAX DELIVERIES HALTED") == ["BA"]

    def test_goldman_sachs(self) -> None:
        assert match_tickers("GOLDMAN SACHS RAISES S&P 500 TARGET TO 6000") == ["GS"]

    def test_eli_lilly(self) -> None:
        assert match_tickers("ELI LILLY FDA APPROVED FOR ADULTS WITH OBESITY") == ["LLY"]

    def test_lockheed_martin(self) -> None:
        assert "LMT" in match_tickers("LOCKHEED MARTIN WINS $2B DEFENSE CONTRACT")

    def test_berkshire_hathaway(self) -> None:
        assert "BRK.B" in match_tickers("BERKSHIRE HATHAWAY REPORTS RECORD CASH PILE")

    def test_ibm(self) -> None:
        assert match_tickers("*IBM 4Q REV. $19.69B, EST. $19.21B") == ["IBM"]

    def test_google_from_alphabet(self) -> None:
        assert "GOOGL" in match_tickers("ALPHABET REPORTS Q4 REVENUE BEAT")

    def test_google_from_google(self) -> None:
        assert "GOOGL" in match_tickers("ANTHROPIC RAISES $3B FROM GOOGLE")

    def test_microsoft_and_meta(self) -> None:
        result = match_tickers("MICROSOFT AND META ANNOUNCE AI PARTNERSHIP")
        assert "MSFT" in result
        assert "META" in result

    def test_deduplicates_name_and_dollar(self) -> None:
        """$NVDA + NVIDIA in same text should give one NVDA."""
        result = match_tickers("$NVDA - BERNSTEIN ON NVIDIA GROQ DEAL")
        assert result.count("NVDA") == 1

    def test_word_boundary(self) -> None:
        """APPLE should not match inside APPLETON."""
        assert match_tickers("APPLETON PARTNERS RAISES FUNDS") == []


# =============================================================================
# Private company matching (~PREFIX)
# =============================================================================


class TestPrivateTickers:
    """Non-public companies prefixed with ~."""

    def test_openai(self) -> None:
        assert "~OPENAI" in match_tickers("JUST IN: OPENAI RAISES $122B")

    def test_spacex(self) -> None:
        assert match_tickers("SPACEX IN TALKS WITH SAUDI PIF FOR $5 BLN ANCHOR STAKE") == [
            "~SPACEX"
        ]

    def test_bytedance_via_tiktok(self) -> None:
        assert "~BYTEDANCE" in match_tickers("TIKTOK BAN SIGNED INTO LAW")

    def test_anthropic_and_google(self) -> None:
        result = match_tickers("ANTHROPIC RAISES $3B FROM GOOGLE")
        assert "~ANTHROPIC" in result
        assert "GOOGL" in result

    def test_mixed_public_and_private(self) -> None:
        result = match_tickers("AMAZON IN TALKS TO INVEST UP TO $50B IN OPENAI")
        assert "AMZN" in result
        assert "~OPENAI" in result


# =============================================================================
# False positive prevention
# =============================================================================


class TestFalsePositivePrevention:
    """Headlines without company mentions should return empty list."""

    def test_macro_data(self) -> None:
        assert match_tickers("US CPI (YOY) (MAR) ACTUAL: 2.5% VS 1.9% PREVIOUS; EST 2.6%") == []
        assert match_tickers("US INITIAL JOBLESS CLAIMS ACTUAL: 202K VS 210K PREVIOUS") == []
        assert match_tickers("*US FEB. ISM SERVICES PMI AT 56.1, HIGHEST SINCE 2022") == []

    def test_fed_policy(self) -> None:
        assert match_tickers("FED CUTS RATES BY 50BPS - SURPRISE MOVE") == []
        assert match_tickers("FOMC DECISION: RATES UNCHANGED") == []

    def test_political(self) -> None:
        assert match_tickers("*TRUMP: PAM BONDI TRANSITIONING TO NEW PRIVATE SECTOR JOB") == []
        assert match_tickers("BIDEN SIGNS EXECUTIVE ORDER ON AI SAFETY") == []

    def test_geopolitical(self) -> None:
        assert match_tickers("IRAN WARNS OF TOTAL RESPONSE AS US BUILDS FORCES") == []
        assert match_tickers("SUPREME COURT STRIKES DOWN TRUMP'S GLOBAL TARIFFS") == []
        assert match_tickers("EU'S KALLAS: MUST SCALE UP EU'S ASPIDES NAVAL MISSION") == []

    def test_oil_commodity(self) -> None:
        assert (
            match_tickers("*DATED BRENT OIL PRICE SOARS TO $141.37/BBL, HIGHEST SINCE 2008") == []
        )
        assert match_tickers("OIL PRICES SURGE AS WAR ESCALATES") == []

    def test_imf_commentary(self) -> None:
        assert match_tickers("IMF EXPECTS US GROWTH TO INCREASE TO 2.4% BY 2026") == []

    def test_citadel_fund(self) -> None:
        assert match_tickers("*CITADEL GFI FUND SANK 8.2%, WELLINGTON DOWN 1.9%") == []

    def test_market_wipeout(self) -> None:
        assert match_tickers("JUST IN: OVER $777 BILLION WIPED OUT FROM US STOCK MARKET") == []


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    def test_empty_string(self) -> None:
        assert match_tickers("") == []

    def test_no_uppercase(self) -> None:
        assert match_tickers("all lowercase text about stocks") == []

    def test_url_not_matched(self) -> None:
        """x.com URLs should not produce ticker matches."""
        assert match_tickers("[...](https://x.com/DeItaone/status/123)") == []


# =============================================================================
# Real marketfeed headlines (integration)
# =============================================================================


class TestMarketfeedIntegration:
    """Real headlines from data/marketfeed.jsonl verified by grep."""

    HEADLINES: list[tuple[str, list[str]]] = [
        ("*NVIDIA INVESTS $2B IN MARVELL TECHNOLOGY", ["NVDA", "MRVL"]),
        ("*OPENAI: ANNOUNCING $110B IN NEW INVESTMENT AT A $730B VALUATION", ["~OPENAI"]),
        ("*AMAZON IN TALKS TO INVEST UP TO $50B IN OPENAI: WSJ", ["AMZN", "~OPENAI"]),
        ("SPACEX HAS HELD TALKS WITH SAUDI PIF FOR ANCHOR INVESTMENT IN 2026 IPO", ["~SPACEX"]),
        ("*IBM 4Q REV. $19.69B, EST. $19.21B", ["IBM"]),
        ("US CPI (YOY) (MAR) ACTUAL: 2.5% VS 1.9% PREVIOUS; EST 2.6%", []),
        ("US INITIAL JOBLESS CLAIMS ACTUAL: 202K VS 210K PREVIOUS; EST 212K", []),
        ("$100 OIL AND $4 GAS LOOM IF STRAIT OF HORMUZ STAYS CLOSED", []),
        ("TRUMP FACES 2,000 TARIFF LAWSUITS FOLLOWING SUPREME COURT LOSS", []),
        ("*DATED BRENT OIL PRICE SOARS TO $141.37/BBL, HIGHEST SINCE 2008", []),
        ("TESLA STOCK SLIDES AS BITCOIN FALLS", ["TSLA"]),
        ("GOLDMAN SACHS RAISES S&P 500 TARGET TO 6000", ["GS"]),
        ("JPMORGAN BEATS Q4 EARNINGS ON STRONG TRADING", ["JPM"]),
    ]

    def test_real_headlines(self) -> None:
        for headline, expected in self.HEADLINES:
            matched = match_tickers(headline)
            assert set(matched) == set(expected), (
                f"Headline: {headline[:60]}\n  Expected: {expected}\n  Got:      {matched}"
            )
