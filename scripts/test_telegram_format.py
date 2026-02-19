#!/usr/bin/env python3
"""Test script to verify enhanced Telegram notification formatting.

Run: uv run python scripts/test_telegram_format.py
"""

from datetime import datetime, timezone

from synesis.notifications.telegram import format_condensed_signal
from synesis.processing.news import (
    Direction,
    MarketEvaluation,
    ResearchQuality,
    SectorImplication,
    SmartAnalysis,
    SourcePlatform,
    TickerAnalysis,
    UnifiedMessage,
)


def create_sample_message() -> UnifiedMessage:
    """Create a sample unified message."""
    return UnifiedMessage(
        external_id="test_12345",
        source_platform=SourcePlatform.telegram,
        source_account="@DeItaone",
        text="*FED CUTS RATES BY 25BPS, AS EXPECTED - Fed funds rate now at 4.00-4.25%\n\nFOMC statement notes continued progress on inflation",
        timestamp=datetime.now(timezone.utc),
        raw={},
    )


def create_sample_analysis() -> SmartAnalysis:
    """Create a sample Stage 2 analysis with rich data."""
    return SmartAnalysis(
        tickers=["SPY", "TLT", "XLF", "QQQ"],
        sectors=["Financials", "Real Estate", "Technology"],
        sentiment=Direction.bullish,
        sentiment_score=0.7,
        primary_thesis="Rate cuts support equity valuations through lower discount rates and reduced borrowing costs, particularly benefiting rate-sensitive sectors like real estate and financials",
        thesis_confidence=0.82,
        ticker_analyses=[
            TickerAnalysis(
                ticker="TLT",
                company_name="iShares 20+ Year Treasury Bond ETF",
                bull_thesis="Lower rates directly benefit long-duration bonds. 25bps cut aligns with expectations, and forward guidance suggests more cuts in 2025",
                bear_thesis="If inflation resurges, Fed may pause or reverse cuts. 'Higher for longer' narrative could return",
                net_direction=Direction.bullish,
                conviction=0.85,
                time_horizon="days",
                catalysts=[
                    "January CPI data",
                    "Employment reports",
                    "Fed dot plot projections",
                ],
                risk_factors=[
                    "Sticky services inflation",
                    "Strong labor market data",
                    "Global rate divergence",
                ],
            ),
            TickerAnalysis(
                ticker="XLF",
                company_name="Financial Select Sector SPDR",
                bull_thesis="Rate cuts improve borrowing conditions and may steepen yield curve, benefiting bank net interest margins over time",
                bear_thesis="Near-term, flattening yield curve compresses margins. Credit concerns if cuts signal economic weakness",
                net_direction=Direction.bullish,
                conviction=0.72,
                time_horizon="weeks",
                catalysts=[
                    "Q4 bank earnings",
                    "Loan growth data",
                    "Credit quality reports",
                ],
                risk_factors=[
                    "Commercial real estate exposure",
                    "Consumer credit deterioration",
                ],
            ),
        ],
        sector_implications=[
            SectorImplication(
                sector="Real Estate",
                direction=Direction.bullish,
                reasoning="Lower rates reduce mortgage costs and cap rates, supporting property valuations across residential and commercial segments",
                subsectors=["REITs", "Homebuilders", "Commercial Real Estate"],
            ),
            SectorImplication(
                sector="Financials",
                direction=Direction.bullish,
                reasoning="While near-term NIM pressure exists, improved credit conditions and increased loan activity support longer-term outlook",
                subsectors=["Regional Banks", "Insurance", "Asset Managers"],
            ),
            SectorImplication(
                sector="Technology",
                direction=Direction.bullish,
                reasoning="Growth stocks benefit from lower discount rates applied to future earnings. Borrowing costs decrease for high-growth companies",
                subsectors=["Software", "Cloud Computing", "AI Infrastructure"],
            ),
        ],
        historical_context="The Fed's December 2024 cut follows a pattern seen in 2019 mid-cycle adjustments. In similar scenarios, equities have rallied 5-8% over the following 3 months, with rate-sensitive sectors outperforming. The 1995 'soft landing' cuts saw SPY gain 34% the following year.",
        typical_market_reaction="Historically, expected rate cuts produce muted immediate reactions but sustained rallies over weeks as liquidity conditions improve. Treasury yields typically fall 10-20bps across the curve in the week following a cut.",
        market_evaluations=[
            MarketEvaluation(
                market_id="poly_fed_2025_1",
                market_question="Will the Fed cut rates again at the January 2025 FOMC meeting?",
                is_relevant=True,
                relevance_reasoning="Directly related to Fed rate policy trajectory following today's cut",
                current_price=0.35,
                estimated_fair_price=0.42,
                edge=0.07,
                verdict="undervalued",
                confidence=0.75,
                reasoning="Market pricing 35% probability is below our estimate of 42%. FOMC dot plot suggests another cut, and Powell's language indicated continued data-dependency favoring cuts if inflation remains subdued.",
                recommended_side="yes",
            ),
            MarketEvaluation(
                market_id="poly_recession_2025",
                market_question="Will the US enter a recession in 2025?",
                is_relevant=True,
                relevance_reasoning="Fed rate cuts often interpreted as response to economic slowdown concerns",
                current_price=0.28,
                estimated_fair_price=0.22,
                edge=-0.06,
                verdict="overvalued",
                confidence=0.68,
                reasoning="Current pricing of 28% seems elevated given strong labor market and consumer spending. Rate cuts appear precautionary rather than reactive to imminent recession.",
                recommended_side="no",
            ),
            MarketEvaluation(
                market_id="poly_sp500_5500",
                market_question="Will the S&P 500 reach 5500 by end of Q1 2025?",
                is_relevant=True,
                relevance_reasoning="Rate cuts historically support equity valuations",
                current_price=0.52,
                estimated_fair_price=0.53,
                edge=0.01,
                verdict="fair",
                confidence=0.55,
                reasoning="Market pricing appears roughly fair. Rate cuts are supportive, but much depends on earnings growth and geopolitical factors.",
                recommended_side="skip",
            ),
        ],
        research_quality=ResearchQuality.high,
    )


def main() -> None:
    """Run the test to display formatted output."""
    message = create_sample_message()
    analysis = create_sample_analysis()

    formatted = format_condensed_signal(message, analysis)

    print("=" * 60)
    print("FORMATTED TELEGRAM MESSAGE")
    print("=" * 60)
    print(formatted)
    print("=" * 60)
    print(f"\nTotal length: {len(formatted)} characters")
    print("Telegram limit: 4096 characters")
    print(f"Will split: {'Yes' if len(formatted) > 4096 else 'No'}")

    if len(formatted) > 4096:
        from synesis.notifications.telegram import _split_message_at_sections

        chunks = _split_message_at_sections(formatted)
        print(f"\nWould split into {len(chunks)} messages:")
        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i}: {len(chunk)} chars")


if __name__ == "__main__":
    main()
