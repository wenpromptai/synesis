"""Integration tests for MacroStrategist against real FRED data.

Tests that the strategist correctly:
1. Fetches all FRED series (VIX, yields, fed funds, unemployment)
2. Consolidates macro themes from different sources
3. Produces a coherent regime assessment with thematic tilts

Run with: uv run pytest tests/integration/test_macro_strategist.py -v -m integration
"""

from __future__ import annotations

from datetime import date

import pytest

from synesis.processing.intelligence.models import MacroView, ThematicTilt
from synesis.processing.intelligence.strategists.macro import (
    MacroStrategistDeps,
    _fetch_fred_data,
    _format_fred_context,
    _format_macro_themes,
    analyze_macro,
)


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
async def fred_client(real_redis):
    """Real FRED client for integration testing."""
    from synesis.providers.fred.client import FREDClient

    client = FREDClient(redis=real_redis)
    yield client
    await client.close()


# ── FRED Data Fetching ───────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fetch_fred_data(fred_client):
    """All key FRED series return recent data."""
    data = await _fetch_fred_data(fred_client)

    # VIX should be available and recent
    assert "VIXCLS" in data
    assert data["VIXCLS"]["value"] is not None
    vix = float(data["VIXCLS"]["value"])
    assert 5 < vix < 100, f"VIX {vix} seems unreasonable"

    # 10Y yield
    assert "DGS10" in data
    assert data["DGS10"]["value"] is not None
    y10 = float(data["DGS10"]["value"])
    assert 0 < y10 < 20, f"10Y yield {y10}% seems unreasonable"

    # 2Y yield
    assert "DGS2" in data
    assert data["DGS2"]["value"] is not None

    # Fed funds rate
    assert "FEDFUNDS" in data
    assert data["FEDFUNDS"]["value"] is not None

    # Unemployment
    assert "UNRATE" in data
    assert data["UNRATE"]["value"] is not None

    # Yield curve spread computed
    assert "yield_curve_spread" in data
    spread = data["yield_curve_spread"]["value"]
    assert isinstance(spread, float)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fred_context_formatting(fred_client):
    """FRED data formats into readable context for LLM."""
    data = await _fetch_fred_data(fred_client)
    context = _format_fred_context(data)

    assert "## Current Economic Indicators (FRED)" in context
    assert "VIX" in context
    assert "10-Year Treasury Yield" in context
    assert "Yield Curve Spread" in context


# ── Theme Consolidation ──────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_format_macro_themes():
    """Macro themes from social + news are consolidated."""
    state = {
        "social_analysis": {
            "macro_themes": [
                {
                    "theme": "risk-off rotation",
                    "sentiment_score": -0.6,
                    "context": "multiple accounts",
                },
                {"theme": "AI capex cycle", "sentiment_score": 0.5, "context": "tech focus"},
            ],
        },
        "news_analysis": {
            "macro_themes": [
                {"theme": "tariff escalation", "sentiment_score": -0.7, "context": "trade war"},
            ],
            "story_clusters": [
                {"headline": "Fed hawkish", "event_type": "macro", "urgency": "critical"},
            ],
        },
    }

    formatted = _format_macro_themes(state)

    assert "## Macro Themes from Layer 1 Analysts" in formatted
    assert "[Social] risk-off rotation" in formatted
    assert "[Social] AI capex cycle" in formatted
    assert "[News] tariff escalation" in formatted
    assert "[News cluster] Fed hawkish" in formatted


# ── Full Pipeline (LLM call) ─────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_macro_analysis(fred_client):
    """Full MacroStrategist pipeline with real FRED data + mock themes.

    Costs ~$0.05-0.10 per run (smart model, small prompt).
    """
    state = {
        "social_analysis": {
            "macro_themes": [
                {"theme": "risk-off sentiment rising", "sentiment_score": -0.5},
            ],
        },
        "news_analysis": {
            "macro_themes": [
                {"theme": "Fed hawkish, rates higher for longer", "sentiment_score": -0.6},
            ],
            "story_clusters": [
                {
                    "headline": "Fed official: no rush to cut rates",
                    "event_type": "macro",
                    "urgency": "high",
                },
            ],
        },
    }

    from unittest.mock import AsyncMock

    mock_db = AsyncMock()
    mock_db.get_upcoming_events = AsyncMock(return_value=[])
    mock_db.get_events_by_date_range = AsyncMock(return_value=[])
    mock_db.get_last_fomc_meeting_date = AsyncMock(return_value=None)

    mock_yf = AsyncMock()
    mock_yf.get_quote = AsyncMock(
        return_value=AsyncMock(
            last=None,
            prev_close=None,
            avg_50d=None,
            avg_200d=None,
        )
    )

    mock_sec = AsyncMock()

    deps = MacroStrategistDeps(
        fred=fred_client,
        db=mock_db,
        yfinance=mock_yf,
        sec_edgar=mock_sec,
        current_date=date.today(),
    )
    result = await analyze_macro(state, deps)

    assert isinstance(result, MacroView)

    # Regime should be assessed
    assert result.regime in ("risk_on", "risk_off", "transitioning", "uncertain")
    assert -1.0 <= result.sentiment_score <= 1.0

    # Key drivers should be populated
    assert len(result.key_drivers) >= 2, "Should have at least 2 key drivers"

    # Sector tilts should exist
    assert len(result.thematic_tilts) >= 1, "Should have at least 1 sector tilt"
    for tilt in result.thematic_tilts:
        assert isinstance(tilt, ThematicTilt)
        assert tilt.theme  # non-empty
        assert -1.0 <= tilt.sentiment_score <= 1.0

    # Risks should be identified
    assert len(result.risks) >= 1, "Should have at least 1 risk"

    print(f"\n{'=' * 60}")
    print("MacroStrategist Result")
    print(f"{'=' * 60}")
    print(f"Regime: {result.regime} (sentiment: {result.sentiment_score:+.1f})")
    print(f"Key drivers: {result.key_drivers}")
    print("Sector tilts:")
    for tilt in result.thematic_tilts:
        print(f"  {tilt.theme}: {tilt.sentiment_score:+.1f} — {tilt.reasoning}")
    print(f"Risks: {result.risks}")
