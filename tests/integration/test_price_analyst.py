"""Integration tests for PriceAnalyst against real AAPL data.

Tests that:
1. yfinance returns bars and quote correctly
2. pandas-ta indicators compute from real bars
3. Massive returns ATM options bars
4. IV self-computation produces reasonable values from Massive EOD prices
5. Full pipeline produces valid PriceAnalysis with LLM narratives

Run with: uv run pytest tests/integration/test_price_analyst.py -v -m integration
"""

from __future__ import annotations

from datetime import date

import pytest

from synesis.processing.intelligence.models import PriceAnalysis
from synesis.processing.intelligence.specialists.price.agent import (
    PriceDeps,
    _derive_options_metrics,
    _gather_massive,
    _gather_yfinance,
    analyze_price,
)
from synesis.processing.intelligence.specialists.price.indicators import (
    bars_to_dataframe,
    compute_indicators,
    compute_iv_from_price,
)

TICKER = "AAPL"


@pytest.fixture
async def yf_client(real_redis):
    from synesis.providers.yfinance.client import YFinanceClient

    return YFinanceClient(redis=real_redis)


@pytest.fixture
async def massive_client(real_redis):
    try:
        from synesis.providers.massive.client import MassiveClient

        return MassiveClient(redis=real_redis)
    except (ValueError, Exception):
        return None


# ── yfinance Data ────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gather_yfinance(yf_client):
    """yfinance returns bars, quote, and realized vol."""
    data = await _gather_yfinance(yf_client, TICKER)

    assert data["quote"].ticker == TICKER
    assert data["quote"].last is not None and data["quote"].last > 0
    assert len(data["bars"]) >= 40


# ── Indicators ───────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_indicators_from_real_bars(yf_client):
    """All key indicators compute from real AAPL bars."""
    bars = await yf_client.get_history(TICKER, period="3mo", interval="1d")
    df = bars_to_dataframe(bars)
    indicators = compute_indicators(df)

    assert "rsi_14" in indicators
    assert 0 <= indicators["rsi_14"] <= 100
    assert "ema_8" in indicators
    assert "ema_21" in indicators
    assert "adx" in indicators
    assert "atr_percent" in indicators
    assert "bb_percent_b" in indicators
    assert "bb_width_percentile" in indicators
    assert "price_zscore" in indicators
    assert "volume_ratio" in indicators
    assert "obv_trend" in indicators
    assert "nearest_support" in indicators
    assert "nearest_resistance" in indicators
    assert "ema_cross" in indicators
    assert "macd_histogram" in indicators


# ── Massive Options + Short Data ─────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gather_massive_options(massive_client, yf_client):
    """Massive returns ATM options bars."""
    if massive_client is None:
        pytest.skip("Massive client not configured")

    quote = await yf_client.get_quote(TICKER)
    data = await _gather_massive(massive_client, TICKER, quote.last)

    # ATM options close prices (from EOD bars)
    assert data["atm_call_close"] is not None, "ATM call close should be available"
    assert data["atm_call_close"] > 0
    assert data["atm_put_close"] is not None, "ATM put close should be available"
    assert data["atm_put_close"] > 0
    assert data["strike"] is not None
    assert data["dte"] is not None and data["dte"] > 20

    # Volume from bars
    assert data["atm_call_volume"] is not None and data["atm_call_volume"] > 0
    assert data["atm_put_volume"] is not None and data["atm_put_volume"] > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_iv_from_massive_eod(massive_client, yf_client):
    """IV computed from Massive EOD close prices is reasonable."""
    if massive_client is None:
        pytest.skip("Massive client not configured")

    quote = await yf_client.get_quote(TICKER)
    spot = quote.last
    data = await _gather_massive(massive_client, TICKER, spot)

    if not data["atm_call_close"] or not data["dte"]:
        pytest.skip("No ATM options data from Massive")

    tte = data["dte"] / 365.0

    call_iv = compute_iv_from_price(
        data["atm_call_close"], spot, data["strike"], tte, option_type="call"
    )
    put_iv = compute_iv_from_price(
        data["atm_put_close"], spot, data["strike"], tte, option_type="put"
    )

    assert call_iv is not None, "Call IV should compute"
    assert put_iv is not None, "Put IV should compute"
    assert 0.05 < call_iv < 1.5, f"Call IV {call_iv:.1%} unreasonable"
    assert 0.05 < put_iv < 1.5, f"Put IV {put_iv:.1%} unreasonable"
    assert abs(call_iv - put_iv) < 0.15, "Call/put IV should be within 15 vol points"

    print(f"\nIV from Massive EOD ({TICKER} {data['strike']} exp {data['expiration']})")
    print(f"  Call: ${data['atm_call_close']:.2f} → IV={call_iv:.1%}")
    print(f"  Put:  ${data['atm_put_close']:.2f} → IV={put_iv:.1%}")
    print(f"  Skew: {put_iv / call_iv:.2f}")
    print(f"  Avg:  {(call_iv + put_iv) / 2:.1%}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_options_metrics_derived(massive_client, yf_client):
    """Options metrics derive correctly from Massive EOD data."""
    if massive_client is None:
        pytest.skip("Massive client not configured")

    quote = await yf_client.get_quote(TICKER)
    spot = quote.last
    massive_data = await _gather_massive(massive_client, TICKER, spot)

    # Get realized vol from yfinance
    yf_data = await _gather_yfinance(yf_client, TICKER)
    rv = yf_data.get("realized_vol_30d")

    metrics = _derive_options_metrics(massive_data, spot, rv)

    if massive_data["atm_call_close"]:
        assert "atm_iv" in metrics, "ATM IV should be computed"
        assert 5 < metrics["atm_iv"] < 150, f"ATM IV {metrics['atm_iv']}% unreasonable"

        if rv:
            assert "iv_rv_spread" in metrics, "IV-RV spread should be computed"

        assert "atm_skew_ratio" in metrics, "Skew should be computed"
        assert 0.5 < metrics["atm_skew_ratio"] < 2.0, "Skew should be reasonable"

    if massive_data["atm_call_volume"] and massive_data["atm_call_volume"] > 0:
        assert "put_call_volume_ratio" in metrics

    print(f"\nOptions metrics for {TICKER}:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")


# ── Full Pipeline ────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_price_analysis(yf_client, massive_client):
    """Full PriceAnalyst pipeline with LLM narratives."""
    deps = PriceDeps(
        yfinance=yf_client,
        massive=massive_client,
        current_date=date.today(),
    )

    result = await analyze_price(TICKER, deps)

    assert isinstance(result, PriceAnalysis)
    assert result.ticker == TICKER
    assert result.spot_price is not None and result.spot_price > 0

    # Technical indicators populated
    assert result.rsi_14 is not None
    assert result.ema_8 is not None
    assert result.ema_21 is not None
    assert result.adx is not None
    assert result.atr_percent is not None

    # Options IV (from Massive EOD if available)
    if massive_client:
        assert result.atm_iv is not None, "ATM IV should be computed from Massive"

    # LLM narratives
    assert result.technical_narrative != ""
    assert result.options_narrative != ""

    print(f"\n{'=' * 60}")
    print(f"PriceAnalyst: {result.ticker} @ ${result.spot_price:.2f}")
    print(f"{'=' * 60}")
    print(f"RSI: {result.rsi_14} | ADX: {result.adx} | ATR%: {result.atr_percent}")
    print(f"EMA cross: {result.ema_cross}")
    print(f"BB %B: {result.bb_percent_b} | BB width pctl: {result.bb_width_percentile}")
    print(f"Z-score: {result.price_zscore} | Vol ratio: {result.volume_ratio}")
    print(
        f"ATM IV: {result.atm_iv}% | RV: {result.realized_vol_30d}% | IV-RV: {result.iv_rv_spread}"
    )
    print(f"P/C Vol: {result.put_call_volume_ratio} | Skew: {result.atm_skew_ratio}")
    print(f"Notable: {result.notable_setups}")
    print(f"\nTechnical: {result.technical_narrative[:300]}...")
    print(f"Options: {result.options_narrative[:300]}...")
