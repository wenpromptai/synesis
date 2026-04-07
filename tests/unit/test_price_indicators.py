"""Tests for PriceAnalyst indicator computation and pattern detection."""

from __future__ import annotations

from datetime import date, timedelta


from synesis.processing.intelligence.specialists.price.indicators import (
    bars_to_dataframe,
    compute_indicators,
    detect_notable_setups,
)


def _make_bars(n: int = 60, start_price: float = 100.0, trend: float = 0.5) -> list:
    """Create mock OHLCBar-like objects for testing."""
    from dataclasses import dataclass

    @dataclass
    class MockBar:
        date: date
        open: float
        high: float
        low: float
        close: float
        volume: int

    bars = []
    price = start_price
    base_date = date(2026, 1, 5)
    for i in range(n):
        price += trend + (i % 3 - 1) * 0.5  # trend + noise
        bars.append(
            MockBar(
                date=base_date + timedelta(days=i),
                open=round(price - 0.5, 2),
                high=round(price + 1.0, 2),
                low=round(price - 1.0, 2),
                close=round(price, 2),
                volume=1000000 + i * 10000,
            )
        )
    return bars


class TestBarsToDataframe:
    """Tests for OHLCV bar conversion."""

    def test_converts_bars(self) -> None:
        bars = _make_bars(30)
        df = bars_to_dataframe(bars)
        assert len(df) == 30
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "date"

    def test_empty_bars(self) -> None:
        df = bars_to_dataframe([])
        assert df.empty

    def test_sorted_by_date(self) -> None:
        bars = _make_bars(30)
        df = bars_to_dataframe(bars)
        assert df.index.is_monotonic_increasing


class TestComputeIndicators:
    """Tests for technical indicator computation."""

    def test_computes_all_indicators(self) -> None:
        bars = _make_bars(60)
        df = bars_to_dataframe(bars)
        indicators = compute_indicators(df)

        # Should have all key indicators
        assert "ema_8" in indicators
        assert "ema_21" in indicators
        assert "ema_cross" in indicators
        assert "rsi_14" in indicators
        assert "atr_percent" in indicators
        assert "volume_ratio" in indicators
        assert "nearest_support" in indicators
        assert "nearest_resistance" in indicators

    def test_rsi_in_valid_range(self) -> None:
        bars = _make_bars(60)
        df = bars_to_dataframe(bars)
        indicators = compute_indicators(df)
        rsi = indicators.get("rsi_14")
        assert rsi is not None
        assert 0 <= rsi <= 100

    def test_ema_cross_describes_direction(self) -> None:
        bars = _make_bars(60, trend=0.5)  # uptrend
        df = bars_to_dataframe(bars)
        indicators = compute_indicators(df)
        ema_cross = indicators.get("ema_cross", "")
        assert "bullish" in ema_cross or "bearish" in ema_cross

    def test_insufficient_bars_returns_empty(self) -> None:
        bars = _make_bars(10)  # too few
        df = bars_to_dataframe(bars)
        indicators = compute_indicators(df)
        assert indicators == {}

    def test_support_resistance_computed(self) -> None:
        bars = _make_bars(60)
        df = bars_to_dataframe(bars)
        indicators = compute_indicators(df)
        assert indicators["nearest_support"] < indicators["nearest_resistance"]

    def test_volume_ratio_positive(self) -> None:
        bars = _make_bars(60)
        df = bars_to_dataframe(bars)
        indicators = compute_indicators(df)
        vol_ratio = indicators.get("volume_ratio")
        assert vol_ratio is not None
        assert vol_ratio > 0


class TestDetectNotableSetups:
    """Tests for pattern detection."""

    def test_rsi_oversold(self) -> None:
        setups = detect_notable_setups({"rsi_14": 25.0}, {})
        assert any("oversold" in s for s in setups)

    def test_rsi_overbought(self) -> None:
        setups = detect_notable_setups({"rsi_14": 75.0}, {})
        assert any("overbought" in s for s in setups)

    def test_rsi_neutral_no_flag(self) -> None:
        setups = detect_notable_setups({"rsi_14": 50.0}, {})
        rsi_setups = [s for s in setups if "RSI" in s]
        assert len(rsi_setups) == 0

    def test_bb_squeeze(self) -> None:
        setups = detect_notable_setups({"bb_width_percentile": 10, "adx": 15}, {})
        assert any("squeeze" in s.lower() for s in setups)

    def test_extended_zscore(self) -> None:
        setups = detect_notable_setups({"price_zscore": 2.5}, {})
        assert any("extended" in s for s in setups)

    def test_depressed_zscore(self) -> None:
        setups = detect_notable_setups({"price_zscore": -2.3}, {})
        assert any("depressed" in s for s in setups)

    def test_unusual_volume(self) -> None:
        setups = detect_notable_setups({"volume_ratio": 3.0}, {})
        assert any("volume" in s.lower() for s in setups)

    def test_options_expensive(self) -> None:
        setups = detect_notable_setups({}, {"iv_rv_spread": 15})
        assert any("expensive" in s for s in setups)

    def test_options_cheap(self) -> None:
        setups = detect_notable_setups({}, {"iv_rv_spread": -8})
        assert any("cheap" in s for s in setups)

    def test_elevated_put_skew(self) -> None:
        setups = detect_notable_setups({}, {"atm_skew_ratio": 1.25})
        assert any("skew" in s for s in setups)

    def test_obv_bearish_divergence(self) -> None:
        setups = detect_notable_setups(
            {"obv_trend": "falling", "ema_cross": "bullish, crossed 3 days ago"}, {}
        )
        assert any("OBV divergence" in s for s in setups)

    def test_no_setups_when_normal(self) -> None:
        setups = detect_notable_setups(
            {
                "rsi_14": 50,
                "price_zscore": 0.5,
                "volume_ratio": 1.0,
                "adx": 25,
                "bb_width_percentile": 50,
            },
            {
                "iv_rv_spread": 3,
                "atm_skew_ratio": 1.08,
            },
        )
        assert len(setups) == 0


class TestComputeIVFromPrice:
    """Tests for Black-Scholes IV computation."""

    def test_atm_call_iv(self) -> None:
        """ATM call produces reasonable IV."""
        from synesis.processing.intelligence.specialists.price.indicators import (
            compute_iv_from_price,
        )

        # AAPL ~$260, ATM call at $4.20, 10 DTE
        iv = compute_iv_from_price(
            option_price=4.20, spot=260.0, strike=260.0, tte=10 / 365, rate=0.0425
        )
        assert iv is not None
        assert 0.15 < iv < 0.50, f"IV {iv:.2%} outside reasonable range for ATM"

    def test_atm_put_iv(self) -> None:
        """ATM put produces reasonable IV."""
        from synesis.processing.intelligence.specialists.price.indicators import (
            compute_iv_from_price,
        )

        iv = compute_iv_from_price(
            option_price=5.00,
            spot=260.0,
            strike=260.0,
            tte=10 / 365,
            rate=0.0425,
            option_type="put",
        )
        assert iv is not None
        assert 0.15 < iv < 0.50

    def test_call_put_iv_similar(self) -> None:
        """ATM call and put IV should be close (put-call parity)."""
        from synesis.processing.intelligence.specialists.price.indicators import (
            compute_iv_from_price,
        )

        call_iv = compute_iv_from_price(
            option_price=4.20,
            spot=260.0,
            strike=260.0,
            tte=10 / 365,
            rate=0.0425,
            option_type="call",
        )
        put_iv = compute_iv_from_price(
            option_price=4.80,
            spot=260.0,
            strike=260.0,
            tte=10 / 365,
            rate=0.0425,
            option_type="put",
        )
        assert call_iv is not None and put_iv is not None
        assert abs(call_iv - put_iv) < 0.10, "ATM call/put IV should be within 10 vol points"

    def test_deep_itm_option(self) -> None:
        """Deep ITM option (mostly intrinsic value) still computes."""
        from synesis.processing.intelligence.specialists.price.indicators import (
            compute_iv_from_price,
        )

        # $200 call on $260 stock = $60 intrinsic + small time value
        iv = compute_iv_from_price(
            option_price=61.0, spot=260.0, strike=200.0, tte=10 / 365, rate=0.0425
        )
        # Should return something, may be high IV due to time value component
        assert iv is not None

    def test_zero_price_returns_none(self) -> None:
        """Zero or negative inputs return None."""
        from synesis.processing.intelligence.specialists.price.indicators import (
            compute_iv_from_price,
        )

        assert compute_iv_from_price(0, 260.0, 260.0, 10 / 365) is None
        assert compute_iv_from_price(4.0, 0, 260.0, 10 / 365) is None
        assert compute_iv_from_price(4.0, 260.0, 260.0, 0) is None
