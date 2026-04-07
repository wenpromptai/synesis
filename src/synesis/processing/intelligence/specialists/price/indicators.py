"""Technical indicator computation from OHLCV bars via pandas-ta.

All functions are pure — take bar data, return computed values.
No API calls, no LLM, no side effects.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pandas_ta  # noqa: F401 — registers df.ta accessor
from scipy.stats import norm

from synesis.core.logging import get_logger

logger = get_logger(__name__)


def bars_to_dataframe(bars: list[Any]) -> pd.DataFrame:
    """Convert yfinance OHLCBar list to a pandas DataFrame for pandas-ta.

    Args:
        bars: List of OHLCBar objects with date, open, high, low, close, volume.

    Returns:
        DataFrame with columns: open, high, low, close, volume, indexed by date.
    """
    if not bars:
        return pd.DataFrame()

    records = []
    for b in bars:
        records.append(
            {
                "date": b.date,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume or 0,
            }
        )

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Drop rows with None values
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def compute_indicators(df: pd.DataFrame) -> dict[str, Any]:
    """Compute all technical indicators from an OHLCV DataFrame.

    Returns a dict of computed values ready for PriceAnalysis model fields.
    """
    if df.empty or len(df) < 20:
        logger.warning("Insufficient bars for indicator computation", bars=len(df))
        return {}

    result: dict[str, Any] = {}
    close = df["close"]
    latest_close = float(close.iloc[-1])

    # ── Trend: EMA crossover ─────────────────────────────────────
    df.ta.ema(length=8, append=True)
    df.ta.ema(length=21, append=True)

    ema8 = df.get("EMA_8")
    ema21 = df.get("EMA_21")
    if ema8 is not None and ema21 is not None:
        result["ema_8"] = round(float(ema8.iloc[-1]), 2)
        result["ema_21"] = round(float(ema21.iloc[-1]), 2)

        # Detect crossover
        above_now = ema8.iloc[-1] > ema21.iloc[-1]
        cross_desc = "bullish" if above_now else "bearish"

        # Count days since cross
        days_since = 0
        for i in range(len(df) - 2, max(0, len(df) - 30), -1):
            was_above = ema8.iloc[i] > ema21.iloc[i]
            if was_above != above_now:
                days_since = len(df) - 1 - i
                break

        if days_since > 0:
            result["ema_cross"] = f"{cross_desc}, crossed {days_since} days ago"
        else:
            result["ema_cross"] = f"{cross_desc}, EMA8 {'above' if above_now else 'below'} EMA21"

    # ── Trend: ADX ───────────────────────────────────────────────
    adx_df = df.ta.adx(length=14)
    if adx_df is not None and "ADX_14" in adx_df.columns:
        result["adx"] = round(float(adx_df["ADX_14"].iloc[-1]), 1)

    # ── Momentum: RSI ────────────────────────────────────────────
    df.ta.rsi(length=14, append=True)
    rsi_col = df.get("RSI_14")
    if rsi_col is not None:
        result["rsi_14"] = round(float(rsi_col.iloc[-1]), 1)

    # ── Momentum: MACD ───────────────────────────────────────────
    macd_df = df.ta.macd(fast=12, slow=26, signal=9)
    if macd_df is not None:
        hist_col = [c for c in macd_df.columns if "MACDh" in c]
        macd_col = [c for c in macd_df.columns if c.startswith("MACD_")]
        signal_col = [c for c in macd_df.columns if "MACDs" in c]

        if hist_col:
            hist_val = float(macd_df[hist_col[0]].iloc[-1])
            result["macd_histogram"] = round(hist_val, 3)

        # Detect signal line crossover
        if macd_col and signal_col:
            macd_vals = macd_df[macd_col[0]]
            signal_vals = macd_df[signal_col[0]]
            above_now = macd_vals.iloc[-1] > signal_vals.iloc[-1]
            was_above = (
                macd_vals.iloc[-2] > signal_vals.iloc[-2] if len(macd_vals) > 1 else above_now
            )
            if above_now and not was_above:
                result["macd_signal_cross"] = "bullish cross"
            elif not above_now and was_above:
                result["macd_signal_cross"] = "bearish cross"
            else:
                result["macd_signal_cross"] = "none"

    # ── Volatility: ATR ──────────────────────────────────────────
    df.ta.atr(length=14, append=True)
    atr_col = df.get("ATRr_14")
    if atr_col is not None and latest_close > 0:
        atr_val = float(atr_col.iloc[-1])
        result["atr_percent"] = round(atr_val / latest_close * 100, 2)

    # ── Volatility: Bollinger Bands ──────────────────────────────
    bb_df = df.ta.bbands(length=20, std=2)
    if bb_df is not None:
        bw_col = [c for c in bb_df.columns if "BBB" in c]
        bp_col = [c for c in bb_df.columns if "BBP" in c]

        # %B (position within bands)
        if bp_col:
            result["bb_percent_b"] = round(float(bb_df[bp_col[0]].iloc[-1]), 3)

        # BB width percentile vs 3mo range
        if bw_col:
            bw_series = bb_df[bw_col[0]].dropna()
            if len(bw_series) > 5:
                current_bw = float(bw_series.iloc[-1])
                percentile = (bw_series < current_bw).sum() / len(bw_series) * 100
                result["bb_width_percentile"] = round(float(percentile), 1)

    # ── Mean Reversion: Z-score vs SMA(20) ───────────────────────
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    if sma20.iloc[-1] is not None and std20.iloc[-1] is not None and std20.iloc[-1] > 0:
        zscore = (latest_close - sma20.iloc[-1]) / std20.iloc[-1]
        result["price_zscore"] = round(float(zscore), 2)

    # ── Volume ───────────────────────────────────────────────────
    vol = df["volume"]
    avg_20d = vol.rolling(20).mean()
    if avg_20d.iloc[-1] is not None and avg_20d.iloc[-1] > 0:
        result["volume_ratio"] = round(float(vol.iloc[-1] / avg_20d.iloc[-1]), 2)

    # OBV trend (10-day slope direction)
    df.ta.obv(append=True)
    obv_col = df.get("OBV")
    if obv_col is not None and len(obv_col) >= 10:
        obv_10d_change = float(obv_col.iloc[-1]) - float(obv_col.iloc[-10])
        if obv_10d_change > 0:
            result["obv_trend"] = "rising"
        elif obv_10d_change < 0:
            result["obv_trend"] = "falling"
        else:
            result["obv_trend"] = "flat"

    # ── Support / Resistance (pivot points) ──────────────────────
    # Use prior day's high/low/close for classic pivots
    if len(df) >= 2:
        prev = df.iloc[-2]
        pivot = (prev["high"] + prev["low"] + prev["close"]) / 3
        s1 = 2 * pivot - prev["high"]
        r1 = 2 * pivot - prev["low"]
        result["nearest_support"] = round(float(s1), 2)
        result["nearest_resistance"] = round(float(r1), 2)

    return result


def detect_notable_setups(indicators: dict[str, Any], options: dict[str, Any]) -> list[str]:
    """Flag notable technical + options patterns.

    Returns list of plain English setup descriptions.
    """
    setups: list[str] = []

    # BB squeeze (low BB width + low ADX)
    bb_pct = indicators.get("bb_width_percentile")
    adx = indicators.get("adx")
    if bb_pct is not None and bb_pct < 20 and adx is not None and adx < 20:
        setups.append(
            f"Volatility squeeze: BB width at {bb_pct:.0f}th percentile with ADX {adx:.0f} — breakout likely"
        )

    # RSI extreme
    rsi = indicators.get("rsi_14")
    if rsi is not None:
        if rsi < 30:
            setups.append(f"RSI oversold at {rsi:.0f} — potential bounce setup")
        elif rsi > 70:
            setups.append(f"RSI overbought at {rsi:.0f} — potential pullback risk")

    # Price extended from mean
    zscore = indicators.get("price_zscore")
    if zscore is not None:
        if zscore > 2.0:
            setups.append(
                f"Price extended: z-score {zscore:+.1f} above 20d mean — mean reversion risk"
            )
        elif zscore < -2.0:
            setups.append(
                f"Price depressed: z-score {zscore:+.1f} below 20d mean — potential bounce"
            )

    # Unusual volume
    vol_ratio = indicators.get("volume_ratio")
    if vol_ratio is not None and vol_ratio > 2.0:
        setups.append(f"Unusual volume: {vol_ratio:.1f}x 20-day average")

    # OBV divergence
    obv_trend = indicators.get("obv_trend")
    ema_cross = indicators.get("ema_cross", "")
    if obv_trend == "falling" and "bullish" in ema_cross:
        setups.append(
            "OBV divergence: money flowing out despite bullish price trend — weakening rally"
        )
    elif obv_trend == "rising" and "bearish" in ema_cross:
        setups.append(
            "OBV divergence: money flowing in despite bearish price trend — potential reversal"
        )

    # IV vs RV spread
    iv_rv = options.get("iv_rv_spread")
    if iv_rv is not None:
        if iv_rv > 10:
            setups.append(
                f"Options expensive: IV-RV spread +{iv_rv:.0f}% — premium selling opportunity"
            )
        elif iv_rv < -5:
            setups.append(f"Options cheap: IV-RV spread {iv_rv:.0f}% — premium buying opportunity")

    # Elevated put skew
    skew = options.get("atm_skew_ratio")
    if skew is not None and skew > 1.20:
        setups.append(f"Elevated put skew ({skew:.2f}x) — unusual demand for downside protection")

    return setups


# ── IV Computation ───────────────────────────────────────────────


def compute_iv_from_price(
    option_price: float,
    spot: float,
    strike: float,
    tte: float,
    rate: float = 0.0425,
    option_type: str = "call",
    max_iter: int = 100,
) -> float | None:
    """Compute implied volatility from an option's market price using Newton-Raphson.

    Used as fallback when yfinance IV is invalid (0% or <1%).

    Args:
        option_price: Option's last trade or close price.
        spot: Underlying stock price.
        strike: Option strike price.
        tte: Time to expiry in years (days / 365).
        rate: Risk-free rate (default ~3mo Treasury).
        option_type: "call" or "put".
        max_iter: Maximum Newton-Raphson iterations.

    Returns:
        Implied volatility as a decimal (e.g. 0.27 = 27%), or None if failed.
    """
    if option_price <= 0 or spot <= 0 or strike <= 0 or tte <= 0:
        return None

    iv = 0.3  # initial guess
    for _ in range(max_iter):
        sqrt_t = math.sqrt(tte)
        d1 = (math.log(spot / strike) + (rate + 0.5 * iv**2) * tte) / (iv * sqrt_t)
        d2 = d1 - iv * sqrt_t

        if option_type == "call":
            price = spot * norm.cdf(d1) - strike * math.exp(-rate * tte) * norm.cdf(d2)
        else:
            price = strike * math.exp(-rate * tte) * norm.cdf(-d2) - spot * norm.cdf(-d1)

        vega = spot * norm.pdf(d1) * sqrt_t
        if vega < 1e-10:
            break

        diff = price - option_price
        if abs(diff) < 1e-6:
            break

        iv -= diff / vega
        iv = max(0.01, min(5.0, iv))

    # Sanity check: IV should be between 1% and 500%
    if 0.01 <= iv <= 5.0:
        return round(iv, 4)
    return None
