"""HMM-based macro regime detector.

Trains a Gaussian HMM on weekly cross-asset features to classify regimes.
Uses FRED + yfinance data that the pipeline already fetches.

Features (6):
  1. SPY weekly log returns (equity risk appetite)
  2. SPY 20-day realized volatility (regime differentiator)
  3. VIX level (fear gauge)
  4. HY credit spread OAS (stress indicator)
  5. Yield curve 10Y-2Y (recession/expansion)
  6. UUP weekly log returns (dollar/global risk)

Usage:
    detector = RegimeDetector(fred_api_key="...")
    await detector.fit(lookback_years=10)
    result = detector.predict_current()
    # result = {"regime": "risk_on", "confidence": 0.85, "probabilities": {...}}
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Any

import httpx
import numpy as np
import pandas as pd
import yfinance as yf

from synesis.core.logging import get_logger

logger = get_logger(__name__)

_FEATURE_NAMES = [
    "spy_ret",
    "spy_vol",
    "vix",
    "hy_oas",
    "t10y2y",
    "uup_ret",
]

_N_STATES = 3
_MIN_WEEKS = 104  # Minimum 2 years for training


class RegimeDetector:
    """HMM-based macro regime detector."""

    def __init__(self, fred_api_key: str) -> None:
        self._fred_api_key = fred_api_key
        self._model: Any = None
        self._regime_labels: dict[int, str] = {}
        self._last_trained: date | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    async def fit(self, lookback_years: int = 10, n_states: int = _N_STATES) -> dict[str, Any]:
        """Fetch data and fit the HMM.

        Args:
            lookback_years: Years of weekly data to use for training.
            n_states: Number of hidden regimes (2, 3, or 4).

        Returns:
            Dict with training summary (n_observations, n_states, regime_labels, log_likelihood).
        """
        from hmmlearn.hmm import GaussianHMM

        start = date.today() - timedelta(days=lookback_years * 365)

        logger.info("Fetching regime data", start=str(start), lookback_years=lookback_years)
        raw = await self._fetch_data(str(start))

        if len(raw) < _MIN_WEEKS:
            raise ValueError(f"Insufficient data: got {len(raw)} weeks, need {_MIN_WEEKS}")

        # Normalize features (expanding z-score, no lookahead)
        normalized = self._normalize(raw)
        X = normalized[_FEATURE_NAMES].values

        logger.info(
            "Fitting HMM", n_states=n_states, observations=len(X), features=len(_FEATURE_NAMES)
        )

        # Fit with multiple initializations
        best_model = None
        best_score = -np.inf
        last_error: Exception | None = None

        for i in range(10):
            try:
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type="diag",
                    n_iter=200,
                    random_state=42 + i,
                )
                model.fit(X)
                score = model.score(X)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                last_error = e
                logger.debug("HMM init failed", attempt=i, error=str(e))
                continue

        if best_model is None:
            logger.error("All HMM initializations failed", last_error=str(last_error))
            raise RuntimeError(f"All HMM initializations failed: {last_error}")

        self._model = best_model
        self._regime_labels = self._label_regimes(best_model, n_states)
        self._last_trained = date.today()

        # Store the raw + normalized data for predict_current
        self._raw_data = raw
        self._normalized_data = normalized

        logger.info(
            "HMM fitted",
            n_states=n_states,
            log_likelihood=round(best_score, 2),
            labels=self._regime_labels,
        )

        return {
            "n_observations": len(X),
            "n_states": n_states,
            "regime_labels": self._regime_labels,
            "log_likelihood": round(best_score, 2),
        }

    def predict_current(self) -> dict[str, Any]:
        """Get current regime assessment from the fitted model.

        Returns:
            Dict with regime, confidence, probabilities, and regime durations.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted — call fit() first")

        X = self._normalized_data[_FEATURE_NAMES].values
        posteriors = self._model.predict_proba(X)
        latest = posteriors[-1]

        if np.any(np.isnan(latest)):
            logger.warning("HMM produced NaN posteriors — model may be degenerate")
            return {"regime": "uncertain", "confidence": 0, "probabilities": {}}

        most_likely = int(np.argmax(latest))

        # Compute how long we've been in the current regime
        states = self._model.predict(X)
        duration = 0
        for s in reversed(states):
            if s == most_likely:
                duration += 1
            else:
                break

        return {
            "regime": self._regime_labels[most_likely],
            "confidence": round(float(latest[most_likely]), 3),
            "probabilities": {
                self._regime_labels[i]: round(float(p), 3) for i, p in enumerate(latest)
            },
            "duration_weeks": duration,
            "last_trained": str(self._last_trained) if self._last_trained else None,
            "data_date": str(self._raw_data.index[-1].date()) if len(self._raw_data) > 0 else None,
        }

    # ── Data Fetching ──────────────────────────────────────────────

    async def _fetch_data(self, start: str) -> Any:
        """Fetch and align all features into a weekly DataFrame."""
        async with httpx.AsyncClient(timeout=30) as client:
            vix_data, hy_data, yc_data = await asyncio.gather(
                self._fetch_fred(client, "VIXCLS", start),
                self._fetch_fred(client, "BAMLH0A0HYM2", start),
                self._fetch_fred(client, "T10Y2Y", start),
            )

        # yfinance (daily, we'll resample)
        spy_df = yf.download("SPY", start=start, progress=False)
        uup_df = yf.download("UUP", start=start, progress=False)

        if spy_df.empty:
            raise ValueError("No SPY data returned from yfinance")
        if uup_df.empty:
            logger.warning("No UUP data from yfinance — dollar feature will be zero-filled")

        # Handle MultiIndex columns from yfinance
        spy_close: pd.Series = pd.Series(spy_df["Close"].squeeze(), dtype=float)
        uup_close: pd.Series = pd.Series(uup_df["Close"].squeeze(), dtype=float)

        # Compute daily features
        spy_ret = pd.Series(np.log(spy_close / spy_close.shift(1)), index=spy_close.index)
        spy_vol = spy_ret.rolling(20).std()
        uup_ret = pd.Series(np.log(uup_close / uup_close.shift(1)), index=uup_close.index)

        # Build daily DataFrame
        daily = pd.DataFrame(
            {
                "spy_ret": spy_ret,
                "spy_vol": spy_vol,
                "uup_ret": uup_ret,
            }
        )

        # Add FRED series (already datetime-indexed)
        for name, data in [("vix", vix_data), ("hy_oas", hy_data), ("t10y2y", yc_data)]:
            if data:
                series = pd.Series(
                    {pd.Timestamp(d): v for d, v in data},
                    dtype=float,
                    name=name,
                )
                daily[name] = series.reindex(daily.index, method="ffill")

        # Resample to weekly (Friday close)
        weekly = pd.DataFrame()
        weekly["spy_ret"] = daily["spy_ret"].resample("W-FRI").sum()
        weekly["spy_vol"] = daily["spy_vol"].resample("W-FRI").last()
        weekly["vix"] = daily["vix"].resample("W-FRI").last()
        weekly["hy_oas"] = daily["hy_oas"].resample("W-FRI").last()
        weekly["t10y2y"] = daily["t10y2y"].resample("W-FRI").last()
        weekly["uup_ret"] = daily["uup_ret"].resample("W-FRI").sum()

        weekly = weekly.ffill().dropna()
        logger.info("Regime data fetched", weeks=len(weekly), start=str(weekly.index[0].date()))
        return weekly

    async def _fetch_fred(
        self, client: httpx.AsyncClient, series_id: str, start: str
    ) -> list[tuple[str, float]]:
        """Fetch a FRED series as daily observations."""
        try:
            resp = await client.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "api_key": self._fred_api_key,
                    "series_id": series_id,
                    "observation_start": start,
                    "sort_order": "asc",
                    "file_type": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            result = []
            for obs in data.get("observations", []):
                val = obs.get("value", ".")
                if val != ".":
                    try:
                        result.append((obs["date"], float(val)))
                    except (ValueError, TypeError):
                        pass
            return result
        except Exception:
            logger.warning(
                "FRED fetch failed for regime detector", series_id=series_id, exc_info=True
            )
            return []

    # ── Feature Preprocessing ──────────────────────────────────────

    def _normalize(self, df: Any) -> Any:
        """Expanding z-score normalization (no lookahead)."""
        normalized = pd.DataFrame(index=df.index)
        for col in _FEATURE_NAMES:
            if col not in df.columns:
                normalized[col] = 0.0
                continue
            expanding_mean = df[col].expanding(min_periods=52).mean()
            expanding_std = df[col].expanding(min_periods=52).std()
            normalized[col] = (df[col] - expanding_mean) / expanding_std.replace(0, 1)

        return normalized.dropna()

    # ── Regime Labeling ────────────────────────────────────────────

    @staticmethod
    def _label_regimes(model: Any, n_states: int) -> dict[int, str]:
        """Auto-label regimes by mean equity return (spy_ret is feature 0)."""
        means = model.means_[:, 0]  # spy_ret column
        sorted_indices = list(np.argsort(means))

        if n_states == 2:
            return {sorted_indices[0]: "risk_off", sorted_indices[1]: "risk_on"}
        elif n_states == 3:
            return {
                sorted_indices[0]: "risk_off",
                sorted_indices[1]: "transitioning",
                sorted_indices[2]: "risk_on",
            }
        else:
            labels = {}
            for i, idx in enumerate(sorted_indices):
                labels[idx] = f"regime_{i}"
            return labels
