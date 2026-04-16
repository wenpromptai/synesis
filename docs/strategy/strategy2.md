# Strategy 2: Mean Reversion (Fade Extremes)

## Overview

Daily-frequency mean reversion strategy. Identify stocks that have moved too far too fast, bet on them snapping back. Buy the most oversold, short the most overbought. GBM predicts reversion probability and magnitude.

**Frequency**: Daily rebalance
**Universe**: S&P 500
**Data required**: Daily OHLCV, daily VIX close, sector labels per ticker
**Holding period**: 1-5 trading days (until reversion or stop-loss)

---

## The Edge

Stocks that deviate significantly from their recent mean tend to revert — especially in range-bound or choppy markets. Overreaction to news, forced selling (margin calls, index rebalancing), and liquidity shocks create temporary mispricings that correct within days.

This strategy naturally hedges Strategy 1 (Momentum Ranking). Momentum works in trending markets; mean reversion works when momentum breaks down.

---

## Data Required

| Data | Granularity | Purpose |
|---|---|---|
| **OHLCV** (Open, High, Low, Close, Volume) | Daily per stock | Features + signal |
| **VIX close** | Daily | Regime context — mean reversion works differently in high vs low vol |
| **Sector label** | Static per ticker | Categorical feature — reversion speed varies by sector |

No intraday, tick, or options data needed.

---

## Features

| Feature | Description |
|---|---|
| `zscore_5d` | Z-score of 5-day return vs 60-day rolling mean/std — the primary "how extreme is this move" signal |
| `zscore_10d` | Same over 10-day window — catches slower dislocations |
| `dist_from_20sma` | % distance from 20-day simple moving average |
| `dist_from_50sma` | % distance from 50-day SMA |
| `bb_position` | Where price sits within Bollinger Bands (0 = lower band, 1 = upper band) |
| `rsi_5` | 5-day RSI — short-term overbought/oversold |
| `vol_21d` | 21-day realized volatility — high-vol stocks revert differently |
| `vol_ratio` | Today's volume / 21-day avg volume — spike volume on the extreme move suggests overreaction |
| `vix` | Daily VIX level — regime context |
| `sector` | GICS sector (categorical) — tech reverts faster than utilities |
| `historical_reversion_rate` | Over the past year, when this stock hit a similar z-score, what % of the time did it revert within 5 days? |

**Target variable**: 5-day forward return from the extreme point (`close[t+5] / close[t] - 1`)

---

## Signal Generation

### GBM's Job

Predict: **"Given how extreme this stock's move is + its historical reversion tendency + current vol regime, what's the expected 5-day return?"**

The model is trained on historical instances where stocks hit z-score extremes. It learns which extremes are likely to revert (and by how much) vs which are the start of a new trend.

### Model

Same stacking approach as Strategy 1 (LightGBM + CatBoost + XGBoost with Ridge meta-learner), or single LightGBM for simplicity.

---

## Entry Rules

### When to Buy (Long)

- Stock's `zscore_5d` < -2.0 (moved more than 2 standard deviations below its recent mean)
- GBM predicts positive 5-day forward return with high confidence
- VIX is not in crisis regime (VIX > 35 = stand aside — extreme moves during crashes often keep going)
- Buy at **market close** (MOC order)

### When to Sell (Short)

- Stock's `zscore_5d` > +2.0 (moved more than 2 standard deviations above)
- GBM predicts negative 5-day forward return with high confidence
- Same VIX filter
- Short at **market close**

### Filtering

Not every extreme is tradeable. The GBM filters out:
- Extremes driven by genuine fundamental shifts (earnings, M&A, FDA decisions) — these tend to NOT revert
- Stocks with low historical reversion rates
- Illiquid names with wide spreads

---

## Exit Rules

| Condition | Action |
|---|---|
| Price reverts to 20-day SMA | Close position (target reached) |
| 5 trading days elapsed | Close position (time stop) |
| Unrealized loss > 3% | Close position (stop-loss) |
| Z-score extends further (e.g., -2.0 → -3.5) | Optional: add to position OR stop out depending on GBM re-prediction |

The key discipline: **don't hold hoping for a bigger reversion**. Take the mean and move on.

---

## Position Sizing

- Equal-weight across positions
- Max 15-20 concurrent positions
- Scale by conviction: GBM predicted return magnitude determines weight (bigger predicted reversion = larger position, capped at 2x base weight)
- Regime scaling: reduce size by 50% when VIX > 25

---

## When It Works vs When It Fails

| Market Condition | Performance |
|---|---|
| Range-bound / choppy | Best — lots of overextensions that revert |
| Low volatility grind | Decent — fewer signals but high hit rate |
| Strong trend (up or down) | Worst — you're fading a trend that keeps going |
| Crash / crisis | Dangerous — extremes keep getting more extreme |

This is why regime detection matters. In trending markets, reduce size or turn off. In choppy markets, increase size.

---

## Validation

Same as Strategy 1:
- Combinatorial Purged Cross-Validation (CPCV) with 5-day purge + 5-day embargo
- PBO (Probability of Backtest Overfitting) must be < 0.5
- Walk-forward confirmation

---

## Risks

- **Catching falling knives**: The biggest risk. A stock down 3 standard deviations might go down 5 more (think: fraud, bankruptcy, sector collapse). Stop-losses are non-negotiable.
- **Crowded trade**: Mean reversion is well-known. During vol spikes, many quants pile into the same oversold names simultaneously, then all exit together.
- **Regime mismatch**: Running this strategy in a strong trend market will bleed money. Regime detection is critical, not optional.
- **Correlation spikes**: In a selloff, all "oversold" stocks are correlated — your 20 "diversified" long positions are really one big bet on a market bounce.

---

## Production Pipeline

```
Daily (after market close):

  1. Pull daily OHLCV + VIX
  2. Compute z-scores and features           (Polars + polars_ta)
  3. Filter for stocks with |zscore_5d| > 2.0
  4. GBM predicts 5-day forward return       (LightGBM or stacked ensemble)
  5. Rank by predicted reversion magnitude
  6. Enter top signals, exit reverted/stopped positions
  7. Execute via MOC orders

Weekly:
  8. Drift monitoring (Evidently)

Monthly:
  9. Retrain model (Optuna + CPCV)
```

---

## Tech Stack

Same as Strategy 1:

| Layer | Tool |
|---|---|
| Data processing | Polars |
| Feature engineering | polars_ta / mintalib |
| Modeling | LightGBM (or stacked LGB + XGB + Cat) |
| Hyperparameter tuning | Optuna |
| Validation | CPCV |
| Backtesting | VectorBT |
| Drift monitoring | Evidently AI |
| Explainability | SHAP |
