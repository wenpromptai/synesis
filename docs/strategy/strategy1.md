# Strategy 1: Regime-Aware Momentum Ranking (Daily)

## Overview

Daily-frequency long/short equity strategy. Rank S&P 500 stocks by predicted 5-day forward return using a stacked gradient boosting ensemble. Scale position sizes based on HMM-detected market regime.

**Frequency**: Daily rebalance
**Universe**: Top 500 US stocks by market cap (recomputed monthly)
**Data required**: Daily OHLCV, daily VIX close, sector labels per ticker, market cap per ticker
**Holding period**: 5 trading days (rolling)

---

## Signal Generation

### Features (all computed from daily OHLCV)

| Feature | Description |
|---|---|
| `ret_5d` | 5-day trailing return |
| `ret_21d` | 21-day (1-month) trailing return |
| `ret_63d` | 63-day (3-month) trailing return |
| `vol_21d` | 21-day rolling standard deviation of daily returns |
| `vol_ratio` | Today's volume / 21-day average volume |
| `rsi_14` | 14-day RSI |
| `macd` | MACD line (12/26 EMA difference) |
| `sector` | GICS sector (categorical) |

**Target variable**: 5-day forward return (`close[t+5] / close[t] - 1`)

### Model: Stacked Gradient Boosting Ensemble

Three base learners, each with different inductive biases:

1. **LightGBM** — leaf-wise growth, catches sharp nonlinearities, fastest training
2. **CatBoost** — ordered boosting (built-in protection against target leakage on time-series), native categorical handling for sector, most resistant to overfitting on noisy return data
3. **XGBoost** — strong L1/L2 regularization, best SHAP integration for understanding what drives predictions

**Meta-learner**: Ridge regression on the 3 base model predictions. Ridge (not another tree) prevents overstacking overfitting. It learns which base model to trust under which conditions.

**Why stack all three**: Each GBM sees the data differently. LightGBM finds sharp local patterns. CatBoost resists overfitting. XGBoost provides stability. Typical improvement: 10-20% better Sharpe than any single model.

### Ranking

Each day, the stacked model predicts 5-day forward return for every stock in the universe. Stocks are ranked by predicted return within that day's cross-section.

### Universe Construction

The universe is the top 500 US equities by total market capitalization, recomputed monthly:

1. Pull all US common stocks via Massive `get_grouped_daily()` (one API call for all tickers on a given date)
2. Filter: common stock only (type = `CS`), active, USD-denominated
3. Fetch `market_cap` from Massive `get_ticker_overview()` for each candidate
4. Rank by market cap descending, take top 500
5. Apply a **buffer rule**: existing members stay unless they drop below rank 550; new entrants must reach rank 450. This prevents excessive churn from stocks hovering around the cutoff.

Recompute on the first trading day of each month. Between recomputes, the universe is fixed.

**Why market-cap-ranked instead of S&P 500**: Historical S&P 500 membership (adds/removes) is not freely available, making survivorship-bias-free backtesting difficult. A market-cap-ranked universe is self-describing — you can reconstruct it at any historical date from price + shares outstanding data alone.

---

## Regime Detection

A 3-state Hidden Markov Model (HMM) runs on two inputs:
- SPY daily returns
- VIX daily close

The HMM identifies three latent market regimes:

| State | Characteristics | Position Scale |
|---|---|---|
| **Bull** | Low volatility, positive drift | 100% |
| **Bear** | High volatility, negative drift | 50% |
| **Crisis** | Extreme volatility, sharp drawdown | 0% (flat) |

The HMM is fit on historical data and updated monthly. Each trading day, the model classifies the current regime based on the most recent observations.

**Why HMM over simple thresholds**: The HMM detects *latent* state transitions — a day can look calm on the surface, but the joint distribution of returns + VIX may already be shifting toward bear. Threshold rules ("VIX > 25 = bear") miss these early transitions.

---

## Entry / Exit Rules

### When to Buy (Long)

- Stock ranks in the **top 20** of the universe by predicted 5-day return
- Current regime is Bull or Bear (not Crisis)
- Equal-weight across the 20 positions, scaled by regime factor

### When to Sell (Short)

- Stock ranks in the **bottom 20** of the universe by predicted 5-day return
- Same regime scaling applies

### Position Sizing

```
position_weight = (1 / 20) * regime_scale

regime_scale:
  Bull  = 1.0  (full allocation)
  Bear  = 0.5  (half allocation)
  Crisis = 0.0 (all cash)
```

For a $100K portfolio in Bull regime: each of 40 positions (20 long + 20 short) gets $2,500.

### Rebalance

- Daily at market close (or use MOC orders)
- 5-day rolling: positions opened today are closed 5 trading days later, unless still ranked in top/bottom 20 (in which case they roll)
- If regime shifts to Crisis mid-holding: close all positions immediately

### Stop Loss

- Per-position: close if unrealized loss exceeds **3%**
- Portfolio: go flat if portfolio drawdown exceeds **10%** from peak, wait for regime to flip back to Bull before re-entering

---

## Validation

### Combinatorial Purged Cross-Validation (CPCV)

Standard k-fold CV leaks future data in time series. CPCV is the gold standard:

- **Purge**: Remove training samples within 5 days of any test sample (prevents feature leakage from overlapping return windows)
- **Embargo**: Add a 5-day gap after each test fold (prevents autocorrelation contamination)
- **Combinatorial**: Test on all possible train/test path combinations, not just one walk-forward path

This produces the **Probability of Backtest Overfitting (PBO)** — the chance your strategy is just noise. **PBO > 0.5 = do not trade.**

### Walk-Forward (Secondary)

Rolling 2-year train / 3-month test windows, advancing 1 month at a time. Used to confirm CPCV results and simulate realistic retraining cadence.

---

## Hyperparameter Tuning

Use Optuna with TPE (Tree-structured Parzen Estimator) sampler. Optimize for **mean Sharpe ratio across CPCV folds**.

Key hyperparameters to tune per base learner:

| Parameter | Search Range |
|---|---|
| `n_estimators` | 200 – 1000 |
| `num_leaves` (LGB) / `max_depth` (XGB, Cat) | 15 – 127 / 4 – 8 |
| `learning_rate` | 0.01 – 0.1 (log scale) |
| `subsample` | 0.6 – 1.0 |
| `min_child_samples` | 20 – 200 |

200 Optuna trials typically beats 10,000 grid search points because TPE learns which regions of hyperparameter space are promising.

---

## Production Pipeline

```
Daily (after market close):

  1. Pull daily OHLCV + VIX close (universe from latest monthly recompute)
  2. Compute features              (Polars + polars_ta, < 1 second)
  3. Detect regime                  (HMM on SPY + VIX, < 0.1 second)

Monthly (first trading day):
  0. Recompute universe            (top 500 by market cap with buffer rule)
  4. Predict 5-day forward returns  (Stacked LGB + XGB + Cat, < 0.5 second)
  5. Rank stocks, apply regime sizing
  6. Generate orders (new entries, exits, stop-loss triggers)
  7. Execute via broker API

Weekly:
  8. Drift monitoring              (Evidently — check feature distribution shift)
     If drift detected → trigger retrain

Monthly:
  9. Full retrain                  (Optuna HPO + CPCV, ~30 min on CPU)
  10. Update HMM regime model
```

Total daily inference: **under 2 seconds**, no GPU needed.

---

## Monitoring & Kill Switches

| Metric | Threshold | Action |
|---|---|---|
| Rolling 30-day Sharpe | < 0 for 2 consecutive weeks | Pause trading |
| Portfolio drawdown | > 10% from peak | Go flat |
| Feature drift (KL divergence) | Significant shift detected | Retrain immediately |
| Model hit rate (% correct direction) | < 48% over 30 days | Review / pause |

---

## Costs & Assumptions

- **Commission**: ~10bps per trade (discount broker)
- **Slippage**: ~10bps estimate for S&P 500 liquid names
- **Turnover**: ~40-60% daily (high — the short holding period means frequent rotation)
- **Shorting**: Requires margin account; borrow costs for hard-to-borrow names can eat alpha
- **Tax**: High turnover = short-term capital gains; more suitable for tax-advantaged accounts or entities

---

## Risks

- **Regime model lag**: HMM detects regimes with a delay; the first days of a crash may be traded at full size
- **Universe drift**: Market-cap-ranked universe may differ from S&P 500 (includes some non-S&P large caps, excludes some smaller S&P members). This is a feature, not a bug — it avoids index committee subjectivity
- **Crowded factors**: Momentum is well-known; alpha decays as more capital chases it
- **Model decay**: Financial relationships are non-stationary; monthly retraining helps but doesn't eliminate this
- **Overfitting**: Despite CPCV, the strategy has many degrees of freedom (3 models x hyperparameters x regime states). PBO check is critical before going live
- **Execution gap**: Paper backtest assumes close-to-close fills; real execution at market close may differ

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data processing | Polars (lazy evaluation) |
| Feature engineering | polars_ta / mintalib |
| Regime detection | pomegranate (HMM) + ruptures (change-point) |
| Modeling | LightGBM + CatBoost + XGBoost, Ridge meta-learner |
| Hyperparameter tuning | Optuna (TPE sampler) |
| Validation | CPCV (purge + embargo) |
| Backtesting | VectorBT |
| Drift monitoring | Evidently AI |
| Explainability | SHAP (TreeSHAP) |
