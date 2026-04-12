# Strategy 4: Macro Sector Rotation

## Overview

Monthly-frequency sector rotation strategy. Instead of picking individual stocks, trade 11 sector ETFs based on macro regime signals. GBM predicts which sectors will outperform next month. Go long the top 3 sectors, short the bottom 3.

**Frequency**: Monthly rebalance
**Universe**: 11 SPDR Sector ETFs (XLK, XLF, XLE, XLV, XLI, XLC, XLY, XLP, XLRE, XLB, XLU)
**Data required**: Daily OHLCV for sector ETFs, daily macro indicators (yield curve, credit spreads, USD, oil, VIX)
**Holding period**: 1 month (rolling)

---

## The Edge

Different sectors respond to macro conditions differently. When interest rates rise, financials outperform and utilities underperform. When oil spikes, energy leads and consumer discretionary lags. When the yield curve inverts, defensives (utilities, staples, healthcare) outperform cyclicals.

These relationships are well-known — the edge comes from:
1. Combining many macro signals simultaneously (a human tracks 2-3, the GBM tracks 20+)
2. Detecting regime shifts faster than monthly macro reports
3. Capturing nonlinear interactions (e.g., rising rates + falling oil = different from rising rates + rising oil)

---

## Data Required

### Sector ETFs (the thing you're trading)

| ETF | Sector |
|---|---|
| XLK | Technology |
| XLF | Financials |
| XLE | Energy |
| XLV | Healthcare |
| XLI | Industrials |
| XLC | Communication Services |
| XLY | Consumer Discretionary |
| XLP | Consumer Staples |
| XLRE | Real Estate |
| XLB | Materials |
| XLU | Utilities |

**Data needed**: Daily OHLCV for each ETF.

### Macro Indicators (the features)

| Indicator | What It Captures | Source |
|---|---|---|
| **US 10Y Treasury yield** | Interest rate level | FRED (DGS10) |
| **2Y-10Y yield spread** | Yield curve slope — inversion predicts recession | FRED (T10Y2Y) |
| **US Investment Grade credit spread** (OAS) | Credit stress — widening = risk-off | FRED (BAMLC0A0CM) |
| **US High Yield credit spread** | Junk bond stress — more sensitive than IG | FRED (BAMLH0A0HYM2) |
| **DXY (US Dollar Index)** | Dollar strength — affects exporters, commodities, EM | Daily OHLCV |
| **WTI Crude Oil** | Energy input — affects energy, transports, consumer | Daily OHLCV |
| **Gold** | Safe haven proxy | Daily OHLCV |
| **VIX** | Implied volatility — fear gauge | Daily close |
| **VIX term structure** (VIX - VIX3M) | Contango vs backwardation — backwardation = stress | Daily |
| **SPY** | Broad market direction | Daily OHLCV |
| **Fed Funds Rate** | Monetary policy stance | FRED (FEDFUNDS, monthly) |
| **ISM Manufacturing PMI** | Economic activity | Monthly release |
| **Initial Jobless Claims** | Labor market health | Weekly release |

All of these are freely available. FRED data is free with an API key. Market data (DXY, oil, gold, VIX) comes from any standard daily data source.

---

## Features

Computed monthly (at month-end, before rebalancing):

### Macro Features

| Feature | Description |
|---|---|
| `yield_10y_level` | Current 10Y yield |
| `yield_10y_1m_change` | 1-month change in 10Y yield |
| `yield_curve_slope` | 2Y-10Y spread |
| `yield_curve_1m_change` | 1-month change in slope |
| `ig_spread_level` | IG credit spread (OAS) |
| `ig_spread_1m_change` | 1-month change |
| `hy_spread_level` | HY credit spread |
| `hy_spread_1m_change` | 1-month change |
| `dxy_ret_1m` | USD index 1-month return |
| `oil_ret_1m` | WTI crude 1-month return |
| `gold_ret_1m` | Gold 1-month return |
| `vix_level` | VIX close |
| `vix_term_structure` | VIX - VIX3M (negative = backwardation = stress) |
| `spy_ret_1m` | S&P 500 1-month return |
| `spy_ret_3m` | S&P 500 3-month return (trend) |
| `ism_pmi` | Latest ISM Manufacturing PMI reading |
| `claims_4wk_avg` | 4-week moving average of initial jobless claims |

### Per-Sector Features

| Feature | Description |
|---|---|
| `sector_ret_1m` | Sector ETF's 1-month return |
| `sector_ret_3m` | Sector ETF's 3-month return |
| `sector_ret_relative_1m` | Sector return minus SPY return (relative strength) |
| `sector_vol_1m` | Sector's 1-month realized volatility |
| `sector_corr_spy_3m` | 3-month rolling correlation with SPY (high = no diversification benefit) |

**Target variable**: Sector's next-month return relative to SPY (`sector_ret_next_1m - spy_ret_next_1m`)

---

## Signal Generation

### GBM's Job

For each of the 11 sectors, predict: **"Given current macro conditions and this sector's recent behavior, will it outperform or underperform the S&P 500 next month, and by how much?"**

The model learns relationships like:
- Rising rates + steepening curve → overweight Financials, underweight Utilities
- Widening credit spreads + rising VIX → overweight Staples/Healthcare, underweight Discretionary
- Falling USD + rising oil → overweight Energy/Materials

### Model

LightGBM (single model is fine — only 11 predictions per month, no need for stacking complexity). Train one model on all sector-months pooled together, with sector as a categorical feature.

---

## Entry Rules

### Monthly Rebalance (Last Trading Day of Month)

1. Compute all features using month-end data
2. GBM predicts next-month relative return for all 11 sectors
3. Rank sectors by predicted relative return

### Positions

- **Long top 3 sectors** — equal-weight (e.g., 33% each of the long leg)
- **Short bottom 3 sectors** — equal-weight
- **Ignore middle 5** — no edge, save on transaction costs

### Execution

- Submit MOC orders on the last trading day of the month
- Or trade at the open of the first trading day of the new month

---

## Exit Rules

| Condition | Action |
|---|---|
| Month ends | Close all positions, recompute, establish new positions |
| Mid-month macro shock (e.g., VIX spikes above 35) | Optional: reduce all positions by 50% |
| Any single position down > 8% | Stop-loss that sector's position |

Simple monthly cadence. No daily monitoring needed (though the macro shock override is prudent).

---

## Position Sizing

- Split portfolio into long leg and short leg (50/50 or 60/40 long-biased)
- Equal-weight within each leg (3 positions per leg)
- Example for $100K portfolio:
  - Long leg: 3 x $16.7K = $50K in top 3 sectors
  - Short leg: 3 x $16.7K = $50K in bottom 3 sectors
  - Net market exposure: ~0 (dollar neutral)
- Optional: tilt long-biased (60/40) if your regime model says Bull

---

## When It Works vs When It Fails

| Market Condition | Performance |
|---|---|
| Macro-driven rotation (rate cycles, oil shocks, recessions) | Best — clear sector winners and losers |
| Low dispersion (everything moves together) | Worst — no spread between top and bottom sectors |
| Gradual regime shifts | Good — model picks up on changing macro trends |
| Sudden black swan (overnight crash) | Monthly rebalance is too slow to react |

---

## Validation

- Combinatorial Purged Cross-Validation with **2-month purge + 1-month embargo** (monthly frequency = wider gaps)
- PBO check < 0.5
- Walk-forward: train on prior 5 years of monthly data, test on next 1 year, roll forward 1 year at a time
- **Small sample warning**: 11 sectors x 20 years = ~2,640 data points. This is small. Keep the model simple (few features, shallow trees) to avoid overfitting.

---

## Risks

- **Small sample size**: Monthly frequency means fewer data points to train on. Overfitting is the primary risk. Use shallow trees (max_depth=4), high min_child_samples, and aggressive regularization.
- **Low dispersion periods**: When macro conditions are benign, all sectors perform similarly. The long/short spread is near zero and transaction costs eat you alive.
- **Regime shift lag**: Monthly rebalance means you're always 1-30 days late to a regime shift. A mid-month crash hits you at full size.
- **Crowded macro trades**: "Overweight energy when oil rises" is not a secret. The edge is in combining many signals and detecting nonlinear interactions, not in any single relationship.
- **ETF tracking error**: Sector ETFs don't perfectly track their underlying sectors. Reconstitution, rebalancing, and fund flows introduce noise.

---

## Advantages Over Strategy 1-3

- **Fewest positions**: Only 6 ETFs at a time (3 long, 3 short)
- **Lowest turnover**: Monthly rebalance, ~30-50% monthly turnover
- **Simplest execution**: ETFs are extremely liquid, tight spreads, easy to borrow for shorting
- **Lowest monitoring burden**: Check once a month, set and forget
- **Uncorrelated**: Different alpha source (macro vs stock-specific) than Strategies 1-3
- **No single-stock risk**: ETF diversification protects against individual company blowups

---

## Production Pipeline

```
Monthly (last trading day):

  1. Pull macro data (FRED API, daily market data)
  2. Compute features for all 11 sectors
  3. GBM predicts next-month relative return per sector
  4. Rank sectors, select top 3 (long) and bottom 3 (short)
  5. Close prior month's positions, enter new positions via MOC orders

Quarterly:
  6. Retrain model (Optuna + CPCV)

Ongoing:
  7. Monitor for mid-month macro shocks (VIX spike → reduce exposure)
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data processing | Polars |
| Macro data | FRED API (free with key) |
| Modeling | LightGBM (single model, keep simple) |
| Hyperparameter tuning | Optuna (fewer trials needed — small dataset) |
| Validation | CPCV (2-month purge) |
| Backtesting | VectorBT |
| Explainability | SHAP |
