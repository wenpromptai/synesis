# Strategy 3: Post-Earnings Announcement Drift (PEAD)

## Overview

Event-driven strategy that trades the persistent drift in stock prices following earnings announcements. The market underreacts to earnings surprises on day 1, and the stock continues drifting in the surprise direction for 5-20 days. GBM predicts drift direction and magnitude.

**Frequency**: Event-driven (trade only around earnings dates)
**Universe**: S&P 500
**Data required**: Daily OHLCV, earnings calendar, analyst consensus EPS estimates
**Holding period**: 5 trading days post-earnings

---

## The Edge

Post-Earnings Announcement Drift (PEAD) is one of the oldest and most persistent anomalies in finance — documented for 40+ years and still present. After a company reports earnings that beat or miss expectations, the stock keeps drifting in the surprise direction for days to weeks.

Why it persists:
- Institutional investors can't rebalance portfolios instantly
- Retail investors underreact to nuance in earnings reports
- Analyst estimate revisions lag behind actual results
- Information diffusion takes time across the market

---

## Data Required

| Data | Granularity | Purpose |
|---|---|---|
| **OHLCV** (Open, High, Low, Close, Volume) | Daily per stock | Price features + gap measurement |
| **Earnings calendar** | Per company, quarterly | Know when each company reports |
| **Analyst consensus EPS estimate** | Pre-earnings snapshot | Compute surprise = actual - expected |
| **Actual reported EPS** | Post-earnings | The other half of the surprise calculation |

Optional (strengthens signal):
| Data | Purpose |
|---|---|
| **Analyst revision history** (30 days pre-earnings) | Revision momentum feature |
| **Historical earnings surprises** (prior 4-8 quarters) | Surprise streak feature |

No intraday, tick, or options data needed.

---

## Features

All features computed **before or on the earnings day**, so the model can predict the subsequent drift.

### Pre-Earnings Features (known before the report)

| Feature | Description |
|---|---|
| `surprise_streak` | How many consecutive quarters the company beat or missed estimates (e.g., +4 = beat 4 in a row) |
| `revision_momentum_30d` | Net direction of analyst estimate revisions in the 30 days before earnings (positive = analysts raising estimates) |
| `pre_earnings_ret_5d` | Stock's 5-day return going into earnings — run-up may front-run the surprise |
| `pre_earnings_vol_compression` | Ratio of 5-day realized vol / 30-day realized vol before earnings — low ratio = coiled spring, bigger move expected |
| `sector_beat_rate` | What % of companies in the same sector that already reported this cycle beat estimates |
| `sector` | GICS sector (categorical) |

### Post-Earnings Features (known after the report, before you trade)

| Feature | Description |
|---|---|
| `eps_surprise_pct` | (Actual EPS - Consensus EPS) / |Consensus EPS| — magnitude of the surprise |
| `gap_direction` | Did the stock gap up or down on the first post-earnings trading day? (+1 / -1) |
| `gap_magnitude` | Size of the gap (%) — bigger gap = stronger initial reaction |
| `day1_volume_ratio` | Earnings day volume / 21-day average volume — high ratio = more conviction in the move |
| `day1_range` | (High - Low) / Close on earnings day — wide range = uncertainty, narrow = consensus |

---

## Signal Generation

### GBM's Job

Predict: **"Given the earnings surprise and these pre/post features, what's the 5-day drift after day 1?"**

The model is NOT predicting earnings. It's predicting **how the market will continue to react** after the initial gap. The surprise has already happened — the question is whether the drift continues.

### Model

LightGBM (or stacked ensemble). Trained on all historical earnings events in the universe with their subsequent 5-day returns.

---

## Entry Rules

### Timeline for Each Earnings Event

```
Day 0:  Company reports earnings (before market open or after close)
Day 1:  First full trading day post-earnings — market digests the news
        → At close of Day 1: compute all features, run GBM prediction
        → If signal is strong: enter position at Day 1 close (MOC order)
Day 2-6: Hold position (the drift window)
Day 6:  Exit at close (MOC order)
```

You wait until end of Day 1 because:
- You need the actual gap and day 1 price action as features
- The initial reaction is noisy — by close of Day 1 you have a clearer picture

### When to Buy (Long)

- Company beat earnings expectations (positive `eps_surprise_pct`)
- Stock gapped up on Day 1 (confirming the market views the beat positively)
- GBM predicts positive 5-day drift with high confidence
- Buy at **close of Day 1** (MOC order)

### When to Sell (Short)

- Company missed earnings expectations
- Stock gapped down on Day 1
- GBM predicts negative 5-day drift
- Short at **close of Day 1**

### Filtering — When NOT to Trade

- Surprise is tiny (|eps_surprise_pct| < 1%) — no edge, noise dominates
- Gap contradicts surprise (beat earnings but gapped down, or vice versa) — something else is going on, stand aside
- GBM confidence is low (predicted return near zero)

---

## Exit Rules

| Condition | Action |
|---|---|
| 5 trading days elapsed (close of Day 6) | Close position — drift window is over |
| Unrealized loss > 3% | Close position (stop-loss) |
| Predicted drift fully realized early | Optional: take profit if stock has already moved the predicted amount |

Fixed 5-day holding period keeps it mechanical. Don't hold through the next earnings hoping for more.

---

## Position Sizing

- 5% of portfolio per position (max)
- Max 5-6 concurrent positions (earnings cluster within seasons)
- Scale by surprise magnitude: bigger surprise % = larger position (capped at 2x base weight)
- During off-season (few earnings): don't force trades. Capital sits idle and that's fine.

---

## The Earnings Calendar

Earnings are seasonal — the bulk of S&P 500 companies report in 4 windows:

| Season | Rough Dates | Trades/Week |
|---|---|---|
| Q4 earnings | Mid-Jan to mid-Feb | 6-10 |
| Q1 earnings | Mid-Apr to mid-May | 6-10 |
| Q2 earnings | Mid-Jul to mid-Aug | 6-10 |
| Q3 earnings | Mid-Oct to mid-Nov | 6-10 |
| Off-season | Between windows | 0-2 |

This strategy is **event-driven, not always-on**. You're active ~16 weeks per year, idle ~36 weeks. This is a feature, not a bug — it means the strategy is uncorrelated to strategies that trade every day (like Strategy 1 and 2).

---

## Validation

- Combinatorial Purged Cross-Validation (CPCV) with **10-day purge + 5-day embargo** (longer purge because earnings are quarterly — you need to ensure no information leakage between quarterly events for the same company)
- PBO (Probability of Backtest Overfitting) must be < 0.5
- Walk-forward by earnings season (train on prior 8 quarters, test on next 2 quarters)

---

## Risks

- **Fading PEAD**: Some studies suggest PEAD has weakened in recent years as more quants trade it. Magnitude may be smaller than historical averages.
- **Earnings season concentration**: All your trades cluster in 4 windows. One bad earnings season (e.g., macro shock during reporting) can dominate annual returns.
- **Data quality**: Consensus EPS estimates vary by provider. Stale or inaccurate consensus = wrong surprise calculation = bad signal.
- **After-hours moves**: Some stocks make their entire post-earnings move in after-hours/pre-market. By the time you enter at Day 1 close, the drift may already be priced in for the fastest-reacting names.
- **Guidance matters more than EPS**: A company can beat EPS but guide down, causing the stock to drop. The gap_direction feature captures this, but the model needs enough history to learn it.

---

## Production Pipeline

```
Daily (during earnings season):

  1. Check earnings calendar — who reported today?
  2. For companies that reported:
     a. Pull actual EPS vs consensus
     b. Wait for Day 1 close
     c. Compute all features (gap, volume, surprise %)
     d. GBM predicts 5-day drift
     e. Enter positions via MOC orders
  3. For existing positions:
     a. Check stop-loss (3%)
     b. Close positions at Day 5

Off-season:
  4. Retrain model on latest earnings data (Optuna + CPCV)
  5. Update analyst consensus data
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data processing | Polars |
| Earnings calendar | NASDAQ API / your existing provider |
| Analyst estimates | Financial data provider (e.g., Alpha Vantage, FMP, or scraped) |
| Modeling | LightGBM (or stacked ensemble) |
| Hyperparameter tuning | Optuna |
| Validation | CPCV (10-day purge) |
| Backtesting | VectorBT |
| Explainability | SHAP |
