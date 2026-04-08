---
up: []
related: ["[[implied-volatility]]", "[[iv-rank-strategy-selection]]", "[[vega]]"]
created: 2026-04-07
type: concept
tags: [options, volatility, systematic]
aliases: [IV rank, IV percentile, IVR]
---

# IV Rank

> [!info] Definition
> A measure of where current [[implied-volatility|IV]] sits relative to its historical range, normalizing IV across different underlyings. ^definition

## Formulas

**IV Rank** (range-based):
$$
IVR = \frac{IV_{current} - IV_{52w\ low}}{IV_{52w\ high} - IV_{52w\ low}} \times 100
$$

**IV Percentile** (time-based):
$$
IVP = \frac{\text{Number of days IV was below current level}}{252} \times 100
$$

## Key Properties

1. **Normalizes across tickers:** NVDA IV of 40% might be low (IVR 20%), while XLU IV of 20% might be high (IVR 80%)
2. **Range: 0-100%** for both measures
3. **IVR vs IVP:** IVR is sensitive to outliers (one spike distorts the range); IVP is more robust
4. **Mean-reverting:** Extreme IVR readings tend to revert — this is the edge for premium sellers

## Strategy Selection Matrix

| IV Rank | Premium | Strategy Type | Examples |
|---------|---------|--------------|----------|
| > 50% | Rich | **Sell premium** | [[iron-condor]], [[short-strangle]], [[covered-call]] |
| 30-50% | Fair | **Directional spreads** | [[bull-call-spread]], [[bear-put-spread]] |
| < 30% | Cheap | **Buy premium** | [[long-straddle]], [[long-strangle]], [[calendar-spread]] |

## In Practice

- **[[iv-rank-strategy-selection]]:** The full framework mapping IVR to optimal strategy
- **[[covered-call]]:** Only sell calls when IVR > 50% — otherwise premium isn't worth the upside cap
- **[[iron-condor]]:** Entry filter: IVR > 50-60% for sufficient credit
- **[[long-straddle]]:** Buy when IVR < 30% and a catalyst is expected

## Data Source

> [!info] Synesis Pipeline
> Compute from yfinance: get 52-week chain IV history via repeated `get_options_chain()` snapshots, or approximate from VIX history for index strategies via `get_history("^VIX", period="1y")`.

---
**See also:** [[implied-volatility]] | [[iv-rank-strategy-selection]] | [[regime-options-matrix]] | [[vega]]
