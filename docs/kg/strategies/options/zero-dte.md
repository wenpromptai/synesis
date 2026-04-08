---
up: []
related: ["[[iron-condor]]", "[[long-straddle]]", "[[butterfly-spread]]"]
created: 2026-04-07
type: strategy
category: options
subcategory: systematic
complexity: advanced
data_source: [yfinance]
tags: [options, systematic, 0DTE, high-gamma, intraday]
---

# Zero DTE (0DTE)

> [!abstract]
> Same-day expiration trades on SPX/SPY. Extreme [[gamma]], rapid [[time-decay|theta decay]], and amplified moves. High risk, requires precise timing and tiny position sizing.

## Core Mechanic

Options expiring today have near-zero extrinsic value and extreme [[gamma]]. ATM 0DTE options can swing from 0 to deep ITM in minutes. Three primary approaches:

1. **Credit spreads** — sell premium, collect accelerated theta
2. **Iron condors** — range bet with defined risk
3. **Directional** — buy calls/puts for leveraged directional exposure

## Why 0DTE Is Different

| Property | Standard (30-45 DTE) | 0DTE |
|----------|---------------------|------|
| [[gamma]] | Moderate | **Extreme** |
| [[theta]] | ~$X/day | **All remaining value** |
| [[delta]] sensitivity | Gradual | **Binary near ATM** |
| [[vega]] | Significant | **Near zero** |
| Time horizon | Weeks | **Hours** |

## Trade Construction

### Credit Spread (Most Common)

1. Sell 1 SPX put/call — **5-15 [[delta]]** (~0.5-1.5% OTM)
2. Buy 1 SPX put/call — **5 points further OTM** (wing)
3. Collect credit, target 50% profit within hours
4. Hard stop: close at 2x credit

### Iron Condor

1. Put spread: sell 10-delta put, buy 5-delta put
2. Call spread: sell 10-delta call, buy 5-delta call
3. Ultra-tight wings (5-10 points)
4. Close at 50% profit or 2 hours before close

| Parameter | Value |
|-----------|-------|
| Underlying | SPX (cash-settled, no assignment) or SPY |
| DTE | 0 (same day) |
| Short leg [[delta]] | 0.05-0.15 |
| Wing width | 5-10 points (SPX) |
| Target exit | 50% of credit within 2-3 hours |
| Hard stop | 2x credit |

### SPX vs SPY

| | SPX | SPY |
|---|---|---|
| Settlement | **Cash** (no assignment risk) | Physical |
| Tax treatment | **60/40** (60% long-term) | Short-term |
| Contract size | 100x index (~$500K notional) | 100x ETF (~$50K notional) |
| Liquidity | Very high on 0DTE | Very high |
| Expirations | Mon/Wed/Fri (+ daily) | Mon/Wed/Fri |

### Sizing

> [!danger] Position Sizing is Critical
> - **0.5-1% of portfolio** per 0DTE trade maximum
> - The extreme gamma means max loss is hit fast
> - Never trade 0DTE with money you can't lose today
> - Treat as high-frequency income, not a core strategy

## Pinning and [[max-pain]]

0DTE options with massive [[open-interest]] at round-number strikes create pinning effects. Check max-pain level for the day — center [[iron-condor]] or [[butterfly-spread]] around it.

> [!danger] Key Risk
> - **Extreme speed:** Can go from max profit to max loss in minutes
> - No time to adjust — by the time you react, the move may be over
> - Gap risk around macro announcements (FOMC, CPI, NFP)
> - Overtrading temptation — the dopamine loop is real

## Data Pipeline

> [!info] Synesis Data
> | Need | Source | Method |
> |------|--------|--------|
> | Today's expiration chain | yfinance | `get_options_chain(ticker, today_exp, greeks=True)` |
> | Spot price | yfinance | `get_quote(ticker)` |
> | [[max-pain]] | yfinance | Compute from [[open-interest]] in chain |

> [!warning] Data Limitation
> yfinance provides delayed chain data (15-20 min). Real 0DTE trading requires real-time data feeds. Use yfinance for analysis/screening, not live execution timing.

---
**Related strategies:** [[iron-condor]] | [[long-straddle]] | [[butterfly-spread]]
**Concepts:** [[gamma]] | [[theta]] | [[time-decay]] | [[max-pain]] | [[open-interest]] | [[delta]]
