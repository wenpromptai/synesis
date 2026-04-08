---
up: []
related: ["[[volatility-risk-premium]]", "[[long-straddle]]", "[[short-strangle]]"]
created: 2026-04-07
type: strategy
category: options
subcategory: systematic
complexity: advanced
data_source: [yfinance]
tags: [options, systematic, volatility, arbitrage]
---

# Volatility Arbitrage

> [!abstract]
> Trade the spread between [[implied-volatility|IV]] and [[realized-volatility|RV]] directly. When IV >> RV, sell options (overpriced). When RV >> IV, buy options (underpriced). Delta-hedge to isolate the vol component.

## Core Mechanic

Unlike [[volatility-risk-premium]] (which systematically sells), vol arb goes both directions based on the IV-RV gap:

| Condition | IV vs RV | Action | Vehicle |
|-----------|----------|--------|---------|
| IV >> RV | Overpriced options | Sell straddle/strangle | [[short-strangle]] |
| RV >> IV | Underpriced options | Buy straddle/strangle | [[long-straddle]] |
| IV ≈ RV | Fair priced | No trade | — |

**Key difference from VRP:** Vol arb delta-hedges continuously to isolate the pure volatility bet, removing directional risk.

## Delta-Hedging

1. Sell (or buy) ATM straddle
2. Immediately hedge [[delta]] to zero with stock
3. Re-hedge delta at regular intervals (daily or when delta exceeds threshold)
4. P&L = premium collected - realized gamma costs (if short) OR realized gamma gains - premium paid (if long)

$$
P\&L_{vol\ arb} \approx \frac{1}{2} \Gamma S^2 (\sigma_{realized}^2 - \sigma_{implied}^2) \cdot T
$$

If $\sigma_{realized} < \sigma_{implied}$: short vol profits. If $\sigma_{realized} > \sigma_{implied}$: long vol profits.

## Trade Construction (Short Vol Arb)

1. Compute 30-day [[realized-volatility|RV]] from price history
2. Get current ATM [[implied-volatility|IV]] from chain
3. If IV > RV by meaningful margin (e.g., IV/RV > 1.2): sell ATM straddle
4. Delta-hedge with stock to neutralize direction
5. Re-hedge daily
6. Close at 50% profit or at expiry

| Parameter | Value |
|-----------|-------|
| IV/RV threshold | > 1.2 (sell) or < 0.8 (buy) |
| Hedge frequency | Daily or when |delta| > 0.10 |
| DTE | 30-45 days |
| Target exit | 50% of premium |

> [!warning] Complexity Warning
> True vol arb requires continuous delta-hedging, which means frequent trading and transaction costs. The edge is real but thin. This is an institutional strategy that retail traders should approach cautiously.

> [!danger] Key Risk
> - Transaction costs from frequent hedging can eat the edge
> - Discrete hedging (daily) vs continuous hedging creates slippage
> - Gamma risk: short gamma positions lose from large moves between hedges
> - Requires discipline and systematic execution

## Data Pipeline

> [!info] Synesis Data
> | Need | Source | Method |
> |------|--------|--------|
> | [[realized-volatility]] (30d) | yfinance | `get_options_snapshot()` or compute from `get_history()` |
> | [[implied-volatility]] (ATM) | yfinance | `get_options_chain(ticker, exp)` → ATM IV |
> | [[delta]] for hedging | yfinance | `get_options_chain(ticker, exp, greeks=True)` |
> | Stock price (for hedging) | yfinance | `get_quote(ticker)` |
> | Price history (RV calc) | yfinance | `get_history(ticker, period="3mo")` |

---
**Related strategies:** [[volatility-risk-premium]] | [[long-straddle]] | [[short-strangle]]
**Concepts:** [[implied-volatility]] | [[realized-volatility]] | [[gamma]] | [[delta]] | [[vega]]
