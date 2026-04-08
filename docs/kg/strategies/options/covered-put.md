---
up: []
related: ["[[covered-call]]", "[[bear-put-spread]]", "[[short-strangle]]"]
created: 2026-04-07
type: strategy
category: options
subcategory: income
complexity: simple
data_source: [yfinance]
tags: [options, income, theta, bearish]
---

# Covered Put

> [!abstract]
> Short stock + short OTM put. Mirror of [[covered-call]] for bearish bias. Collect premium, cap downside profit.

## Core Mechanic

Short the stock, then sell a put below current price. Premium income offsets the risk of the stock rising. Profit is capped at the put strike minus short sale price plus premium.

```
P&L
 |─────────  max loss = unlimited (stock rises)
 |          \
 |           \
─┼────────────\──────────────── Price →
 |             \
 |              ────────────── capped gain = (short price - K) + premium
 ↓        K (short put)   short price
```

**Max profit:** (Short price - strike) + premium
**Max loss:** Unlimited (stock rises)
**Breakeven:** Short price + premium received

## Greeks Profile

| Greek | Exposure | Meaning |
|-------|----------|---------|
| [[delta]] | Net negative (~-0.65 to -0.75) | Short stock delta plus short put delta |
| [[gamma]] | Net negative | Short put gamma; large downside moves cap profit |
| [[theta]] | Net positive | Short put decays in your favor |
| [[vega]] | Net negative | IV increase hurts |

## Trade Construction

1. Short the stock
2. Sell 1 put per 100 shares short — **30-45 DTE, [[delta]] -0.25 to -0.35**
3. Close at 50% of premium
4. Roll down/out if stock drops near short put strike

| Parameter | Value |
|-----------|-------|
| DTE | 30-45 days |
| Short put [[delta]] | -0.25 to -0.35 |
| Target exit | 50% of premium |

> [!danger] Key Risk
> - **Unlimited upside risk** from short stock position
> - Less common than [[covered-call]] — short selling has margin requirements and borrow costs
> - Short squeeze risk in heavily shorted names

## Data Pipeline

> [!info] Synesis Data
> | Need | Source | Method |
> |------|--------|--------|
> | [[options-chain]] | yfinance | `get_options_chain(ticker, exp)` |
> | Greeks | yfinance | `get_options_chain(ticker, exp, greeks=True)` |

---
**Related strategies:** [[covered-call]] | [[bear-put-spread]] | [[short-strangle]]
**Concepts:** [[delta]] | [[theta]] | [[implied-volatility]] | [[time-decay]]
