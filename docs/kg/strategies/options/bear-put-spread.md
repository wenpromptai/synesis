---
up: []
related: ["[[bull-call-spread]]", "[[protective-put]]", "[[risk-reversal]]"]
created: 2026-04-07
type: strategy
category: options
subcategory: directional
complexity: simple
data_source: [yfinance]
tags: [options, directional, bearish, defined-risk]
---

# Bear Put Spread

> [!abstract]
> Long higher-strike put + short lower-strike put. Defined-risk bearish bet. Mirror of [[bull-call-spread]].

## Core Mechanic

Buy a put, sell a lower-strike put at the same expiry. Net debit. Profit if stock drops below the long put strike.

```
P&L
 |──────────   max loss = debit paid
 |          \
─┼───────────\──────────────── Price →
 |            \
 |             \
 ↓              ────────────── max gain = width - debit
 |         K1 (short)    K2 (long)
```

**Max profit:** (K2 - K1) - net debit
**Max loss:** Net debit paid
**Breakeven:** K2 - net debit

## Greeks Profile

| Greek | Exposure | Meaning |
|-------|----------|---------|
| [[delta]] | Net negative | Profits from downward move |
| [[gamma]] | Net positive (small) | Spread dampens vs naked put |
| [[theta]] | Net negative | Time works against you |
| [[vega]] | Net positive | IV expansion helps (stocks drop → IV rises) |

## Trade Construction

1. Buy 1 put ATM or slightly ITM — **[[delta]] -0.55 to -0.65**
2. Sell 1 put further OTM — **[[delta]] -0.30 to -0.40**
3. Same expiry: **30-60 DTE**

| Parameter | Value |
|-----------|-------|
| Long put [[delta]] | -0.55 to -0.65 |
| Short put [[delta]] | -0.30 to -0.40 |
| DTE | 30-60 days |
| Target exit | 50-75% of max profit |

> [!tip] When to Use
> - Bearish thesis with defined downside target
> - Cheaper than buying naked puts
> - Preferred over [[protective-put]] when you don't hold the stock
> - [[vega]] benefit: IV typically rises as stocks fall, helping this position

> [!danger] Key Risk
> - Max loss = entire debit if stock stays above K2 at expiry
> - Capped profit — won't benefit from a crash below K1

## Data Pipeline

> [!info] Synesis Data
> | Need | Source | Method |
> |------|--------|--------|
> | [[options-chain]] | yfinance | `get_options_chain(ticker, exp)` |
> | Greeks | yfinance | `get_options_chain(ticker, exp, greeks=True)` |

---
**Related strategies:** [[bull-call-spread]] | [[protective-put]] | [[risk-reversal]]
**Concepts:** [[delta]] | [[vega]] | [[breakeven]] | [[moneyness]]
