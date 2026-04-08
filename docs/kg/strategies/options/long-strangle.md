---
up: []
related: ["[[long-straddle]]", "[[short-strangle]]", "[[iron-condor]]"]
created: 2026-04-07
type: strategy
category: options
subcategory: volatility
complexity: medium
data_source: [yfinance]
tags: [options, volatility, long-vol]
---

# Long Strangle

> [!abstract]
> Long OTM call + long OTM put. Cheaper than [[long-straddle]] but needs a bigger move to profit. Wider breakevens, lower max loss.

## Core Mechanic

Buy an OTM call and an OTM put. Both legs are cheaper than ATM, so total cost is lower. But the stock must move further to reach profitability.

```
P&L
 ↑
 |  \                                 /
 |    \                             /
 |      \                         /
─┼────────\───────────────────────/──── Price →
 |          ─────────────────────  max loss = total premium
 |        K1 (long put)    K2 (long call)
 |       BE1                        BE2
```

**Max profit:** Unlimited
**Max loss:** Total premium paid
**Breakevens:** K1 - premium (down) / K2 + premium (up)

## Greeks Profile

| Greek | Exposure | Meaning |
|-------|----------|---------|
| [[delta]] | Near zero | OTM call and put deltas roughly cancel |
| [[gamma]] | Net positive | Less than [[long-straddle]] (OTM = less gamma) |
| [[theta]] | Net negative | Less daily cost than straddle but still paying decay |
| [[vega]] | Net positive | IV expansion benefits both legs |

## Trade Construction

1. Buy 1 OTM put — **[[delta]] -0.25 to -0.35**
2. Buy 1 OTM call — **[[delta]] 0.25 to 0.35**
3. Same expiry: **21-45 DTE**

| Parameter | Value |
|-----------|-------|
| Put [[delta]] | -0.25 to -0.35 |
| Call [[delta]] | 0.25 to 0.35 |
| DTE | 21-45 days |
| Target exit | 25-50% profit |

> [!tip] Strangle vs Straddle
> | | [[long-straddle]] | Long Strangle |
> |---|---|---|
> | Cost | Higher | **Lower** |
> | [[breakeven]] width | Narrower | **Wider** |
> | Max [[gamma]] | Higher | Lower |
> | Win rate | Higher | **Lower** |
> | Preferred when | Tight consolidation | Expecting large move, cost-sensitive |

> [!danger] Key Risk
> - Wider breakevens mean you need a bigger move to profit
> - Both legs expire worthless if stock stays between strikes
> - Same [[time-decay]] and IV crush risks as [[long-straddle]]

## Data Pipeline

> [!info] Synesis Data
> | Need | Source | Method |
> |------|--------|--------|
> | [[options-chain]] | yfinance | `get_options_chain(ticker, exp, greeks=True)` |
> | [[iv-rank]] | yfinance | VIX history |

---
**Related strategies:** [[long-straddle]] | [[short-strangle]] | [[iron-condor]]
**Concepts:** [[delta]] | [[gamma]] | [[vega]] | [[theta]] | [[breakeven]]
