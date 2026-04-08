---
up: []
related: ["[[covered-call]]", "[[bear-put-spread]]", "[[risk-reversal]]"]
created: 2026-04-07
type: strategy
category: options
subcategory: hedging
complexity: simple
data_source: [yfinance]
tags: [options, hedging, protection, insurance]
---

# Protective Put

> [!abstract]
> Long stock + long put. Portfolio insurance — limits downside to the put strike while maintaining unlimited upside. Costs premium.

## Core Mechanic

Buy a put on stock you already own. The put gains value as the stock drops, offsetting losses below the strike. Equivalent to a long call via [[put-call-parity]].

```
P&L
 ↑                              unlimited upside
 |                             /
 |                            /
 |                           /
─┼──────────────────────────/──── Price →
 |──────────────────────── /
 |  max loss = (cost - K) + premium
 |                    K (long put)
```

**Max profit:** Unlimited (stock rises)
**Max loss:** (Stock cost - strike) + premium paid
**Breakeven:** Stock cost + premium paid

## Greeks Profile

| Greek | Exposure | Meaning |
|-------|----------|---------|
| [[delta]] | Net positive (reduced) | Long stock delta partially offset by long put |
| [[gamma]] | Net positive | Long put provides positive gamma — accelerating protection |
| [[theta]] | Net negative | You pay [[time-decay]] for the insurance |
| [[vega]] | Net positive | IV expansion benefits (puts get more valuable in panics) |

## Trade Construction

1. Hold long equity
2. Buy 1 put per 100 shares — **5-10% OTM, 30-90 DTE**
3. Longer DTE = lower daily cost but more capital tied up
4. Consider rolling 7-14 days before expiry

| Parameter | Value |
|-----------|-------|
| Strike | 5-10% below current price |
| DTE | 30-90 days |
| [[delta]] | -0.20 to -0.35 |

> [!tip] When to Use
> - Holding a concentrated position through a risky event
> - Portfolio protection during elevated macro uncertainty
> - VIX < 20 = puts are cheaper (lower [[implied-volatility|IV]])
> - Prefer over selling stock when you want to maintain upside exposure

> [!danger] Key Risk
> - **Premium cost** — buying puts every month is expensive (3-5% annually)
> - [[time-decay]] works against you constantly
> - Timing the hedge: too early = wasted premium; too late = puts are expensive
> - Consider [[bear-put-spread]] to reduce cost (but caps the protection)

## Data Pipeline

> [!info] Synesis Data
> | Need | Source | Method |
> |------|--------|--------|
> | [[options-chain]] | yfinance | `get_options_chain(ticker, exp)` |
> | Greeks | yfinance | `get_options_chain(ticker, exp, greeks=True)` |
> | VIX level | yfinance | `get_quote("^VIX")` |

---
**Related strategies:** [[covered-call]] | [[bear-put-spread]] | [[risk-reversal]]
**Concepts:** [[delta]] | [[gamma]] | [[vega]] | [[theta]] | [[put-call-parity]] | [[time-decay]]
