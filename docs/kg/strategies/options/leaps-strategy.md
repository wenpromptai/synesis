---
up: []
related: ["[[bull-call-spread]]", "[[covered-call]]", "[[protective-put]]"]
created: 2026-04-07
type: strategy
category: options
subcategory: directional
complexity: medium
data_source: [yfinance]
tags: [options, directional, long-term, stock-replacement]
---

# LEAPS Strategy

> [!abstract]
> Long-dated (6-24 month) deep ITM calls as stock replacement. Lower capital outlay than owning shares, with similar upside exposure.

## Core Mechanic

Buy a deep ITM call with 6-24 months to expiry. High [[delta]] (0.70-0.80) means the option moves nearly dollar-for-dollar with the stock. You control 100 shares for a fraction of the capital.

```
P&L
 ↑                              / similar to stock above breakeven
 |                             /
 |                            /
─┼───────────────────────────/──── Price →
 |──────────────────────────
 |  max loss = premium paid (much less than stock cost)
 |                         BE (strike + premium)
```

**Max profit:** Unlimited
**Max loss:** Premium paid (significantly less than stock purchase price)
**Breakeven:** Strike + premium paid

## Key Properties

1. **Capital efficiency:** Control $50,000 of stock for $10,000-$15,000
2. **High delta:** 0.70-0.80 — near stock-like participation
3. **Low [[theta]]:** Far-dated options decay slowly — minimal time erosion
4. **No dividends:** You don't receive dividends (factor into analysis)
5. **Leverage:** ~3-5x notional leverage built in

## Trade Construction

1. Choose stock with strong long-term thesis
2. Buy 1 deep ITM call — **[[delta]] 0.70-0.80, 12-24 months to expiry**
3. Strike: 15-25% below current price (deep ITM)
4. Roll at 6 months remaining to maintain long DTE

| Parameter | Value |
|-----------|-------|
| [[delta]] | 0.70-0.80 |
| DTE | 12-24 months (LEAPS) |
| [[moneyness]] | Deep ITM (15-25% below spot) |
| Roll | At 6 months remaining |

> [!tip] When to Use
> - Long-term bullish thesis but want less capital at risk
> - Replace stock position with defined max loss
> - Pair with short-term [[covered-call]] overlay on the LEAPS (poor man's covered call)

> [!danger] Key Risk
> - Premium lost if stock drops below strike at expiry
> - No dividends (relevant for high-yield stocks)
> - Wide bid-ask spreads on LEAPS (less liquid)
> - Leverage cuts both ways — percentage loss is amplified

## Data Pipeline

> [!info] Synesis Data
> | Need | Source | Method |
> |------|--------|--------|
> | Expirations (find 12-24mo) | yfinance | `get_options_expirations(ticker)` |
> | Chain for deep ITM strikes | yfinance | `get_options_chain(ticker, exp, greeks=True)` |

---
**Related strategies:** [[bull-call-spread]] | [[covered-call]] | [[protective-put]]
**Concepts:** [[delta]] | [[theta]] | [[moneyness]] | [[time-decay]] | [[breakeven]]
