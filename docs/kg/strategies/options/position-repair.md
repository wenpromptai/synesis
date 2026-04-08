---
up: []
related: ["[[covered-call]]", "[[bull-call-spread]]"]
created: 2026-04-07
type: strategy
category: options
subcategory: hedging
complexity: medium
data_source: [yfinance]
tags: [options, hedging, repair, cost-basis]
---

# Position Repair

> [!abstract]
> Ratio call spread on an underwater long stock position. Reduces breakeven without adding cash. Buy 1 ATM call + sell 2 OTM calls for net zero cost.

## Core Mechanic

You're long stock at $100, it's now at $85. Instead of averaging down (more capital at risk), overlay a 1x2 call spread:
- Buy 1 call at $85 (ATM)
- Sell 2 calls at $92.50 (OTM)
- Net cost: ~$0 (2 short calls fund the 1 long call)

```
P&L
 ↑            ╱╲
 |           ╱  ╲  new breakeven = ~$92.50 (was $100)
 |          ╱    ╲
─┼─────────╱──────╲──────── Price →
 |        ╱        ╲  capped above $92.50
 |───────╱
 |    $85      $92.50   $100 (original cost)
```

**Effect:** Breakeven drops from $100 to ~$92.50 at no additional cost. Upside capped at $92.50.

## Trade Construction

1. Hold underwater long stock (bought at $X, now at lower price)
2. Buy 1 call at current price (ATM)
3. Sell 2 calls at midpoint between current price and original cost
4. Target net debit: $0 or small credit
5. Same expiry: **45-60 DTE** (needs time for stock to recover)

| Parameter | Value |
|-----------|-------|
| Long call strike | Current stock price (ATM) |
| Short call strike | (Current + Original cost) / 2 |
| Ratio | 1:2 (buy 1, sell 2) |
| Net cost | ~$0 |
| DTE | 45-60 days |

> [!danger] Key Risk
> - **Caps upside** at the short call strike — if stock fully recovers, you miss the last leg
> - Above the short strike, you have a naked short call (undefined risk on the extra leg)
> - Only works if stock recovers partially; if it keeps dropping, the spread expires worthless
> - Assignment risk on 2 short calls if stock rallies hard

> [!tip] When to Use
> - Stock down 10-20% from entry, you believe partial recovery likely
> - Better than averaging down when you don't want more capital at risk
> - NOT for stocks in structural decline — repair assumes recovery

## Data Pipeline

> [!info] Synesis Data
> | Need | Source | Method |
> |------|--------|--------|
> | [[options-chain]] | yfinance | `get_options_chain(ticker, exp)` |
> | Greeks | yfinance | `get_options_chain(ticker, exp, greeks=True)` |

---
**Related strategies:** [[covered-call]] | [[bull-call-spread]]
**Concepts:** [[delta]] | [[breakeven]] | [[moneyness]] | [[time-decay]]
