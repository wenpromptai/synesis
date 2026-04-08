---
up: []
related: ["[[delta]]", "[[options-chain]]", "[[time-decay]]"]
created: 2026-04-07
type: concept
tags: [options, mechanics]
aliases: [ITM, OTM, ATM, in-the-money, out-of-the-money, at-the-money, moneyness]
---

# Moneyness

> [!info] Definition
> The relationship between an option's strike price and the underlying's current price. Determines intrinsic value and probability profile. ^definition

## States

| State | Call | Put | Intrinsic Value |
|-------|------|-----|-----------------|
| **ITM** (in-the-money) | Strike < Spot | Strike > Spot | Positive |
| **ATM** (at-the-money) | Strike = Spot | Strike = Spot | Zero |
| **OTM** (out-of-the-money) | Strike > Spot | Strike < Spot | Zero |

## Impact on Greeks

| Moneyness | [[delta]] | [[gamma]] | [[theta]] | [[vega]] |
|-----------|-----------|-----------|-----------|----------|
| Deep ITM | ~1.0 / -1.0 | Low | Low | Low |
| ATM | ~0.50 / -0.50 | **Highest** | **Highest** | **Highest** |
| OTM | ~0.05-0.30 | Moderate | Moderate | Moderate |
| Deep OTM | ~0.00 | Near zero | Near zero | Near zero |

## In Practice

- **[[covered-call]]:** Sell OTM calls (delta 0.25-0.35) — above current price
- **[[iron-condor]]:** Both short legs OTM (delta 0.15-0.20 each side)
- **[[long-straddle]]:** Buy ATM call + ATM put — maximum [[gamma]] and [[vega]]
- **[[leaps-strategy]]:** Buy deep ITM calls (delta 0.70-0.80) — stock replacement
- **[[protective-put]]:** Buy OTM puts (5-10% below spot) — insurance

## Data Source

> [!info] Synesis Pipeline
> yfinance chain includes `in_the_money` boolean field per contract. Calculate moneyness ratio: strike / spot price.

---
**See also:** [[delta]] | [[options-chain]] | [[time-decay]] | [[breakeven]]
