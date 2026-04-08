---
up: []
related: ["[[moneyness]]", "[[time-decay]]", "[[options-chain]]"]
created: 2026-04-07
type: concept
tags: [options, mechanics]
aliases: [breakeven point, BE, break-even]
---

# Breakeven

> [!info] Definition
> The underlying price at which a strategy produces zero profit/loss at expiration. Every options position has one or two breakeven points that define the boundary between profit and loss. ^definition

## Breakevens by Strategy

| Strategy | Breakeven(s) |
|----------|-------------|
| **Long call** | Strike + premium paid |
| **Long put** | Strike - premium paid |
| **[[covered-call]]** | Stock cost - premium received |
| **[[protective-put]]** | Stock cost + premium paid |
| **[[bull-call-spread]]** | Lower strike + net debit |
| **[[bear-put-spread]]** | Upper strike - net debit |
| **[[long-straddle]]** | Strike +/- total premium (two BEs) |
| **[[long-strangle]]** | Call strike + premium / Put strike - premium (two BEs) |
| **[[iron-condor]]** | Short put strike - credit / Short call strike + credit (two BEs) |
| **[[butterfly-spread]]** | Lower strike + debit / Upper strike - debit (two BEs) |

## Key Properties

1. **At expiration only:** Breakeven is exact only at expiry; before expiry, extrinsic value shifts the actual breakeven
2. **Probability of profit:** Distance from current price to breakeven indicates probability of profit
3. **More breakevens = more complex:** Spreads have 2 BEs, condors have 2, butterflies have 2
4. **Cost basis reduction:** Strategies like [[covered-call]] and [[wheel-strategy]] reduce breakeven over repeated cycles

## In Practice

- **Entry filter:** Check that breakevens are achievable given expected move and time horizon
- **Risk assessment:** Distance from spot to downside breakeven = buffer before loss
- **Strategy comparison:** Compare breakevens across strategies for the same directional view

---
**See also:** [[moneyness]] | [[time-decay]] | [[options-chain]]
