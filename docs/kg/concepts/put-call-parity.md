---
up: []
related: ["[[risk-reversal]]", "[[covered-call]]", "[[protective-put]]", "[[delta]]"]
created: 2026-04-07
type: concept
tags: [options, theory]
aliases: [put-call parity, PCP]
---

# Put-Call Parity

> [!info] Definition
> The fundamental no-arbitrage relationship linking European call and put prices at the same strike and expiry to the underlying price and risk-free rate. ^definition

## Formula

$$
C - P = S - K \cdot e^{-rT}
$$

Where: $C$ = call price, $P$ = put price, $S$ = spot price, $K$ = strike, $r$ = risk-free rate, $T$ = time to expiry.

Rearranged: $C + K \cdot e^{-rT} = P + S$ (long call + bond = long put + stock)

## Key Properties

1. **No-arbitrage constraint:** If violated, risk-free profit exists (quickly arbitraged away)
2. **Synthetic equivalence:** Any position can be replicated using the other three components
3. **American options:** Parity is approximate due to early exercise — holds as inequality

## Synthetic Positions

| Desired Position | Equivalent Construction |
|-----------------|----------------------|
| Long stock | Long call + short put + bond |
| Short stock | Short call + long put - bond |
| Long call | Long stock + long put - bond |
| Long put | Short stock + long call + bond |

## In Practice

- **[[covered-call]]** (long stock + short call) = **short put** at the same strike — identical P&L
- **[[risk-reversal]]:** Directly constructed from the C - P relationship — short put + long call ≈ synthetic long
- **[[protective-put]]** (long stock + long put) = **long call** + bond — same payoff
- **Arbitrage detection:** If parity is violated in yfinance chain data, it's usually a stale quote, not a real opportunity

---
**See also:** [[risk-reversal]] | [[covered-call]] | [[protective-put]] | [[delta]]
