---
up: []
related: ["[[gamma]]", "[[moneyness]]", "[[options-chain]]"]
created: 2026-04-07
type: concept
tags: [options, greeks]
aliases: [Delta]
---

# Delta

> [!info] Definition
> The rate of change of an option's price with respect to a $1 move in the underlying asset. ^definition

## Formula

$$
\Delta = \frac{\partial V}{\partial S}
$$

Where $V$ = option value, $S$ = underlying price.

## Key Properties

1. **Range:** 0 to 1 for calls, 0 to -1 for puts
2. **Probability proxy:** Delta approximates the probability of expiring [[moneyness|ITM]]
3. **Hedge ratio:** A delta of 0.50 means 50 shares of stock to hedge 1 contract
4. **Moneyness dependent:** Deep [[moneyness|ITM]] approaches +/-1, deep [[moneyness|OTM]] approaches 0, [[moneyness|ATM]] near +/-0.50
5. **Not static:** Changes with price ([[gamma]]), time ([[theta]]), and [[implied-volatility|IV]] (vanna)

## Delta by Moneyness

| Moneyness | Call Delta | Put Delta |
|-----------|-----------|-----------|
| Deep ITM | 0.80-1.00 | -0.80 to -1.00 |
| ATM | ~0.50 | ~-0.50 |
| OTM | 0.01-0.30 | -0.01 to -0.30 |
| Deep OTM | ~0.00 | ~0.00 |

## In Practice

- **Strike selection:** [[covered-call]] sells 0.25-0.35 delta calls; [[iron-condor]] sells 0.16 delta on each side
- **Position sizing:** Delta-weighted notional = contracts x delta x 100 x stock price
- **Hedging:** [[protective-put]] adds negative delta to offset long stock (+1 delta per 100 shares)
- **Stock replacement:** [[leaps-strategy]] buys 0.70-0.80 delta deep ITM calls
- **Directional bets:** [[bull-call-spread]] has net positive delta; [[bear-put-spread]] has net negative delta

## Common Misconceptions

> [!warning] Watch Out
> Delta is NOT exactly the probability of expiring ITM — it's a mathematical derivative that approximates probability under risk-neutral pricing. Real-world probability differs due to skew and drift.

---
**See also:** [[gamma]] | [[theta]] | [[vega]] | [[moneyness]] | [[options-chain]]
