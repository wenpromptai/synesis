---
up: []
related: ["[[delta]]", "[[theta]]", "[[zero-dte]]"]
created: 2026-04-07
type: concept
tags: [options, greeks]
aliases: [Gamma]
---

# Gamma

> [!info] Definition
> The rate of change of [[delta]] with respect to a $1 move in the underlying. Measures how fast delta accelerates. ^definition

## Formula

$$
\Gamma = \frac{\partial^2 V}{\partial S^2} = \frac{\partial \Delta}{\partial S}
$$

## Key Properties

1. **Always positive** for long options (calls and puts)
2. **Highest ATM** and near expiration — gamma explodes in final days
3. **Gamma-theta tradeoff:** Long gamma costs [[theta]]; short gamma earns theta
4. **Acceleration:** High gamma means delta changes rapidly — position behaves non-linearly

## Gamma by Time to Expiry

| DTE | ATM Gamma | Behaviour |
|-----|-----------|-----------|
| 90+ days | Low | Delta moves slowly, position is stable |
| 30 days | Moderate | Standard options trading range |
| 7 days | High | Delta becomes sensitive to small moves |
| 0 DTE | Extreme | ATM options flip between 0 and 1 delta rapidly |

## In Practice

- **[[zero-dte]]:** Pure gamma trading — ATM 0DTE options have extreme gamma, allowing rapid directional profits
- **[[long-straddle]]:** Long gamma position — profits from moves in either direction because delta adjusts favorably
- **[[iron-condor]]:** Short gamma — profits when price stays still; large moves cause both legs to go deeper ITM
- **[[short-strangle]]:** Short gamma with undefined risk — the worst case for short gamma
- **[[calendar-spread]]:** Long back-month gamma is lower than short front-month gamma — net gamma depends on DTE gap

## The Gamma-Theta Tradeoff

> [!tip] Core Insight
> You cannot be long gamma without paying [[theta]]. You cannot earn theta without being short gamma. This is the fundamental tension in options trading.
>
> - **Long gamma, short theta:** Pay daily for the right to profit from moves ([[long-straddle]])
> - **Short gamma, long theta:** Earn daily but risk large losses from moves ([[iron-condor]])

---
**See also:** [[delta]] | [[theta]] | [[vega]] | [[zero-dte]] | [[time-decay]]
