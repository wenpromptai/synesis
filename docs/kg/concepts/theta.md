---
up: []
related: ["[[time-decay]]", "[[gamma]]", "[[implied-volatility]]"]
created: 2026-04-07
type: concept
tags: [options, greeks]
aliases: [Theta]
---

# Theta

> [!info] Definition
> The rate of change of an option's price with respect to the passage of one day, all else equal. Measures [[time-decay]]. ^definition

## Formula

$$
\Theta = \frac{\partial V}{\partial t}
$$

Negative for long options (value erodes), positive for short options (value earned).

## Key Properties

1. **Non-linear decay:** Theta accelerates — most decay occurs in the final 30 days
2. **Highest ATM:** ATM options have the most extrinsic value to lose
3. **Proportional to IV:** Higher [[implied-volatility|IV]] = higher theta (more extrinsic value)
4. **Weekend effect:** Theta is priced continuously but markets are closed weekends — Friday premium includes weekend decay

## Theta Decay Curve

```
Extrinsic Value
  |
  |────────────────╲
  |                  ╲
  |                    ╲
  |                      ╲
  |                        ╲
  |                          ╲
  |                            ╲
  └──────────────────────────────── DTE
  90      60      30      15    0
```

~1/3 of value decays in final 1/4 of the option's life.

## In Practice

- **[[covered-call]]:** Positive theta is the core income mechanism — premium decays in your favor
- **[[iron-condor]]:** Net positive theta across 4 legs; earn daily as long as price stays inside wings
- **[[short-strangle]]:** Highest theta income but undefined risk
- **[[long-straddle]]:** Negative theta — you pay daily; need the underlying to move enough to overcome decay
- **[[calendar-spread]]:** Exploits theta differential: front-month decays faster than back-month

---
**See also:** [[gamma]] | [[time-decay]] | [[implied-volatility]] | [[vega]]
