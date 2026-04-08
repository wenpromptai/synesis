---
up: []
related: ["[[implied-volatility]]", "[[realized-volatility]]", "[[volatility-risk-premium]]"]
created: 2026-04-07
type: concept
tags: [options, greeks]
aliases: [Vega]
---

# Vega

> [!info] Definition
> The rate of change of an option's price with respect to a 1 percentage-point change in [[implied-volatility|implied volatility]]. ^definition

## Formula

$$
\nu = \frac{\partial V}{\partial \sigma}
$$

Where $\sigma$ = implied volatility.

## Key Properties

1. **Always positive** for long options — IV increase benefits buyers
2. **Highest ATM** and with longer DTE — far-dated ATM options are most IV-sensitive
3. **Per-point basis:** Vega of 0.15 means option price changes $0.15 per 1% IV change
4. **Decreases near expiry:** Short-dated options are less sensitive to IV

## In Practice

- **[[long-straddle]]:** Long vega — profits from IV expansion (pre-event setup)
- **[[iron-condor]]:** Short vega — IV expansion is the primary enemy; both short legs become more expensive to close
- **[[volatility-risk-premium]]:** Systematically short vega to harvest the IV > RV gap
- **[[calendar-spread]]:** Net long vega — back-month has more vega than front-month
- **[[earnings-options-systematic]]:** Pre-earnings = long vega (IV expands into event); post-earnings = short vega (IV crush)

## Vega and Strategy Selection

| IV Environment | Vega Stance | Strategies |
|---------------|-------------|------------|
| High [[iv-rank]] (>50%) | Short vega | [[iron-condor]], [[short-strangle]], [[covered-call]] |
| Low IV rank (<30%) | Long vega | [[long-straddle]], [[long-strangle]], [[calendar-spread]] |
| Mid IV rank | Neutral vega | [[bull-call-spread]], [[bear-put-spread]] |

---
**See also:** [[implied-volatility]] | [[realized-volatility]] | [[iv-rank]] | [[theta]]
