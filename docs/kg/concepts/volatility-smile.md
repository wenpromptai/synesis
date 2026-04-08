---
up: []
related: ["[[implied-volatility]]", "[[moneyness]]", "[[options-chain]]"]
created: 2026-04-07
type: concept
tags: [options, volatility]
aliases: [vol smile, volatility skew, skew, vol skew]
---

# Volatility Smile

> [!info] Definition
> The pattern where [[implied-volatility|IV]] varies across strike prices at a given expiration. In equity markets, OTM puts typically have higher IV than OTM calls, creating a "smirk" rather than a symmetric smile. ^definition

## Shape

```
IV
 |
 |  ╲                        (equity skew / smirk)
 |    ╲
 |      ╲
 |        ╲─────────╱        (slight upturn far OTM calls)
 |                 ╱
 └──────────────────────── Strike
  OTM puts   ATM    OTM calls
```

## Why It Exists

1. **Demand for protection:** Portfolio hedgers buy OTM puts → drives up put IV
2. **Fat left tail:** Markets crash more often/harder than they rally — skew reflects this
3. **Leverage effect:** Stock drops → company leverage increases → volatility increases
4. **Supply-demand:** More natural put buyers (hedgers) than sellers → put premium stays elevated

## Key Metrics

- **25-delta skew:** IV(25-delta put) - IV(25-delta call) — standard skew measure
- **Skew steepness:** Steep = market is fearful; flat = complacent

## In Practice

- **[[iron-condor]]:** Skew asymmetry means put spreads collect more credit than call spreads for same delta
- **[[covered-call]]:** Steep call skew (rare) = better premium; steep put skew = safer environment
- **[[risk-reversal]]:** Skew determines the net cost — steep put skew makes bullish risk-reversals cheaper
- **[[volatility-arbitrage]]:** Skew changes can be traded via butterfly positions

## Observable From Data

> [!info] Synesis Pipeline
> Plot IV from yfinance `get_options_chain()` across strikes at a single expiration. The shape reveals the smile/skew structure.

---
**See also:** [[implied-volatility]] | [[moneyness]] | [[options-chain]] | [[iron-condor]]
