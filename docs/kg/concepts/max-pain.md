---
up: []
related: ["[[open-interest]]", "[[zero-dte]]", "[[butterfly-spread]]"]
created: 2026-04-07
type: concept
tags: [options, mechanics]
aliases: [max pain, maximum pain, pin risk]
---

# Max Pain

> [!info] Definition
> The strike price at which option holders collectively lose the most money at expiration — equivalently, where option writers profit the most. Price tends to gravitate toward max pain near expiry ("pinning"). ^definition

## Calculation

For each strike $K_i$:
$$
\text{Pain}(K_i) = \sum_{\text{calls}} OI_j \times \max(K_i - K_j, 0) + \sum_{\text{puts}} OI_j \times \max(K_j - K_i, 0)
$$

Max pain = strike that minimizes total pain (or maximizes writer profit).

## Why It Works (Sometimes)

1. **Dealer hedging:** Market makers who sold options delta-hedge; as expiry nears, their hedging activity pushes price toward max pain
2. **Gamma exposure:** Near expiry, [[gamma]] is concentrated at high-OI strikes, creating magnetic-like effects
3. **Self-fulfilling:** Enough participants watching max pain creates actual gravitational pull

## In Practice

- **[[zero-dte]]:** Max pain is most relevant on 0DTE — extreme gamma concentration at high-OI strikes
- **[[butterfly-spread]]:** Target the body (short strikes) at or near max pain for highest probability of pinning
- **[[iron-condor]]:** Center the profit zone around max pain for additional edge
- **Not predictive far from expiry:** Max pain only matters in the final 1-2 days

> [!warning] Watch Out
> Max pain is a tendency, not a guarantee. Large directional flows (earnings, macro events) override pinning mechanics. Works best on quiet, low-catalyst expiration days.

## Data Source

> [!info] Synesis Pipeline
> Compute from yfinance `get_options_chain()` — sum [[open-interest]] across all strikes and calculate pain at each level.

---
**See also:** [[open-interest]] | [[zero-dte]] | [[gamma]] | [[butterfly-spread]]
