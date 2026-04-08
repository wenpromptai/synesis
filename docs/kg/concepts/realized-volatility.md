---
up: []
related: ["[[implied-volatility]]", "[[volatility-risk-premium]]", "[[volatility-arbitrage]]"]
created: 2026-04-07
type: concept
tags: [options, volatility]
aliases: [RV, historical volatility, HV, realized vol]
---

# Realized Volatility

> [!info] Definition
> The actual observed volatility of an asset, measured as the annualized standard deviation of log returns over a lookback period. ^definition

## Formula

$$
RV = \sigma_{realized} = \sqrt{\frac{252}{n-1} \sum_{i=1}^{n} (r_i - \bar{r})^2}
$$

Where $r_i = \ln(P_i / P_{i-1})$, $n$ = number of trading days, 252 = annualization factor.

## Key Properties

1. **Backward-looking:** Measures what already happened, not what will happen
2. **Window-dependent:** 10-day RV vs 30-day RV vs 90-day RV give different readings
3. **Compare to [[implied-volatility|IV]]:** The gap (IV - RV) = [[volatility-risk-premium]]
4. **Clustering:** Volatility clusters — high RV periods tend to persist (GARCH effect)

## Common Windows

| Window | Use |
|--------|-----|
| 10-day | Short-term trading, [[zero-dte]] context |
| 30-day | Standard comparison to [[implied-volatility\|IV]] (VIX equivalent) |
| 60-day | Medium-term regime assessment |
| 252-day | Annual baseline |

## In Practice

- **[[volatility-risk-premium]]:** When IV >> RV, sell premium (overpriced insurance)
- **[[volatility-arbitrage]]:** Trade the IV - RV spread directly with delta-hedged positions
- **[[regime-options-matrix]]:** RV level helps confirm whether current regime is truly low-vol or high-vol
- **[[covered-call]]:** Low RV expected vs current IV = ideal environment (premium rich, moves small)

## Data Source

> [!info] Synesis Pipeline
> yfinance `get_options_snapshot(ticker)` provides `.realized_vol_30d` (30-day RV). For custom windows, compute from `get_history(ticker, period)` price data.

---
**See also:** [[implied-volatility]] | [[volatility-risk-premium]] | [[volatility-arbitrage]] | [[vega]]
