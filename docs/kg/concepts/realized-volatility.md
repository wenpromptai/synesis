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
| 30-day | Standard comparison to [[implied-volatility|IV]] (VIX equivalent) |
| 60-day | Medium-term regime assessment |
| 252-day | Annual baseline |

## In Practice

- **[[volatility-risk-premium]]:** When IV >> RV, sell premium (overpriced insurance)
- **[[volatility-arbitrage]]:** Trade the IV - RV spread directly with delta-hedged positions
- **[[regime-options-matrix]]:** RV level helps confirm whether current regime is truly low-vol or high-vol
- **[[covered-call]]:** Low RV expected vs current IV = ideal environment (premium rich, moves small)

## IV-RV Premium Screening

The gap between [[implied-volatility|IV]] and RV is the core input for premium-selling strategies. Pipeline briefs now provide systematic IV-RV data across debated tickers, enabling screening for premium-selling opportunities.

**Observed IV-RV spreads (2026-04-13 brief):**
| Ticker | ATM IV | 30d RV | IV-RV Spread | Signal |
|--------|--------|--------|-------------|--------|
| [[tickers/AMD]] | 63.4% | 49.7% | **+13.7%** | Rich premium — strong selling opportunity |
| [[tickers/GS]] | 39.9% | 31.9% | +8.0% | Moderately rich — event risk (earnings) |
| [[tickers/GOOG]] | 38.0% | 33.7% | +4.3% | Mildly rich |
| [[tickers/AVGO]] | 47.4% | 45.8% | +1.6% | Near fair value |
| [[tickers/NVDA]] | 34.7% | 34.6% | +0.1% | Fair — no edge |

**Screening rules of thumb:**
- IV-RV > +10%: Strong premium-selling signal. Consider [[covered-call]], [[iron-condor]], or [[short-strangle]] depending on directional view
- IV-RV +5% to +10%: Moderate — look for catalysts (earnings, events) that justify elevated IV before selling
- IV-RV < +5%: Fair to cheap — no premium-selling edge. Consider premium-buying strategies if directionally convicted
- IV-RV < 0%: Rare — realized vol exceeding implied. Strong signal for long vol ([[long-straddle]], [[long-strangle]])

See [[volatility-risk-premium]] for the systematic strategy and [[iv-rank-strategy-selection]] for combining IV-RV with [[iv-rank]].

## Data Source

> [!info] Synesis Pipeline
> yfinance `get_options_snapshot(ticker)` provides `.realized_vol_30d` (30-day RV). For custom windows, compute from `get_history(ticker, period)` price data.

> [!quote]- Sources
> - [[sources/brief-2026-04-13]] — IV-RV data across 5 debated tickers

---
**See also:** [[implied-volatility]] | [[volatility-risk-premium]] | [[volatility-arbitrage]] | [[vega]] | [[iv-rank-strategy-selection]]
