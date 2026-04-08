---
up: []
related: ["[[realized-volatility]]", "[[iv-rank]]", "[[vega]]", "[[volatility-smile]]"]
created: 2026-04-07
type: concept
tags: [options, volatility]
aliases: [IV, implied vol, implied volatility]
---

# Implied Volatility

> [!info] Definition
> The market's expectation of future price volatility, derived (implied) from current option prices via a pricing model. Higher IV = more expensive options. ^definition

## How It Works

IV is "backed out" of the Black-Scholes formula. Given the observed market price of an option, solve for the volatility parameter $\sigma$ that makes the model price match:

$$
C_{market} = BS(S, K, T, r, \sigma_{implied})
$$

IV is expressed as an annualized percentage — IV of 30% means the market expects the underlying to move within a ~30% range over the next year (1 standard deviation).

## Key Properties

1. **Forward-looking:** Unlike [[realized-volatility|RV]], IV reflects expectations, not history
2. **Mean-reverting:** IV tends to revert to a long-term average; extreme readings are temporary
3. **Negatively correlated with price:** Stock drops → IV spikes (fear premium)
4. **Not uniform across strikes:** Varies by strike and expiry — see [[volatility-smile]]
5. **Per-contract:** Each option has its own IV; yfinance chain provides this per contract

## IV vs Realized Volatility

| | IV | [[realized-volatility|RV]] |
|---|---|---|
| Direction | Forward-looking | Backward-looking |
| Source | Option prices | Price history |
| Typical relationship | IV > RV ~80% of the time | — |
| Gap | = [[volatility-risk-premium]] | — |

## In Practice

- **[[iv-rank-strategy-selection]]:** IV level determines strategy — high IV = sell premium, low IV = buy premium
- **[[volatility-risk-premium]]:** IV systematically overestimates RV; selling options harvests this gap
- **[[long-straddle]]:** Buy when IV is low, expecting expansion
- **[[iron-condor]]:** Sell when IV is high, expecting contraction

## Data Source

> [!info] Synesis Pipeline
> yfinance `get_options_chain(ticker, expiration)` returns IV per contract in the `.implied_volatility` field. VIX (`get_quote("^VIX")`) represents SPX 30-day IV.

---
**See also:** [[realized-volatility]] | [[iv-rank]] | [[vega]] | [[volatility-smile]] | [[volatility-risk-premium]]
