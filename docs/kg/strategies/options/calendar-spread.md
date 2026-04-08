---
up: []
related: ["[[iron-condor]]", "[[butterfly-spread]]", "[[long-straddle]]"]
created: 2026-04-07
type: strategy
category: options
subcategory: volatility
complexity: medium
data_source: [yfinance]
tags: [options, volatility, time-spread, theta]
---

# Calendar Spread

> [!abstract]
> Short near-term option + long far-term option at the same strike. Profit from differential [[time-decay]] and IV term structure. Also called a time spread or horizontal spread.

## Core Mechanic

The front-month option decays faster than the back-month. You earn the [[time-decay]] differential. Also benefits if back-month [[implied-volatility|IV]] increases relative to front-month.

```
P&L (at front-month expiry)
 ↑
 |           ╱╲
 |          ╱  ╲  max profit at strike (price pins)
 |         ╱    ╲
 |        ╱      ╲
─┼───────╱────────╲──────── Price →
 |──────╱          ╲────── max loss = net debit
 |    BE1    K     BE2
```

**Max profit:** Occurs if stock pins at the strike at front-month expiry (back-month retains value, front-month worthless)
**Max loss:** Net debit paid
**Breakevens:** Approximately strike +/- a range (depends on back-month IV)

## Greeks Profile

| Greek | Exposure | Meaning |
|-------|----------|---------|
| [[delta]] | Near zero | Delta-neutral at same strike |
| [[gamma]] | Mixed | Short front-month (high gamma) + long back-month (low gamma) |
| [[theta]] | Net positive | Front-month decays faster — this is the profit mechanism |
| [[vega]] | Net positive | Back-month has more [[vega]] than front-month — IV expansion helps |

## Trade Construction

1. Sell 1 near-term option (call or put) — **30 DTE or less**
2. Buy 1 far-term option at **same strike** — **60-90 DTE**
3. Use ATM strike for maximum [[theta]] differential
4. Close at front-month expiry or at 25-50% profit

| Parameter | Value |
|-----------|-------|
| Strike | ATM (same for both legs) |
| Front-month DTE | 21-30 days |
| Back-month DTE | 50-90 days |
| Target exit | 25-50% profit or at front-month expiry |

> [!tip] When to Use
> - Front-month [[implied-volatility|IV]] elevated vs back-month (IV term structure in backwardation)
> - Expecting stock to stay near current price short-term
> - Pre-event: sell the pre-event expiry, hold through the event with back-month
> - Lower risk than [[iron-condor]] with similar theta profile

> [!danger] Key Risk
> - Large move away from strike kills the position (both legs lose time value at the strike)
> - Front-month can be assigned if ITM near expiry
> - Back-month IV collapse hurts (you're long vega on the back leg)
> - More complex to manage than single-expiry spreads

## Data Pipeline

> [!info] Synesis Data
> | Need | Source | Method |
> |------|--------|--------|
> | Expirations (multiple) | yfinance | `get_options_expirations(ticker)` |
> | Chains at 2 different expiries | yfinance | `get_options_chain(ticker, exp1)` + `get_options_chain(ticker, exp2)` |
> | Greeks at both expiries | yfinance | `get_options_chain(ticker, exp, greeks=True)` |
> | IV comparison across expiries | yfinance | Compare ATM IV between front and back months |

---
**Related strategies:** [[iron-condor]] | [[butterfly-spread]] | [[long-straddle]]
**Concepts:** [[theta]] | [[vega]] | [[time-decay]] | [[implied-volatility]] | [[gamma]]
