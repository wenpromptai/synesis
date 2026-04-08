---
up: []
related: ["[[iron-condor]]", "[[long-strangle]]", "[[covered-call]]", "[[volatility-risk-premium]]"]
created: 2026-04-07
type: strategy
category: options
subcategory: income
complexity: medium
data_source: [yfinance]
tags: [options, income, theta, premium-selling, undefined-risk]
---

# Short Strangle

> [!abstract]
> Short OTM put + short OTM call on the same underlying. Highest premium income strategy but undefined risk on both sides.

## Core Mechanic

Sell both an OTM put and an OTM call. Profit if the stock stays between the two strikes. You collect premium from both sides.

```
P&L
 ↑
 |          ┌────────────────┐  max gain = total credit
 |         /                  \
─┼────────/────────────────────\────── Price →
 |───────/                      \────  max loss = unlimited
 |     K1 (short put)      K2 (short call)
 ↓
```

**Max profit:** Total credit received (price between K1 and K2 at expiry)
**Max loss:** Unlimited (price moves far beyond either strike)
**Breakevens:** K1 - credit (downside) / K2 + credit (upside)

## Greeks Profile

| Greek | Exposure | Meaning |
|-------|----------|---------|
| [[delta]] | Near zero | Delta-neutral at initiation |
| [[gamma]] | Net negative | Large moves in either direction hurt |
| [[theta]] | Net positive | Both legs decay — highest theta of income strategies |
| [[vega]] | Net negative | IV expansion is the primary enemy |

## Trade Construction

1. Sell 1 OTM put — **[[delta]] ~0.16** (1 SD OTM)
2. Sell 1 OTM call — **[[delta]] ~0.16** (1 SD OTM)
3. Same expiry: **30-45 DTE**
4. Close at 50% of total credit
5. Close if either leg's delta exceeds 0.30 (tested)
6. Hard stop: 2x credit received

| Parameter | Value |
|-----------|-------|
| DTE | 30-45 days |
| Short leg [[delta]] | 0.16 each side |
| Target exit | 50% of credit |
| Adjustment trigger | Delta > 0.30 on either leg |

### Screener Criteria

| Filter | Threshold |
|--------|-----------|
| [[iv-rank]] | > 50% (ideally > 60%) |
| DTE | 30-45 days |
| Short leg delta | 0.15-0.20 |
| Earnings within DTE | No |
| Underlying | High liquidity (SPY, QQQ, blue chips) |

### Sizing

Margin requirement = max(put side, call side) notional. Target max loss per strangle at 2-3% of portfolio. The undefined risk means sizing must be conservative.

> [!danger] Key Risk
> - **Unlimited risk** on both sides — a crash or squeeze can create catastrophic loss
> - Requires margin; margin calls possible during stress
> - Psychologically difficult during drawdowns
> - Consider [[iron-condor]] for defined-risk alternative with similar but smaller payoff

## Data Pipeline

> [!info] Synesis Data
> | Need | Source | Method |
> |------|--------|--------|
> | [[options-chain]] | yfinance | `get_options_chain(ticker, exp)` |
> | [[delta]] for strike selection | yfinance | `get_options_chain(ticker, exp, greeks=True)` |
> | [[iv-rank]] | yfinance | VIX history or chain IV history |

---
**Related strategies:** [[iron-condor]] | [[long-strangle]] | [[covered-call]] | [[volatility-risk-premium]]
**Concepts:** [[delta]] | [[theta]] | [[vega]] | [[implied-volatility]] | [[iv-rank]] | [[breakeven]]
