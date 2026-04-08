---
up: []
related: ["[[covered-put]]", "[[wheel-strategy]]", "[[protective-put]]", "[[short-strangle]]", "[[position-repair]]"]
created: 2026-04-07
type: strategy
category: options
subcategory: income
complexity: simple
data_source: [yfinance]
tags: [options, income, theta, premium-selling]
---

# Covered Call

> [!abstract]
> Long stock + short OTM call. Collect premium now, cap upside at the strike. The simplest options income strategy.

## Core Mechanic

Sell a call against shares you own. You earn premium (income) but give up gains above the strike price. Equivalent to a short put at the same strike via [[put-call-parity]].

```
P&L
 ↑              ────────────── capped gain = (K - cost) + premium
 |             /
 |            /
─┼───────────/──────────────── Price →
 |          /
 |─────────  max loss = cost - premium (if stock → 0)
 |   cost      K (short call strike)
```

**Max profit:** (Strike - stock cost) + premium received
**Max loss:** Stock cost - premium (stock goes to zero)
**Breakeven:** Stock cost - premium received

## Greeks Profile

| Greek | Exposure | Meaning |
|-------|----------|---------|
| [[delta]] | Net positive (~0.65-0.75) | Long stock delta minus short call delta |
| [[gamma]] | Net negative | Short call creates negative gamma; large upside moves cap out |
| [[theta]] | Net positive | Short call premium decays in your favor — core income mechanism |
| [[vega]] | Net negative | IV increase hurts (short call costs more to close) |

## When It Works

- Sideways to mildly bullish market
- **[[iv-rank]] > 50%** — premium is rich enough to be worth capping upside
- You want income on a position you plan to hold anyway
- Low [[realized-volatility|realized vol]] expected vs current [[implied-volatility|IV]]

## Trade Construction

1. Hold long equity (or buy the stock)
2. Sell 1 call per 100 shares — **30-45 DTE, [[delta]] 0.25-0.35** (~15-20% OTM)
3. Close at 50% of premium collected
4. Roll up/out if stock rallies within 1 strike of short call
5. Repeat monthly

| Parameter | Value |
|-----------|-------|
| DTE | 30-45 days |
| Short call [[delta]] | 0.25-0.35 |
| Target exit | 50% of premium |
| Roll trigger | Stock within 1 strike of short |

### Screener Criteria

| Filter | Threshold |
|--------|-----------|
| [[iv-rank]] | > 50% |
| DTE | 30-45 days |
| Short call delta | 0.25-0.35 |
| Stock trend | Neutral to mildly bullish |
| Earnings within DTE | No |
| Annualized premium yield | > 12% |
| Bid-ask spread | < 5% of mid |

### Sizing

Overlay calls on 50-100% of equity position. 100% maximizes income but eliminates all upside. 50% overlay balances income and participation.

> [!danger] Key Risk
> - **Caps upside** in strong rallies — you participate only to the strike
> - Gap-up risk: stock jumps above strike before you can adjust
> - Assignment risk if deep ITM near expiry
> - Wrong strategy if very bullish — use [[bull-call-spread]] instead

## Data Pipeline

> [!info] Synesis Data
> | Need | Source | Method |
> |------|--------|--------|
> | [[options-chain]] | yfinance | `get_options_chain(ticker, exp)` |
> | [[delta]] for strike selection | yfinance | `get_options_chain(ticker, exp, greeks=True)` |
> | [[iv-rank]] | yfinance | Compute from VIX history or chain IV history |
> | Stock price | yfinance | `get_quote(ticker)` |

---
**Related strategies:** [[covered-put]] | [[wheel-strategy]] | [[protective-put]] | [[short-strangle]]
**Concepts:** [[delta]] | [[theta]] | [[implied-volatility]] | [[iv-rank]] | [[time-decay]] | [[put-call-parity]]
