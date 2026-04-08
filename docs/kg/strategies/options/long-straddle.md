---
up: []
related: ["[[long-strangle]]", "[[iron-butterfly]]", "[[calendar-spread]]"]
created: 2026-04-07
type: strategy
category: options
subcategory: volatility
complexity: medium
data_source: [yfinance]
tags: [options, volatility, long-vol, event-driven]
---

# Long Straddle

> [!abstract]
> Long ATM call + long ATM put at the same strike and expiry. Profit from a big move in either direction. Classic pre-event volatility play.

## Core Mechanic

Buy both a call and put at the same ATM strike. You don't care which direction — you just need the stock to move more than the total premium paid.

```
P&L
 ↑
 |  \                             /
 |    \                         /
 |      \                     /
 |        \                 /
─┼──────────\─────────────/──────── Price →
 |            \         /
 |              ╲─────╱  max loss = total premium
 |                 K (ATM strike)
 |             BE1          BE2
```

**Max profit:** Unlimited (in either direction)
**Max loss:** Total premium paid (call + put)
**Breakevens:** K - total premium (downside) / K + total premium (upside)

## Greeks Profile

| Greek | Exposure | Meaning |
|-------|----------|---------|
| [[delta]] | Near zero | Delta-neutral at initiation (call + put cancel) |
| [[gamma]] | Net positive (maximum) | ATM options have highest [[gamma]] — delta adjusts favorably |
| [[theta]] | Net negative (maximum) | Paying maximum [[time-decay]] — the clock is your enemy |
| [[vega]] | Net positive (maximum) | IV expansion directly profits the position |

## When It Works

- Before known catalysts: earnings, FDA decisions, legal rulings
- [[iv-rank]] < 30% — options are cheap, IV likely to expand
- You expect a big move but don't know the direction
- Stock has been consolidating (compressed range → breakout)

## Trade Construction

1. Buy 1 ATM call + 1 ATM put — **same strike, same expiry**
2. DTE: **21-45 days** (enough time for the move, manageable decay)
3. For earnings plays: buy 7-14 days before event, close before/on event
4. Target: 25-50% of debit or close pre-event (capture IV expansion)

| Parameter | Value |
|-----------|-------|
| Strike | ATM (nearest to current price) |
| DTE | 21-45 days (or 7-14 days pre-event) |
| Target exit | 25-50% profit or pre-event close |
| Stop loss | Close at 50% of debit lost |

> [!danger] Key Risk
> - **[[time-decay]] is brutal** — you pay maximum theta daily
> - If the stock doesn't move enough, both legs decay to zero
> - Post-event IV crush can kill the position even if the stock moves (IV drop > directional gain)
> - Consider [[long-strangle]] for cheaper entry (but wider breakevens)

## Data Pipeline

> [!info] Synesis Data
> | Need | Source | Method |
> |------|--------|--------|
> | [[options-chain]] (ATM strikes) | yfinance | `get_options_chain(ticker, exp)` |
> | [[iv-rank]] (timing) | yfinance | VIX history |
> | [[implied-volatility|IV]] per contract | yfinance | chain `.implied_volatility` |
> | ATM snapshot | yfinance | `get_options_snapshot(ticker)` |

---
**Related strategies:** [[long-strangle]] | [[iron-butterfly]] | [[calendar-spread]] | [[earnings-options-systematic]]
**Concepts:** [[gamma]] | [[theta]] | [[vega]] | [[implied-volatility]] | [[breakeven]] | [[time-decay]]
