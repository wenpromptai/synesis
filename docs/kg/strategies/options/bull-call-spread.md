---
up: []
related: ["[[bear-put-spread]]", "[[risk-reversal]]", "[[covered-call]]"]
created: 2026-04-07
type: strategy
category: options
subcategory: directional
complexity: simple
data_source: [yfinance]
tags: [options, directional, bullish, defined-risk]
---

# Bull Call Spread

> [!abstract]
> Long lower-strike call + short higher-strike call. Defined-risk bullish bet with capped profit and capped loss.

## Core Mechanic

Buy a call, sell a higher-strike call at the same expiry. Net debit position. Profit if stock rises above the long call strike.

```
P&L
 ↑              ────────────── max gain = width - debit
 |             /
 |            /
─┼───────────/──────────────── Price →
 |──────────   max loss = debit paid
 |         K1 (long)    K2 (short)
```

**Max profit:** (K2 - K1) - net debit
**Max loss:** Net debit paid
**Breakeven:** K1 + net debit

## Greeks Profile

| Greek | Exposure | Meaning |
|-------|----------|---------|
| [[delta]] | Net positive | Profits from upward move |
| [[gamma]] | Net positive (small) | Less gamma than naked call — spread dampens it |
| [[theta]] | Net negative | Time works against you (debit spread) |
| [[vega]] | Net positive (small) | Slight IV benefit, less than naked call |

## Trade Construction

1. Buy 1 call at or slightly ITM — **[[delta]] 0.55-0.65**
2. Sell 1 call at target price — **[[delta]] 0.30-0.40**
3. Same expiry: **30-60 DTE**
4. Risk/reward: typically 1:1 to 1:2 (risk debit, gain width minus debit)

| Parameter | Value |
|-----------|-------|
| Long call [[delta]] | 0.55-0.65 (ATM or slightly ITM) |
| Short call [[delta]] | 0.30-0.40 |
| DTE | 30-60 days |
| Target exit | 50-75% of max profit |

### Screener Criteria

| Filter | Threshold |
|--------|-----------|
| [[iv-rank]] | < 50% (options not overpriced) |
| Bullish bias | Confirmed by trend/thesis |
| Spread width | Match to expected move |
| Bid-ask | < 10% of spread width |

> [!danger] Key Risk
> - Max loss = entire debit if stock stays below K1 at expiry
> - [[time-decay]] works against you — close losing trades before expiry
> - Capped profit — if very bullish, naked calls have more upside (but more risk)

## Data Pipeline

> [!info] Synesis Data
> | Need | Source | Method |
> |------|--------|--------|
> | [[options-chain]] | yfinance | `get_options_chain(ticker, exp)` |
> | Greeks | yfinance | `get_options_chain(ticker, exp, greeks=True)` |

---
**Related strategies:** [[bear-put-spread]] | [[risk-reversal]] | [[leaps-strategy]]
**Concepts:** [[delta]] | [[theta]] | [[vega]] | [[breakeven]] | [[moneyness]]
