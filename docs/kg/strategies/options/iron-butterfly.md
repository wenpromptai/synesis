---
up: []
related: ["[[butterfly-spread]]", "[[iron-condor]]", "[[long-straddle]]"]
created: 2026-04-07
type: strategy
category: options
subcategory: volatility
complexity: medium
data_source: [yfinance]
tags: [options, volatility, pin, defined-risk, premium-selling]
---

# Iron Butterfly

> [!abstract]
> Short ATM straddle + long OTM wings. Higher credit than [[iron-condor]] but narrower profit zone. Max profit if stock pins at the center strike.

## Core Mechanic

Sell an ATM call and ATM put (straddle), buy OTM wings for protection. Essentially an [[iron-condor]] with the short strikes at the same level (ATM).

```
P&L
 ↑
 |              ╱╲
 |             ╱  ╲  max gain = net credit
 |            ╱    ╲
 |           ╱      ╲
─┼──────────╱────────╲────── Price →
 |─────────╱          ╲───── max loss = wing width - credit
 |       K1     K2      K3
 |      long  short AT  long
 |      put   put+call  call
```

**Max profit:** Net credit received (stock at K2 at expiry)
**Max loss:** Wing width - net credit
**Breakevens:** K2 - credit (down) / K2 + credit (up)

## Greeks Profile

| Greek | Exposure | Meaning |
|-------|----------|---------|
| [[delta]] | Zero | Short straddle is delta-neutral |
| [[gamma]] | Net negative (extreme) | Short ATM straddle = maximum short gamma |
| [[theta]] | Net positive (high) | Maximum theta collection of any defined-risk strategy |
| [[vega]] | Net negative (high) | Maximum IV sensitivity — IV expansion is devastating |

## Trade Construction

1. Sell 1 ATM put + Sell 1 ATM call (short straddle)
2. Buy 1 OTM put wing (5-10% below ATM)
3. Buy 1 OTM call wing (5-10% above ATM)
4. Same expiry: **21-45 DTE**

| Parameter | Value |
|-----------|-------|
| Short strikes | ATM |
| Wing width | 5-10% of underlying |
| DTE | 21-45 days |
| Target exit | 25-50% of credit |

> [!tip] Iron Butterfly vs Iron Condor
> | | Iron Butterfly | [[iron-condor]] |
> |---|---|---|
> | Credit | **Higher** | Lower |
> | Profit zone | **Narrower** | Wider |
> | Max gain | Higher | Lower |
> | Probability of profit | **Lower** | Higher |
> | Best for | High conviction of pin | Range-bound markets |

> [!danger] Key Risk
> - Narrow profit zone — any significant move erodes profits
> - Maximum short gamma of defined-risk strategies
> - IV expansion is the primary enemy (short ATM vega is maximum)

## Data Pipeline

> [!info] Synesis Data
> | Need | Source | Method |
> |------|--------|--------|
> | [[options-chain]] | yfinance | `get_options_chain(ticker, exp, greeks=True)` |
> | ATM strike identification | yfinance | `get_options_snapshot(ticker)` for spot |

---
**Related strategies:** [[butterfly-spread]] | [[iron-condor]] | [[long-straddle]]
**Concepts:** [[gamma]] | [[theta]] | [[vega]] | [[max-pain]] | [[breakeven]]
