---
up: []
related: ["[[iron-butterfly]]", "[[iron-condor]]", "[[calendar-spread]]"]
created: 2026-04-07
type: strategy
category: options
subcategory: volatility
complexity: medium
data_source: [yfinance]
tags: [options, volatility, pin, defined-risk]
---

# Butterfly Spread

> [!abstract]
> Long 1 lower wing + short 2 middle body + long 1 upper wing (all calls or all puts). Low-cost bet that the stock pins at the body strike. High reward-to-risk ratio.

## Core Mechanic

Three strikes, same expiry. The body (2 short) is your profit target. Wings (1 long each side) define max loss. Very cheap to enter, big payout if stock pins.

```
P&L
 ↑
 |              ╱╲
 |             ╱  ╲  max gain = body width - debit
 |            ╱    ╲
 |           ╱      ╲
─┼──────────╱────────╲────── Price →
 |─────────╱          ╲───── max loss = net debit
 |       K1     K2      K3
 |      long   2x short  long
```

**Max profit:** (K2 - K1) - net debit (at body strike K2)
**Max loss:** Net debit paid (small)
**Breakevens:** K1 + debit / K3 - debit

## Greeks Profile

| Greek | Exposure | Meaning |
|-------|----------|---------|
| [[delta]] | Near zero | Delta-neutral at body |
| [[gamma]] | Mixed/negative near expiry | Short gamma at body, long at wings |
| [[theta]] | Net positive near expiry | Benefits from time decay if near body |
| [[vega]] | Net negative | IV expansion hurts (makes pinning less likely) |

## Trade Construction

1. Buy 1 lower call (or put) — wing
2. Sell 2 middle calls (or puts) — body
3. Buy 1 upper call (or put) — wing
4. **Equal spacing:** K2 - K1 = K3 - K2
5. DTE: **14-30 days** (needs time pressure for pinning)

| Parameter | Value |
|-----------|-------|
| Body strike | Target pin price or [[max-pain]] level |
| Wing width | $2-5 (narrow = cheaper, higher max gain/risk ratio) |
| DTE | 14-30 days |
| Target exit | 50-100% of max profit |

> [!tip] Targeting
> Center the body at [[max-pain]] for highest probability of pinning. Works best on quiet expiration days with no catalyst.

> [!danger] Key Risk
> - Stock must pin near the body — any move away from center loses
> - Low probability of max profit
> - Difficult to exit cleanly (3 legs, potential liquidity issues on inner legs)
> - See [[iron-butterfly]] for a credit-based alternative with similar profile

## Data Pipeline

> [!info] Synesis Data
> | Need | Source | Method |
> |------|--------|--------|
> | [[options-chain]] | yfinance | `get_options_chain(ticker, exp, greeks=True)` |
> | [[max-pain]] | yfinance | Compute from [[open-interest]] across chain |
> | 3 strikes with liquidity | yfinance | Filter by OI > 100 per strike |

---
**Related strategies:** [[iron-butterfly]] | [[iron-condor]] | [[calendar-spread]]
**Concepts:** [[gamma]] | [[theta]] | [[max-pain]] | [[open-interest]] | [[breakeven]]
