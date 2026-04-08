---
up: []
related: ["[[bull-call-spread]]", "[[bear-put-spread]]", "[[covered-call]]"]
created: 2026-04-07
type: strategy
category: options
subcategory: directional
complexity: medium
data_source: [yfinance]
tags: [options, directional, synthetic]
---

# Risk Reversal

> [!abstract]
> Short OTM put + long OTM call (bullish) or vice versa. Near-zero cost synthetic directional exposure. Leveraged bet with undefined risk on the short leg.

## Core Mechanic (Bullish)

Sell an OTM put, use the premium to fund an OTM call. The result is a synthetic long position entered for near-zero net cost. Connected to [[put-call-parity]]: C - P ≈ S - K.

```
P&L (bullish risk reversal)
 ↑                           / unlimited upside (long call)
 |                          /
 |                         /
─┼────────────────────────/──── Price →
 |                       /
 | ────────────────────/  zero zone between strikes
 |/
 ↓  unlimited downside (short put)
   K1 (short put)     K2 (long call)
```

**Max profit:** Unlimited (stock rises above K2)
**Max loss:** K1 x 100 - premium (stock drops below K1, assigned on put)
**Breakeven:** Depends on net credit/debit; approximately K2 - net credit or K1 + net debit

## Greeks Profile

| Greek | Exposure | Meaning |
|-------|----------|---------|
| [[delta]] | Net positive (bullish) | Both legs contribute positive delta |
| [[gamma]] | Mixed | Long call gamma + short put negative gamma |
| [[theta]] | Near neutral | Short put theta offsets long call theta |
| [[vega]] | Net long | Benefits from [[volatility-smile\|skew]] — put IV > call IV makes this cheaper |

## Trade Construction (Bullish)

1. Sell 1 put — **[[delta]] 0.20-0.30**, 30-60 DTE
2. Buy 1 call — **[[delta]] 0.20-0.30**, same expiry
3. Target: net zero cost (put premium funds call)
4. Manage short put like a [[wheel-strategy]] entry — accept assignment if tested

| Parameter | Value |
|-----------|-------|
| Put [[delta]] | 0.20-0.30 (sell) |
| Call [[delta]] | 0.20-0.30 (buy) |
| DTE | 30-60 days |
| Net cost | ~$0 (put premium ≈ call premium) |

> [!tip] Skew Advantage
> [[volatility-smile|Skew]] in equity markets means OTM puts have higher IV than OTM calls. This makes bullish risk reversals cheaper — you sell expensive put IV and buy cheap call IV.

> [!danger] Key Risk
> - **Undefined downside risk** from short put — same risk as owning stock
> - Requires margin for the short put
> - If stock drops, you're assigned at the put strike

## Data Pipeline

> [!info] Synesis Data
> | Need | Source | Method |
> |------|--------|--------|
> | [[options-chain]] | yfinance | `get_options_chain(ticker, exp)` |
> | Greeks | yfinance | `get_options_chain(ticker, exp, greeks=True)` |
> | [[volatility-smile|Skew]] | yfinance | Compare IV across strikes in chain |

---
**Related strategies:** [[bull-call-spread]] | [[covered-call]] | [[wheel-strategy]]
**Concepts:** [[delta]] | [[put-call-parity]] | [[volatility-smile]] | [[moneyness]]
