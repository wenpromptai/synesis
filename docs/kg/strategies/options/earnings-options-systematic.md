---
up: []
related: ["[[long-straddle]]", "[[iron-condor]]", "[[iv-rank-strategy-selection]]"]
created: 2026-04-07
type: strategy
category: options
subcategory: systematic
complexity: medium
data_source: [yfinance]
tags: [options, systematic, earnings, event-driven]
---

# Earnings Options Systematic

> [!abstract]
> Systematic pre/post earnings volatility trades. Pre-earnings: buy straddles (IV expansion). Post-earnings: sell straddles (IV crush). Exploit the predictable IV cycle around earnings.

## The Earnings IV Cycle

```
IV
 ↑
 |                    ╱╲ ← earnings date
 |                   ╱  ╲
 |                  ╱    ╲ IV crush
 |                 ╱      ╲
 |               ╱         ╲────── post-earnings IV
 |  ─────── ───╱
 | baseline IV
 └──────────────────────────────── Time
    -30d    -14d  -7d  E  +1d  +7d
```

## Strategy 1: Pre-Earnings (Long Vol)

**Thesis:** IV expands into earnings as market prices in uncertainty. Buy options early, sell before the event.

1. Buy [[long-straddle]] or [[long-strangle]] — **14-21 days before earnings**
2. Use the expiration that includes the earnings date
3. **Exit 1-2 days BEFORE earnings** — capture IV expansion, avoid the binary event
4. Target: 15-30% profit from IV expansion alone

| Parameter | Value |
|-----------|-------|
| Entry | 14-21 days pre-earnings |
| Exit | 1-2 days pre-earnings (before event) |
| Strategy | [[long-straddle]] or [[long-strangle]] |
| Target | 15-30% of debit |

> [!tip] Key Insight
> You're NOT betting on the earnings outcome. You're betting that [[implied-volatility|IV]] will rise as the event approaches. Exit before the event to avoid the crush.

## Strategy 2: Post-Earnings (Short Vol)

**Thesis:** After earnings, IV collapses (crush) as uncertainty resolves. Sell options immediately after to capture the crush.

1. Wait for earnings report + market reaction
2. Sell [[iron-condor]] or [[short-strangle]] — **morning after earnings**
3. Use the nearest monthly expiry (30-45 DTE)
4. IV crush is immediate — close at 50% profit (often within days)

| Parameter | Value |
|-----------|-------|
| Entry | Morning after earnings report |
| Strategy | [[iron-condor]] or [[short-strangle]] |
| DTE | 30-45 days |
| Target exit | 50% of credit (often 3-7 days) |

> [!warning] Post-Earnings Risk
> If the stock gaps and continues trending, the short position can lose. Set hard stops at 2x credit. Avoid selling into the move — wait for the first session to settle.

## Screening Criteria

| Filter | Pre-Earnings | Post-Earnings |
|--------|-------------|---------------|
| [[iv-rank]] | < 50% (cheap options) | > 60% (still elevated post-report) |
| Typical IV expansion | > 20% rise into earnings | > 30% crush post-earnings |
| Stock behavior | History of big moves | History of mean-reversion post-event |
| [[open-interest]] | > 1,000 at ATM | > 1,000 at ATM |
| Bid-ask | < 5% of mid | < 5% of mid |

## Data Pipeline

> [!info] Synesis Data
> | Need | Source | Method |
> |------|--------|--------|
> | Earnings dates | NASDAQ API | Earnings calendar endpoint |
> | [[options-chain]] | yfinance | `get_options_chain(ticker, exp, greeks=True)` |
> | Historical IV (pre-earnings pattern) | yfinance | Track chain IV over time |
> | [[iv-rank]] | yfinance | Current vs 52-week IV range |

---
**Related strategies:** [[long-straddle]] | [[iron-condor]] | [[iv-rank-strategy-selection]] | [[calendar-spread]]
**Concepts:** [[implied-volatility]] | [[iv-rank]] | [[vega]] | [[theta]] | [[time-decay]]
