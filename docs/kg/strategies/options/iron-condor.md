---
up: []
related: ["[[short-strangle]]", "[[iron-butterfly]]", "[[butterfly-spread]]", "[[calendar-spread]]"]
created: 2026-04-07
type: strategy
category: options
subcategory: income
complexity: medium
data_source: [yfinance]
tags: [options, income, theta, defined-risk, premium-selling]
---

# Iron Condor

> [!abstract]
> Bull put spread + bear call spread on the same underlying and expiry. Defined risk on both sides. Profit if price stays inside the wings.

## Core Mechanic

Four legs: buy OTM put (wing), sell closer put (credit), sell OTM call (credit), buy further OTM call (wing). Collect net credit. Defined max loss = spread width - credit.

```
P&L
 ↑
 |        ┌──────────────────┐  max gain = net credit
 |       /                    \
─┼──────/──────────────────────\────── Price →
 |─────/                        \──── max loss = width - credit
 |   K1     K2              K3     K4
 | long    short            short  long
 | put     put              call   call
```

**Max profit:** Net credit (price between K2 and K3)
**Max loss:** Spread width - credit (price outside K1 or K4)
**Breakevens:** K2 - credit (downside) / K3 + credit (upside)

## Greeks Profile

| Greek | Exposure | Meaning |
|-------|----------|---------|
| [[delta]] | Near zero | Delta-neutral at initiation |
| [[gamma]] | Net negative | Short gamma on both sides — large moves hurt |
| [[theta]] | Net positive | Time decay is primary income source |
| [[vega]] | Net negative | IV expansion is the primary enemy |

## When It Works

- **Post-event** when IV crush is expected
- [[iv-rank]] > 50-60%: premium is rich
- Range-bound markets, no major upcoming catalyst
- SPY/QQQ monthly condors — one of the most widely-deployed income strategies

**IV Skew note:** Steep put skew means the bull put spread collects more credit than the bear call spread for the same delta. Consider a skew-adjusted condor: put spread closer to ATM (more credit), call spread further OTM.

## Trade Construction

**Downside:** Sell 0.16-delta put, buy 0.05-delta put (bull put spread)
**Upside:** Sell 0.16-delta call, buy 0.05-delta call (bear call spread)
Same expiry: **30-45 DTE**

| Rule | Action |
|------|--------|
| Profit target | Close at **50% of credit** |
| Tested leg | Close/roll if short leg delta > 0.30 |
| Hard stop | Close full condor at 2x credit |
| Time stop | Close at 21 DTE if not at target |

### Screener Criteria

| Filter | Threshold |
|--------|-----------|
| [[iv-rank]] | > 50% (ideally > 60%) |
| DTE | 30-45 days |
| Short leg [[delta]] | 0.15-0.20 each side |
| Wing width | 5-10% of underlying per spread |
| Expected move (1 SD) | Short strikes at 1.0-1.25x expected move |
| Earnings within DTE | No |
| Bid-ask per spread | < 10% of credit collected |
| [[open-interest]] | > 100 per strike |

### Sizing

Risk per condor = spread width - credit. On a 10-wide condor receiving $2.00, max loss = $800. Target max loss per condor at 1% of portfolio. Spread across 2-3 underlyings. Target total portfolio [[theta]] from condors: 0.1-0.2% of portfolio value per day.

> [!danger] Key Risk
> - Both spreads can lose in a big move
> - Total max loss = (spread width x 2) - total credit
> - Assignment risk on short legs near expiry
> - Gap events blow through wings

## Data Pipeline

> [!info] Synesis Data
> | Need | Source | Method |
> |------|--------|--------|
> | [[options-chain]] | yfinance | `get_options_chain(ticker, exp)` |
> | Greeks for all 4 legs | yfinance | `get_options_chain(ticker, exp, greeks=True)` |
> | [[iv-rank]] | yfinance | VIX history |
> | [[volatility-smile|Skew]] assessment | yfinance | IV across strikes from chain |

---
**Related strategies:** [[short-strangle]] | [[iron-butterfly]] | [[butterfly-spread]] | [[calendar-spread]]
**Concepts:** [[delta]] | [[gamma]] | [[theta]] | [[vega]] | [[iv-rank]] | [[volatility-smile]] | [[breakeven]]
