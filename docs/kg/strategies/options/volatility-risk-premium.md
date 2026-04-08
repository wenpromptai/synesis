---
up: []
related: ["[[volatility-arbitrage]]", "[[iron-condor]]", "[[short-strangle]]", "[[covered-call]]"]
created: 2026-04-07
type: strategy
category: options
subcategory: systematic
complexity: advanced
data_source: [yfinance]
tags: [options, systematic, VRP, short-vol, premium-selling]
---

# Volatility Risk Premium

> [!abstract]
> Harvest the persistent gap between [[implied-volatility|IV]] and [[realized-volatility|RV]]. IV > RV ~80% of the time. This is the "insurance premium" that option sellers earn systematically.

## Core Mechanic

```
Implied Vol (VIX)     ╭──╮        ╭╮    ╭──╮
                     ╱    ╲      ╱  ╲  ╱    ╲
                    ╱      ╲    ╱    ╲╱      ╲
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  ← Realised vol
 VRP = gap = profit for sellers
```

**Why it exists:**
1. Hedgers overpay for protection (portfolio managers buy puts regardless of "fair" price)
2. Loss aversion — downside fear creates structural demand for puts
3. Natural supply-demand imbalance: more vol buyers than sellers

## Harvesting VRP (Options-Based)

### Strategy 1: Short SPY Strangles (Monthly)
- Sell 30-45 DTE, 10-15 [[delta]] put + call
- Collect premium; profit if SPY moves less than implied range
- Close at 50% profit; hard stop at 2x credit
- See [[short-strangle]]

### Strategy 2: Systematic [[covered-call|Covered Calls]]
- Buy SPY/QQQ, sell OTM calls monthly
- Captures 30-50% of VRP (call side only)
- Lowest risk VRP harvest

### Strategy 3: Put Spread Selling (Defined Risk)
- Sell 30 DTE put, buy further OTM put for protection
- Example: SPY at 500 → sell 490P / buy 480P for $3.00 credit, $7.00 max loss
- **Best risk-adjusted VRP harvest** for systematic use

## Regime Filter — THE MOST IMPORTANT PART

> [!danger] Never Harvest VRP Blindly
> | Condition | Action |
> |-----------|--------|
> | VIX < 20 AND normal term structure | **Full size** (75-100%) |
> | VIX 20-25 | **Half size** (50%) |
> | VIX > 25 | **No new short vol (0%)** |
> | VIX > 30 | Consider **buying** vol, not selling |

**Additional signals:**
- VIX 52-week percentile > 80% → reduce size
- Credit spreads widening (macro stress) → reduce
- Multiple consecutive -2% days → pause entirely

## Position Sizing

- **Win rate:** ~80%
- **Avg win:** 1-2% of notional
- **Avg loss:** 10-15% in stress events
- **Practical sizing:** 1-2% of portfolio per VRP trade (quarter-Kelly)
- **Total short vol book:** Never > 10-15% of portfolio NAV
- **Always use defined-risk strategies** (spreads, not naked shorts)

## Systematic Monthly Cycle (VIX < 20)

1. Day after monthly expiry → sell 30-45 DTE SPY strangle (10-15 delta)
2. Target: 0.5-1.0% of portfolio credit per month
3. Close at 50% profit (~15-20 days typical)
4. Close at 2x credit max loss
5. If VIX rises > 25 mid-cycle → close all positions
6. Repeat → 6-12% annualized in normal conditions

> [!danger] Historical Blowups
> | Event | VIX Move | Impact |
> |-------|----------|--------|
> | 2008 Financial Crisis | 60 → 80+ | -70%+ |
> | 2015 Flash Crash | 12 → 40 | -50%+ |
> | 2018 Volmageddon | 17 → 50 | Catastrophic |
> | 2020 COVID | 15 → 82 | -80%+ |
>
> **Pattern:** Low VIX + complacency → sudden spike → forced liquidation cascade. This is why sizing is tiny.

## Data Pipeline

> [!info] Synesis Data
> | Need | Source | Method |
> |------|--------|--------|
> | VIX level | yfinance | `get_quote("^VIX")` |
> | VIX history (regime) | yfinance | `get_history("^VIX", period="1y")` |
> | [[realized-volatility]] | yfinance | `get_options_snapshot(ticker)` → `.realized_vol_30d` |
> | [[implied-volatility]] | yfinance | `get_options_chain()` → ATM IV |
> | SPY/SPX chain | yfinance | `get_options_chain("SPY", exp, greeks=True)` |

---
**Related strategies:** [[volatility-arbitrage]] | [[iron-condor]] | [[short-strangle]] | [[covered-call]]
**Concepts:** [[implied-volatility]] | [[realized-volatility]] | [[iv-rank]] | [[vega]] | [[theta]]
