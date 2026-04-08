---
up: []
related: ["[[options-chain]]", "[[max-pain]]", "[[moneyness]]"]
created: 2026-04-07
type: concept
tags: [options, data, liquidity]
aliases: [OI, open interest]
---

# Open Interest

> [!info] Definition
> The total number of outstanding (open) option contracts at a given strike and expiration. Increases when new positions are opened, decreases when positions are closed. ^definition

## Key Properties

1. **Liquidity proxy:** High OI = tight bid-ask spreads, easier fills
2. **Not volume:** Volume = contracts traded today; OI = total open positions
3. **Directional signal:** Rising OI + rising price = new longs entering (bullish); Rising OI + falling price = new shorts entering (bearish)
4. **Pinning effect:** High OI at a strike can cause price to gravitate toward it near expiry — see [[max-pain]]

## Minimum OI Thresholds

| Strategy | Minimum OI per leg |
|----------|-------------------|
| [[covered-call]] | > 500 |
| [[iron-condor]] | > 100 per strike |
| [[zero-dte]] | > 1,000 (SPX/SPY) |
| Any spread | > 100 per strike |

## In Practice

- **Liquidity filter:** All strategies should check OI before entry — low OI = wide spreads = slippage
- **[[max-pain]] calculation:** Sum OI across all strikes to find the pin price
- **[[zero-dte]]:** Massive OI at round-number SPX strikes creates pinning behavior
- **Unusual OI:** Sudden OI spike at a specific strike can signal institutional positioning

## Data Source

> [!info] Synesis Pipeline
> yfinance `get_options_chain(ticker, expiration)` returns `open_interest` per contract.

---
**See also:** [[options-chain]] | [[max-pain]] | [[moneyness]]
