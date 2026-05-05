---
up: []
related: ["[[tickers/NEM]]", "[[themes/geopolitical-energy-risk]]", "[[concepts/momentum]]"]
created: 2026-05-05
type: concept
tags: [concept, cftc, futures, positioning, contrarian]
---

# CFTC COT Positioning Percentile

The CFTC publishes weekly Commitment of Traders (COT) reports every Friday (data as of Tuesday) showing aggregate futures positions across three participant categories:

- **Commercial hedgers** — producers/consumers using futures to offset business risk; "smart money" on fundamentals.
- **Large speculators** — hedge funds, CTAs, managed futures; trend-following and momentum-driven. This is the actionable contrarian leg.
- **Small speculators** — retail; non-reportable, least informative.

## Positioning Percentile Calculation

For a given contract (e.g., GC gold futures), rank the current speculator net position (longs minus shorts) within its historical lookback window (typically 1–3 years):

```
Percentile = (current_net − min_net) / (max_net − min_net) × 100
```

- **0th percentile** = speculators are as net-short as they've been in the lookback window.
- **100th percentile** = speculators are as net-long as they've been.

## Contrarian Signal Framework

Speculators are momentum-chasers — they pile into trends and are frequently wrong at extremes:

| Percentile | Interpretation | Signal |
|------------|----------------|--------|
| 0–20th | Crowded short — speculators have capitulated | Long bias; any bullish catalyst triggers short-covering squeeze |
| 20–80th | Neutral zone | No positioning edge |
| 80–100th | Crowded long — speculative longs stretched | Short bias or reduce longs |

For gold (GC), a 0th-percentile reading historically precedes sharp rallies as longs re-enter and shorts cover simultaneously.

## In This KG

- As of the **2026-05-01 through 2026-05-04 briefs**: GC at **0th percentile** (crowded short), ES at 28th, NQ at 25th.
- [[tickers/NEM]] is the actionable vehicle for the GC contrarian setup — mining operating leverage amplifies spot gold upside.
- GC positioning interacts with [[themes/geopolitical-energy-risk]] — each Iran/Hormuz escalation spike bids gold, forcing shorts to cover.

## Caveats

- **Trend can persist** — Extreme positioning can stay extreme for months in a strong macro trend; this is a setup identifier, not a timing signal alone.
- **Lookback sensitivity** — A 1-year window vs a 5-year window can yield very different percentiles for the same absolute position.
- **Combine with catalyst** — Pair with price structure or a macro trigger; positioning alone does not determine when the squeeze fires.
- **Report lag** — Data is 3 days stale on release (Tuesday cut-off, Friday publish); fast-moving markets can render the reading obsolete.
- **Commercial inversion** — Commercials being extremely net-short is often bullish (locking in high prices); the speculator leg is the actionable side.

**See also:** [[tickers/NEM]] | [[themes/geopolitical-energy-risk]] | [[concepts/momentum]]
