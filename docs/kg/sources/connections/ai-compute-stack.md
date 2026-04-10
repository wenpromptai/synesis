---
up: []
related: ["[[themes/ai-infrastructure]]", "[[themes/customer-concentration]]"]
created: 2026-04-10
type: connection
nodes: ["[[tickers/NVDA]]", "[[tickers/AVGO]]", "[[tickers/TSM]]", "[[tickers/MRVL]]", "[[tickers/ARM]]"]
discovered_from: kg-lint
tags: [connection, ai, semiconductor, supply-chain]
---

# AI Compute Stack — Core Silicon Supply Chain

> [!abstract]
> Five companies form the critical path of AI compute silicon: NVDA designs GPUs, AVGO designs custom ASICs (e.g. Google TPU), ARM licenses CPU IP, MRVL provides custom silicon + interconnect, and TSM fabricates them all. A disruption or demand slowdown at any point in this stack affects the entire chain. Conversely, all five benefit from the same hyperscaler capex cycle.

## Nodes Involved

| Ticker | Role in Stack | Key Risk |
|--------|--------------|----------|
| [[tickers/NVDA]] | GPU compute — dominant AI training/inference silicon | Customer concentration (~42% one distributor), $95.2B supply commitments |
| [[tickers/AVGO]] | Custom ASIC (TPU) + networking + VMware software | Distributor concentration |
| [[tickers/TSM]] | Foundry — fabricates chips for NVDA, AVGO, ARM, MRVL | Geopolitical (Taiwan), stretched valuation through earnings |
| [[tickers/MRVL]] | Custom silicon + AI interconnect (Celestial AI, XConn) | Newer AI revenue, still proving scale |
| [[tickers/ARM]] | CPU/IP licensing for AI-edge compute; royalty model | Valuation premium on AI narrative |

## Why This Pattern Matters

1. **Correlated demand exposure**: All five derive growth from the same hyperscaler capex cycle (MSFT, GOOG, META, AMZN). If any major hyperscaler pauses or slows capex, the entire stack reprices.
2. **Supply chain dependency**: TSM is the sole advanced-node foundry for the group. ARM IP is embedded in most AI-adjacent chips. The stack is vertically interdependent.
3. **Valuation clustering**: All five trade at significant premiums to historical multiples, pricing in sustained AI capex growth through 2027-2028.
4. **Brief's approach**: The 2026-04-10 Trader used defined-risk call spreads across NVDA, MRVL, and SMH (which contains all five) rather than outright equity — reflecting conviction in the theme but caution on valuation/macro.

## Implication

Size positions in this cluster as a single correlated bet. Owning call spreads in NVDA + MRVL + SMH is effectively triple-counting AI compute exposure. The brief's sector tilt framework should account for aggregate AI-compute-stack exposure across all portfolio positions.

> [!quote]- Sources
> - [[sources/brief-2026-04-10]] — all five analyzed with bull/bear debate

---
**See also:** [[themes/ai-infrastructure]] | [[themes/customer-concentration]] | [[sources/connections/ai-optics-concentration-wcap]]
