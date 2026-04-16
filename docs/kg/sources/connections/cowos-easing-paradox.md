---
up: []
related: ["[[themes/ai-infrastructure]]", "[[themes/custom-silicon-tpu-displacement]]"]
created: 2026-04-14
type: connection
nodes: ["[[tickers/AMD]]", "[[tickers/NVDA]]", "[[tickers/AVGO]]", "[[tickers/TSM]]"]
discovered_from: compilation
tags: [connection, semiconductor, cowos, packaging]
---

# CoWoS Capacity Easing: Bullish or Bearish for Semis?

> [!abstract]
> The same supply-chain event (CoWoS advanced packaging capacity easing) is simultaneously the bull case for AMD and the bear case for NVIDIA's pricing power — creating a paradox where capacity expansion can be either bullish or bearish depending on which stock you're analyzing.

## Nodes Involved

- [[tickers/AMD]] — Bull: CoWoS easing converts pent-up demand into visible shipments and operating leverage. Bear: easing intensifies competition, erodes pricing power
- [[tickers/NVDA]] — Bull: delays extend Blackwell scarcity and pricing power. Bear: easing removes scarcity premium, heterogeneous compute (NVLink Fusion) commoditizes parts of the stack
- [[tickers/AVGO]] — Custom ASIC demand grows as CoWoS capacity expands to serve TPU/ASIC programs alongside GPUs
- [[tickers/TSM]] — Direct beneficiary of CoWoS capacity buildout regardless of who wins the allocation

## Evidence

**Capacity data (Morgan Stanley, Apr 2026):**
- TSMC CoWoS scaling from ~75-80K wafers/month (late 2025) to 130-150K wafers/month by end 2026
- 2026 total CoWoS demand: ~1M wafers (+49% YoY) — demand growing faster than supply
- **Allocation:** NVDA ~595K (60%), AVGO ~150K (15%), AMD ~105K (11%)
- AMD's 2026 allocation is **double** its 2025 level, but still dwarfed by NVDA

**The paradox:**
1. **For AMD (bull):** Doubling CoWoS allocation from 2025 → 2026 unlocks MI355/MI400 ramp. Revenue gated by packaging, not demand. Easing converts pent-up hyperscaler demand into shipments → operating leverage on 52.5% gross margin. AMD at ISSCC 2026: MI355X "matching GB200 performance."
2. **For AMD (bear):** If CoWoS eases for everyone, NVDA gets more too (595K wafers). AMD can't out-supply NVDA. Pricing power erodes as scarcity premium fades. 58.3x EV/EBITDA leaves no room for execution friction.
3. **For NVDA (bull):** Even with easing, NVDA has 60% of CoWoS allocation. Blackwell stays scarce through 2026 as Rubin ramps behind it. Scarcity is structural, not temporary.
4. **For NVDA (bear):** NVLink Fusion explicitly enables heterogeneous compute — once CoWoS frees up custom ASIC production (AVGO for Google TPU, etc.), alternative silicon paths become viable. Pricing power compression as TAM becomes multi-vendor.

**Key insight:** CoWoS easing is unambiguously bullish for **TSM** (more wafers at premium pricing) and **AVGO** (custom ASIC programs get capacity). For GPU companies, it's a zero-sum reallocation game where the question is whether volume growth offsets pricing normalization.

## Implications

- When analyzing semiconductor names, ask: "Does this company benefit from scarcity or from abundance?" TSM and AVGO benefit from abundance (more volume). NVDA benefits from scarcity (pricing power). AMD is ambiguous (needs volume but can't compete on pricing).
- Supply may catch up by late 2026 (~1.56M annualized capacity vs ~1M demand), but new products (Rubin, MI400) could re-tighten. Watch for TSMC's quarterly guidance on CoWoS utilization.
- The debate will recur in every earnings cycle for these names through at least 2027.

> [!quote]- Sources
> - [[sources/brief-2026-04-13]] — AMD/NVDA CoWoS debates
> - Morgan Stanley CoWoS capacity analysis (Charlie Chan, Apr 2026) — allocation data
> - TrendForce — TSMC CoWoS booking status

---
**See also:** [[themes/ai-infrastructure]] | [[themes/custom-silicon-tpu-displacement]] | [[tickers/TSM]]
