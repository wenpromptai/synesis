---
up: []
related: ["[[concepts/co-packaged-optics]]", "[[concepts/silicon-photonics]]", "[[tickers/LITE]]", "[[tickers/COHR]]", "[[tickers/AXTI]]", "[[themes/optical-photonics-supply-chain]]"]
created: 2026-04-10
type: concept
aliases: [InP, Indium Phosphide, InP Substrates]
tags: [concept, materials, semiconductor, optics, supply-chain]
---

# Indium Phosphide (InP) Substrates

The critical III-V compound semiconductor substrate for fabricating lasers used in datacenter optical interconnects. Every 800G and 1.6T transceiver — whether pluggable or [[concepts/co-packaged-optics|CPO]] — requires InP-based laser chips.

## Why It Matters

InP is the "silicon of photonics" — the base material on which EML and CW lasers are grown via epitaxy. No InP → no lasers → no optical transceivers → no AI datacenter networking at 800G+.

## Supply Chain Structure (as of 2026-04-10)

| Layer | Key Players | Bottleneck |
|-------|------------|-----------|
| **Substrate wafers** | AXT/Tongmei (~35%), Sumitomo, JX Nippon | China export controls on AXT |
| **Epitaxial growth** | Lumentum, Coherent, MACOM, Sumitomo | Capacity pre-allocated by NVIDIA |
| **Laser chip fab** | Lumentum (sole 200G EML volume), Coherent, Broadcom | Lead times >2027 |
| **Module assembly** | AAOI, Coherent, Fabrinet, Foxconn | Customer concentration risk |

## Critical Constraints

- **China export controls**: InP added to China export control list Feb 2025. AXT's Beijing Tongmei subsidiary requires Ministry of Commerce permit for each customer order. Processing time: ~60 business days. Q4 2025 revenue missed guidance due to fewer permits issued.
- **NVIDIA pre-allocation**: NVIDIA aggressively securing EML laser capacity at top suppliers, pushing lead times past 2027 and triggering industry-wide supply shortage (SDxCentral, March 2026).
- **Single-source risk**: Lumentum is currently the only supplier shipping 200G-per-lane EMLs at volume. Analysts expect double-digit price increases on 200G EMLs in 2026 due to no viable second source.
- **AXT capacity**: Doubling capacity through 2026, targeting $35M quarterly InP revenue run-rate. VGF-method 6-inch InP mass production achieved.

## EML vs. CW Laser (InP in both)

| Laser Type | Architecture | Used In | Key Supplier |
|-----------|-------------|---------|-------------|
| EML (externally modulated) | InP laser + InP modulator | 1.6T pluggable transceivers | [[tickers/LITE\|Lumentum]] (sole 200G volume) |
| CW (continuous wave) | InP laser (constant beam) | [[concepts/co-packaged-optics\|CPO]] external light source | [[tickers/COHR\|Coherent]], Lumentum |

> [!danger] Geopolitical chokepoint
> ~35% of global InP substrate supply flows through Beijing. If China restricts exports as trade leverage (mirroring gallium/germanium controls), the entire optical supply chain faces disruption regardless of downstream demand.

> [!quote]- Sources
> - [[sources/cpo-sector-deep-dive-2026-04-10]] — sector research
> - SDxCentral: "Nvidia's aggressive laser procurement spurs supply chain fears" (March 2026)
> - Semiconductor Today: AXT Q4/2025 export permit constraints (March 2026)
