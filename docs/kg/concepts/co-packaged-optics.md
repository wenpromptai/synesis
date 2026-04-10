---
up: []
related: ["[[concepts/indium-phosphide-substrates]]", "[[concepts/silicon-photonics]]", "[[concepts/optical-circuit-switch]]", "[[themes/optical-photonics-supply-chain]]", "[[themes/ai-infrastructure]]"]
created: 2026-04-10
type: concept
aliases: [CPO, Co-Packaged Optics]
tags: [concept, optics, cpo, networking, datacenter]
---

# Co-Packaged Optics (CPO)

Integration of optical engines directly into the switch ASIC package, eliminating the traditional pluggable transceiver module. Optical modulation happens on a [[concepts/silicon-photonics|silicon photonics]] chip adjacent to the switch die, fed by external [[concepts/indium-phosphide-substrates|InP]] continuous-wave (CW) lasers.

## Why It Matters

Traditional pluggable transceivers require long electrical traces between switch ASIC and optical engine → signal loss, power waste, latency. CPO collapses that gap, enabling:
- Higher bandwidth density per switch (51.2 Tb/s demonstrated)
- Lower power per bit (critical at datacenter scale)
- Reduced electromagnetic interference vs. copper

## Architecture (Pluggable vs. CPO)

| Attribute | Pluggable (current) | CPO (emerging) |
|-----------|-------------------|----------------|
| Laser type | EML (externally modulated) | CW (continuous wave) + SiPh modulator |
| Laser material | [[concepts/indium-phosphide-substrates\|InP]] | InP (CW source) + silicon (modulator) |
| Integration | Separate module in faceplate cage | On-substrate, inside switch package |
| Serviceability | Hot-swappable | Requires board-level replacement |
| Volume production | Now (800G/1.6T shipping) | 2027–2028 expected |

## Key Platforms (as of 2026-04-10)

- **NVIDIA Quantum-X800 InfiniBand**: 144 ports of 800G CPO, 63× signal-integrity gain vs. OSFP pluggable modules
- **Broadcom Bailly**: Tomahawk-5 ASIC + 8× 6.4 Tbps optical engines = 51.2 Tb/s total; 50K+ switches shipped in 2025
- **TSMC COUPE**: CPO platform verified for cloud customers (2026)
- **Coherent**: 6.4T (32×200G) socketed CPO demo at OFC 2026 using SiPh + InP CW lasers

## Market Sizing (2026-04-10 estimates)

| Source | TAM | Timeframe | CAGR |
|--------|-----|-----------|------|
| IDTechEx | >$20B | by 2036 | 37% |
| ResearchAndMarkets | $2.9B | by 2032 | 29.7% |
| Morgan Stanley | $90B (total AI optics) | by 2028 | — |
| CPO penetration | 4% → 30% | 2026 → 2029 | 153% |

## Adoption Timeline

- **2026**: Initial deployments, small volumes. First year of CPO technology introduction.
- **2027**: Manufacturing at high volume becomes feasible. Scale-up CPO volumes inflect Q4 2027.
- **2028+**: Volume production. Analysts expect scale-up CPO 3-4× bigger impact than scale-out.
- **>5 years**: CPO begins cannibalizing pluggables in scale-out applications.

> [!warning] CPO replaces copper first, not fiber
> Initial CPO deployments target copper interconnects inside racks, not existing fiber-based pluggable modules. Pluggable 800G/1.6T modules will dominate scale-out and scale-across networks for the remainder of this decade. CPO's impact on pluggable optics is nonexistent in the near term.

## Common Misconceptions

1. **"CPO kills pluggables"** — Not this decade. They develop in parallel; CPO is additive (incremental TAM), not replacement.
2. **"CPO eliminates InP dependency"** — False. Both CW lasers (CPO) and EMLs (pluggable) require InP substrates. CPO may actually increase total InP demand.
3. **"Any optics company benefits equally"** — The value chain shifts. EML laser makers (LITE) benefit from pluggable. CW laser + SiPh integration (COHR) benefits from CPO. Different winners at different phases.

> [!quote]- Sources
> - [[sources/cpo-sector-deep-dive-2026-04-10]] — sector research
> - IDTechEx CPO 2026-2036 report (2026-04-10)
> - OFC 2026 conference demonstrations (March 2026)
> - SemiAnalysis CPO deep dive (2026)
