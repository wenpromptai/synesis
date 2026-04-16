---
up: []
related: ["[[concepts/co-packaged-optics]]", "[[concepts/indium-phosphide-substrates]]", "[[tickers/COHR]]", "[[tickers/TSEM]]", "[[themes/optical-photonics-supply-chain]]"]
created: 2026-04-10
type: concept
aliases: [SiPh, Silicon Photonics]
tags: [concept, photonics, semiconductor, foundry]
---

# Silicon Photonics (SiPh)

Using CMOS-compatible silicon fabrication to integrate optical waveguides, modulators, and detectors on a silicon chip. Leverages existing semiconductor manufacturing infrastructure for high-volume, lower-cost optical components.

## Core Principle

Silicon is transparent at telecom wavelengths (1310 nm, 1550 nm) and can guide light through waveguides. However, silicon cannot efficiently emit light — it still requires an [[concepts/indium-phosphide-substrates|InP]] laser as the light source. The breakthrough is integrating everything *except* the laser onto a silicon chip.

## Role in CPO

In [[concepts/co-packaged-optics|CPO]] architectures, the silicon photonics chip sits on the switch substrate and handles modulation/detection. An external InP CW laser feeds light into the SiPh chip via fiber or waveguide coupling. This combines InP's laser efficiency with silicon's manufacturing scalability.

## Key Players (as of 2026-04-10)

| Company | Role | Evidence |
|---------|------|----------|
| [[tickers/COHR|Coherent]] | SiPh integration + InP CW lasers | 6.4T CPO demo at OFC 2026; NVIDIA Spectrum-X collaborator |
| [[tickers/TSEM|Tower Semi]] | SiPh foundry services | Revenue ~$52M in Q3 2025 (~70% YoY growth) |
| TSMC | COUPE CPO platform | Verified for cloud customers (2026) |
| Intel | Integrated photonics R&D | Potential disruptor if on-chip laser achieved |
| Broadcom | Bailly CPO switch SiPh integration | 50K+ Tomahawk-5 CPO switches shipped 2025 |

## Competitive Threat

If Intel, TSMC, or GlobalFoundries achieve a fully monolithic solution (laser + modulator + detector all on silicon), the InP substrate moat narrows significantly. This remains a long-term research challenge — no commercially viable on-chip silicon laser exists at datacenter scale as of 2026.

> [!quote]- Sources
> - [[sources/cpo-sector-deep-dive-2026-04-10]] — sector research
> - Coherent OFC 2026 CPO demonstrations (March 2026)
> - ExoSwan: "Top Silicon Photonics Stocks 2026" (2026)
