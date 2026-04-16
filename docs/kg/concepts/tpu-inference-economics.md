---
up: []
related: ["[[themes/custom-silicon-tpu-displacement]]", "[[themes/ai-infrastructure]]", "[[concepts/implied-volatility]]"]
created: 2026-04-14
type: concept
tags: [ai, inference, custom-silicon, tpu]
aliases: [inference economics, cost per token, TPU economics]
---

# TPU Inference Economics

> [!info] Definition
> The cost structure of running AI inference on custom silicon (Google TPUs, AWS Trainium) versus general-purpose GPUs, measured in cost-per-token, performance-per-dollar, or total cost of ownership (TCO). ^definition

## Overview

Inference now dominates AI compute budgets — accounting for roughly two-thirds of all AI compute in 2026 (up from ~50% in 2025), with inference infrastructure spending at $20.6B (55% of the $37.5B AI cloud infrastructure market). Over a model's lifetime, inference accounts for 80-90% of total compute costs. This shift makes inference economics the key battleground for AI infrastructure investment.

## Key Properties

1. **Compute-mix advantage:** Hyperscalers building custom silicon (TPUs, Trainium) gain structural cost advantages by eliminating NVIDIA's margin layer and optimizing silicon for their specific inference workloads
2. **Stack optimization matters more than raw silicon:** Google's internal inference stack is far more optimized for TPUs than open-source alternatives — real-world cost differences stem from vertical integration (custom silicon + custom software + at-scale deployment), not from chips alone
3. **Vendor claims diverge from independent benchmarks:** Google claims TPU v6e delivers 4x better price-performance than H100; independent testing (Artificial Analysis) showed NVIDIA B200/H100 at ~5x better tokens-per-dollar — the gap reflects stack optimization, not silicon capability

## Current Generation Comparison (as of Apr 2026)

| Platform | Peak Compute | Memory | TCO Estimate | Workload Fit |
|----------|-------------|--------|--------------|-------------|
| Google TPU v7 (Ironwood) | 4,614 TFLOPS | 192 GB HBM, 7.2 TB/s | ~44% lower than GB200 (SemiAnalysis) | Inference-first design |
| NVIDIA Blackwell Ultra | Record MLPerf inference | HBM3e | Premium pricing | Training + inference |
| AWS Trainium2 | — | — | ~50% less than H100 instances | AWS-native inference |
| AMD MI355X | Matching GB200 (ISSCC 2026) | HBM3e | CoWoS-gated supply | General AI workloads |

## Ironwood (TPU v7) — Inference-First Design

Google's latest TPU, generally available early 2026, is purpose-built for inference:
- 4,614 TFLOPS per chip (5x vs v6e's 918 TFLOPS)
- 192 GB HBM at 7.2 TB/s (6x memory vs v6e)
- 9,216-chip pod: 42.5 exaflops peak compute
- SemiAnalysis estimates all-in TCO per chip is roughly 44% lower than a GB200 server

## NVIDIA's Response: NVLink Fusion

Rather than competing purely on silicon, NVIDIA is becoming the **interconnect standard** for heterogeneous compute:
- **NVLink Fusion** (launched May 2025) lets hyperscalers plug custom ASICs into NVIDIA's NVLink fabric
- AWS Trainium4 will be the first to use it — creating hybrid GPU+ASIC racks
- **Vera Rubin** platform (shipping 2026-2027): 72 Rubin GPUs per rack via NVLink 6 at 3.6 TB/s per GPU
- Strategy: even if custom silicon takes inference share, NVIDIA retains ecosystem lock-in at the network layer

## Investment Implications

The TPU inference economics debate directly informs the bull/bear cases for:
- [[tickers/GOOG]] — Bull: compute-mix advantage makes AI margin-neutral/accretive. Bear: capex/depreciation ramp offsets unit-cost savings
- [[tickers/AVGO]] — Bull: custom ASIC design partner through 2031 (~3.5 GW Google TPU capacity from 2027). Bear: customer leverage compresses incremental margins
- [[tickers/NVDA]] — Bull: NVLink Fusion preserves ecosystem even in heterogeneous world. Bear: inference shift to custom silicon erodes GPU pricing power

> [!warning] Key Uncertainty
> Google did not submit TPU results to MLPerf inference rounds — making independent cost-per-token comparisons difficult. Most cited "advantages" come from vendor claims or limited third-party testing.

> [!quote]- Sources
> - [[sources/brief-2026-04-13]] — GOOG TPU inference debate, AVGO 2031 commitment
> - Google Cloud blog — Ironwood announcement (Apr 2026)
> - SemiAnalysis — "TPUv7: The 900lb Gorilla" — TCO analysis
> - Artificial Analysis — TPU v6e vs B200/H100 tokens-per-dollar benchmark
> - NVIDIA developer blog — Blackwell Ultra MLPerf Inference v5.1 records

---
**See also:** [[themes/custom-silicon-tpu-displacement]] | [[themes/ai-infrastructure]] | [[sources/connections/cowos-easing-paradox]]
