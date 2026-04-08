---
up: []
raw_file: "raw/multibagger-stocks.md"
created: 2026-04-07
type: source
tags: [academic-paper, equity, factor-investing, multibagger]
---

# Yartseva (2025) — The Alchemy of Multibagger Stocks

> [!info] Source
> **Title:** The Alchemy of Multibagger Stocks: Understanding the Drivers of Stock Market Outperformance
> **Authors:** Viktoriya Yartseva
> **Year:** 2025
> **Type:** Academic paper
> **Raw file:** `raw/multibagger-stocks.md`

## Summary

Comprehensive empirical study of 464 "enduring multibagger" stocks (10x+ return) on NYSE/NASDAQ from 2009-2024 (11,600 company-year observations). Tests the Fama-French five-factor model and extends it with novel explanatory variables using panel regression (static and dynamic), general-to-specific modelling, and GMM estimators. Out-of-sample validated on 2023-2024 data.

## Key Findings

1. **[[fcf-yield]] is the most important driver** of multibagger returns — highest coefficients across all model specifications
2. **[[size-effect]] confirmed:** Small caps outperform; median starting market cap $348M
3. **[[value-factor]] plays the biggest role** overall — B/M and FCF/P both strongly significant with positive signs
4. **[[profitability-factor]] confirmed:** EBITDA margin (static) and ROA (dynamic) are significant; negative profitability is the strongest red flag
5. **[[investment-pattern]] reversed from FF5:** Aggressive investment is good, but asset growth must not exceed EBITDA growth (unique finding)
6. **[[momentum]] is complex:** Only 1-month momentum is positive; 3-6 month momentum is negative (reversal); entry near 12-month low is optimal
7. **Interest rates matter:** Rising Fed rate depresses multibagger returns by ~10pp
8. **Earnings growth is NOT predictive:** Despite popular belief, all forms of earnings growth (EPS, revenue, EBITDA CAGR) are statistically insignificant
9. **P/E is unreliable:** Breaks for loss-makers, distorts other coefficients — FCF/P and B/M are superior
10. **Standard FF5 fails:** Large unexplained intercept (α=83.9) — additional factors needed

## Nodes Created

| Node | Type | Action |
|------|------|--------|
| [[fcf-yield]] | concept | Created |
| [[fama-french-factors]] | concept | Created |
| [[size-effect]] | concept | Created |
| [[value-factor]] | concept | Created |
| [[profitability-factor]] | concept | Created |
| [[investment-pattern]] | concept | Created |
| [[momentum]] | concept | Created |
| [[multibagger-screening]] | strategy | Created |

## Nodes Updated

None — all nodes were new.

## Notable Methodology

- Fama-French five-factor sorting (27 portfolios: 3×3×3 on size/value/profitability)
- Pooled GLS regression → fixed effects panel → Arellano-Bond difference GMM → Blundell-Bond system GMM
- 150+ variables tested via Hendry's general-to-specific methodology
- Out-of-sample prediction: models never predicted rise when stocks fell; consistently pessimistic (favorable bias)

## Research Connections

- **Piotroski F-Score** (2000): Accounting-based screen — this paper's model supersedes it with more factors
- **Sentiment analysis suggested:** Paper recommends incorporating social media/news sentiment (aligns with Synesis pipeline capabilities)
- **AI integration suggested:** Paper recommends neural networks/ensemble methods for improved prediction (future pipeline enhancement)

---
**See also:** [[multibagger-screening]] | [[fcf-yield]] | [[fama-french-factors]]
