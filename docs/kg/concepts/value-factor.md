---
up: []
related: ["[[fama-french-factors]]", "[[fcf-yield]]", "[[size-effect]]", "[[multibagger-screening]]"]
created: 2026-04-07
type: concept
tags: [equity, factor-investing, valuation]
aliases: [value factor, HML, high minus low, value effect, book-to-market]
---

# Value Factor

> [!info] Definition
> The empirical tendency for stocks trading at low prices relative to their fundamental value (high book-to-market, high FCF yield) to outperform expensive "growth" stocks over time. One of the original Fama-French factors (HML — High Minus Low). ^definition

## Key Valuation Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| **B/M** (Book-to-Market) | Book Value / Market Cap | FF5 standard; B/M > 1 = undervalued |
| **[[fcf-yield]]** (FCF/P) | Free Cash Flow / Price | Most powerful predictor for multibaggers |
| **P/E** (Price-to-Earnings) | Price / EPS | Widely used but problematic (breaks for loss-makers) |
| **P/S** (Price-to-Sales) | Price / Revenue per Share | Works for unprofitable firms |
| **EV/EBITDA** | Enterprise Value / EBITDA | Debt-aware, significant in some models |

## Key Properties

1. **Plays the biggest role** in explaining multibagger returns across both static and dynamic models (Yartseva, 2025)
2. **FCF/P is superior to B/M:** Both are significant, but FCF yield has higher coefficients and combines valuation with profitability
3. **P/E is unreliable:** Statistically insignificant, distorts other coefficients — avoid in quantitative models
4. **Growth vs value is a false dichotomy:** Multibaggers must be *both* growth and value stocks — high growth companies that are also cheap relative to fundamentals
5. **Practical threshold:** B/M > 0.40 combined with positive operating profitability dramatically improves odds of positive excess returns
6. **Starting valuations of multibaggers:** Median P/S 0.6, P/B 1.1, forward P/E 11.3, PEG 0.8 — all indicating value at entry

## In Practice

- **[[multibagger-screening]]:** Screen for high B/M (> 0.40) and high [[fcf-yield]] as primary value filters
- **[[fama-french-factors]]:** The HML factor — one of the five systematic risk factors
- **[[size-effect]]:** Small + value is the strongest combination for outperformance
- **Short candidates:** Low B/M (≤ 0) + weak profitability + small cap = worst performers, potential shorts

> [!quote]- Sources
> - Fama & French (2015) — HML as a systematic risk factor
> - Yartseva (2025) "The Alchemy of Multibagger Stocks" — value factor has highest coefficients across all models; FCF/P identified as superior valuation proxy; 1% increase in B/M or FCF/P associated with 7-52% increase in future returns; P/E unreliable; median multibagger starting valuations very low

## Data Source

> [!info] Synesis Pipeline
> yfinance `get_quote(ticker)` provides `priceToBook`, `forwardPE`, `priceToSalesTrailing12Months`. Compute B/M as `1 / priceToBook`. FCF yield from `freeCashflow / marketCap`.

---
**See also:** [[fama-french-factors]] | [[fcf-yield]] | [[size-effect]] | [[profitability-factor]] | [[multibagger-screening]]
