---
up: []
related: ["[[fama-french-factors]]", "[[fcf-yield]]", "[[value-factor]]", "[[multibagger-screening]]"]
created: 2026-04-07
type: concept
tags: [equity, factor-investing, fundamentals]
aliases: [profitability factor, RMW, robust minus weak, profitability effect, quality factor]
---

# Profitability Factor

> [!info] Definition
> The empirical tendency for companies with robust operating profitability to generate higher future stock returns than companies with weak profitability. One of the Fama-French five factors (RMW — Robust Minus Weak). ^definition

## Profitability Metrics Tested

| Metric | Significance for Multibaggers | Notes |
|--------|------------------------------|-------|
| **EBITDA margin** | Significant in static models | Replaced operating profitability in upgraded FF5 |
| **ROA** | Significant in dynamic models | Preferred when momentum is included |
| **[[fcf-yield]]** | Most significant overall | Also interpretable as a profitability measure |
| Operating profit margin | Not significant | Too narrow |
| ROE, ROC, Cash ROIC | Not significant | Dropped during model reduction |
| Gross/net margin | Not significant | Too noisy |

## Key Properties

1. **Confirmed for multibaggers:** Controlling for other factors, portfolios with robust profitability generate 16-41% excess returns vs 10-29% for weak profitability (Yartseva, 2025)
2. **EBITDA > operating profit:** Replacing operating profitability with EBITDA margin dramatically increases the coefficient from ~0 to 0.709
3. **Critical avoidance signal:** All portfolios with negative price returns had weak profitability — negative operating profitability is the strongest red flag
4. **Threshold:** Positive operating profitability + B/M > 0.40 classifies into portfolios with significantly higher excess return probability
5. **Median multibagger profitability at start:** Gross margin 34.8%, operating margin 3.9%, ROE 9.0%, ROC 6.5% — average, not exceptional

## In Practice

- **[[multibagger-screening]]:** Require positive EBITDA and positive operating profitability as minimum quality filters
- **[[fama-french-factors]]:** The RMW factor — robust profitability outperforms weak
- **[[fcf-yield]]:** High FCF yield implicitly captures profitability (can't have strong cash flow without profits)
- **Avoid:** Companies with negative operating profitability, especially when combined with small size and low B/M — these are the worst performers

> [!quote]- Sources
> - Fama & French (2015) — RMW factor in the five-factor model
> - Yartseva (2025) "The Alchemy of Multibagger Stocks" — profitability confirmed across all sorts; EBITDA margin preferred over operating profitability; all negative-return portfolios had weak profitability; ROA significant in dynamic models

## Data Source

> [!info] Synesis Pipeline
> yfinance `get_quote(ticker)` provides `ebitdaMargins`, `operatingMargins`, `returnOnAssets`, `returnOnEquity`. Quarterly data via `get_financials()`.

---
**See also:** [[fama-french-factors]] | [[fcf-yield]] | [[value-factor]] | [[investment-pattern]] | [[multibagger-screening]]
