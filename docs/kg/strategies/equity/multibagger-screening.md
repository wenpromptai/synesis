---
up: []
related: ["[[fcf-yield]]", "[[fama-french-factors]]", "[[size-effect]]", "[[value-factor]]", "[[profitability-factor]]", "[[investment-pattern]]", "[[momentum]]", "[[yartseva-2025-multibaggers]]"]
created: 2026-04-07
type: strategy
tags: [equity, screening, factor-investing, quantitative]
category: equity-long
complexity: intermediate
aliases: [multibagger screening, multibagger screen, 10-bagger screener]
---

# Multibagger Screening

> [!info] Core Idea
> A systematic factor-based screening strategy for identifying stocks with the potential to return 10x+, based on empirical analysis of 464 multibagger stocks on NYSE/NASDAQ from 2009-2024 (Yartseva, 2025). Combines traditional Fama-French factors with novel predictors: [[fcf-yield]], [[investment-pattern]], [[momentum]] reversal, and macroeconomic environment.

## Screening Criteria

### Must-Have Filters (High Confidence)

| Factor | Screen | Rationale |
|--------|--------|-----------|
| **[[fcf-yield]]** | FCF/P in top quartile | Most important single predictor of future returns |
| **[[size-effect]]** | Market cap < $2B (or TEV-based) | Small caps outperform; median multibagger started at $348M |
| **[[value-factor]]** | B/M > 0.40 | High B/M consistently outperforms; avoids negative-equity traps |
| **[[profitability-factor]]** | Positive operating profitability | All negative-return portfolios had weak profitability |
| **[[profitability-factor]]** | Positive EBITDA margin | EBITDA is the preferred profitability metric |

### Strong-Signal Filters

| Factor | Screen | Rationale |
|--------|--------|-----------|
| **[[investment-pattern]]** | Asset growth > 0% | Shrinking-asset companies are worst performers |
| **[[investment-pattern]]** | Asset growth ≤ EBITDA growth | Unaffordable investment reduces returns by 4-11pp |
| **[[momentum]]** | Price near 12-month low | Entry point critical; near-low = highest future returns |
| **[[momentum]]** | 3-6 month price decline | Reversal effect: recent losers become future winners |

### Macro Overlay

| Factor | Screen | Rationale |
|--------|--------|-----------|
| Interest rate environment | Fed rate stable or declining | Rising rates depress multibagger returns by ~10pp |

### Avoidance Criteria (Short Candidates)

| Condition | Risk |
|-----------|------|
| B/M ≤ 0 (negative equity) | Liabilities exceed assets |
| Negative operating profitability | All loss-making portfolios underperformed |
| Market cap < $200M + negative profitability | Worst losses (-18.1% annual for smallest, weakest) |
| Asset growth negative (shrinking assets) | Underinvestment spiral |

## Empirical Performance

Based on 464 enduring multibaggers (10x+ return, 2009-2024):
- **Average 15-year return:** 26x (21.4% CAGR), including 24 100-baggers
- **Starting valuations:** Median P/S 0.6, P/B 1.1, forward P/E 11.3, PEG 0.8
- **Sector distribution:** Broadly diversified (not tech-concentrated)
- **Growth rates (median 15yr CAGR):** Revenue 11.1%, Operating profit 17.3%, Net profit 22.9%, EPS 20.0%
- **Model predictions:** Never predicted a rise when stocks fell — consistently errs on side of caution

## Key Insights

1. **Growth vs value is a false dichotomy:** Multibaggers are high-growth companies that are also cheap at entry
2. **Earnings growth is NOT predictive:** Despite popular belief, past earnings growth (EPS, revenue, EBITDA growth rates) is statistically insignificant for predicting future multibagger returns
3. **Cash flow > earnings:** [[fcf-yield]] captures what matters — actual cash generation relative to price
4. **Entry timing matters more than trend-following:** Buy near lows, not near highs
5. **Debt/solvency metrics insignificant:** Altman score, debt ratios, leverage metrics are not predictive
6. **Dividends are a feature:** 58% paid dividends at start, growing to 78% by 2024
7. **R&D spending not predictive:** Higher R&D allocation does not predict higher returns

## Comparison with Other Screeners

| Screener | Approach | Key Difference |
|----------|----------|----------------|
| **Piotroski F-Score** | 9 binary accounting signals | No momentum, no FCF yield, no macro |
| **Mohanram G-Score** | Growth-focused accounting | Designed for growth stocks, less robust |
| **Lynch "Ten-Bagger"** | Qualitative (5 elements) | Not statistically validated |
| **This model** | Panel regression, 150+ variables tested | Empirically validated, out-of-sample tested |

> [!quote]- Sources
> - Yartseva (2025) "The Alchemy of Multibagger Stocks: Understanding the Drivers of Stock Market Outperformance" — 464 multibaggers on NYSE/NASDAQ (2009-2024), 11,600 company-year observations, static + dynamic panel regressions, out-of-sample validation 2023-2024
> - Phelps (1972), Lynch (1988), Mayer (2018) — qualitative predecessors
> - Fama & French (2015) — factor framework foundation

## Data Source

> [!info] Synesis Pipeline
> All screening factors available via yfinance: `get_quote()` for market cap, TEV, margins, ratios, 52-week range; `get_financials()` for FCF, total assets, EBITDA (historical); `get_history()` for price momentum. FRED for Fed rate environment.

---
**See also:** [[fcf-yield]] | [[fama-french-factors]] | [[size-effect]] | [[value-factor]] | [[profitability-factor]] | [[investment-pattern]] | [[momentum]]
