---
up: []
related: ["[[fama-french-factors]]", "[[profitability-factor]]", "[[multibagger-screening]]", "[[size-effect]]", "[[value-factor]]", "[[momentum]]", "[[fcf-yield]]"]
created: 2026-04-07
type: concept
tags: [equity, factor-investing, fundamentals]
aliases: [investment pattern, CMA, conservative minus aggressive, investment factor, asset growth]
---

# Investment Pattern

> [!info] Definition
> The relationship between a company's investment rate (year-on-year growth of total assets) and its future stock returns. For multibagger stocks, this relationship is *reversed* from the standard Fama-French prediction: aggressive investment leads to higher returns, but only when backed by corresponding EBITDA growth. ^definition

## The Reversal

**Standard Fama-French (2015):** Conservative investment → higher returns (CMA factor is positive)
**Multibagger stocks (Yartseva, 2025):** Aggressive investment → higher returns in **100% of pairwise comparisons**

This is a distinctive feature of multibagger stocks not observed in general stock samples.

## The Investment Dummy

The critical nuance: investment must be *affordable*.

$$
\text{Inv Dummy} = \begin{cases} 1 & \text{if asset growth} > \text{EBITDA growth} \\ 0 & \text{otherwise} \end{cases}
$$

When Inv Dummy = 1 (assets growing faster than EBITDA), future stock returns drop by **4-11 percentage points** controlling for other factors. The coefficient (-22.789 in the upgraded FF5 model) is strongly significant.

**Translation:** Multibaggers must invest aggressively — but the investment must be covered by growing earnings. Expand your assets, but don't outrun your EBITDA.

## Key Properties

1. **Reversed from FF5:** Aggressive investment is *good* for multibaggers (opposite of standard CMA factor)
2. **Affordability constraint:** Asset growth must not exceed EBITDA growth — this is the unique multibagger investment pattern
3. **Red flag: shrinking assets:** All loss-making portfolios had conservative (actually negative) asset growth averaging -6.8% vs +40.0% for outperformers
4. **Underinvestment kills:** Companies not investing enough to maintain production capabilities are the worst performers
5. **Strongly significant:** The investment dummy appears in both static and dynamic models with large negative coefficients

## In Practice

- **[[multibagger-screening]]:** Screen for positive asset growth, then filter out companies where asset growth > EBITDA growth
- **[[fama-french-factors]]:** The CMA factor — reversed for multibaggers
- **[[profitability-factor]]:** Weak profitability causes underinvestment, which causes further decline — a vicious cycle

> [!quote]- Sources
> - Fama & French (2015) — CMA factor predicts conservative investment → higher returns
> - Yartseva (2025) "The Alchemy of Multibagger Stocks" — reversed investment effect found in 100% of pairwise comparisons among 464 multibaggers; investment dummy (asset growth > EBITDA growth) reduces future returns by 4-11pp; shrinking-asset companies are worst performers

## Data Source

> [!info] Synesis Pipeline
> yfinance `get_financials(ticker)` provides total assets and EBITDA from income/balance sheet statements. Compute YoY growth rates and compare. `get_quote()` provides `totalAssets`.

---
**See also:** [[fama-french-factors]] | [[profitability-factor]] | [[multibagger-screening]]
