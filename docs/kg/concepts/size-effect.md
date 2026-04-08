---
up: []
related: ["[[fama-french-factors]]", "[[value-factor]]", "[[multibagger-screening]]", "[[momentum]]", "[[investment-pattern]]", "[[fcf-yield]]"]
created: 2026-04-07
type: concept
tags: [equity, factor-investing, fundamentals]
aliases: [size effect, SMB, small minus big, small-cap effect, size factor]
---

# Size Effect

> [!info] Definition
> The empirical tendency for small-cap stocks to generate higher returns than large-cap stocks over time, after controlling for other risk factors. One of the original Fama-French factors (SMB — Small Minus Big). ^definition

## How It Works

Size is typically measured by market capitalisation or total enterprise value (TEV). Smaller companies have more room to grow, are less efficiently priced (lower analyst coverage), and carry higher risk — which the market compensates with higher expected returns.

## Key Properties

1. **Strongly confirmed for multibaggers:** Small-cap multibaggers generate the highest excess returns across all factor sorts (Yartseva, 2025)
2. **TEV superior to market cap:** Total enterprise value (which includes debt) is a better size proxy than market cap alone in upgraded factor models
3. **Amplifies other factors:** Small size magnifies both upside (when combined with high value/profitability) and downside (when combined with weak profitability)
4. **Median multibagger starting size:** $348M market cap, $702M revenue (2009 values)
5. **Negative coefficient:** In regression models, larger TEV → lower future returns, confirming the effect

## Risk Warning

> [!warning] Double-Edged
> Small size amplifies losses too. Small-cap stocks with weak profitability and negative equity experience the worst performance: -18.1% annual price decline for the smallest, least profitable cohort. The smaller the company, the more severe the losses when fundamentals are weak.

## In Practice

- **[[multibagger-screening]]:** Screen for small-to-mid cap companies (starting market cap under ~$2B) as the primary universe
- **[[fama-french-factors]]:** The SMB factor — one of the five systematic risk factors
- **[[value-factor]]:** Small + undervalued is the strongest combination for outperformance

> [!quote]- Sources
> - Fama & French (1993) — original documentation of the size premium
> - Yartseva (2025) "The Alchemy of Multibagger Stocks" — small-cap multibaggers outperform medium and large across all factor sorts; median starting market cap $348M; TEV preferred over market cap as size proxy

## Data Source

> [!info] Synesis Pipeline
> yfinance `get_quote(ticker)` provides `marketCap` and `enterpriseValue` (TEV). Screen small caps with `marketCap < 2_000_000_000`.

---
**See also:** [[fama-french-factors]] | [[value-factor]] | [[profitability-factor]] | [[multibagger-screening]]
