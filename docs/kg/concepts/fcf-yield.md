---
up: []
related: ["[[value-factor]]", "[[profitability-factor]]", "[[multibagger-screening]]", "[[size-effect]]", "[[momentum]]", "[[investment-pattern]]", "[[fama-french-factors]]"]
created: 2026-04-07
type: concept
tags: [equity, valuation, fundamentals]
aliases: [FCF yield, free cash flow yield, FCF/P, cash flow yield]
---

# Free Cash Flow Yield

> [!info] Definition
> The ratio of free cash flow per share to the current stock price (FCF/P). Measures how much disposable cash a company generates relative to its market valuation. Higher FCF yield = cheaper stock relative to its cash-generating ability. ^definition

## How It Works

$$
\text{FCF Yield} = \frac{\text{Free Cash Flow per Share}}{\text{Stock Price}} \times 100\%
$$

Equivalently:
$$
\text{FCF Yield} = \frac{\text{Levered FCF}}{\text{Market Cap}} \times 100\%
$$

FCF yield combines valuation and profitability into a single metric. A high FCF yield means the company is both undervalued and cash-generative — two properties that independently predict outperformance.

## Key Properties

1. **Dual signal:** Captures both value (low price relative to fundamentals) and quality (actual cash generation, not just accounting profit)
2. **Robust to accounting manipulation:** FCF is harder to distort than earnings — it represents real cash flowing through the business
3. **Superior to P/E:** P/E breaks down for loss-making companies (negative earnings) and distorts when earnings are near zero. FCF yield avoids these issues
4. **Most important multibagger predictor:** Empirically the strongest single driver of future multibagger returns (Yartseva, 2025)
5. **Granger-causal:** Past FCF yield Granger-causes future stock returns — it is predictive, not merely correlated

## FCF Yield vs Other Valuation Metrics

| Metric | Strengths | Weaknesses |
|--------|-----------|------------|
| **FCF/P** | Cash-based, dual signal, robust | Volatile for capex-heavy firms |
| P/E | Widely used, intuitive | Breaks for loss-makers, near-zero earnings |
| B/M | Fama-French standard, stable | Doesn't capture cash generation |
| EV/EBITDA | Enterprise-level, debt-aware | EBITDA ignores capex, tax |
| P/S | Works for unprofitable firms | Ignores margins entirely |

## In Practice

- **[[multibagger-screening]]:** FCF yield is the primary screening factor — stocks with high FCF/P ratios have the highest probability of delivering 10x+ returns
- **[[value-factor]]:** FCF yield is an alternative (and empirically superior) proxy for the value factor alongside B/M
- **[[profitability-factor]]:** High FCF yield implicitly requires profitability — a company cannot have strong cash flow without being profitable

> [!quote]- Sources
> - Yartseva (2025) "The Alchemy of Multibagger Stocks" — FCF/P identified as the most important driver of multibagger returns with highest regression coefficients across all model specifications (static and dynamic). Coefficient of 7-52% increase in future returns per 1% increase in FCF/P.

## Data Source

> [!info] Synesis Pipeline
> yfinance `get_quote(ticker)` provides `freeCashflow` and `marketCap` fields. Compute FCF yield as `freeCashflow / marketCap`. Historical FCF available via `get_financials()` cash flow statement.

---
**See also:** [[value-factor]] | [[profitability-factor]] | [[multibagger-screening]]
