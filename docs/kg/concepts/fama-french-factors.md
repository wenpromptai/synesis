---
up: []
related: ["[[size-effect]]", "[[value-factor]]", "[[profitability-factor]]", "[[investment-pattern]]", "[[multibagger-screening]]"]
created: 2026-04-07
type: concept
tags: [equity, factor-investing, asset-pricing]
aliases: [Fama-French, FF5, five-factor model, factor model]
---

# Fama-French Five-Factor Model

> [!info] Definition
> An asset pricing model that explains stock returns through five systematic risk factors: market risk (beta), size (SMB), value (HML), profitability (RMW), and investment (CMA). Proposed by Fama and French (2015) as an extension of their earlier three-factor model. ^definition

## The Five Factors

| Factor | Abbreviation | Proxy | Effect |
|--------|-------------|-------|--------|
| Market | Rm - Rf | S&P 500 return minus T-bill | Stocks move with the market |
| Size | SMB (Small Minus Big) | Market cap | Small caps outperform large caps |
| Value | HML (High Minus Low) | Book-to-market ratio | High B/M outperforms low B/M |
| Profitability | RMW (Robust Minus Weak) | Operating profitability | Profitable firms outperform |
| Investment | CMA (Conservative Minus Aggressive) | Asset growth rate | Conservative investors outperform |

## Key Properties

1. **Explains ~90% of return variation** in diversified portfolios — the dominant framework in academic finance
2. **Size, value, and profitability confirmed** for multibagger stocks (Yartseva, 2025)
3. **Investment factor reversed** for multibaggers: aggressive investment leads to *higher* returns (opposite of standard FF5 prediction), provided it is backed by EBITDA growth — see [[investment-pattern]]
4. **Incomplete for high-growth stocks:** The five factors leave a large unexplained intercept (α = 83.9) when applied to multibaggers, indicating additional drivers exist
5. **Upgraded version:** Replacing market cap with TEV, B/M with P/E, and operating profitability with EBITDA margin significantly improves model fit

## Limitations for Multibaggers

The standard FF5 model fails to fully capture multibagger returns. Key gaps:
- **Missing FCF yield:** [[fcf-yield]] is the strongest single predictor but absent from FF5
- **Investment factor reversed:** FF5 predicts conservative investment → higher returns; multibaggers show the opposite
- **No momentum:** Price [[momentum]] and entry-point effects are absent
- **No macro:** Interest rate environment has significant impact but is excluded

## In Practice

- **[[multibagger-screening]]:** FF5 factors form the foundation, but additional factors (FCF yield, momentum, interest rates) are needed
- **[[size-effect]]:** The SMB factor — small caps outperform
- **[[value-factor]]:** The HML factor — undervalued stocks outperform
- **[[profitability-factor]]:** The RMW factor — profitable firms outperform
- **[[investment-pattern]]:** The CMA factor — reversed for multibaggers

> [!quote]- Sources
> - Fama & French (2015) "A five-factor asset pricing model" — original specification
> - Yartseva (2025) "The Alchemy of Multibagger Stocks" — tested FF5 on 464 multibaggers (2009-2024), found large unexplained intercept, proposed upgraded model with TEV, EBITDA margin, and investment dummy

---
**See also:** [[size-effect]] | [[value-factor]] | [[profitability-factor]] | [[investment-pattern]] | [[fcf-yield]]
