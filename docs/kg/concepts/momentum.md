---
up: []
related: ["[[fama-french-factors]]", "[[multibagger-screening]]", "[[value-factor]]", "[[size-effect]]", "[[profitability-factor]]", "[[investment-pattern]]", "[[fcf-yield]]"]
created: 2026-04-07
type: concept
tags: [equity, technical, factor-investing]
aliases: [momentum, price momentum, trend reversal, overreaction hypothesis]
---

# Momentum

> [!info] Definition
> The tendency for stocks that have performed well (poorly) in the recent past to continue performing well (poorly) in the near future. For multibagger stocks, momentum effects are complex: short-lived positive persistence followed by rapid trend reversals. ^definition

## How It Works (General)

The standard momentum effect (Jegadeesh & Titman, 1993):
- Buy past winners, sell past losers → generates abnormal returns over 3-12 month horizons
- Widely documented across asset classes (Asness et al., 2013)
- Often attributed to investor underreaction to information followed by overreaction

## Multibagger Momentum: The Reversal

For multibagger stocks, momentum is **not the simple positive persistence** assumed by industry practitioners:

| Lookback | Coefficient | Interpretation |
|----------|------------|----------------|
| 1-month | Positive (weak) | Only significant in 1 model — very short-lived |
| 3-month | **Negative** | Recent winners tend to reverse |
| 6-month | **Negative** | Strongest reversal signal |
| 12-month range | **Negative** | Closer to 12-month high → lower future returns |

**Key insight:** If a multibagger stock has been rising for 3-6 months, it is more likely to decline next year. The closer to its 12-month high, the worse the expected return.

## Entry Point Matters

> [!important] Critical Finding
> The stock should be **close to its 12-month low** at purchase and ideally have **fallen considerably in the preceding 6 months**. This is the optimal entry point for multibaggers.

The 12-month price range variable:
$$
\text{Price Range} = \frac{\text{Current Price} - \text{12mo Low}}{\text{12mo High} - \text{12mo Low}} \times 100\%
$$

Low price range (near 12-month low) → higher future returns. This aligns with the Overreaction Hypothesis (De Bondt & Thaler, 1985).

## Key Properties

1. **Short-lived positive momentum:** Only 1-month momentum is positive, and barely significant
2. **Rapid reversal:** 3-6 month momentum is negative — winners reverse quickly
3. **Entry point critical:** Buy near 12-month lows, not highs
4. **Term structure of returns:** Multibaggers have complex, non-linear return dynamics
5. **Challenges EMH:** These patterns suggest markets do not efficiently incorporate past price information

## In Practice

- **[[multibagger-screening]]:** Filter for stocks near 12-month lows with recent 3-6 month price declines — these are the best entry points
- **[[value-factor]]:** Momentum reversal is consistent with value investing — buy when the market has overreacted to the downside
- **Exit signal:** Proximity to 12-month high should trigger caution or position reduction

> [!quote]- Sources
> - Jegadeesh & Titman (1993) — original momentum effect documentation
> - De Bondt & Thaler (1985) — overreaction hypothesis
> - Yartseva (2025) "The Alchemy of Multibagger Stocks" — 1-month momentum positive but weak; 3-6 month momentum negative; price range strongly negative; entry point near 12-month low is optimal for multibaggers

## Data Source

> [!info] Synesis Pipeline
> yfinance `get_history(ticker, period="1y")` provides daily OHLCV. Compute momentum as `(price_now / price_n_months_ago) - 1`. Price range from `fiftyTwoWeekHigh` and `fiftyTwoWeekLow` in `get_quote()`.

---
**See also:** [[fama-french-factors]] | [[value-factor]] | [[multibagger-screening]]
