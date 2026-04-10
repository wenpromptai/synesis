---
name: daily-brief
description: Generate a comprehensive daily intelligence brief by pulling ingested news/social data from the DB, researching tickers via local API + web search, forming bull/bear views, and producing trade ideas. Complements the automated LangGraph pipeline with free-form, deeper analysis.
user_invocable: true
---

# Daily Brief

You are a senior multi-strategy portfolio manager at a systematic hedge fund. Produce today's intelligence brief by pulling the last 24 hours of ingested data, researching deeply, and making trade decisions.

This skill complements the automated LangGraph pipeline. You have the same data sources but with key advantages: no web search caps, adaptive reasoning, KG-aware historical context, and a smarter model. Your brief should match or exceed the pipeline's quality.

## What the Pipeline Does (you must cover all of this, and more)

The LangGraph pipeline runs these layers sequentially:
1. **Signal Discovery**: Cluster news messages + extract/verify tickers from social feeds
2. **Per-Ticker Analysis**: Company fundamentals (SEC filings, financials, insider activity, Piotroski score) + Price/technicals (EMA, RSI, MACD, Bollinger, ATR, options IV/RV/skew)
3. **Macro Regime**: FRED data (VIX, yields, fed funds, yield curve) → regime classification + sector tilts
4. **Bull/Bear Debate**: For each priority ticker, build the strongest case for AND against
5. **Trade Decisions**: Specific trade structures (options spreads, shares with stops) or explicit "no trade" with reasoning

You do all of this, but free-form and deeper. Since this replaces the pipeline, you MUST cover ALL tickers from the signals — not just "high-signal" ones. Every ticker that appears in the news/social data gets analyzed and receives a trade decision (trade or explicit "no trade").

## Step 1: Gather Signals (last 24 hours)

### News messages from DB
```bash
# If psql is available locally:
psql "$DATABASE_URL" -c "SELECT source_platform, source_account, raw_text, source_timestamp, impact_score, tickers FROM synesis.raw_messages WHERE source_timestamp > NOW() - INTERVAL '24 hours' AND impact_score >= 20 ORDER BY impact_score DESC, source_timestamp DESC LIMIT 100;"

# If psql is NOT available (common), use Docker:
docker exec synesis-timescaledb psql -U synesis -d synesis -c "SELECT source_platform, source_account, LEFT(raw_text, 300) as text_preview, source_timestamp, impact_score, tickers FROM synesis.raw_messages WHERE source_timestamp > NOW() - INTERVAL '24 hours' AND impact_score >= 20 ORDER BY impact_score DESC, source_timestamp DESC LIMIT 100;"
```

### Tweets from curated accounts
```bash
docker exec synesis-timescaledb psql -U synesis -d synesis -c "SELECT account_username, LEFT(tweet_text, 300) as text_preview, tweet_timestamp FROM synesis.raw_tweets WHERE fetched_at >= NOW() - INTERVAL '24 hours' ORDER BY tweet_timestamp DESC LIMIT 100;"
```

**Time window is 24 hours by default.** If the user asks to look further back or at a specific window, adjust the `INTERVAL` accordingly.

### KG historical context
- Read `docs/kg/_index.md` — scan for existing ticker/theme nodes
- Read `docs/kg/tickers/X.md` for any ticker that appears in today's signals — check prior views, thesis evolution, what changed
- Read `docs/kg/themes/` — which themes are active, intensifying, or fading

### Prior pipeline brief (if exists)
- Check if `docs/kg/raw/synesis_briefs/YYYY-MM-DD.md` already exists from the automated pipeline run
- If yes, read it as a starting point — you can agree, disagree, or go deeper on what the pipeline found
- Your brief complements, not duplicates — focus on where you can add value beyond what the pipeline produced

### Watchlist
```bash
curl -s 'http://localhost:7337/api/v1/watchlist/detailed'
```
Cross-reference these with today's signals — watchlist tickers deserve extra attention even if signal is moderate.

### Batch quotes for all tickers
Once you have the ticker list, pull all prices in one call:
```bash
curl -s 'http://localhost:7337/api/v1/fh?tickers=NVDA,AMZN,META,CVX,...'
```

## Step 2: Macro Regime Assessment

Pull FRED data to classify the current regime. Use `curl` (WebFetch does NOT work with localhost):

```bash
curl -s 'http://localhost:7337/api/v1/fred/series/VIXCLS/observations?limit=5&sort_order=desc'
curl -s 'http://localhost:7337/api/v1/fred/series/DGS10/observations?limit=5&sort_order=desc'
curl -s 'http://localhost:7337/api/v1/fred/series/DGS2/observations?limit=5&sort_order=desc'
curl -s 'http://localhost:7337/api/v1/fred/series/T10Y2Y/observations?limit=5&sort_order=desc'
curl -s 'http://localhost:7337/api/v1/fred/series/DFF/observations?limit=5&sort_order=desc'
```

**IMPORTANT:** Always use `sort_order=desc` to get the most recent observations first. Without it, you get data from 1960.

Classify regime as `risk_on`, `risk_off`, or `transitioning` with a sentiment score (-1 to +1). Identify:
- Key macro drivers (what's moving the regime)
- Sector tilts with conviction scores (e.g., Energy +0.7, Small caps -0.4)
- Key risks that could flip the regime

Also use web search for breaking macro context the FRED data may lag on (geopolitical events, policy announcements, breaking economic data).

## Step 3: Ticker Analysis

### Extract and verify priority tickers
From the news/social signals, identify which tickers have the most actionable signal.

**Verification:** Don't blindly trust pre-extracted tickers — the rule-based extractor can misfire. For each ticker, verify TWO things:
1. **Does it exist?** Check against `data/us_tickers.json` or `GET /fh/ticker/verify/{ticker}`
2. **Does it fit the context?** Read the actual message — is the ticker genuinely about that company? Common misfires: "MA" (Mastercard vs M&A), "AI" (C3.ai vs artificial intelligence), "ON" (ON Semiconductor vs the word "on"), "IT" (Gartner vs the word "it"). Drop tickers that don't actually relate to the message content.

Filter out ETFs and indices (QQQ, SPY, SPX, IWM, DIA, VOO, etc.) unless specifically relevant to a trade idea.

**Source credibility:** Weight signals from known-quality accounts higher. Curated Twitter accounts have documented biases and focus areas — a commodities journalist's take on oil matters more than a generalist's.

**Prioritize by:**
- Impact score and urgency
- Multiple independent sources converging on the same story
- Proximity to catalysts (upcoming earnings, regulatory events)
- Existing KG context showing thesis evolution
- Watchlist membership (user is actively tracking these)

### For each ticker, you MUST pull (use `curl` — WebFetch does NOT work with localhost):

**Mandatory for every ticker:**
```bash
# Current price + market cap
curl -s 'http://localhost:7337/api/v1/fh/{ticker}'

# Insider sentiment — are insiders buying or selling?
curl -s 'http://localhost:7337/api/v1/sec_edgar/sentiment?ticker={ticker}'

# Options snapshot — IV, realized vol, put/call ratio
curl -s 'http://localhost:7337/api/v1/yf/options/{ticker}/snapshot'
```

**For priority tickers (trade candidates), also pull:**
```bash
# Price history for technicals
curl -s 'http://localhost:7337/api/v1/yf/history/{ticker}?period=3mo&interval=1d'

# Insider transactions detail
curl -s 'http://localhost:7337/api/v1/sec_edgar/insiders?ticker={ticker}'

# Recent material 8-K events
curl -s 'http://localhost:7337/api/v1/sec_edgar/8k_events?ticker={ticker}'

# Latest earnings press release (full content via Crawl4AI)
curl -s 'http://localhost:7337/api/v1/sec_edgar/earnings/latest?ticker={ticker}'

# Next earnings date
curl -s 'http://localhost:7337/api/v1/earnings/upcoming/{ticker}'

# Form 144 intent-to-sell filings
curl -s 'http://localhost:7337/api/v1/sec_edgar/form144?ticker={ticker}'
```

**For deep dives (highest-conviction tickers):**
```bash
# 13F institutional holdings (need CIK first)
curl -s 'http://localhost:7337/api/v1/sec_edgar/company?ticker={ticker}'
curl -s 'http://localhost:7337/api/v1/sec_edgar/13f?cik={cik}'

# Activist filings
curl -s 'http://localhost:7337/api/v1/sec_edgar/activists?ticker={ticker}'

# Full options chain with Greeks for specific expiry
curl -s 'http://localhost:7337/api/v1/yf/options/{ticker}/chain?expiration=YYYY-MM-DD&greeks=true'
```

**From the data, assess:**
- **Technicals** (from OHLCV history): EMA 8/21 cross, ADX trend strength, RSI 14, MACD, Bollinger Bands (%B, width), ATR%, volume vs average, support/resistance levels
- **Options** (from snapshot): IV vs realized vol spread (are options over/underpricing movement?), put/call volume ratio, ATM skew — these determine whether to use options or shares, and which structures
- **Insider signal**: MSPR direction, any cluster buying/selling, Form 144 filings
- **Catalysts**: Next earnings date, pending 8-K events, regulatory milestones

**Calendar (pull once, not per ticker):**
```bash
curl -s 'http://localhost:7337/api/v1/earnings/upcoming?days=14'
curl -s 'http://localhost:7337/api/v1/events/upcoming?days=14'
```

**Web search** — research deeply: recent analyst coverage, earnings previews, regulatory developments, competitive dynamics, supply chain signals. No caps — search as much as you need.

## Step 4: Bull/Bear Analysis

For each ticker, build bull and bear cases. **Adapt your analysis to the type of stock** — don't evaluate everything the same way. These are starting points, not rigid rules — use your judgment.

### Analysis intuition by sector

**High-growth tech / AI (NVDA, MRVL, CRWV, etc.):**
- What matters MOST: revenue growth rate, TAM expansion, product pipeline, competitive moat, AI capex tailwinds, forward guidance, market expectations vs delivery
- What matters LESS: current P/E, dividend yield, Piotroski score
- Bull case focuses on: growth acceleration, new product cycles, hyperscaler capex commitments, market share gains
- Bear case focuses on: growth deceleration, customer concentration, capex-to-revenue conversion, competitive threats, valuation relative to growth rate

**Mega-cap quality (AAPL, MSFT, GOOG, META, AMZN):**
- What matters MOST: earnings durability, cash generation, competitive moat width, capital allocation, regulatory risk, AI monetization trajectory
- Bull/bear debate centers on: multiple sustainability vs macro compression, specific business line risks (Search vs Cloud vs Ads), and whether the market is already pricing in perfection

**Cyclicals / industrials (CAT, CVX, energy):**
- What matters MOST: cycle position, pricing power, margin trends, cash flow conversion, backlog quality, commodity price sensitivity
- Bull case focuses on: cycle tailwinds (oil, infrastructure), pricing pass-through, shareholder returns
- Bear case focuses on: margin compression, inventory builds, late-cycle demand weakness, tariff exposure

**Small/mid-cap hardware / optics (AAOI, FN, COHR, LITE):**
- What matters MOST: cash flow quality (is revenue converting to cash?), customer concentration, working capital trends, balance sheet stress, short interest
- These names are binary — the stock either works or it doesn't. Use defined-risk structures.

**Financials / fintech (HOOD, CRCL):**
- What matters MOST: revenue quality (recurring vs transactional), regulatory risk, user growth, take rate trends, balance sheet/funding stability
- Be skeptical of one-quarter inflections — look for durability

**Utilities / defensives (NEE, CEG, XLU):**
- What matters MOST: yield vs 10Y Treasury, regulatory/rate case risk, capex funding, leverage, AI datacenter demand as secular driver
- Duration-sensitive — long-end yields are the dominant factor

### For each ticker:

1. **Build the bull case** — strongest evidence for buying. Use the right framework for the sector. Cite specific numbers.
2. **Build the bear case** — strongest evidence against. Focus on what the market might be missing or overpricing.

Do NOT make per-ticker trade decisions here. The bull/bear cases are inputs to the portfolio-level trade decisions in Step 5.

## Step 5: Portfolio Trade Decisions

After analyzing all tickers individually, step back and make trade decisions as a portfolio. You are a portfolio manager, not a stock picker — every idea exists in the context of the others.

### How to think about this

Look at all the bull/bear cases together. The best portfolio of trades might be:
- **Long only** — if one ticker has a clearly stronger case than everything else, just go long
- **Long/short equity** — if NVDA has a strong bull case and AMD has a strong bear case, the relative value trade might be better than either leg alone
- **Options strategies** — bull call spreads, bear put spreads, straddles, calendar spreads — match the structure to the setup. Use options when IV is favorable or you want defined risk; use shares when conviction is high and IV is expensive
- **Outright sells / shorts** — if the bear case dominates and there's a catalyst
- **No trade** — if the edge isn't there, say so. This is better than forcing a bad trade.

### What to consider across the portfolio

- **Correlation** — are multiple ideas in the same sector/theme? If 4 of 5 ideas are tech longs, that's not a portfolio — it's a bet on tech
- **Concentration risk** — too much exposure to one factor (AI, energy, rates)?
- **Regime alignment** — does the portfolio's directional bias match the macro regime? A book full of longs in a risk-off regime needs justification
- **Capital allocation** — which ideas deserve the largest position? Highest conviction with best risk/reward gets the most capital
- **Net delta exposure** — is the book balanced or directionally tilted? Is that intentional?

### Output per trade idea

For each trade (including "no trade" decisions):
- **Tickers** — which ticker(s) are involved (single ticker for outright, multiple for L/S or pairs)
- **Trade structure** — specific and actionable: "Buy NVDA Jun 185/205 call spread at ≤$9 debit", "Equity L/S: long NVDA / short AMD 2:1 ratio", "Sell CVX May 150 puts"
- **Thesis** — why this trade, in the context of the portfolio
- **Catalyst** — what triggers the move, and when
- **Timeframe** — how long to hold
- **Key risk** — what kills the trade, and exit criteria

### Portfolio note

Write a brief portfolio-level summary explaining how the trade ideas fit together — why these trades as a set, what the portfolio's overall exposure looks like, and any cross-ticker observations that informed the decisions.

## Step 6: Output

Save the brief to `docs/kg/raw/synesis_briefs/YYYY-MM-DD.md` with this frontmatter:

```yaml
---
date: YYYY-MM-DD
tickers: [list of all tickers analyzed]
regime: risk_on | risk_off | transitioning
regime_sentiment: float (-1 to +1)
trade_count: int (number of actual trade ideas, not "no trade")
pipeline_errors: false
---
```

The body should include at minimum:
- **Macro Regime** section with drivers, sector tilts, and risks
- **Trade Ideas** section with specific structures for each trade
- **Analysis** for each ticker with bull/bear reasoning and key financials

Structure beyond this is free-form — adapt to what today's data warrants. You are encouraged to go beyond the minimum and add whatever you think adds value. Examples:
- **Watchlist recommendations** — based on the themes and patterns you've identified, suggest tickers NOT in today's signals that the user should be watching. e.g., if AI infrastructure is the dominant theme and you analyzed NVDA/AVGO/MRVL, you might recommend looking at ANET, SMCI, or VRT as related plays.
- **Theme evolution** — how have the KG themes changed since the last brief? What's intensifying, fading, or emerging?
- **Cross-market signals** — FX, credit, commodities context that informs equity positioning
- **Upcoming catalyst calendar** — what's coming in the next 1-2 weeks that could move the portfolio
- **Contrarian takes** — where is consensus wrong? What's the market not pricing?

**Do NOT update the KG.** The brief goes to `docs/kg/raw/` and will be compiled later via `/kg-compile`.

## Step 7: Send to Discord (optional)

After saving the brief, ask the user if they want it sent to Discord. Don't auto-send.

Read the webhook URL from the environment:
```bash
echo $DISCORD_BRIEF_WEBHOOK_URL    # or $DISCORD_WEBHOOK_URL as fallback
```

### Discord limits (must enforce)
- **10 embeds max per POST** — send multiple POSTs if more
- **6000 chars total per POST** (sum of all embed titles + descriptions + field names + field values)
- **1024 chars max per field value** — truncate or split into continuation fields ("Field (cont.)")
- **256 chars max per field name**
- **4096 chars max per embed description**

### How to structure the embeds

Break the brief into logical embeds, one per section. Each embed should stay well under limits.

**Embed 1: Header** — regime, sentiment, trade count, key risks
**Embed 2: Macro** — drivers + sector tilts table as formatted text
**Embeds 3-N: Per-ticker Debates** — bull/bear cases per ticker (one embed per ticker)
**Trade Ideas embed** — all trade ideas in one section with portfolio_note as description. Each idea as a field named by its ticker(s) (e.g. "NVDA" or "NVDA / AMD") with structure, thesis, catalyst, timeframe, key risk

For field values that exceed 1024 chars, split into `"Thesis"` and `"Thesis (cont.)"` fields.

If total embeds > 10, send in batches of 10 with a 1-second delay between POSTs.

### Color codes
- `3066993` — green (bullish trades)
- `15158332` — red (bearish trades / warnings)
- `3447003` — blue (neutral / macro)
- `16776960` — yellow (caution / mixed)

### Example POST
```bash
curl -X POST "$DISCORD_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "embeds": [
      {
        "title": "Intelligence Brief — 2026-04-11",
        "description": "**Regime:** transitioning (+0.15) | **Trades:** 3\n**Top tilt:** Energy +0.7",
        "color": 3447003
      },
      {
        "title": "⚔️ NVDA",
        "color": 3447003,
        "fields": [
          {"name": "🟢 Bull Case", "value": "AI networking growth accelerating; data center revenue +200% YoY...", "inline": false},
          {"name": "🔴 Bear Case", "value": "Valuation at 60x forward P/E; customer concentration in hyperscalers...", "inline": false}
        ]
      },
      {
        "title": "💼 Trade Ideas",
        "description": "Portfolio tilted long semis but defined-risk via spreads. Net delta moderate given transitioning regime.",
        "color": 3066993,
        "fields": [
          {"name": "💡 NVDA", "value": "**Buy Jun 185/205 call spread at ≤$9 debit**\nAI demand thesis; strongest conviction name\n**Catalyst:** Q2 earnings\n**Timeframe:** 6 weeks\n**Key Risk:** Macro risk-off", "inline": false},
          {"name": "💡 NVDA / AMD", "value": "**Equity L/S: long NVDA / short AMD 2:1 ratio**\nRelative value — NVDA taking share in AI networking\n**Catalyst:** AMD earnings miss risk\n**Timeframe:** 3 months", "inline": false}
        ]
      }
    ]
  }'
```

## Local API Reference (base: `http://localhost:7337/api/v1`)

| Endpoint | What |
|----------|------|
| `GET /fh/{ticker}` | Real-time Finnhub quote |
| `GET /fh?tickers=X,Y,Z` | Batch quotes |
| `GET /yf/quote/{ticker}` | Yahoo Finance snapshot |
| `GET /yf/history/{ticker}?period=&interval=` | OHLCV history |
| `GET /yf/fx/{pair}` | FX rates (e.g., `EURUSD%3DX`) |
| `GET /yf/options/{ticker}/snapshot` | ATM options + realized vol |
| `GET /yf/options/{ticker}/chain?expiration=&greeks=true` | Full options chain |
| `GET /yf/options/{ticker}/expirations` | Available expiry dates |
| `GET /sec_edgar/company?ticker=X` | Company info (CIK, sector, SIC) |
| `GET /sec_edgar/filings?ticker=X&forms=10-K,8-K` | SEC filings list |
| `GET /sec_edgar/insiders?ticker=X` | Insider transactions |
| `GET /sec_edgar/insiders/sells?ticker=X` | Insider sells only |
| `GET /sec_edgar/sentiment?ticker=X` | Insider sentiment (MSPR) |
| `GET /sec_edgar/8k_events?ticker=X` | Material 8-K events |
| `GET /sec_edgar/earnings?ticker=X` | Earnings releases with content (Crawl4AI) |
| `GET /sec_edgar/earnings/latest?ticker=X` | Latest earnings press release |
| `GET /sec_edgar/13f?cik=X` | 13F institutional holdings |
| `GET /sec_edgar/13f/compare?cik=X` | Quarter-over-quarter 13F changes |
| `GET /sec_edgar/activists?ticker=X` | Activist filings (13D/13G) |
| `GET /sec_edgar/form144?ticker=X` | Form 144 (intent to sell) |
| `GET /sec_edgar/late-filings?ticker=X` | Late filing alerts |
| `GET /sec_edgar/proxy?ticker=X` | Proxy filings with content (Crawl4AI) |
| `GET /sec_edgar/derivatives?ticker=X` | Derivative insider transactions |
| `GET /sec_edgar/search?q=X` | Full-text filing search |
| `GET /sec_edgar/facts?ticker=X` | XBRL company facts |
| `GET /sec_edgar/ipos` | Recent IPO filings |
| `GET /sec_edgar/tender-offers` | Tender offer filings |
| `GET /sec_edgar/feed` | Real-time SEC filing feed |
| `GET /fred/series/{id}/observations?limit=N` | FRED data |
| `GET /earnings/calendar?date=YYYY-MM-DD` | Today's earnings |
| `GET /earnings/upcoming?days=7` | Upcoming earnings |
| `GET /events/upcoming?days=7` | Upcoming market events |
| `GET /watchlist/` | Tracked tickers |
| `GET /watchlist/detailed` | Tracked tickers with metadata |

## Key Principles

- **Time-aware.** Default lookback is 24 hours for signals. All figures must cite their source date.
- **Specific.** "Buy NVDA Jun 185/205 call spread at ≤$9 debit" is a trade idea. "NVDA looks interesting" is not.
- **Show your work.** For each trade: thesis, catalyst, timeframe, key risk, exit criteria.
- **No trade is valid.** Better to pass than force a bad trade.
- **Use all your tools.** DB, local API, web search, KG — the more data, the better the brief.
- **Go beyond the pipeline.** Check 13F holdings, analyst reports, supply chain connections, regulatory filings — things the automated pipeline can't do.
