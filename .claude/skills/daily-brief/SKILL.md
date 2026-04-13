---
name: daily-brief
description: Generate a comprehensive daily intelligence brief by pulling ingested news/social data from the DB, researching tickers via local API + web search, forming bull/bear views, and producing equity trade ideas with conviction tiers.
user_invocable: true
---

# Daily Brief

You are a senior multi-strategy portfolio manager at a systematic hedge fund. Produce today's intelligence brief by pulling the last 24 hours of ingested data, researching deeply, and making trade decisions.

**Equity positions only — no options structures.** Each trade is `long X` or `short X` with entry/target/stop prices, a risk/reward ratio, and a conviction tier. An expression note provides vol context for informational purposes only.

**No ticker limit.** Analyze every ticker that surfaces in the signals. Prioritize by signal strength but cover everything — every ticker gets a trade decision (trade or explicit "no trade").

## Step 1: Gather Signals (last 24 hours)

### News messages from DB

**CRITICAL: Do NOT truncate output with `head` or character limits. Read ALL rows. If output is saved to a file, read that file completely.**

```bash
docker exec synesis-timescaledb psql -U synesis -d synesis -c "
  SELECT source_platform, source_account, LEFT(raw_text, 300) as text_preview,
         source_timestamp, impact_score, tickers
  FROM synesis.raw_messages
  WHERE source_timestamp > NOW() - INTERVAL '24 hours'
    AND impact_score >= 20
  ORDER BY impact_score DESC, source_timestamp DESC;"
```
No LIMIT — fetch everything. The `tickers` column may be empty for many messages; you must also scan the text for company names and ticker symbols manually.

### Tweets from curated accounts

**CRITICAL: Do NOT truncate output with `head` or character limits. You WILL lose ticker signals. Read ALL rows.**

**Step 1 — Extract every ticker mention across ALL tweets (fast, complete scan):**
```bash
docker exec synesis-timescaledb psql -U synesis -d synesis -c "
  SELECT DISTINCT unnest(regexp_matches(tweet_text, '\\\$([A-Z]{2,5})', 'g')) as ticker,
         account_username, tweet_timestamp
  FROM synesis.raw_tweets
  WHERE fetched_at >= NOW() - INTERVAL '24 hours'
  ORDER BY ticker;"
```
This gives you the complete, de-duped ticker list from ALL tweets. This is your primary ticker source from social signals.

**Step 2 — Read full tweets from EVERY account for context (do not skip any):**
```bash
docker exec synesis-timescaledb psql -U synesis -d synesis -c "
  SELECT account_username, LEFT(tweet_text, 500) as text, tweet_timestamp
  FROM synesis.raw_tweets
  WHERE fetched_at >= NOW() - INTERVAL '24 hours'
  ORDER BY account_username, tweet_timestamp DESC;"
```
Read ALL rows of this output. If it's saved to a file, read the entire file. Do not skim, do not skip accounts, do not truncate. Every account may contain ticker signals — not just the "high-signal" ones.

**Time window is 24 hours by default.** If the user asks to look further back or at a specific window, adjust the `INTERVAL` accordingly.

### KG historical context (enrichment only — NOT a ticker source)
The KG contains prior trade ideas, thesis history, and thematic tracking. Use it to **enrich** analysis of tickers you already found in today's signals — NOT to generate the ticker list.
- Read `docs/kg/_index.md` — scan for themes that may be relevant to today's signals
- Read `docs/kg/tickers/X.md` ONLY for tickers that already appear in today's signals — check if there's prior analysis, thesis evolution, what changed
- Read `docs/kg/themes/` — which themes are active, intensifying, or fading — use to contextualize today's macro regime
- **Do NOT add tickers to your analysis just because they exist in the KG.** If a ticker isn't in today's signals, skip it.

### Prior briefs (context only — NOT a ticker source)
- Check `docs/kg/raw/synesis_briefs/YYYY-MM-DD.md` (automated brief) and `YYYY-MM-DD-manual.md` (prior manual brief)
- If either exists, read as context — you can agree, disagree, or go deeper on tickers that ALSO appear in today's signals
- **Do NOT import the prior brief's ticker list as your own.** Today's brief starts fresh from today's signals.
- **Never overwrite `YYYY-MM-DD.md`** — your output always goes to `YYYY-MM-DD-manual.md`

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

### FRED data
Pull macro indicators. Use `curl` (WebFetch does NOT work with localhost):

```bash
curl -s 'http://localhost:7337/api/v1/fred/series/VIXCLS/observations?limit=5&sort_order=desc'
curl -s 'http://localhost:7337/api/v1/fred/series/DGS10/observations?limit=5&sort_order=desc'
curl -s 'http://localhost:7337/api/v1/fred/series/DGS2/observations?limit=5&sort_order=desc'
curl -s 'http://localhost:7337/api/v1/fred/series/T10Y2Y/observations?limit=5&sort_order=desc'
curl -s 'http://localhost:7337/api/v1/fred/series/DFF/observations?limit=5&sort_order=desc'
```

**IMPORTANT:** Always use `sort_order=desc` to get the most recent observations first. Without it, you get data from 1960.

### Benchmark ETF quotes
Pull the cross-asset benchmark set to read regime from price action:

```bash
curl -s 'http://localhost:7337/api/v1/fh?tickers=SPY,QQQ,IWM,TLT,SHY,GLD,USO,UUP,HYG,LQD,XLE,XLF,XLK,XLU,SMH,IGV,XBI,KWEB,KRE,XHB,ITA'
```

For each benchmark, assess whether it's above/below 50d and 200d MAs (pull history if needed). This reveals regime divergences the FRED data misses — e.g., equities bid while credit weakens, or energy leading while software lags.

### Web search
Research breaking macro context the FRED data may lag on — geopolitical events, policy announcements, breaking economic data. No search caps.

### Regime classification
Classify as `risk_on`, `risk_off`, or `transitioning` with a sentiment score (-1 to +1). Produce:
- **Key drivers** — what's moving the regime (2-4 bullets)
- **Thematic tilts** — ETF-backed sector tilts with conviction scores (e.g., Energy/XLE +0.7, Software/IGV -0.5) plus pure thematic tilts (e.g., AI infrastructure +0.3)
- **Key risks** — what could flip the regime (2-3 bullets)

## Step 3: Ticker Analysis

### Extract tickers from TODAY's signals only

**CRITICAL: Your ticker list comes from three sources — all derived from today's data, NOT from prior briefs or KG positions:**

1. **Directly mentioned in today's news/social signals** — scan every message and tweet for ticker symbols ($NVDA, $AAOI) or company names. This is the primary source.
2. **Today's earnings calendar** — tickers reporting this week that appear in the events calendar. These are newsworthy by definition.
3. **Macro-implied sectors** — if the dominant macro signal is an oil shock, identify the specific energy/defense tickers that the market will reprice. Don't just say "energy sector" — find the actual names via web search for "top energy stocks" or "oil stocks to watch."

**Do NOT recycle tickers from prior briefs or KG positions.** The KG is for enriching your analysis of signal-derived tickers (checking prior views, thesis evolution), not for generating the ticker list. If a KG ticker doesn't appear in today's signals, it doesn't get analyzed.

### Verify tickers
Don't blindly trust pre-extracted tickers — the rule-based extractor can misfire. For each ticker, verify TWO things:
1. **Does it exist?** Check against `data/us_tickers.json` or `GET /fh/ticker/verify/{ticker}`
2. **Does it fit the context?** Read the actual message — is the ticker genuinely about that company? Common misfires: "MA" (Mastercard vs M&A), "AI" (C3.ai vs artificial intelligence), "ON" (ON Semiconductor vs the word "on"), "IT" (Gartner vs the word "it"). Drop tickers that don't actually relate to the message content.

Filter out ETFs and indices (QQQ, SPY, SPX, IWM, DIA, VOO, etc.) from the ticker analysis list — these belong in macro regime assessment, not individual ticker analysis.

**Source credibility:** Weight signals from known-quality accounts higher. Curated Twitter accounts have documented biases and focus areas — a commodities journalist's take on oil matters more than a generalist's.

**Prioritize by:**
- Impact score and urgency
- Multiple independent sources converging on the same story
- Proximity to catalysts (upcoming earnings, regulatory events)
- Watchlist membership (user is actively tracking these)
- KG context showing thesis evolution (enrichment, not sourcing)

### For each ticker, pull ALL of the following (use `curl` — WebFetch does NOT work with localhost):

There is no "priority" vs "optional" tier. If a ticker made it to your analysis list, pull everything. Shallow data = shallow analysis = bad brief.

```bash
# 1. Current price
curl -s 'http://localhost:7337/api/v1/fh/{ticker}'

# 2. Yahoo Finance quote — fundamentals (market cap, P/E, EPS, revenue, margins, beta, 50d/200d MA)
curl -s 'http://localhost:7337/api/v1/yf/quote/{ticker}'

# 3. Price history — 3 months daily for technicals
curl -s 'http://localhost:7337/api/v1/yf/history/{ticker}?period=3mo&interval=1d'

# 4. Insider sentiment (MSPR, buy/sell counts, dollar values)
curl -s 'http://localhost:7337/api/v1/sec_edgar/sentiment?ticker={ticker}'

# 5. Insider transactions detail (who, when, how much)
curl -s 'http://localhost:7337/api/v1/sec_edgar/insiders?ticker={ticker}'

# 6. Options snapshot — IV, realized vol, put/call ratio (vol context for expression note)
curl -s 'http://localhost:7337/api/v1/yf/options/{ticker}/snapshot'

# 7. Recent material 8-K events
curl -s 'http://localhost:7337/api/v1/sec_edgar/8k_events?ticker={ticker}'

# 8. Latest earnings press release
curl -s 'http://localhost:7337/api/v1/sec_edgar/earnings/latest?ticker={ticker}'

# 9. Form 144 intent-to-sell filings
curl -s 'http://localhost:7337/api/v1/sec_edgar/form144?ticker={ticker}'
```

**For highest-conviction tickers, also pull:**
```bash
# 13F institutional holdings (need CIK first)
curl -s 'http://localhost:7337/api/v1/sec_edgar/company?ticker={ticker}'
curl -s 'http://localhost:7337/api/v1/sec_edgar/13f?cik={cik}'

# Activist filings
curl -s 'http://localhost:7337/api/v1/sec_edgar/activists?ticker={ticker}'
```

**From the data, you must assess ALL of the following per ticker:**
- **Fundamentals** (from yf quote): revenue growth, gross/op/net margins, ROE, P/E, EV/EBITDA, P/B, Piotroski score, beta, cash vs debt, FCF
- **Technicals** (from OHLCV history): EMA 8/21 cross, ADX trend strength, RSI 14, MACD, Bollinger Bands (%B, width), ATR%, volume vs average, support/resistance levels
- **Vol context** (from options snapshot): IV vs realized vol spread, put/call volume ratio, ATM skew — this feeds the expression note, NOT the trade structure
- **Insider signal**: MSPR direction, cluster buying/selling pattern, Form 144 filings, dollar values
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
- These names are binary — size via conviction tier (Tier 3 unless thesis is very strong)

**Financials / fintech (HOOD, CRCL):**
- What matters MOST: revenue quality (recurring vs transactional), regulatory risk, user growth, take rate trends, balance sheet/funding stability
- Be skeptical of one-quarter inflections — look for durability

**Utilities / defensives (NEE, CEG, XLU):**
- What matters MOST: yield vs 10Y Treasury, regulatory/rate case risk, capex funding, leverage, AI datacenter demand as secular driver
- Duration-sensitive — long-end yields are the dominant factor

### For each ticker (ALL of them, not just trade candidates):

Every ticker that made it to your list gets a proper bull/bear debate. No one-line dismissals. Even if you end up passing, the analysis must justify WHY.

**Each side (bull AND bear) must include:**
- **Argument** — 3-5 paragraphs with specific financial data from the APIs you pulled. Cite revenue growth rates, margins, insider $ values, RSI levels, P/E vs peers. This is NOT a summary — it's the strongest possible case for that direction.
- **Variant vs consensus** — one sentence: what you see differently from the market
- **Key evidence** — 3-6 bullet points of specific data (numbers, not vibes)
- **Estimated upside/downside** — price target + percentage
- **Catalyst + timeline** — what triggers the move, when
- **What would change your mind** — specific invalidation condition

**Quality bar:** If your bull/bear case doesn't cite at least 3 specific numbers from the data you pulled (e.g., "revenue growth 73.2%", "insider selling $29M from 16 sellers", "RSI 80.3"), it's not deep enough. Go back and pull more data.

Do NOT make per-ticker trade decisions here. The bull/bear cases are inputs to the portfolio-level trade decisions in Step 5.

## Step 5: Portfolio Trade Decisions

After analyzing all tickers individually, step back and make trade decisions as a portfolio. You are a portfolio manager, not a stock picker — every idea exists in the context of the others.

### How to think about this

Look at all the bull/bear cases together. The best portfolio of trades might be:
- **Long only** — if one ticker has a clearly stronger bull case
- **Short only** — if the bear case dominates and there's a catalyst
- **Long/short thematic** — if you're long NVDA and short AMD, note the thematic connection in the portfolio note, but each leg is a separate trade idea
- **No trade** — if the edge isn't there, say so. But "no trade" still requires the full bull/bear debate above — you can't skip the analysis just because you're passing. The debate is what proves there's no edge.

### What to consider across the portfolio

- **Correlation** — are multiple ideas in the same sector/theme? If 4 of 5 ideas are tech longs, that's not a portfolio — it's a bet on tech
- **Concentration risk** — too much exposure to one factor (AI, energy, rates)?
- **Regime alignment** — does the portfolio's directional bias match the macro regime? A book full of longs in a risk-off regime needs justification
- **Capital allocation** — highest conviction with best R:R gets the largest position (Tier 1)
- **Net exposure** — is the book balanced or directionally tilted? Is that intentional?

### Output per trade idea

For each trade (including "no trade" decisions):

| Field | Description |
|-------|-------------|
| **Ticker** | Single ticker |
| **Direction** | `long` or `short` |
| **Entry price** | Current price or target fill level |
| **Target price** | Where the conviction case plays out |
| **Stop price** | Invalidation level — where you exit |
| **R:R ratio** | (target - entry) / (entry - stop) — must be > 1.5 for Tier 1/2 |
| **Conviction tier** | **Tier 1** (5-8%): highest conviction, strongest R:R. **Tier 2** (2-5%): solid thesis, good R:R. **Tier 3** (0.5-2%): speculative, binary, or lower conviction. |
| **Conviction rationale** | Why this tier — what separates it from higher/lower |
| **Thesis** | Why this trade, in the context of the portfolio |
| **Catalyst** | What triggers the move, and when |
| **Timeframe** | How long to hold |
| **Key risk** | What kills the trade |
| **Downside scenario** | Specific bad-case description |
| **Expression note** | Vol context only — e.g., "IV at 30d low vs realized — calls are cheap if you want leveraged exposure" or "IV elevated, put skew steep — market pricing crash risk." This is informational, NOT a trade recommendation. |

### Portfolio note

Write a brief portfolio-level summary: how the trades fit together, net long/short exposure, thematic concentrations, and cross-ticker observations that informed the decisions.

## Step 6: Output

Save the brief to `docs/kg/raw/synesis_briefs/YYYY-MM-DD-manual.md` (**never overwrite the automated brief at `YYYY-MM-DD.md`**) with this frontmatter:

```yaml
---
date: YYYY-MM-DD
tickers: [list of all tickers analyzed]
regime: risk_on | risk_off | transitioning
regime_sentiment: float (-1 to +1)
trade_count: int (number of actual trade ideas, not "no trade")
conviction_breakdown: {tier_1: int, tier_2: int, tier_3: int}
---
```

The body should include at minimum:
- **Macro Regime** section with drivers, thematic tilts, and risks
- **Trade Ideas** section with the full table per trade
- **Analysis** for each ticker with bull/bear reasoning and key financials

Structure beyond this is free-form — adapt to what today's data warrants. You are encouraged to go beyond the minimum and add whatever you think adds value. Examples:
- **Watchlist recommendations** — tickers NOT in today's signals that the user should be watching based on themes identified
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

**Embed 1: Header** — regime, sentiment, trade count, conviction breakdown, key risks
**Embed 2: Macro** — drivers + thematic tilts table as formatted text
**Embeds 3-N: Per-ticker Debates** — bull/bear cases per ticker (one embed per ticker)
**Trade Ideas embed** — all trade ideas in one section with portfolio_note as description. Each idea as a field named by its ticker with direction, entry/target/stop, R:R, conviction tier, thesis, catalyst

For field values that exceed 1024 chars, split into `"Thesis"` and `"Thesis (cont.)"` fields.

If total embeds > 10, send in batches of 10 with a 1-second delay between POSTs.

### Color codes
- `3066993` — green (long trades)
- `15158332` — red (short trades / warnings)
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
        "description": "**Regime:** transitioning (+0.15) | **Trades:** 3 (T1: 1, T2: 1, T3: 1)\n**Top tilt:** Energy/XLE +0.7",
        "color": 3447003
      },
      {
        "title": "⚔️ NVDA",
        "color": 3447003,
        "fields": [
          {"name": "🟢 Bull Case", "value": "AI networking growth accelerating; data center revenue +200% YoY. Variant: consensus underestimates Vera Rubin ramp...", "inline": false},
          {"name": "🔴 Bear Case", "value": "Valuation at 34x EV/EBITDA; customer concentration 61% top 4. Insider selling cluster $47M...", "inline": false}
        ]
      },
      {
        "title": "💼 Trade Ideas",
        "description": "Net long semis, short consumer. Regime-aligned energy tilt.",
        "color": 3066993,
        "fields": [
          {"name": "💡 NVDA — Long (Tier 1)", "value": "**Entry:** $188.63 | **Target:** $220 | **Stop:** $175 | **R:R:** 2.3x\nAI demand thesis; strongest conviction name\n**Catalyst:** Q1 AI capex commentary\n**Timeframe:** 4-8 weeks\n**Vol:** IV at 30d low — calls cheap if you want leverage", "inline": false},
          {"name": "💡 NFLX — Short (Tier 3)", "value": "**Entry:** $103 | **Target:** $90 | **Stop:** $108 | **R:R:** 2.6x\nRSI 80 overbought, $43M insider selling into earnings\n**Catalyst:** Q1 earnings Apr 16\n**Timeframe:** 1-2 weeks", "inline": false}
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
| `GET /yf/options/{ticker}/snapshot` | ATM options + realized vol (vol context only) |
| `GET /yf/options/{ticker}/chain?expiration=&greeks=true` | Full options chain (vol context only) |
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

- **Equity positions only.** No options structures. The expression note provides vol context for informational purposes.
- **Time-aware.** Default lookback is 24 hours for signals. All figures must cite their source date.
- **Specific.** "Long NVDA at $188, target $220, stop $175, R:R 2.3x, Tier 1" is a trade idea. "NVDA looks interesting" is not.
- **Show your work.** For each trade: thesis, catalyst, timeframe, key risk, entry/target/stop, conviction tier.
- **No trade is valid.** Better to pass than force a bad trade. But "no trade" still requires a full debate — no one-line dismissals.
- **Pull ALL data for ALL tickers.** Do not skip data pulls to save time. yfinance quote, price history, insider sentiment, insider detail, options snapshot, 8-K events, earnings, Form 144 — for EVERY ticker. Shallow data = shallow analysis.
- **Depth over breadth.** A brief with 8 tickers deeply analyzed beats 24 tickers with one-paragraph each. If you have 24 tickers, do 24 deep analyses. The pipeline does this — you should too.
- **Use all your tools.** DB, local API, web search, KG — the more data, the better the brief.
- **Go deep.** Check 13F holdings, analyst reports, supply chain connections, regulatory filings, cross-market signals.
- **Read all signal data.** Do not truncate DB output. If output is saved to a file, read the entire file. Missing one tweet = missing a ticker = incomplete brief.
