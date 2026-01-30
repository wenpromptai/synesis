# Polymarket Trading Intelligence

Research compiled from @thejayden, community analysis, and profitable trader patterns.

---

## Power Users to Follow

### Tier 1: Verified Alpha (Linked Profiles + Track Records)

| Handle | Focus | Why Follow | Verification |
|--------|-------|-----------|--------------|
| @thejayden | Tools/Analysis | Whale Watcher, PolyInsiderBot, latency strategies | 351K views on pinned article, open-source tools |
| @CarOnPolymarket | Trading/Building | Top 0.0001% trader, builds @PredictFolio | Links profile, runs discord.gg/polyalpha |
| @nicoco89poly | Trading | Started $410 → top 0.1%, $80K+ PnL, rank 87/1.8M | Specific verifiable stats in bio |
| @mombil | Trading | German trader, +$350K PnL from $10K deposit (July 2024) | Transparent deposit/PnL ratio |
| @r_gopfan | Trading | Professional trader, $1M+ PnL | Claims professional status |
| @cashyPoly | Geopolitics | Top 0.01% trader, Middle East specialist | Domain specialization + rank claim |
| @MomentumKevin | Market Making | Polymarket MM, Top 500 Hyperliquid all-time | Cross-platform verification |
| @Frosen | Trading | Top 0.01% bettor/predictor | Links main profile |
| @TraderMush247 | Trading/DeFi | Top 1% PnL, DLMM Maxi | Specific rank claim |
| @MazinoTower | Analytics/Sports | Sports whale tracking, shinsu.xyz dashboards | Detailed PnL breakdowns, trades @BullpenFi |
| @the_smart_ape | Trading/Analysis | Profile analysis, trader breakdowns | Technical analysis, high engagement |
| @ArchiveExplorer | Analysis | Trade breakdowns, strategy documentation | Polymarket community member |

### Tier 2: Tool Builders (Verified by Product)

| Handle | Focus | Tool/Product | Notes |
|--------|-------|--------------|-------|
| @0xdotdot | Trading Terminal | @nexustoolsfun | Pro terminal for Polymarket |
| @NevuaMarkets | Alerts | Watchlists + TG/Discord alerts | Builders program winner 🥇 |
| @zharkov_crypto | Analytics | @PolyHuntApp | Web3 dev, 7K+ community |
| @CoyodyUgly | Analytics | @PolyScout | Founder, also trades |
| @pizzintwatch | OSINT | Pentagon Pizza Watch | Unique alt-data for geopolitics |
| @whalewatchpoly | Whale Tracking | mobyscreener.com | Powered by @mobyagent |
| @poly_data | Official Analytics | polymarketanalytics.com | Supported by Polymarket, powered by @goldskyio |
| @PolyScopeBot | AI Analytics | Wallet tracker | AI for insider detection, whale tracking |
| @jayka003 | Backtesting | Repla & AXB Zero | Free backtesting software, 134K views on demo |

### Tier 3: Analysts & Researchers

| Handle | Focus | Notes |
|--------|-------|-------|
| @xatacrypt | US Politics/Crypto | Analyst with @zscdao, election insights |
| @journoverax | Research | Pure analyst/researcher focus |
| @luishXYZ | Research/Building | Trader, builder, and researcher |
| @cryptouncommon | Fundamentals | FA threads, quality over quantity approach |

### Tier 4: News & Community (Useful, Not Alpha)

| Handle | Focus | Notes |
|--------|-------|-------|
| @polymarketbet | Analysis/Research | PredictTrader - Top 0.001% trader, deep trade analysis threads |
| @PolymarketIntel | Breaking News | Geopolitical insights, community-run |
| @PolymarketTrade | Traders Community | Official traders account |
| @predictionarc | Community | TG: t.me/thepredictionarc |
| @wasabiboat | Investigations | news.polymarket.com |
| @polymarketinfo | Newsletter | The Oracle - news & views |
| @poly_archive | Entertainment | Archive of trader comments |
| @Polynoob_ | Education | Guides, strategies, lore |

### Legacy (Original List)

| Handle | Focus | Notes |
|--------|-------|-------|
| @PolyInsiderBot | Insider Alerts | Telegram bot tracking suspected insider activity |
| @SecureZer0 | Arbitrage | AlertPilot arbitrage visualizations |
| @zethesx | Insider Analysis | Geopolitical bet tracking |
| @holy_moses7 | Insider Exposure | Rank 515 trader, $1→$1M journey |
| @Lookonchain | On-chain | Whale tracking, on-chain analytics |
| @hoeem | Pred Markets Dev | Early PolyInsiderBot advocate |
| @PolymarketBuild | Community | Polymarket Builders community |

---

## Notable Traders (Study Their Patterns)

| Profile | Stats | Strategy Notes |
|---------|-------|----------------|
| @kch123 | $6M all-time PnL, $3.1M/month | High volume, custom dashboard needed |
| @hal15617 | +$100K single BTC market | NO shares only, no hedge, YOLO style |
| "French Whale" Theo | $85M from Trump election | 11+ accounts, political specialization |
| @a4385 | $233K overnight (Jan 2026) | Bot liquidity drain, 15-min crypto manipulation |
| beachboy4 | +$5,785,064 (4 trades, Jan 2026) | Sports specialist, high-conviction betting |
| 0x492442Ea | +$5,348,163 (19 trades, Jan 2026) | Sports specialist, $5.6M invested → $11M returned |
| 432614799197 | +$2,777,937 (6 trades, Jan 2026) | Sports specialist |
| Pimping | +$1,267,349 (3 trades, Jan 2026) | Sports specialist, highest profit-per-trade ratio |

### Sports Whale Insight (Jan 2026)

**Source:** @MazinoTower analysis

5 traders made **$16,271,556** in one week on sports markets alone:
- Not arbitrage bots or latency setups
- Just **domain expertise** in sports
- Average: 3-19 trades per trader (high conviction, few bets)
- Top 50 sports traders: Invested $23.8M → Returned $44.3M (+$20.4M profit)

**Key Pattern:** Sports specialists making massive PnL with very few trades. This is domain expertise, not volume.

---

## Trading Strategies

### 1. Latency Arbitrage (45s Advantage)

**Source:** @thejayden pinned article (351K views)

The livestream delay gap:
- Real Event: T = 0s
- Data Feed: T = +3s (official stats)
- Livestream: T = +45-60s (Twitch/YouTube/TV)

**Edge:** Scrape data feeds directly via Chrome DevTools Protocol
- Launch Chrome with `--remote-debugging-port=9222`
- Use Playwright to connect via CDP
- Poll data every 0.1s, alert on changes
- 30-60 second edge over video watchers

**Best for:** Live sports, esports markets

---

### 2. Insider Detection & Copy-Trading

**Sources:** @RetroValix, @PolyInsiderBot, @Polysights, @spacexbt, @DidiTrading, @gemchange_ltd

#### Insider Behavior Patterns

Insiders typically exhibit these behaviors:

| Pattern | Description | Detection Signal |
|---------|-------------|------------------|
| **Fresh Wallet + Big Bet** | New account (<24h) places >$10K on single market | Account age + bet size + market count |
| **Single Market Focus** | Only trades ONE specific market | Market diversity = 1 |
| **99% Padding** | Small bets on near-certain markets to look normal | High-probability markets with small size |
| **Wash Trading Cover** | Buy/sell same markets to inflate trade count | Same market buy+sell within short window |
| **Card/External Funding** | Uses card instead of traceable EVM wallet | No on-chain deposit history |
| **Pre-News Positioning** | Large position hours before news breaks | Entry timing vs news timestamp |
| **100% Win Rate** | Perfect record in specific niche category | Win rate = 100% in narrow vertical |

#### 4-Step Detection Process (@RetroValix Method)

**Step 1: Filter for Suspicious Wallets**
Use Polysights or custom filters:
```python
suspicious_wallets = filter_wallets(
    markets_traded=1,          # Only one market
    total_trades=1,            # Single trade
    bet_size_min=1000,         # >$1K position
    account_age_days_max=7,    # New account
)
```

**Step 2: Find EVM Wallet**
- Copy Polymarket internal wallet address
- Use [Safe Global](https://app.safe.global) → "Watch any account"
- Select Polygon network → paste address
- This reveals the source EVM wallet that funded the account

**Step 3: Analyze Wallet History**
- Paste EVM wallet into [DeBank](https://debank.com)
- Check Transactions → All Chains
- Look for:
  - Connections to other suspicious wallets
  - ENS domains (Google the ENS)
  - Round number funding amounts
  - Same source funding multiple accounts

**Step 4: Build Insider Cluster**
- Go to market's "Top Holders" section
- Analyze all top wallets for similar patterns
- Cross-reference funding sources, timing, behavior
- Combine into cluster to confirm coordinated activity

#### Wallet Clustering Analysis (@gemchange_ltd Method)

Advanced detection using graph analysis:
```python
# Louvain clustering + Jaccard similarity
# Maps wallet relationships into hypergraph

def build_wallet_hypergraph(top_wallets: list[str]) -> Graph:
    """
    Identify who's copying who by analyzing:
    - Trade timing correlation
    - Position similarity (same markets, same direction)
    - Funding source overlap
    - Entry/exit timing patterns
    """
    edges = []
    for wallet_a, wallet_b in combinations(top_wallets, 2):
        similarity = jaccard_similarity(
            get_positions(wallet_a),
            get_positions(wallet_b)
        )
        timing_corr = timing_correlation(wallet_a, wallet_b)

        if similarity > 0.7 or timing_corr > 0.8:
            edges.append((wallet_a, wallet_b, similarity))

    return create_graph(edges)

# Detect clusters using Louvain algorithm
clusters = louvain_communities(graph)
```

**Output:** "Wallets A, B, C, D are likely controlled by same entity"

#### Copy Score Formula (@predictingtop)

Rank traders by consistency, not luck:
```python
copy_score = (
    r_squared *           # Consistency of returns
    win_rate *            # Percentage of winning trades
    max_drawdown_factor * # Risk management
    profit_factor         # Gross profit / gross loss
)
```

#### Real-Time Detection Signals

| Signal | Threshold | Action |
|--------|-----------|--------|
| New wallet + >$50K bet | Account <24h, single market | **HIGH ALERT** - Likely insider |
| 100% win rate + niche market | Perfect record in obscure category | **FLAG** - Research wallet |
| Multiple wallets, same timing | >3 wallets trading within 60s | **CLUSTER ALERT** - Coordinated |
| Pre-news large position | >$10K position, news breaks <6h later | **RETROSPECTIVE FLAG** - Track wallet |
| Card funding + single market | No EVM history, one prediction | **SUSPICIOUS** - Identity hiding |

#### Documented Insider Examples

| Case | Details | Profit | Detection Method |
|------|---------|--------|------------------|
| Maduro Attack | $35K → $442K in hours | 12.6x | @spacexbt tool flagged 5 alerts pre-event |
| Iran Strike | $160K position, 40-min-old account | Unknown | Fresh wallet + large bet |
| Israel-Iran Whale | 100% win rate on Israel bets, dormant 7 months | $400K+ | Win rate anomaly |
| Lighter Insiders | Cluster of wallets, all predicting same obscure market | Unknown | @RetroValix cluster analysis |

#### Tools for Insider Detection

| Tool | Features | Link |
|------|----------|------|
| **Polysights** | Wallet filtering, 85% accuracy on flagged trades | @Polysights |
| **Safe Global** | Find EVM wallet from Polymarket address | app.safe.global |
| **DeBank** | Transaction history, all chains | debank.com |
| **PolyInsiderBot** | Telegram alerts for suspicious activity | t.me/PredictionIns |
| **PolyScope** | AI trader profiling, insider detection | @PolyScopeBot |

#### GitHub Open Source

- [NickNaskida/polymarket-insider-bot](https://github.com/NickNaskida/polymarket-insider-bot) - Async bot with ML scoring
- [pselamy/polymarket-insider-tracker](https://github.com/pselamy/polymarket-insider-tracker) - Tracks fresh wallets, unusual sizing

---

### 3. Bonding Strategy (High-Probability Bonds)

**Source:** Web research

90% of $10K+ orders occur at prices >$0.95

Strategy:
- Buy near-certain outcomes (>95% probability)
- ~5% return per trade
- 2 trades/week = 520% simple annual return
- Some traders earn $150K+/year

---

### 4. Market Rebalancing Arbitrage

Buy YES + NO when combined price < $1.00
- Guarantees profit when market resolves
- Requires monitoring spreads >2.5-3% (after fees)
- Bots dominate this strategy

---

### 5. Cross-Platform Arbitrage

Compare prices: Polymarket vs Kalshi vs Probable
- Same event, different odds = opportunity
- GitHub tools exist for automation

---

### 6. Domain Specialization

Build information edge in specific verticals:
- Politics (highest documented PnL)
- Crypto (price prediction markets)
- Sports (requires latency edge)
- Geopolitics (Iran, Russia/Ukraine)

---

### 7. Bot Liquidity Drain (Weekend Manipulation)

**Source:** @polymarketbet thread (Jan 18, 2026)

Exploit thin weekend liquidity to trap market-making bots on 15-minute crypto markets.

**The Setup (by @a4385 - $233K overnight):**
1. Trade Saturday night when Binance spot order books are shallow
2. Pick a 15-minute crypto market (e.g., "XRP Up or Down")
3. Aggressively buy one side (UP) at any price, pushing it to ~70¢
4. Market-making bots see "opportunity" and sell more shares into your position
5. Accumulate large position (~77K shares at ~48¢ average)
6. 2 minutes before settlement: manipulate underlying on Binance spot (~$1M buy)
7. Push price 0.5% in your favor, win the market
8. Immediately sell spot position back

**Economics:**
- Cost: ~$6,200 (0.25% slippage × 2 + Binance VIP fees 0.06%)
- Profit: $233K
- ROI: ~3,700%

**Requirements:**
- Binance VIP 4+ account (for low fees)
- ~$1M capital to move spot price
- Weekend timing (thin liquidity)
- Speed to execute spot trade before settlement

**Victim:** @aleksandmoney bot lost a full year of profits in one night.

**Wallet:** `0x506bce138df20695c03cd5a59a937499fb00b0fe`

**Warning:** This may violate Polymarket ToS and could be considered market manipulation. For research purposes only.

---

## Tools

### Analysis Platforms

| Tool | URL | Use |
|------|-----|-----|
| Polymarket Analytics | polymarketanalytics.com | Trader leaderboards, 1M+ traders |
| PolyTrack | polytrackhq.app | Follow top traders |
| Polywhaler | polywhaler.com | Whale tracking |

### @thejayden Open Source Tools

| Tool | Link | Purpose |
|------|------|---------|
| Whale Watcher | pastebin.com/degSYc3T | Scrape trades by wallet |
| Trade Dashboard | pastebin.com/KkZpQxCa | Visualize trade history |
| 45s Scraper | In pinned article | Real-time data feed scraping |

### Bots & Trading Tools

**Telegram Bots:**

| Bot | Handle/Link | Use Case |
|-----|-------------|----------|
| PolyInsiderBot | t.me/PredictionIns | Insider activity alerts |
| insiders.bot | @insidersdotbot, t.me/polyinsiders | Social trading, elite analytics |
| PolyX Bot | @PolyxBot, polyxbot.org | Advanced TG trading bot |
| TradePolyBot | @TradePolyBot | Fast trading interface |
| Velori AI | @VeloriAIPredict | AI predictions (Perplexity/Gemini powered) |
| Polycule | polycule.trade | TG bot by @top_jeet_ |
| PolyScope | @PolyScopeBot | AI trader profiling, insider detection |

**Whale/Alert Platforms:**

| Platform | Handle | Features |
|----------|--------|----------|
| Whale Watch | @whalewatchpoly | Live feed of top trader bets |
| Nevua Markets | @NevuaMarkets | Custom watchlists, TG/Discord/Webhook alerts |
| PolyAlertHub | @PolyAlertHub | Insider, whale, market alerts |
| Polymarket Scanner | @PloyPulseBot | Real-time big trade tracking |

**Trading Terminals:**

| Tool | Handle | Notes |
|------|--------|-------|
| Nexus Tools | @nexustoolsfun | Pro terminal by @0xdotdot |
| PredictFolio | @PredictFolio | By @CarOnPolymarket |
| PolyScout | @PolyScout | By @CoyodyUgly |
| PolyHunt | @PolyHuntApp | By @zharkov_crypto |

**Legacy:**
- AlertPilot - Arbitrage execution (referenced by @SecureZer0)

### 2026 New Tools (Jan 2026)

| Tool | Handle | Features | Innovation |
|------|--------|----------|------------|
| PolyMoon | @PolyMoonio, polymoon.io | Bloomberg Terminal for Polymarket, copy top wallets, 0% fees | Pro-grade UI |
| Polycool | @PolycoolApp | 1,000 OGs leaderboard with X accounts linked, no bots/ghost wallets | Identity verification |
| TradeOnSight | @tradeonsight | Curated trader "bench", follow/fade lists, one-tap copy in TG | Trader curation |
| PolyGun | TG Bot | Faster execution via Telegram | Speed focus |

---

## Tool Features Analysis & Innovation Opportunities

### Current Tool Capabilities (Researched Jan 2026)

**1. Nexus Tools (nexustools.fun)**
- Market discovery dashboard with real-time data
- Columns: Outcomes, Whale Volume, Smart Volume, 15m Volume, Liquidity
- Time filters (Last 15 min, etc.)
- Wallet tracker (requires login)
- One-click trading via Safe wallet
- Live trade feed + watchlist
- Portfolio tracking

**2. insiders.bot**
- Signals feed by category (Finance, Crypto, Politics, Sports, Tech)
- Wallet tracking
- Copy trading
- Multi-column customizable dashboard
- Pro tier for premium features

**3. MobyScreener (mobyscreener.com)**
- Smart Money Predictions Feed - tracks top traders by win rate
- Real-time trade visibility
- Mobile apps (iOS/Android)
- Smart-wallet tracking algorithm
- Watchlists
- $MOBY token ecosystem

**4. Common Features Across Tools**
- Whale/large trade alerts
- Wallet tracking
- Copy trading
- Telegram integration
- Leaderboards
- Category filtering

### Innovation Opportunities (Gaps in Market)

**1. Wallet Categorization by Specialty**
- **Gap:** No tool categorizes wallets by what they're GOOD at predicting
- **Opportunity:** Auto-tag wallets by win rate per category (Politics, Sports, Crypto, Geopolitics)
- **Value:** Follow specialists, not generalists

**2. News-to-Market Correlation**
- **Gap:** Tools track trades but not WHY trades happen
- **Opportunity:** Link news/social signals to market movements
- **Value:** Understand catalysts, not just reactions

**3. Insider Detection AI**
- **Gap:** PolyInsiderBot flags manually, others use simple heuristics
- **Opportunity:** ML model trained on known insider patterns
- **Value:** Earlier detection, fewer false positives

**4. Cross-Platform Arbitrage Scanner**
- **Gap:** Manual comparison across Polymarket/Kalshi/PredictIt
- **Opportunity:** Real-time price diff alerts with execution
- **Value:** Guaranteed profit opportunities

**5. Social Signal Aggregation**
- **Gap:** Traders manually scan X, Telegram, news
- **Opportunity:** LLM-powered news→market relevance scoring
- **Value:** Information edge without manual monitoring

**6. Fade List Intelligence**
- **Gap:** Tools only track "follow" - who to copy
- **Opportunity:** Identify consistently WRONG traders to fade
- **Value:** Contrarian alpha source

**7. Resolution Risk Scoring**
- **Gap:** No tool predicts market resolution disputes
- **Opportunity:** Flag markets with ambiguous resolution criteria
- **Value:** Avoid resolution risk

**8. Time-Series Pattern Recognition**
- **Gap:** Static leaderboards, no trend analysis
- **Opportunity:** Track trader performance over time, identify hot/cold streaks
- **Value:** Dynamic wallet quality scoring

---

## Wallet Tracking Framework

### Categorization Approach

Track wallets by their **specialty** (where they have edge):

| Category | Signals to Track | Example Patterns |
|----------|------------------|------------------|
| **Politics** | Elections, legislation, appointments | Large bets before debate, policy announcements |
| **Crypto** | Price predictions, ETF approvals | Correlated with on-chain whale moves |
| **Sports** | Game outcomes, player props | Bets close to game time, injury info edge |
| **Geopolitics** | Conflicts, sanctions, treaties | Activity before news breaks (insider risk) |
| **Weather/Climate** | Hurricane paths, temperature records | Domain expertise |
| **Entertainment** | Awards, TV ratings | Industry insider patterns |

### Wallet Quality Metrics

| Metric | What It Measures | How to Calculate |
|--------|------------------|------------------|
| **Win Rate** | % of resolved bets won | Wins / Total Resolved |
| **ROI** | Return on capital | PnL / Total Invested |
| **Category Win Rate** | Win rate per category | Filter by market tags |
| **Conviction Score** | Bet size relative to portfolio | Avg bet size / Portfolio |
| **Timing Edge** | Buys before price moves | Entry price vs final price |
| **Consistency** | Stable performance over time | Std dev of monthly returns |

### Known Wallets by Specialty (To Research)

| Wallet/Profile | Specialty | Notable | Source |
|---------------|-----------|---------|--------|
| 0x31a56 | Politics | $409K profit, 4 predictions | @zeroqfer thread |
| "French Whale" Theo | US Politics | $85M Trump election | WSJ reporting |
| @cashyPoly | Middle East | Top 0.01% | X bio |
| @r_gopfan | General | $1M+ PnL | X bio |

**TODO:** Build automated scraper to categorize top 1000 wallets by category performance.

---

## Synesis Innovation Roadmap

Based on research of existing tools, here's what we will **build in-house** that's differentiated.

**Philosophy: Build everything ourselves. No external tool dependencies.**

### Our Core Edge: News → Market Intelligence

**What exists:** Tools track WHAT traders do (copy trading, whale alerts)
**What's missing:** Understanding WHY trades happen (signal intelligence)
**Our play:** Be the "brain" that connects news/social signals to market opportunities

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        SYNESIS STACK                            │
├─────────────────────────────────────────────────────────────────┤
│  INGESTION LAYER (Built In-House)                               │
│  ├── Twitter/X Listener (API + Scraping fallback)               │
│  ├── Telegram Channel Monitor                                   │
│  ├── RSS/News Feed Aggregator                                   │
│  ├── Sports Data Feeds (ESPN, injury reports, live scores)      │
│  └── Polymarket WebSocket (trades, prices, orderbook)           │
├─────────────────────────────────────────────────────────────────┤
│  PROCESSING LAYER (Built In-House)                              │
│  ├── LLM Analysis Pipeline (Claude/OpenAI)                      │
│  ├── Entity Extraction & Market Matching                        │
│  ├── Signal Scoring Engine                                      │
│  ├── Wallet Classifier & Tracker                                │
│  └── Deduplication & Noise Filter                               │
├─────────────────────────────────────────────────────────────────┤
│  TRADING LAYER (Built In-House)                                 │
│  ├── Polymarket CLOB API Client                                 │
│  ├── Position Manager                                           │
│  ├── Risk Management Engine                                     │
│  └── Execution Optimizer                                        │
├─────────────────────────────────────────────────────────────────┤
│  STORAGE LAYER                                                  │
│  ├── PostgreSQL + TimescaleDB (time-series trades)              │
│  ├── pgvector (semantic search for markets)                     │
│  └── Redis (real-time cache, pub/sub)                           │
├─────────────────────────────────────────────────────────────────┤
│  API LAYER                                                      │
│  ├── FastAPI REST endpoints                                     │
│  ├── WebSocket for real-time signals                            │
│  └── Telegram Bot for alerts                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 1: Signal Intelligence Layer

#### 1.1 Real-Time News-to-Market Matcher

```
News Event → LLM Analysis → Relevant Markets → Trading Signal
```

**Features:**
- Ingest X/Twitter, Telegram, RSS feeds in real-time
- LLM extracts entities (people, events, outcomes)
- Match to active Polymarket markets via semantic search
- Score relevance + sentiment + urgency

**Differentiation:** No tool currently does news→market matching. They all require manual discovery.

**Example Flow:**
```
Signal: "@Reuters: Fed Chair hints at rate pause in March"
→ LLM extracts: {entity: "Fed", event: "rate decision", timeframe: "March"}
→ Matches: "Will Fed cut rates in March 2026?" market
→ Output: {market_id, current_price, sentiment: bullish, confidence: 0.8}
```

#### 1.2 Signal Quality Scoring

| Signal Type | Quality Factors | Score Components |
|-------------|-----------------|------------------|
| Breaking News | Source credibility, first-mover | Reuters > random blog |
| Insider Leak | Account age, historical accuracy | New account + big bet = flag |
| Expert Opinion | Domain credentials, track record | Verified expert > anon |
| Social Sentiment | Volume, velocity, influencer ratio | Viral + expert engagement |

#### 1.3 Alert Prioritization Engine

```python
priority = (
    market_liquidity * 0.2 +
    signal_confidence * 0.3 +
    time_sensitivity * 0.3 +
    expected_edge * 0.2
)
```

#### 1.4 AI vs Market Mispricing Detection

**Source:** Inspired by polymarketnews.xyz (shows AI Est vs Market Price gaps)

When our AI assessment differs significantly from market price, that's tradeable alpha:

```python
def calculate_mispricing_signal(market_id: str, news_items: list[NewsItem]) -> MispricingSignal | None:
    """
    Find markets where AI assessment differs from market price.

    Example: AI thinks 20% probability, market says 5.5% = 14.5% edge
    """
    # Aggregate sentiment from all relevant news
    ai_probability = aggregate_news_sentiment(news_items)
    market_price = get_current_price(market_id)

    discrepancy = ai_probability - market_price

    # Only signal if gap is significant (>10%)
    if abs(discrepancy) < 0.10:
        return None

    # More news sources = higher confidence
    source_diversity = len(set(n.source for n in news_items))
    recency_weight = calculate_recency_weight(news_items)

    confidence = min(
        (len(news_items) / 10) *      # Volume of news
        (source_diversity / 5) *       # Diversity of sources
        recency_weight,                # Recent news weighted higher
        1.0
    )

    return MispricingSignal(
        market_id=market_id,
        ai_estimate=ai_probability,
        market_price=market_price,
        discrepancy=discrepancy,
        direction="BUY_YES" if discrepancy > 0 else "BUY_NO",
        edge=abs(discrepancy),
        confidence=confidence,
        news_count=len(news_items),
        reasoning=generate_reasoning(news_items, discrepancy)
    )


def scan_all_markets_for_mispricing() -> list[MispricingSignal]:
    """
    Continuously scan top markets for AI vs Market discrepancies.

    Priority markets:
    1. High volume (>$1M 7d) - enough liquidity to trade
    2. Recent news activity - fresh information edge
    3. Time-sensitive (resolution within 30 days) - faster feedback
    """
    signals = []

    for market in get_active_markets(min_volume_7d=1_000_000):
        news = get_relevant_news(market, hours=72)

        if len(news) >= 3:  # Need minimum news coverage
            signal = calculate_mispricing_signal(market.id, news)
            if signal and signal.confidence > 0.5:
                signals.append(signal)

    return sorted(signals, key=lambda s: s.edge * s.confidence, reverse=True)
```

**Trading Logic:**

| Discrepancy | Confidence | Action |
|-------------|------------|--------|
| >15% gap | High (>0.7) | Auto-execute (if enabled) |
| 10-15% gap | High (>0.7) | Strong alert, recommend trade |
| 10-15% gap | Medium (0.5-0.7) | Alert, manual review |
| <10% gap | Any | Log only, no action |

**Example:**
```
Market: "Khamenei out as Supreme Leader by Jan 31?"
Market Price: 5.5%
AI Estimate: 20.0% (based on 12 news articles about protests, instability)
Discrepancy: +14.5%
Direction: BUY YES
Confidence: 0.8 (high news volume, diverse sources)
→ SIGNAL: Strong mispricing detected, recommend BUY YES
```

### Phase 2: Smart Wallet Intelligence

#### 2.1 Wallet Specialty Classifier

Auto-categorize wallets by analyzing their trading history:

```python
for wallet in top_wallets:
    trades = get_trades(wallet)
    for trade in trades:
        category = classify_market(trade.market)  # Politics, Sports, Crypto, etc.
        outcome = trade.resolved_pnl

    wallet.specialty = category_with_highest_win_rate(trades)
    wallet.specialty_score = win_rate_in_specialty / overall_win_rate
```

**Output:** "This wallet has 78% win rate in Politics vs 45% overall → Politics Specialist"

#### 2.2 Wallet-Signal Correlation

Track which wallets respond to which signal types:

| Wallet | Responds To | Avg Response Time | Edge |
|--------|-------------|-------------------|------|
| 0x123 | Breaking political news | 2-5 min | +12% |
| 0x456 | Crypto whale moves | <1 min | +8% |
| 0x789 | Sports injury reports | 10-30 min | +15% |

**Value:** Know WHO to watch for WHICH signal types.

#### 2.3 Fade List Generator

Identify consistently wrong traders:

```python
fade_candidates = wallets.filter(
    win_rate < 0.4 AND
    total_bets > 50 AND
    avg_bet_size > $1000
)
```

**Strategy:** When fade_candidate buys YES, consider NO.

#### 2.4 Insider Detection Engine

**Sources:** Research from @RetroValix, @Polysights, @spacexbt, @gemchange_ltd

Automated system to detect and copy-trade suspected insiders:

```python
class InsiderDetector:
    """
    Multi-signal insider detection engine.
    Combines fresh wallet detection, cluster analysis, and timing patterns.
    """

    def __init__(self):
        self.alert_thresholds = {
            "fresh_wallet_large_bet": {"account_age_hours": 24, "bet_size_min": 10_000},
            "single_market_focus": {"market_count": 1, "bet_size_min": 5_000},
            "perfect_win_rate": {"win_rate": 1.0, "resolved_bets_min": 3},
            "cluster_correlation": {"timing_window_seconds": 60, "min_wallets": 3},
        }

    def scan_for_insiders(self, market_id: str) -> list[InsiderAlert]:
        """Scan market's top holders for insider patterns."""
        alerts = []
        top_holders = get_market_top_holders(market_id)

        for wallet in top_holders:
            score = 0
            flags = []

            # Check 1: Fresh wallet + large bet
            if wallet.account_age_hours < 24 and wallet.position_size > 10_000:
                score += 40
                flags.append("FRESH_WALLET_LARGE_BET")

            # Check 2: Single market focus
            if wallet.markets_traded == 1:
                score += 30
                flags.append("SINGLE_MARKET_FOCUS")

            # Check 3: Perfect or near-perfect win rate in niche
            if wallet.category_win_rate > 0.95 and wallet.category_bets >= 3:
                score += 25
                flags.append("HIGH_WIN_RATE_NICHE")

            # Check 4: Card/external funding (no EVM history)
            if not wallet.has_evm_deposit_history:
                score += 20
                flags.append("HIDDEN_FUNDING_SOURCE")

            # Check 5: Wash trading cover activity
            if detect_wash_trading(wallet):
                score += 15
                flags.append("WASH_TRADING_COVER")

            if score >= 50:
                alerts.append(InsiderAlert(
                    wallet=wallet,
                    score=score,
                    flags=flags,
                    confidence=min(score / 100, 1.0),
                    action="COPY" if score >= 70 else "MONITOR"
                ))

        return sorted(alerts, key=lambda a: a.score, reverse=True)

    def detect_wallet_clusters(self, wallets: list[str]) -> list[WalletCluster]:
        """
        Use Louvain clustering + Jaccard similarity to find coordinated wallets.
        """
        graph = build_similarity_graph(wallets)
        clusters = louvain_communities(graph)

        suspicious_clusters = []
        for cluster in clusters:
            if len(cluster) >= 3:
                # Multiple wallets trading same markets at same time = coordinated
                timing_correlation = calculate_cluster_timing_correlation(cluster)
                position_similarity = calculate_position_similarity(cluster)

                if timing_correlation > 0.8 or position_similarity > 0.7:
                    suspicious_clusters.append(WalletCluster(
                        wallets=cluster,
                        timing_correlation=timing_correlation,
                        position_similarity=position_similarity,
                        likely_single_entity=True
                    ))

        return suspicious_clusters
```

**Alert Levels:**

| Score | Level | Action |
|-------|-------|--------|
| 70-100 | **CRITICAL** | Auto-copy trade (if enabled), immediate alert |
| 50-69 | **HIGH** | Alert + recommend following |
| 30-49 | **MEDIUM** | Add to watchlist |
| <30 | **LOW** | Log for analysis |

**Integration with Trading Layer:**
```python
# When insider detected, optionally auto-copy
if insider_alert.score >= 70 and settings.auto_copy_insiders:
    position = insider_alert.wallet.position
    execute_trade(
        market_id=position.market_id,
        direction=position.direction,
        size=calculate_copy_size(position.size, settings.max_copy_size),
        reason=f"Insider copy: {insider_alert.flags}"
    )
```

### Phase 3: Automated Trading Signals

#### 3.1 Signal Confidence Framework

| Confidence | Criteria | Action |
|------------|----------|--------|
| **High (0.8+)** | Breaking news + whale activity + expert consensus | Auto-execute (if enabled) |
| **Medium (0.5-0.8)** | Single strong signal OR multiple weak signals | Alert + recommend |
| **Low (<0.5)** | Noise, unverified, conflicting signals | Log only |

#### 3.2 Risk-Adjusted Position Sizing

```python
position_size = (
    base_size *
    signal_confidence *
    market_liquidity_factor *
    (1 - correlation_to_existing_positions)
)
```

#### 3.3 Entry/Exit Optimization

- **Entry:** Wait for price to stabilize after news (avoid slippage)
- **Exit:** Set take-profit at expected resolution price, stop-loss at -X%

#### 3.4 High-Conviction Trade Detection

**Source:** @MazinoTower sports whale analysis (Jan 2026)

Sports whales don't make many trades—they make **few, large, high-conviction bets**:

```python
def detect_high_conviction_trade(trade, wallet):
    """Flag trades that indicate domain expertise + conviction."""

    # High conviction = bet size >> average
    size_ratio = trade.size / wallet.avg_bet_size

    # Specialists have few active positions
    active_positions = len(wallet.open_positions)

    if size_ratio > 5 and active_positions < 5:
        return {
            "type": "high_conviction_specialist",
            "confidence": min(size_ratio / 10, 1.0),
            "category": classify_market(trade.market),
            "action": "FOLLOW if wallet is category specialist"
        }
```

**Pattern:** beachboy4 made +$5.7M with just 4 trades. Not volume—conviction.

### Phase 3.5: Sports Intelligence Layer

**Why Sports?** Jan 2026 data shows sports is the highest PnL vertical:
- 5 traders: $16.2M profit in 1 week
- Top 50 sports traders: $20.4M profit
- Few trades, massive returns = domain expertise edge

#### 3.5.1 Sports Data Ingestion

```python
SPORTS_DATA_SOURCES = {
    "live_scores": ["ESPN API", "theScore", "Sportradar"],
    "injury_reports": ["official team feeds", "beat reporters"],
    "betting_lines": ["Vegas consensus", "sharp money moves"],
    "weather": ["game-day conditions for outdoor sports"],
}
```

#### 3.5.2 Sports-Specific Signals

| Signal Type | Source | Latency Edge |
|-------------|--------|--------------|
| Injury news | Beat reporter tweets | 5-30 min before markets adjust |
| Lineup changes | Official team accounts | 10-60 min edge |
| Weather shifts | NWS/weather APIs | Hours of edge for outdoor games |
| Sharp money | Betting line movements | Real-time correlation |

#### 3.5.3 Sports Wallet Tracking

Track wallets with high sports win rates:
- beachboy4, 0x492442Ea, Pimping → add to "Sports Specialists" watchlist
- Alert when they place new sports bets
- Cross-reference with live game data

### Phase 3.6: Backtesting & Strategy Validation

**Source:** @jayka003's backtesting tool (134K views, high demand)

Before deploying strategies live, validate with historical data:

#### 3.6.1 Backtesting Framework

```python
class StrategyBacktester:
    """Replay historical signals against historical market data."""

    def __init__(self, strategy, start_date, end_date):
        self.strategy = strategy
        self.historical_signals = load_signals(start_date, end_date)
        self.historical_prices = load_market_prices(start_date, end_date)

    def run(self):
        results = []
        for signal in self.historical_signals:
            # What would strategy have done?
            action = self.strategy.evaluate(signal)

            # What was actual outcome?
            outcome = self.get_resolution(signal.market_id)

            results.append({
                "signal": signal,
                "action": action,
                "outcome": outcome,
                "pnl": self.calculate_pnl(action, outcome)
            })

        return BacktestReport(results)
```

#### 3.6.2 Paper Trading Mode

- Real signals, simulated execution
- Track theoretical PnL before going live
- Validate signal quality without risk

### Phase 3.7: Profile Popularity Tracking

**Source:** @the_smart_ape analysis (320K profile views = most-watched trader)

High-attention traders create following effects:

#### 3.7.1 Profile Attention Signals

```python
def track_profile_attention(wallet_id):
    """Monitor which profiles are gaining attention."""

    metrics = {
        "profile_views_24h": get_polymarket_views(wallet_id),
        "x_mentions": count_twitter_mentions(wallet_id),
        "copy_traders": count_followers(wallet_id),
        "leaderboard_position_delta": get_rank_change(wallet_id),
    }

    # High attention + new trade = potential following effect
    if metrics["profile_views_24h"] > 100_000:
        return {
            "type": "high_attention_trader",
            "signal": "trades_may_move_market",
            "action": "monitor for front-running opportunities"
        }
```

#### 3.7.2 Attention-Based Signals

| Metric | Threshold | Signal |
|--------|-----------|--------|
| Profile views/24h | >100K | "Hot trader" - others watching |
| New followers/day | >500 | Rising influence |
| Leaderboard jump | +100 ranks | Momentum trader |
| X mentions | >50/day | Community buzz |

### Phase 4: Unique Data Moats

#### 4.1 Pentagon Pizza Index (Inspired by @pizzintwatch)

Alternative data signals for geopolitical markets:
- Late-night pizza orders at government buildings
- Flight tracking (Air Force One, diplomatic jets)
- Unusual activity patterns

#### 4.2 Social Graph Intelligence

Map relationships between:
- Polymarket wallets ↔ X/Twitter accounts (like Polycool does)
- Insider networks (who follows who, who copies who)
- Information flow patterns (who breaks news first)

#### 4.3 Resolution Oracle

Predict resolution disputes before they happen:
- Analyze market description for ambiguity
- Track historical disputes on similar markets
- Flag markets with unclear criteria

### Implementation Priority

| Priority | Feature | Effort | Impact | Dependencies |
|----------|---------|--------|--------|--------------|
| **P0** | News-to-Market Matcher | Medium | High | LLM, Market API |
| **P0** | Real-time X/Telegram ingestion | Medium | High | Twitter API, TG bot |
| **P0** | Sports Data Ingestion | Medium | **Very High** | ESPN API, injury feeds |
| **P0** | AI vs Market Mispricing Detection | Medium | **Very High** | News matcher, Market prices |
| **P1** | Wallet Specialty Classifier | Medium | High | Historical trade data |
| **P1** | Signal Confidence Scoring | Low | Medium | News matcher |
| **P1** | High-Conviction Detection | Low | High | Wallet data, trade stream |
| **P1** | Backtesting Module | Medium | High | Historical data, strategy engine |
| **P1** | Insider Detection Engine | Medium | **Very High** | Wallet data, Polysights API, DeBank |
| **P1** | Wallet Cluster Analysis | Medium | High | Graph analysis, timing correlation |
| **P2** | Fade List Generator | Low | Medium | Wallet data |
| **P2** | Profile Popularity Tracking | Low | Medium | Polymarket API, X API |
| **P2** | Auto-execution | High | High | CLOB API, risk mgmt |
| **P3** | Alt-data signals | High | Medium | Custom scrapers |

### Competitive Positioning

```
                    Manual                    Automated
                      ↑                           ↑
         ┌────────────┼───────────────────────────┤
         │            │                           │
 Whale   │  Existing  │                           │
 Tracking│   Tools    │       ← We compete here   │
         │  (Nexus,   │         on automation     │
         │  MobyScr.) │                           │
         │            │                           │
         ├────────────┼───────────────────────────┤
         │            │                           │
 Signal  │  Manual    │      ★ SYNESIS ★          │
 Intel-  │  Twitter   │   News → Market → Trade   │
 ligence │  scanning  │   Fully automated         │
         │            │                           │
         └────────────┴───────────────────────────┘
```

**Our moat:** LLM-powered signal interpretation + automated market matching.
Others track trades. We explain WHY and act FIRST.

---

## Warnings & Reality Check

### Sobering Statistics

- Only **0.51%** of wallets have PnL > $1,000
- Whale accounts (>$50K volume) = only 1.74%
- Bots achieve 85%+ win rates vs humans ~60%
- Bot turned $313 to $414K in one month

### Edge Erosion

Per @thejayden (Jan 16):
> "Most edges are gone. The only one that still exists and it's getting rarer by the day is..."

### Platform Changes

- Polymarket introduced **dynamic taker fees** on 15-min crypto markets
- Fees up to 3.15% at 50/50 odds (kills latency arbitrage margin)
- Designed to neutralize bot advantage

---

## Telegram Channels

### Signal & Alert Channels

| Channel | Link | Focus | Notes |
|---------|------|-------|-------|
| New Polymarkets | t.me/newpolymarkets | New market alerts | Get notified when new markets launch |
| YN Signals | t.me/YNSignals | Alpha signals | 24/7 aggregated signals, odds anomalies, insider alerts |
| Polytrage | t.me/PoIytrage | Arbitrage alerts | AI-powered arbitrage detection every 15 minutes |
| PolyAnalysis | t.me/the_PolyAnalysis01 | Data/analytics | Analysis and data support for traders |
| PolyInsiderBot | t.me/PredictionIns | Insider activity | Real-time insider detection alerts |

### Community Groups

| Group | Link | Focus | Notes |
|-------|------|-------|-------|
| Polymarket EN Support | t.me/PolymarketENSupport | Official support | ~10K members, tips and community help |
| Prediction Arc | t.me/thepredictionarc | Builders/traders | Active builder and trader community |
| insiders.bot | t.me/polyinsiders | Social trading | Copy trading and elite analytics |
| Pentagon Pizza Watch | t.me/pizzintwatchers | Geopolitical OSINT | Unique alt-data for geopolitics |
| Polymarket Bros | t.me/BrosOnPM | Community opinions | General discussion |
| Polymarket CN | t.me/polymarketCNGroup | Chinese community | Chinese-language traders |
| Polymarket Builders | Official | Builder program | Builder program updates |

---

## Evaluating Polymarket Influencers

### Green Flags (Likely Real Alpha)

1. **Links actual Polymarket profile** with verifiable PnL
2. **Specific strategies** with logic explained, not just "buy this"
3. **Built tools** actively used by community (check GitHub, product links)
4. **Transparent about losses** - discusses what didn't work
5. **Domain expertise** - deep focus on one area (politics, sports, geopolitics)
6. **Followed by credible accounts** - @thejayden, @poly_data, verified traders

### Red Flags (Likely Engagement Farming)

1. **No linked profile** - claims "top trader" without proof
2. **Heavy referral link pushing** - every post is "join via my link"
3. **Generic hype** - "prediction markets are the future" without substance
4. **Memecoin shilling** mixed with Polymarket content
5. **Reposts news** without analysis or original insight
6. **New account** with no trading history but bold claims
7. **Paid promotion** - sponsored posts for other projects

### Verification Steps

1. Check if they link their Polymarket profile
2. Look up profile on polymarketanalytics.com/traders
3. Search their handle + "polymarket" to see if others reference them
4. Check if tools they claim to build actually exist
5. Look for specific trade calls with timestamps (can verify later)

---

## Key Insight

The most profitable traders are:
1. **Insiders** (questionable legality)
2. **Bots** (require infrastructure)
3. **Domain specialists** with real information edges
4. **Patient bonders** collecting 5% on near-certain outcomes

**Retail without edge = gambling.**

---

*Last updated: 2026-01-24 (Added comprehensive insider detection module with @RetroValix 4-step method, wallet clustering via @gemchange_ltd, copy score formula, sports whale analysis, AI vs Market mispricing detection, backtesting module, profile popularity tracking)*
