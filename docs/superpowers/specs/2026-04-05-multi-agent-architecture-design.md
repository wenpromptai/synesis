# Multi-Agent Architecture Design for Synesis

## Context

Synesis currently runs 4 independent processing pipelines (News, Twitter, Market Brief, Event Radar) that don't cross-reference each other's output in any meaningful way. The Market Brief reads diary entries from other flows, but this is one-directional and shallow. Meanwhile, data providers (SEC EDGAR insider/XBRL/ownership data, Massive technicals/short interest, FRED macro trends) have deep capabilities that no pipeline touches.

The goal is to upgrade the system into a multi-agent architecture that:
1. **Specializes** — dedicated expert agents focus on their data domain
2. **Synthesizes** — a strategist layer connects dots across all specialist reports
3. **Challenges** — high-conviction trade ideas go through adversarial bull/bear debate before surfacing
4. **Produces** — daily intelligence briefs + actionable trade recommendations (options + stocks), no auto-execution

## Architecture: Hybrid Layered + Selective Debate

```
LAYER 1 — Specialist Analysts (parallel, daily)
  ├── SocialSentimentAnalyst   (upgraded Twitter agent)
  ├── EventCalendarAnalyst     (existing Event Radar)
  ├── InsiderFlowAnalyst       (NEW — SEC Form 4, 13F, XBRL)
  └── TechnicalAnalyst         (NEW — Massive technicals + yfinance options/vol)

  News Flow (Flow 1) remains real-time, independent.
  Its recent signals feed into Layer 2 as context.

LAYER 2 — Strategists (sequential, after Layer 1)
  ├── MacroStrategist          → MacroView
  └── EquityStrategist         → EquityIdeas (each with conviction float)

GATE: conviction >= 0.7 AND ticker appears in 2+ Layer 1 reports

LAYER 2.5 — Bull/Bear Debate (only for gated ideas, 0-2/day typical)
  ├── BullAdvocate(idea, context)    → BullCase
  ├── BearAdvocate(idea, bull_case)  → BearCase
  └── Adjudicator(bull, bear)        → AdjudicatedIdea

LAYER 3 — Brief Compiler (deterministic Python, no LLM)
  └── Assembles DailyIntelligenceBrief from all outputs
```

## Layer 1: Specialist Analysts

All Layer 1 agents run in parallel via `asyncio.gather`, following the existing `yesterday/__init__.py` sub-analyzer pattern. Each produces a typed Pydantic output model.

### SocialSentimentAnalyst

**What it replaces**: The current `TwitterAgentAnalyzer` in `processing/twitter/analyzer.py`.

**What changes**: Per the existing `docs/twitterplan.md` refactor — remove `get_quote` and `get_options_snapshot` tools, remove yfinance dependency, replace `direction`+`conviction` with `sentiment_score` float (-1 to +1). The agent focuses purely on extracting themes, sentiment, and trade ideas from tweets + web research.

**Tools**: `verify_ticker`, `web_search` (5-7 calls), `web_read` (paired with each search)
**Model**: Sonnet (vsmart tier)
**Output**: `TwitterAgentAnalysis` (existing model, with sentiment_score update from twitterplan.md)

### EventCalendarAnalyst

**What it is**: The existing Event Radar digest pipeline (`processing/events/`). No structural changes — it already uses the sub-analyzer → consolidator pattern.

**Output**: `YesterdayBriefAnalysis` (existing model)

### InsiderFlowAnalyst (NEW)

**Purpose**: Surface insider trading patterns, institutional position changes, and corporate action signals from SEC EDGAR data that currently goes unused.

**Data sources** (all existing in `providers/sec_edgar/client.py`):
- `InsidersMixin.get_insider_transactions(ticker)` — Form 4 buys/sells
- `InsidersMixin.get_derivative_transactions(ticker)` — options exercises
- `InsidersMixin.get_insider_sentiment(ticker)` — monthly share purchase ratio
- `ThirteenFMixin.get_13f_holdings(filing, cik)` — institutional holdings
- `XBRLMixin.get_historical_eps(ticker)` — EPS trend
- `OwnershipMixin.get_form144_filings(ticker)` — insider sale intentions
- `FeedsMixin.get_filing_feed(form_type)` — real-time filing feed

**Tools** (wrapped as PydanticAI tools on the agent):
- `get_insider_transactions(ticker)` — recent Form 4 filings
- `get_filing_feed()` — last 24h of 8-K and Form 4 filings
- `get_insider_sentiment(ticker)` — MSPR conviction metric
- `get_company_fundamentals(ticker)` — XBRL EPS/revenue trend
- `get_form144(ticker)` — insider sale intentions

**Scope**: Scans all active watchlist tickers (typically 10-30). The agent identifies:
- Cluster selling/buying (multiple insiders in same company within 7 days)
- Unusual transaction sizes relative to the insider's history
- Form 144 filings (advance notice of intended sales)
- Convergence with earnings dates (insider sells before earnings = red flag)

**Model**: Haiku (structured data extraction, not deep reasoning)
**Output**: `InsiderFlowAnalysis`

```python
class InsiderSignal(BaseModel):
    ticker: str
    signal_type: Literal["cluster_buy", "cluster_sell", "large_transaction", "form144_sale", "options_exercise"]
    insiders: list[str]  # names
    total_value_usd: float
    sentiment_score: float  # -1 to +1
    context: str  # what happened and why it matters
    days_to_earnings: int | None = None

class InsiderFlowAnalysis(BaseModel):
    signals: list[InsiderSignal]
    summary: str  # 2-3 sentence overview
    tickers_scanned: int
```

### TechnicalAnalyst (NEW)

**Purpose**: Surface technical setups, volatility signals, and short interest data from Massive.com and yfinance — providers that are currently unwired.

**Data sources**:
- `MassiveClient.get_technical_indicators(ticker, type, timespan)` — SMA, EMA, MACD, RSI, Bollinger, ADX
- `MassiveClient.get_macd(ticker, timespan)` — MACD with histogram
- `MassiveClient.get_short_interest(ticker)` — short interest ratio
- `MassiveClient.get_bars(ticker, multiplier, timespan)` — OHLCV for volume analysis
- `YFinanceClient.get_options_snapshot(ticker)` — ATM options with Greeks (IV data)

**Tools**:
- `get_technicals(ticker)` — fetches RSI + MACD + Bollinger in one call
- `get_short_interest(ticker)` — short interest data
- `get_volume_profile(ticker)` — recent OHLCV bars for volume spike detection
- `get_options_vol(ticker)` — ATM IV from yfinance options snapshot

**Rate limit constraint**: Massive is 5 calls/min. The agent scans top 5 watchlist tickers by recency/conviction. Each ticker needs ~2-3 Massive API calls, so batch carefully with delays.

**Model**: Haiku (structured data extraction)
**Output**: `TechnicalScanResult`

```python
class TechnicalSetup(BaseModel):
    ticker: str
    rsi: float | None = None
    macd_signal: Literal["bullish_cross", "bearish_cross", "neutral"] | None = None
    bollinger_position: Literal["above_upper", "below_lower", "within"] | None = None
    short_interest_ratio: float | None = None
    iv_rank: float | None = None  # current IV vs 52-week range
    volume_vs_avg: float | None = None  # ratio vs 20-day avg
    setup_type: Literal["breakout", "breakdown", "oversold_bounce", "squeeze", "high_iv", "none"]
    summary: str

class TechnicalScanResult(BaseModel):
    setups: list[TechnicalSetup]
    notable_tickers: list[str]  # tickers with actionable setups
    scan_timestamp: str
```

## Layer 2: Strategists

Layer 2 agents receive all Layer 1 outputs formatted as text context in their system prompt. They run sequentially after Layer 1 completes.

### MacroStrategist

**Purpose**: Identify the macro regime and how it affects sectors/positioning.

**Input context**: EventCalendarAnalyst themes (macro category) + SocialSentiment macro themes + recent news signals with macro tags + FRED data

**Tools**:
- `get_fred_observations(series_id)` — fetch FRED time series (CPI, unemployment, GDP, rates)
- `web_search(query)` — 2 calls max, for verification of macro claims

**Task**: Assess the current regime (risk-on/risk-off, tightening/easing, growth/recession), map sector implications, flag regime change signals.

**Model**: Sonnet
**Output**: `MacroView`

```python
class MacroView(BaseModel):
    regime: Literal["risk_on", "risk_off", "transitioning", "uncertain"]
    regime_confidence: float  # 0-1
    key_drivers: list[str]  # 3-5 macro factors driving the regime
    sector_tilts: list[SectorTilt]  # which sectors benefit/suffer
    risks: list[str]  # what could change the regime
    macro_trade_ideas: list[MacroTradeIdea]  # 1-3 macro-level ideas

class SectorTilt(BaseModel):
    sector: str
    direction: Literal["overweight", "underweight", "neutral"]
    reasoning: str

class MacroTradeIdea(BaseModel):
    thesis: str
    instruments: list[str]  # tickers or asset classes
    direction: Literal["long", "short", "hedge"]
    catalyst: str
    timeframe: str
```

### EquityStrategist

**Purpose**: Find convergence signals — tickers that appear in 2+ Layer 1 reports — and synthesize into actionable trade ideas.

**Input context**: All Layer 1 outputs, filtered to highlight overlapping tickers. The orchestrator pre-computes a convergence map — a `dict[str, list[str]]` mapping each ticker to the Layer 1 reports that mention it (e.g., `{"NVDA": ["social_sentiment", "insider_flow", "technical"]}`) — and includes it in the EquityStrategist's system prompt so the agent knows where to focus.

**Tools**:
- `get_quote(ticker)` — current price (yfinance)
- `get_options_snapshot(ticker)` — ATM options with Greeks (3 calls max)
- `web_search(query)` — 2 calls max

**Task**: For each converging ticker, synthesize the signals. Example: NVDA mentioned in Twitter (bullish AI thesis) + insider selling (Form 4) + bearish MACD crossover = conflicting signals worth flagging. Produce ranked trade ideas with conviction scores.

**Model**: Sonnet
**Output**: `EquityIdeas`

```python
class TradeIdea(BaseModel):
    ticker: str
    direction: Literal["long", "short", "hedge", "watch"]
    conviction: float  # 0.0 to 1.0
    thesis: str  # 2-3 sentences
    supporting_signals: list[str]  # which Layer 1 reports support this
    conflicting_signals: list[str]  # what argues against
    structure: str  # "buy shares", "buy calls", "put spread", etc.
    catalyst: str  # what triggers the move
    timeframe: str  # "1-3 days", "1-2 weeks", etc.
    entry_context: str  # price level, IV context, support/resistance
    source_count: int  # number of Layer 1 reports mentioning this ticker

class EquityIdeas(BaseModel):
    ideas: list[TradeIdea]  # ranked by conviction
    convergence_summary: str  # overview of cross-signal patterns
```

## Layer 2.5: Multi-Round Bull/Bear Debate (LangGraph Loop)

### Gate Criteria

A `TradeIdea` enters the debate if ALL of:
1. `conviction >= 0.7`
2. `source_count >= 2` (appears in 2+ Layer 1 reports)
3. `direction` is `"long"` or `"short"` (not `"watch"` or `"hedge"`)

Typical: 0-2 ideas per day pass the gate. On busy days (earnings + macro event): maybe 3.

### Debate Flow (2 rounds default, configurable)

```
Round 1: BullAdvocate → BearAdvocate → (counter increments)
Round 2: BullAdvocate (rebuttal) → BearAdvocate (rebuttal) → (counter increments)
Exit:    Adjudicator (synthesizes all rounds)
```

The bear advocate node increments `debate_round`. The conditional edge `should_continue_debate` checks `debate_round < MAX_DEBATE_ROUNDS` — if so, loops back to bull; if not, routes to adjudicator.

In round 2+, the bull and bear see the full debate history (accumulated via `Annotated[list[str], add]` reducer on `debate_messages`), so the bull can rebut the bear's specific arguments and vice versa. This catches strawman arguments that a single pass would miss.

### BullAdvocate

**Model**: Haiku (no tools, pure argumentation)
**Input**: The `TradeIdea` + Layer 1 context + prior `debate_messages` (empty on round 1)
**System prompt**: Round 1: "Make the strongest possible case for this trade. Be specific about entry, sizing, catalysts, and expected payoff." Round 2+: "The bear has responded. Address their specific counterpoints. What did they get wrong? What evidence supports your thesis over theirs?"
**Output appended to `debate_messages`**

### BearAdvocate

**Model**: Haiku (no tools)
**Input**: The `TradeIdea` + Layer 1 context + all `debate_messages` (including latest bull)
**System prompt**: Round 1: "Find what the bull case missed. What data contradicts the thesis? What scenarios lose money?" Round 2+: "The bull has rebutted. Were their counterpoints valid? What are they still anchored on? Sharpen your strongest arguments."
**Output appended to `debate_messages`**, increments `debate_round`

### Adjudicator

**Model**: Sonnet
**Tools**: `web_search` (1 call max — only to verify a specific factual dispute)
**Input**: Original `TradeIdea` + full `debate_messages` (all rounds)
**System prompt**: "Two analysts have debated this trade over multiple rounds. Weigh both final positions. Adjust conviction. If the bear case reveals a fatal flaw, reject the idea. If the bull case holds after scrutiny, refine the trade structure."
**Output**: `AdjudicatedIdea`

```python
class AdjudicatedIdea(BaseModel):
    ticker: str
    original_conviction: float
    revised_conviction: float  # post-debate
    final_direction: Literal["long", "short", "pass"]
    verdict: str  # 2-3 sentence summary of why
    bull_summary: str  # strongest bull arguments that survived debate
    bear_summary: str  # strongest bear arguments that survived debate
    refined_structure: str  # final trade structure recommendation
    key_risk: str  # the single biggest risk to monitor
    stop_loss_trigger: str  # what would make you exit
    position_sizing_note: str  # relative sizing guidance
```

## Layer 3: Brief Compiler (Deterministic)

No LLM. A Python function that:

1. Takes `MacroView` + `EquityIdeas` + `AdjudicatedIdea`s (if any) + Layer 1 summaries
2. Ranks ideas by final conviction (adjudicated conviction for debated ideas, raw conviction for others)
3. Assembles `DailyIntelligenceBrief`
4. Formats Discord embeds
5. Persists to diary table

```python
class DailyIntelligenceBrief(BaseModel):
    headline: str  # from MacroStrategist
    date: str
    macro_regime: MacroView
    trade_ideas: list[TradeIdea | AdjudicatedIdea]  # ranked by conviction
    quick_takes: list[str]  # low-conviction or watch-only ideas, 1 line each
    risk_radar: list[str]  # aggregated from MacroView risks + BearAdvocate risks
    watchlist_updates: list[str]  # tickers added/removed with reason
    insider_highlights: list[InsiderSignal]  # top insider signals
    technical_setups: list[TechnicalSetup]  # notable setups
```

### Discord Output Format

- **Header embed**: Headline + macro regime + sector tilts
- **Trade idea embeds** (one per idea): Ticker, direction, conviction bar, thesis, structure. For adjudicated ideas: expandable bull/bear summary
- **Supporting context embed**: Insider highlights, technical setups, risk radar
- **Watchlist embed**: Changes with reasons

## Database: `raw_tweets` Table

New table for persisting raw tweets before analysis. The Twitter agent job fetches last 24h of tweets from curated accounts, saves each tweet as a row, then the SocialSentimentAnalyst reads from this table.

```sql
CREATE TABLE raw_tweets (
    id BIGSERIAL PRIMARY KEY,
    account_username TEXT NOT NULL,
    tweet_text TEXT NOT NULL,
    tweet_timestamp TIMESTAMPTZ NOT NULL,     -- when the tweet was posted
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- when we fetched it
    tweet_url TEXT,
    UNIQUE (account_username, tweet_timestamp, md5(tweet_text))
);

CREATE INDEX idx_raw_tweets_fetched ON raw_tweets (fetched_at DESC);
CREATE INDEX idx_raw_tweets_account ON raw_tweets (account_username, tweet_timestamp DESC);
```

**Retention**: Keep 30 days, prune via scheduler job (or TimescaleDB retention policy if converted to hypertable).

## Orchestration: LangGraph StateGraph

The pipeline is orchestrated as a LangGraph `StateGraph` rather than manual asyncio. This gives us:
- Shared state dict with reducers for safe concurrent writes
- Conditional edges for debate routing
- Counter-based loop control for multi-round debate
- Fan-out/fan-in for parallel specialist execution

### State Definition

```python
from __future__ import annotations
from typing import Annotated, TypedDict
from operator import add

class IntelligenceState(TypedDict):
    # --- Inputs (set at invocation) ---
    watchlist_tickers: list[str]
    raw_tweets: list[dict]        # from raw_tweets table (last 24h)
    recent_signals: list[dict]    # from signals table (Flow 1 outputs, last 24h)
    calendar_events: list[dict]   # from calendar_events table

    # --- Layer 1 reports (appended by each specialist via reducer) ---
    reports: Annotated[list[dict], add]

    # --- Layer 2 outputs ---
    macro_view: dict
    equity_ideas: list[dict]

    # --- Debate state ---
    debate_round: int
    debate_messages: Annotated[list[str], add]
    adjudicated_ideas: Annotated[list[dict], add]

    # --- Final output ---
    brief: dict
```

### Graph Topology

```python
from langgraph.graph import StateGraph, START, END

def build_intelligence_graph(
    sec_edgar: SECEdgarClient,
    massive: MassiveClient,
    yfinance: YFinanceClient,
    fred: FREDClient,
    db: Database,
) -> CompiledGraph:
    """Build the intelligence pipeline. Called once at startup."""

    # Node functions capture provider deps via closure
    async def social_sentiment_node(state): ...
    async def event_calendar_node(state): ...
    async def insider_flow_node(state): ...
    async def technical_node(state): ...
    async def macro_strategist_node(state): ...
    async def equity_strategist_node(state): ...
    async def bull_advocate_node(state): ...
    async def bear_advocate_node(state): ...
    async def adjudicator_node(state): ...
    async def compiler_node(state): ...

    graph = StateGraph(IntelligenceState)

    # Layer 1: parallel fan-out from START
    graph.add_node("social_sentiment", social_sentiment_node)
    graph.add_node("event_calendar", event_calendar_node)
    graph.add_node("insider_flow", insider_flow_node)
    graph.add_node("technical", technical_node)
    graph.add_edge(START, "social_sentiment")
    graph.add_edge(START, "event_calendar")
    graph.add_edge(START, "insider_flow")
    graph.add_edge(START, "technical")

    # Layer 2: fan-in (waits for all Layer 1), then sequential
    graph.add_node("macro_strategist", macro_strategist_node)
    graph.add_node("equity_strategist", equity_strategist_node)
    graph.add_edge("social_sentiment", "macro_strategist")
    graph.add_edge("event_calendar", "macro_strategist")
    graph.add_edge("insider_flow", "macro_strategist")
    graph.add_edge("technical", "macro_strategist")
    graph.add_edge("macro_strategist", "equity_strategist")

    # Gate: conditional routing after equity strategist
    graph.add_conditional_edges("equity_strategist", gate_for_debate)

    # Debate loop
    graph.add_node("bull_advocate", bull_advocate_node)
    graph.add_node("bear_advocate", bear_advocate_node)
    graph.add_node("adjudicator", adjudicator_node)
    graph.add_edge("bull_advocate", "bear_advocate")
    graph.add_conditional_edges("bear_advocate", should_continue_debate)
    graph.add_edge("adjudicator", "compiler")

    # Compiler
    graph.add_node("compiler", compiler_node)
    graph.add_edge("compiler", END)

    return graph.compile()
```

### Multi-Round Debate (2 rounds default)

```python
MAX_DEBATE_ROUNDS = 2  # bull → bear → bull → bear = 2 full rounds

def should_continue_debate(state: IntelligenceState) -> Literal["bull_advocate", "adjudicator"]:
    if state["debate_round"] < MAX_DEBATE_ROUNDS:
        return "bull_advocate"
    return "adjudicator"

def gate_for_debate(state: IntelligenceState) -> Literal["bull_advocate", "compiler"]:
    gated = [
        i for i in state["equity_ideas"]
        if i["conviction"] >= 0.7 and i["source_count"] >= 2
        and i["direction"] in ("long", "short")
    ]
    return "bull_advocate" if gated else "compiler"
```

### Daily Invocation

```python
async def run_daily_intelligence(compiled_graph, db, redis, watchlist):
    """Triggered daily by scheduler."""
    # 1. Fetch tweets and save to raw_tweets table
    tweets = await fetch_and_store_tweets(db)

    # 2. Gather inputs
    state = {
        "watchlist_tickers": await watchlist.get_active_tickers(),
        "raw_tweets": tweets,
        "recent_signals": await db.get_recent_signals(hours=24),
        "calendar_events": await db.get_calendar_events_for_digest(),
        "reports": [],
        "macro_view": {},
        "equity_ideas": [],
        "debate_round": 0,
        "debate_messages": [],
        "adjudicated_ideas": [],
        "brief": {},
    }

    # 3. Run the graph
    result = await compiled_graph.ainvoke(state, config={"recursion_limit": 25})

    # 4. Persist + notify
    await db.upsert_diary_entry("daily_brief", result["brief"])
    await send_intelligence_brief_discord(result["brief"])
```

### Scheduler Integration

```python
# Before: 3 separate daily jobs
# scheduler.add_job(twitter_agent_job, ...)
# scheduler.add_job(event_digest_job, ...)
# scheduler.add_job(market_brief_job, ...)

# After: 1 unified pipeline + event fetch stays separate
# scheduler.add_job(event_fetch_job, CronTrigger(hour=22), ...)
# scheduler.add_job(run_daily_intelligence, CronTrigger(hour=14, minute=30), ...)
```

### News Flow (Flow 1) Stays Independent

Flow 1 (real-time news → impact scoring → LLM analysis) continues to run independently on its own Redis queue. Its outputs are stored in the `signals` table and read by Layer 2 as "recent news signals." This preserves real-time responsiveness for breaking news.

### Market Brief (Flow 3) Is Subsumed

The current Market Brief (`processing/market/`) is replaced by the intelligence pipeline. Its responsibilities — market snapshot, event context, twitter context, web search, diary persistence — are distributed across the new layers. The MacroStrategist covers macro context, the EquityStrategist covers ticker-level synthesis, and the Brief Compiler handles diary persistence and Discord output. The `market_brief_job` scheduler entry is removed.

## New Files

```
src/synesis/processing/intelligence/
├── __init__.py
├── graph.py                 # build_intelligence_graph() — LangGraph StateGraph definition
├── state.py                 # IntelligenceState TypedDict
├── models.py                # All new Pydantic models (InsiderSignal, TechnicalSetup, MacroView, etc.)
├── specialists/
│   ├── __init__.py
│   ├── social_sentiment.py  # SocialSentimentAnalyst (refactored Twitter agent)
│   ├── insider_flow.py      # InsiderFlowAnalyst agent + tools
│   └── technical.py         # TechnicalAnalyst agent + tools
├── strategists/
│   ├── __init__.py
│   ├── macro.py             # MacroStrategist agent + tools
│   └── equity.py            # EquityStrategist agent + tools
├── debate/
│   ├── __init__.py
│   ├── bull.py              # BullAdvocate agent
│   ├── bear.py              # BearAdvocate agent
│   └── adjudicator.py       # Adjudicator agent
└── compiler.py              # Brief compilation (deterministic, no LLM)
```

**Modified files**:
- `database/init.sql` — add `raw_tweets` table + `sentiment_score` column on watchlist
- `storage/database.py` — tweet CRUD, recent signals query, diary persistence
- `processing/twitter/job.py` — fetch + save tweets to `raw_tweets` table
- `processing/twitter/models.py` — sentiment_score replaces direction+conviction
- `processing/twitter/analyzer.py` — remove yfinance tools, pure analysis
- `agent/scheduler.py` — replace 3 daily jobs with 1 unified pipeline trigger
- `agent/__main__.py` — wire up LangGraph compiled graph + providers
- `notifications/discord.py` — new embed formatters for intelligence brief
- `api/routes/` — new endpoint to trigger intelligence pipeline manually

**New dependency**: `langgraph` (v1.1+)

## Cost Estimate

| Agent | Model | Est. Cost |
|---|---|---|
| SocialSentimentAnalyst | Sonnet (vsmart) | ~$0.15 |
| EventCalendarAnalyst | Sonnet x3 + vsmart | ~$0.15 |
| InsiderFlowAnalyst | Haiku | ~$0.01 |
| TechnicalAnalyst | Haiku | ~$0.01 |
| MacroStrategist | Sonnet | ~$0.08 |
| EquityStrategist | Sonnet | ~$0.12 |
| Debate (avg 1.5 ideas/day, 2 rounds) | Haiku x4 + Sonnet | ~$0.20 |
| Brief Compiler | None | $0.00 |
| **Total** | | **~$0.70-1.20/day** |

Wall clock: ~80-130s (Layer 1 parallel ~30-60s, Layer 2 ~20-30s, Debate ~20-30s per idea with 2 rounds).

## Verification

1. **Unit tests**: Each specialist, strategist, and debate agent gets its own test file with mocked provider responses
2. **Integration test**: `run_daily_intelligence()` end-to-end with real API calls (marked `@pytest.mark.integration`)
3. **Graph visualization**: `app.get_graph().draw_mermaid()` to verify topology
4. **Manual trigger**: `/api/v1/intelligence/trigger` endpoint for on-demand runs
5. **Lint/type check**: `uv run ruff check --fix . && uv run ruff format . && uv run mypy src/`
6. **Output review**: Discord embed visual inspection
7. **Cost monitoring**: Log token usage per agent per run

## Implementation Phases

### Phase 1: Twitter Refactor + Tweets Table (standalone, ships independently)
1. Add `raw_tweets` table to `database/init.sql` + apply via ALTER TABLE
2. Add `sentiment_score` column to watchlist table
3. Refactor Twitter models (`sentiment_score` replaces `direction`+`conviction`)
4. Refactor Twitter analyzer (remove yfinance tools, pure analysis)
5. Refactor Twitter job (fetch → save to `raw_tweets` → analyze)
6. Update database layer (tweet CRUD, `upsert_watchlist_ticker` with sentiment_score)
7. Update Discord formatter, API routes, tests

### Phase 2: New Specialist Agents (standalone, testable independently)
8. Add `langgraph` dependency
9. Create `intelligence/models.py` — all new Pydantic output types
10. Build InsiderFlowAnalyst — wire SEC EDGAR insider/XBRL/Form 144 data
11. Build TechnicalAnalyst — wire Massive technicals + yfinance IV
12. Unit tests for both specialists with mocked providers

### Phase 3: LangGraph Pipeline (wires everything together)
13. Define `IntelligenceState` TypedDict with reducers
14. Build MacroStrategist + EquityStrategist (Layer 2 agents)
15. Build BullAdvocate + BearAdvocate + Adjudicator (debate agents)
16. Build Brief Compiler (deterministic)
17. Build `graph.py` — full StateGraph with fan-out, fan-in, gate, debate loop
18. Integration test the compiled graph end-to-end

### Phase 4: Production Integration
19. Update scheduler — replace 3 daily jobs with 1 pipeline trigger
20. Update `agent/__main__.py` — compile graph at startup, wire providers
21. New Discord embed formatters for intelligence brief
22. API endpoint for manual trigger (`/api/v1/intelligence/trigger`)
23. Retire Market Brief (subsumed by pipeline)
