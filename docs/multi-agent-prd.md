# Multi-Agent Intelligence Pipeline — PRD

## Vision

Upgrade Synesis from 4 independent pipelines into a LangGraph-orchestrated multi-agent system that produces a daily intelligence brief + actionable trade recommendations (stocks + options).

## Phases

### Phase 1: Pure Tweet Data Collection [DONE]
**Goal**: Twitter pipeline becomes pure data collection — fetch and store tweets, no LLM.

**Changes**:
- `raw_tweets` table with composite PK `(account_username, tweet_id)` — stores all fetched tweets, DB handles dedup
- Twitter job stripped to: fetch tweets from accounts → store to `raw_tweets` → done
- Deleted `analyzer.py` (LLM agent) and `models.py` (output types) — analysis moves to Phase 3 SocialSentimentAnalyst
- Removed Twitter Discord digest, watchlist auto-add, diary persistence
- `accounts.py` preserved — account profiles/biases reused in Phase 3

**Patterns preserved for Phase 3 SocialSentimentAnalyst**:
- Account bias/credibility weighting (accounts.py + get_profile())
- Web search + web_read tool pattern with budget caps
- verify_ticker tool
- Tweet formatting grouped by account with timestamps
- Cross-confirmation priority (themes from multiple accounts)
- From TradingAgents: mandatory structured table output, current date injection

**Files**: `database/init.sql`, `storage/database.py`, `twitter/job.py`, `agent/__main__.py`, `notifications/discord.py`, tests

---

### Phase 2: New Specialist Agents
**Goal**: Build 2 new data-gathering agents that wire up underused providers.

**InsiderFlowAnalyst** (Haiku):
- Scans watchlist tickers via SEC EDGAR: Form 4 insider transactions, insider sentiment (MSPR), Form 144 sale intentions, XBRL EPS/revenue, real-time filing feed
- Identifies: cluster selling/buying, unusual transaction sizes, pre-earnings insider activity
- Output: `InsiderFlowAnalysis` (list of `InsiderSignal`)

**TechnicalAnalyst** (Haiku):
- Scans top 5 watchlist tickers via Massive.com + yfinance: RSI, MACD, Bollinger, short interest, volume profile, ATM IV
- Identifies: breakouts, breakdowns, oversold bounces, IV squeezes
- Output: `TechnicalScanResult` (list of `TechnicalSetup`)
- Rate limit: Massive is 5 calls/min, batch carefully

**New files**: `processing/intelligence/models.py`, `specialists/insider_flow.py`, `specialists/technical.py`
**New dependency**: `langgraph` (v1.1+)

---

### Phase 3: LangGraph Pipeline
**Goal**: Wire all agents into a LangGraph StateGraph with multi-round debate.

**LangGraph State Machine**:
```
Layer 1 (parallel fan-out):
  SocialSentimentAnalyst | EventCalendarAnalyst | InsiderFlowAnalyst | TechnicalAnalyst

Layer 2 (fan-in, sequential):
  MacroStrategist → EquityStrategist

Gate: conviction >= 0.7 AND 2+ source convergence

Layer 2.5 (debate loop, 2 rounds):
  BullAdvocate ↔ BearAdvocate → Adjudicator

Layer 3 (deterministic):
  Brief Compiler → DailyIntelligenceBrief
```

**New agents**:
- **MacroStrategist** (Sonnet): Regime assessment (risk-on/off), sector tilts, macro trade ideas. Tools: FRED observations, web search
- **EquityStrategist** (Sonnet): Cross-signal convergence synthesis. Pre-computed convergence map shows which tickers appear in 2+ Layer 1 reports. Tools: quotes, options snapshots, web search
- **BullAdvocate** (Haiku): Argues strongest case for each gated trade idea. No tools, pure argumentation
- **BearAdvocate** (Haiku): Attacks the thesis, finds contradictions. No tools
- **Adjudicator** (Sonnet): Weighs both sides, adjusts conviction, produces final recommendation. 1 web search max
- **Brief Compiler** (no LLM): Ranks ideas by conviction, assembles `DailyIntelligenceBrief`

**State**: `IntelligenceState` TypedDict with `Annotated` reducers for safe concurrent writes. Deps passed via closure factory.

**New files**: `intelligence/state.py`, `intelligence/graph.py`, `strategists/macro.py`, `strategists/equity.py`, `debate/bull.py`, `debate/bear.py`, `debate/adjudicator.py`, `intelligence/compiler.py`

---

### Phase 4: Production Integration
**Goal**: Replace existing scheduler jobs with the unified pipeline. Ship to Discord.

**Changes**:
- Replace 3 daily scheduler jobs (twitter, events, market brief) with 1 unified pipeline trigger
- Compile LangGraph at startup in `agent/__main__.py`, wire all provider clients via closure
- New Discord embed formatters for intelligence brief (header + trade ideas + supporting context + watchlist)
- Debated ideas show bull/bear summary in embeds
- API endpoint: `POST /api/v1/intelligence/trigger` for manual runs
- Retire Market Brief (subsumed by MacroStrategist + EquityStrategist + Compiler)
- Event fetch job stays separate (populates DB before pipeline runs)

---

## Output: Daily Intelligence Brief

Lands in Discord each morning (~10:30am ET):
1. **Macro Regime** — risk-on/off, confidence, key drivers, sector tilts
2. **Trade Ideas** (ranked by conviction) — ticker, direction, thesis, structure, catalyst. Debated ideas include bull/bear summary + stop-loss trigger
3. **Quick Takes** — lower conviction / watch-only ideas, 1 line each
4. **Supporting Context** — top insider signals, technical setups, risk radar
5. **Watchlist Updates** — tickers added/removed with reasons

## Cost & Performance

| Component | Model | Est. Cost/Day |
|---|---|---|
| SocialSentimentAnalyst | Sonnet (vsmart) | ~$0.15 |
| EventCalendarAnalyst | Sonnet x3 + vsmart | ~$0.15 |
| InsiderFlowAnalyst | Haiku | ~$0.01 |
| TechnicalAnalyst | Haiku | ~$0.01 |
| MacroStrategist | Sonnet | ~$0.08 |
| EquityStrategist | Sonnet | ~$0.12 |
| Debate (avg 1.5 ideas/day, 2 rounds) | Haiku x4 + Sonnet | ~$0.20 |
| **Total** | | **~$0.70-1.20** |

Wall clock: ~80-130s end-to-end.

## Specs & Skills

- Architecture spec: `docs/superpowers/specs/2026-04-05-multi-agent-architecture-design.md`
- Phase 1 detail: `docs/twitterplan.md`
- LangGraph skill: `.claude/skills/langgraph-developing/`
