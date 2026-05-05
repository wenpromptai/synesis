# Intelligence Pipeline Architecture

On-demand LangGraph state machine that takes a list of tickers and produces structured per-ticker bull/bear briefs with configurable multi-round debate and trade idea generation.

Source: `src/synesis/processing/intelligence/`  
Triggered via: `POST /api/v1/intelligence/analyze`

## Graph Topology

```
                         START
                         /   \
          ticker_research     route_to_L2()               ← parallel from START
                         \    /  \
                          L2:    L2: ...                   ← Send per ticker
                   company_analyst  price_analyst          ← Layer 2 (parallel fan-out)
                         \        /
                          l2_join                          ← Sync barrier (defer=True)
                              |
                        debate_router()                    ← Conditional: configurable by debate_rounds
                       /        |        \
            [rounds=0]    [rounds>=1]  [no tickers]
                |              |              |
     +-----+-----+      ticker_debate         |
     |           |       subgraph:            |
bull_researcher  |    bull⇄bear loop          |            ← Layer 3: Debate
bear_researcher  |    N rounds                |
(per ticker,     |         |                  |
 parallel)       |         |                  |
     +-----+-----+         |                  |
              \             |                /
                      trader_gate                          ← Sync barrier (defer=True)
                            |
                     trader_router()                       ← Conditional: configurable by trader_mode
                     /              \
          [tickers exist]      [no tickers]
           /          \              |
    [per_ticker]  [portfolio]        |                     ← Layer 4: Trader
         |              |            |
      trader          trader         |
    (per ticker     (single call     |
     via Send)       all tickers)    |
         |              |            |
         +--------------+           /
                  \                /
                      analyze_compiler                     ← Deterministic assembly (defer=True)
                            |
                           END
```

Ref: `graph.py` — `build_analyze_graph()` function

## State (`AnalyzeState`)

Defined in `state.py`.

| Field              | Type                        | Reducer        | Writer                                          |
| ------------------ | --------------------------- | -------------- | ----------------------------------------------- |
| `current_date`     | `str`                       | overwrite      | input                                           |
| `target_tickers`   | `list[str]`                 | overwrite      | input (from POST body)                          |
| `ticker_research`  | `dict`                      | overwrite      | ticker_research node                            |
| `company_analyses` | `list[dict]`                | `add` (append) | company_analyst × N                             |
| `price_analyses`   | `list[dict]`                | `add` (append) | price_analyst × N                               |
| `bull_analyses`    | `list[dict]`                | `add` (append) | bull_researcher × N or ticker_debate × N        |
| `bear_analyses`    | `list[dict]`                | `add` (append) | bear_researcher × N or ticker_debate × N        |
| `trade_ideas`      | `list[dict]`                | `add` (append) | trader × N                                      |
| `portfolio_note`   | `str`                       | overwrite      | trader (portfolio mode only)                    |
| `brief`            | `dict`                      | overwrite      | analyze_compiler                                |

All `add` reducer fields accumulate results from parallel Send fan-out nodes (one per ticker). When using multi-round debate (`rounds>=1`), each ticker contributes multiple bull/bear entries — compiler takes the last round per ticker.

## Ticker Research (parallel with Layer 2)

Source: `specialists/ticker_research/agent.py`

Runs in parallel with the Layer 2 fan-out from START. Researches all requested tickers in a single call using Twitter and web search to surface recent catalysts, sentiment, and news context.

- **Input**: `target_tickers` list, optional Twitter client
- **Tools**: `web_search`, `web_read`, `get_tweets` (if Twitter key available)
- **Output**: `ticker_research` dict — per-ticker research context used by debate and trader agents

## Layer 2: Multi-Agent Analysis (parallel fan-out)

`route_to_L2()` conditional edge uses `Send` to fan out per ticker:
- One `company_analyst` + one `price_analyst` per ticker via `Send("node", {**state, "ticker": t})`

All run in parallel with `ticker_research`.

### CompanyAnalyst (per ticker)

Source: `specialists/company/agent.py`, scoring in `specialists/company/scoring.py`

Three-phase pipeline — only Phase 3 uses LLM:

| Phase                  | What                                                                                                                                                                                                   | Source                        |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------- |
| 1. Data Gather         | `asyncio.gather`: yfinance fundamentals + quarterly financials + analyst ratings, SEC EDGAR insider txns + sentiment + Form 144 + late filing alerts, 10-K/10-Q text, 8-K events | yfinance, SEC EDGAR, Crawl4AI |
| 2. Deterministic Score | Piotroski F-Score (9 signals from quarterly data), InsiderSignal (MSPR, buy/sell counts, cluster detection, C-suite activity), AnalystConsensus (Buy/Hold/Sell counts, price targets, recent upgrades/downgrades), RedFlag detection (late filings, cash flow divergence) | Phase 1 data |
| 3. LLM Synthesis       | business_summary, earnings_quality, risk_assessment, geographic_exposure, key_customers_suppliers, forward_outlook, competitive_position, insider_vs_financials, disclosure_consistency, primary_thesis, key_risks, monitoring_triggers | Phase 1+2 as prompt context |

**Output**: `CompanyAnalysis` — merges Phase 2 deterministic fields (financial_health, insider_signal, analyst_consensus, red_flags) + Phase 3 LLM fields.

### PriceAnalyst (per ticker)

Source: `specialists/price/agent.py`, indicators in `specialists/price/indicators.py`

Three-phase pipeline — only Phase 3 uses LLM:

| Phase                    | What                                                                                                                                                                                                                                                     | Source                      |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| 1. Data Gather           | yfinance quote + 3mo daily bars + options snapshot; Massive ATM options contracts (nearest monthly DTE > 20)                                                                                                                                             | yfinance, Massive           |
| 2. Deterministic Compute | Technical: EMA 8/21, ADX, RSI 14, MACD, ATR%, Bollinger Bands, z-score, volume ratio, OBV, support/resistance. Options: ATM IV via Black-Scholes inversion (Newton-Raphson on Massive EOD), realized vol 30d, IV/RV spread, put/call vol ratio, ATM skew | pandas-ta, custom BS solver |
| 3. LLM Synthesis         | technical_narrative (3-5 sentences), options_narrative (3-5 sentences)                                                                                                                                                                                   | Phase 1+2 as prompt context |

**Output**: `PriceAnalysis` — all deterministic indicators + LLM narratives + notable_setups[].

## Layer 2 Join + Routing

### `l2_join` (sync barrier)

No-op node with `defer=True`. All Layer 2 nodes (`company_analyst`, `price_analyst`, `ticker_research`) edge into it. The `defer` flag makes it wait for all upstream Send fan-out nodes to complete before proceeding.

### `debate_router` (conditional)

Routes from `l2_join` based on ticker availability and `settings.debate_rounds`:

- **No tickers** → `["analyze_compiler"]` (skip debate entirely)
- **`debate_rounds=0`** → `Send("bull_researcher", ...)` + `Send("bear_researcher", ...)` per ticker (parallel, no debate)
- **`debate_rounds>=1`** → `Send("ticker_debate", ...)` per ticker (multi-round debate subgraph)

Wired via `add_conditional_edges`.

## Layer 3: Debate

Configurable via `settings.debate_rounds` (default: 0).

### Mode A: Parallel independent (`debate_rounds=0`)

Each researcher is invoked **once per ticker** via Send. No debate history — bull and bear work independently.

### Mode B: Multi-round debate subgraph (`debate_rounds>=1`)

Source: `debate/subgraph.py`

A compiled LangGraph **subgraph** is invoked per ticker from the `ticker_debate` node in `graph.py`. The subgraph runs a cyclic bull⇄bear loop:

```
START → bull_debate → bear_debate → _should_continue
                                       ├─ round < max_rounds → bull_debate  (loop)
                                       └─ round >= max_rounds → END
```

#### Debate Subgraph State (`DebateState`)

| Field              | Type         | Reducer        | Purpose                                           |
| ------------------ | ------------ | -------------- | ------------------------------------------------- |
| `ticker`           | `str`        | —              | Target ticker (from Send)                         |
| `current_date`     | `str`        | —              | Pipeline date                                     |
| `company_analyses` | `list[dict]` | —              | Layer 2 context (read-only)                       |
| `price_analyses`   | `list[dict]` | —              | Layer 2 context (read-only)                       |
| `ticker_research`  | `dict`       | —              | Research context (read-only)                      |
| `debate_history`   | `list[dict]` | `add` (append) | Accumulates all bull/bear arguments across rounds |
| `round`            | `int`        | overwrite      | Current round (incremented after each bear turn)  |
| `max_rounds`       | `int`        | —              | Configured debate_rounds value                    |

Each round: bull argues (seeing all prior history) → bear argues (seeing all prior history including bull's latest) → round increments. The `_should_continue` function loops back to `bull_debate` if `round < max_rounds`, otherwise ends.

The `ticker_debate` node splits the final `debate_history` into `bull_analyses` + `bear_analyses` for the main graph state.

### Shared: BullResearcher and BearResearcher

Source: `debate/bull.py`, `debate/bear.py`

Both agents accept an optional `debate_history` parameter:

- **Without history** (rounds=0): standard per-ticker analysis, no debate instructions in system prompt
- **With history** (rounds>=1): prior arguments formatted via `format_debate_history()` and appended to prompt

Per-ticker context formatters (in `context.py`):
- `format_company_context_for_ticker(state, ticker)`
- `format_price_context_for_ticker(state, ticker)`
- `format_ticker_research_for_ticker(state, ticker)`

- **Tools**: `web_search` (1 call — `_WEB_SEARCH_CAP`), `web_read`
- **Output**: `TickerDebate(role="bull"|"bear")` — argument, key_evidence, round number

## Trader Gate + Routing

### `trader_gate` (sync barrier)

No-op node with `defer=True`. All debate paths (`bull_researcher`, `bear_researcher`, `ticker_debate`) edge into it. Waits for all debate output to complete before routing to the Trader.

### `trader_router` (conditional)

Routes from `trader_gate` based on ticker availability and `settings.trader_mode`:

- **No tickers** → `["analyze_compiler"]` (skip trader entirely)
- **`trader_mode="per_ticker"`** → `Send("trader", {**state, "ticker": t, "mode": "per_ticker"})` per ticker
- **`trader_mode="portfolio"`** → single `Send("trader", {**state, "tickers": [...], "mode": "portfolio"})`

Wired via `add_conditional_edges`.

## Layer 4: Trader

Source: `trader/trader.py`

The **sole decision maker** in the pipeline. Receives full debate history and ticker research context and produces actionable `TradeIdea` outputs.

### Per-Ticker Mode (`trader_mode="per_ticker"`)

One Send per ticker. Each call receives debate history + ticker research for that ticker.

### Portfolio Mode (`trader_mode="portfolio"`)

Single Send with all tickers. Enables:
- Cross-ticker correlation analysis
- Concentration risk assessment
- Capital allocation via conviction tiers
- `portfolio_note` field for cross-ticker observations (including natural L/S pairs)

### Output

- `TraderOutput` in `models.py` — `trade_ideas[]`, `portfolio_note`, `analysis_date`
- `TradeIdea` in `models.py` — equity-only, one ticker per idea:
  - `tickers[]`, `trade_structure` ("long NVDA" or "short AMD")
  - `entry_price`, `target_price`, `stop_price`, `risk_reward_ratio`
  - `conviction_tier` (1-3), `conviction_rationale`, `downside_scenario`
  - `expression_note` (vol context for optional options enhancement)
  - `thesis`, `catalyst`, `timeframe`, `key_risk`, `analysis_date`
- **Tools**: `web_search` (7 calls — `_WEB_SEARCH_CAP`), `web_read`

## Compilation (deterministic, defer=True)

Source: `compiler.py`

No LLM. `defer=True` waits for all trader nodes (or receives control directly from `debate_router`/`trader_router` when no tickers). Groups bull + bear by ticker, **sorted by round** so last round wins. Assembles final brief dict returned to the caller and saved to `docs/kg/raw/synesis_briefs/YYYY-MM-DD-tradeideas.md`.

## Agent Summary

| Agent                    | Model Tier | Tools                                          | Web Budget  | Output Type            |
| ------------------------ | ---------- | ---------------------------------------------- | ----------- | ---------------------- |
| TickerResearchAnalyst    | smart      | web_search, web_read, get_tweets               | unbounded   | `ticker_research` dict |
| CompanyAnalyst (Phase 3) | smart      | none (deterministic Phase 1-2)                 | 0           | `CompanyAnalysis`      |
| PriceAnalyst (Phase 3)   | smart      | none (deterministic Phase 1-2)                 | 0           | `PriceAnalysis`        |
| BullResearcher           | vsmart     | web_search, web_read                           | 1 per round | `TickerDebate`         |
| BearResearcher           | vsmart     | web_search, web_read                           | 1 per round | `TickerDebate`         |
| Trader                   | vsmart     | web_search, web_read                           | 7           | `TraderOutput`         |

## Key Design Decisions

1. **Trader is the sole decision maker** — all upstream agents (analysts, researchers) gather data and argue cases. Only the Trader produces actionable trade ideas with specific trade structures (see `trader/trader.py`).

2. **Two-mode trader** — `per_ticker` mode gives independent per-ticker decisions; `portfolio` mode enables cross-ticker analysis, concentration awareness, and conviction-based capital allocation. One `TradeIdea` per ticker (equity-only), with `portfolio_note` for cross-ticker observations.

3. **Ticker research runs parallel with Layer 2** — `ticker_research` starts at `START` alongside the Layer 2 fan-out, so Twitter + web research completes by the time debate begins with no serial overhead.

4. **Configurable debate depth** — `debate_rounds=0` gives fast parallel bull/bear; `debate_rounds>=1` enables adversarial multi-round debate where each side must directly counter the other's arguments.

5. **Debate history accumulation** — `DebateState.debate_history` uses `add` reducer to accumulate all arguments across rounds. Both researchers see the full history each turn, enabling increasingly targeted counter-arguments.

6. **Last-round-wins compilation** — compiler sorts by round and takes the final argument per ticker per side (see `compiler.py`), so the most refined version of each case goes into the brief.

7. **Two sync barriers** — `l2_join` separates Layer 2 from debate; `trader_gate` separates debate from Trader. Both use `defer=True` to wait for all upstream fan-out nodes.

8. **Web search budgets** — hard caps per agent per round prevent runaway API costs (Debate: 1 per round per side, Trader: 7). All agents use `web_search_config()` from `processing/common/llm.py`.

9. **Graceful degradation** — EDGAR failures don't block CompanyAnalyst; Massive failures don't block PriceAnalyst; Twitter failures don't block TickerResearch. All failures return `{"error": True}` for downstream visibility.

10. **Closure capture** — provider clients (yfinance, SEC EDGAR, Massive, Crawl4AI) captured at graph build time via `build_analyze_graph()` closure, never serialized in state.

11. **Send semantics for Layer 2 + Layer 3 + Layer 4** — `Send("node", {**state, "ticker": t})` creates independent state copies per ticker. Results merge back via `Annotated[list, add]` reducers in `state.py`.
