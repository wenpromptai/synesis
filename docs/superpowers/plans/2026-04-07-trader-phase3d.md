# Trader Agent (Phase 3D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Trader node that synthesizes debate output + macro context into scored, actionable trade ideas — the only scored output in the pipeline.

**Architecture:** The Trader sits between debate and compiler as a new pipeline stage. A `trader_gate` (defer=True) sync barrier ensures all debate output is available before the Trader runs. A `trader_router` conditional edge dispatches per-ticker Sends or a single portfolio Send based on `TRADER_MODE` config. The compiler is extended to include `trade_ideas` ranked by conviction.

**Tech Stack:** LangGraph v1.1+ (StateGraph, Send, defer), PydanticAI (Agent, RunContext), OpenAI gpt-5.2 (vsmart tier)

**Spec:** `docs/superpowers/specs/2026-04-07-trader-design.md`

---

### Task 1: Add TradeIdea + TraderOutput Models

**Files:**
- Modify: `src/synesis/processing/intelligence/models.py:238` (after TickerDebate)
- Test: `tests/unit/test_intelligence_graph.py`

- [ ] **Step 1: Write failing model validation tests**

Add to `tests/unit/test_intelligence_graph.py`:

```python
from synesis.processing.intelligence.models import TradeIdea, TraderOutput

class TestTradeIdeaModel:
    """Tests for TradeIdea model validation."""

    def test_valid_trade_idea(self) -> None:
        idea = TradeIdea(
            ticker="NVDA",
            sentiment_score=0.8,
            thesis="Strong AI demand",
            trade_structure="bull call spread",
            catalyst="Earnings beat",
            timeframe="2-4 weeks",
            key_risk="Valuation",
            analysis_date=date(2026, 4, 7),
        )
        assert idea.ticker == "NVDA"
        assert idea.sentiment_score == 0.8

    def test_sentiment_score_bounded(self) -> None:
        with pytest.raises(Exception):
            TradeIdea(
                ticker="NVDA",
                sentiment_score=1.5,
                analysis_date=date(2026, 4, 7),
            )
        with pytest.raises(Exception):
            TradeIdea(
                ticker="NVDA",
                sentiment_score=-1.5,
                analysis_date=date(2026, 4, 7),
            )

    def test_ticker_requires_min_length(self) -> None:
        with pytest.raises(Exception):
            TradeIdea(
                ticker="",
                sentiment_score=0.5,
                analysis_date=date(2026, 4, 7),
            )

    def test_trader_output_with_skipped(self) -> None:
        output = TraderOutput(
            trade_ideas=[
                TradeIdea(
                    ticker="NVDA",
                    sentiment_score=0.8,
                    analysis_date=date(2026, 4, 7),
                ),
            ],
            skipped_tickers=["AAPL"],
            portfolio_note="Correlated tech exposure",
            analysis_date=date(2026, 4, 7),
        )
        assert len(output.trade_ideas) == 1
        assert output.skipped_tickers == ["AAPL"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_intelligence_graph.py::TestTradeIdeaModel -v`
Expected: FAIL with `ImportError` — `TradeIdea` not defined

- [ ] **Step 3: Add models to models.py**

Add after `TickerDebate` class (after line 238) in `src/synesis/processing/intelligence/models.py`:

```python
# =============================================================================
# Trader Output (Phase 3D — the ONLY scored output in the pipeline)
# =============================================================================


class TradeIdea(BaseModel):
    """A trade recommendation from the Trader — the ONLY scored output."""

    ticker: str = Field(min_length=1)
    sentiment_score: float = Field(
        ge=-1.0,
        le=1.0,
        description="Direction + conviction. -1.0 (strong sell) to 1.0 (strong buy). 0 = skip.",
    )
    thesis: str = ""
    trade_structure: str = ""
    catalyst: str = ""
    timeframe: str = ""
    key_risk: str = ""
    analysis_date: date


class TraderOutput(BaseModel):
    """Full Trader output — wraps one or more TradeIdeas."""

    trade_ideas: list[TradeIdea] = Field(default_factory=list)
    skipped_tickers: list[str] = Field(default_factory=list)
    portfolio_note: str = ""
    analysis_date: date
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_intelligence_graph.py::TestTradeIdeaModel -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Lint + type check**

Run: `uv run ruff check --fix src/synesis/processing/intelligence/models.py && uv run mypy src/synesis/processing/intelligence/models.py`
Expected: All checks passed

- [ ] **Step 6: Commit**

```bash
git add src/synesis/processing/intelligence/models.py tests/unit/test_intelligence_graph.py
git commit -m "feat: add TradeIdea + TraderOutput models (Phase 3D)"
```

---

### Task 2: Add Config + State Fields

**Files:**
- Modify: `src/synesis/config.py:79` (after debate_rounds)
- Modify: `src/synesis/processing/intelligence/state.py:39` (after bear_analyses)
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing config test**

Add to `tests/unit/test_config.py`:

```python
def test_trader_mode_env_mapping():
    """TRADER_MODE env var maps to trader_mode setting."""
    assert "TRADER_MODE" in [
        f.validation_alias or f.alias or name.upper()
        for name, f in Settings.model_fields.items()
        if name == "trader_mode"
    ] or "trader_mode" in Settings.model_fields
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config.py::test_trader_mode_env_mapping -v`
Expected: FAIL — `trader_mode` not in model_fields

- [ ] **Step 3: Add config field**

Add after `debate_rounds` field (after line 79) in `src/synesis/config.py`:

```python
    trader_mode: str = Field(
        default="per_ticker",
        description="Trader evaluation mode: 'per_ticker' or 'portfolio'",
    )
```

- [ ] **Step 4: Add state field**

Add after `bear_analyses` field (line 39) in `src/synesis/processing/intelligence/state.py`:

```python
    # Trader output (multiple parallel writers in per_ticker mode, needs reducer)
    trade_ideas: Annotated[list[dict[str, Any]], add]
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/test_config.py tests/unit/test_intelligence_graph.py -v`
Expected: PASS — existing tests still pass, new config test passes

- [ ] **Step 6: Commit**

```bash
git add src/synesis/config.py src/synesis/processing/intelligence/state.py tests/unit/test_config.py
git commit -m "feat: add trader_mode config + trade_ideas state field"
```

---

### Task 3: Build Trader Agent

**Files:**
- Create: `src/synesis/processing/intelligence/trader/__init__.py`
- Create: `src/synesis/processing/intelligence/trader/trader.py`

- [ ] **Step 1: Create `__init__.py`**

Create `src/synesis/processing/intelligence/trader/__init__.py` (empty file):

```python
```

- [ ] **Step 2: Create `trader.py` with agent, tools, and public API**

Create `src/synesis/processing/intelligence/trader/trader.py`:

```python
"""Trader — the sole decision maker in the intelligence pipeline.

Receives macro regime + debate arguments (bull/bear) + company/price data
and produces scored TradeIdea outputs. Supports per-ticker and portfolio modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pydantic_ai import Agent, RunContext

from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model
from synesis.processing.common.web_search import (
    Recency,
    format_search_results,
    read_web_page,
    search_market_impact,
)
from synesis.processing.intelligence.context import (
    format_company_context_for_ticker,
    format_debate_history,
    format_macro_context,
    format_news_context_for_ticker,
    format_price_context_for_ticker,
    format_social_context_for_ticker,
)
from synesis.processing.intelligence.models import TraderOutput

logger = get_logger(__name__)

_WEB_SEARCH_CAP = 3

_PER_TICKER_PROMPT = """\
You are the senior Trader at a multi-strategy hedge fund. You are the SOLE \
decision maker — every other agent in this pipeline gathered data or argued \
a case. You decide.

Today's date: {current_date}

## Context

Below you will find the macro regime assessment, plus bull and bear debate \
arguments and supporting analyst data for this ticker.

## Your Job

1. **Read both sides of the debate.** Who made the stronger case? Where did \
the evidence actually point?
2. **Make a decisive call.** Buy, sell, or skip. Do NOT default to "hold" — \
if you're not convinced either way, skip the ticker entirely.
3. **Score your conviction.** sentiment_score: -1.0 (strong sell) to 1.0 \
(strong buy). Calibration: ±0.8-1.0 = high conviction, ±0.4-0.7 = moderate, \
±0.1-0.3 = weak, 0 = skip.
4. **Suggest a trade structure.** Be specific: "buy shares", "bull call \
spread 30/35 June exp", "protective put", etc. Consider the macro regime \
and current IV when choosing between shares and options.
5. **Name the catalyst and timeframe.** What triggers the move, and when?

{mode_instructions}

## Tools
- `web_search(query, recency)` — verify a specific claim. Budget: \
{web_search_cap} calls.
- `web_read(url)` — read full article content (~4000 chars). Unlimited.

## Rules
- Ground your decision in the debate evidence. Do not fabricate data.
- If one side of the debate is missing (error), note this and decide \
with what you have — or skip if insufficient.
- Be specific with trade structure — no vague "consider options".
"""

_PORTFOLIO_INSTRUCTIONS = """\
## Portfolio Mode
You are reviewing ALL tickers together. Consider:
- Cross-ticker correlation (are multiple ideas in the same sector/theme?)
- Concentration risk (too much exposure to one factor?)
- Capital allocation (which ideas deserve the largest position?)
Add a portfolio_note with cross-ticker observations."""

_PER_TICKER_INSTRUCTIONS = ""


@dataclass
class TraderDeps:
    """Dependencies for Trader."""

    current_date: date
    web_search_calls: int = field(default=0, init=False)


async def _tool_web_search(
    ctx: RunContext[TraderDeps],
    query: str,
    recency: str = "day",
) -> str:
    """Search the web for verification context."""
    if ctx.deps.web_search_calls >= _WEB_SEARCH_CAP:
        return f"Web search budget exhausted ({_WEB_SEARCH_CAP} calls used)."
    ctx.deps.web_search_calls += 1

    valid_recencies = ("day", "week", "month", "year", "none")
    recency_val: Recency = recency if recency in valid_recencies else "day"  # type: ignore[assignment]
    results = await search_market_impact(query, recency=recency_val)
    if not results:
        return "No search results found."
    return format_search_results(results)


async def _tool_web_read(ctx: RunContext[TraderDeps], url: str) -> str:
    """Read a web page for full article content."""
    return await read_web_page(url)


def _format_debate_for_ticker(
    state: dict[str, Any], ticker: str
) -> str:
    """Format bull + bear debate arguments for a single ticker."""
    bull_analyses = state.get("bull_analyses", [])
    bear_analyses = state.get("bear_analyses", [])

    # Find the latest (highest round) argument for each side
    bull = None
    for item in sorted(bull_analyses, key=lambda x: x.get("round", 0)):
        if not item.get("error") and item.get("ticker") == ticker:
            bull = item
    bear = None
    for item in sorted(bear_analyses, key=lambda x: x.get("round", 0)):
        if not item.get("error") and item.get("ticker") == ticker:
            bear = item

    lines: list[str] = []
    if bull:
        lines.append("## Bull Case")
        lines.append(bull.get("argument", ""))
        evidence = bull.get("key_evidence", [])
        if evidence:
            lines.append("\nKey evidence:")
            for e in evidence:
                lines.append(f"- {e}")
    else:
        lines.append("## Bull Case\n[Not available — analysis failed or missing]")

    if bear:
        lines.append("\n## Bear Case")
        lines.append(bear.get("argument", ""))
        evidence = bear.get("key_evidence", [])
        if evidence:
            lines.append("\nKey evidence:")
            for e in evidence:
                lines.append(f"- {e}")
    else:
        lines.append("\n## Bear Case\n[Not available — analysis failed or missing]")

    return "\n".join(lines)


def _build_per_ticker_prompt(state: dict[str, Any], ticker: str) -> str:
    """Build the user prompt for per-ticker mode."""
    parts = [
        f"## Ticker: {ticker}",
        format_macro_context(state),
        _format_debate_for_ticker(state, ticker),
        format_company_context_for_ticker(state, ticker),
        format_price_context_for_ticker(state, ticker),
    ]
    return "\n\n".join(parts)


def _build_portfolio_prompt(state: dict[str, Any], tickers: list[str]) -> str:
    """Build the user prompt for portfolio mode."""
    parts = [format_macro_context(state)]
    for ticker in tickers:
        parts.append(f"---\n\n## Ticker: {ticker}")
        parts.append(_format_debate_for_ticker(state, ticker))
        parts.append(format_company_context_for_ticker(state, ticker))
        parts.append(format_price_context_for_ticker(state, ticker))
    return "\n\n".join(parts)


async def analyze_trade_per_ticker(
    state: dict[str, Any],
    current_date: date,
) -> TraderOutput:
    """Run Trader for a single ticker (per_ticker mode)."""
    ticker = state["ticker"]
    logger.info("Starting Trader (per_ticker)", ticker=ticker)

    deps = TraderDeps(current_date=current_date)
    user_prompt = _build_per_ticker_prompt(state, ticker)

    agent: Agent[TraderDeps, TraderOutput] = Agent(
        model=create_model(tier="vsmart"),
        deps_type=TraderDeps,
        output_type=TraderOutput,
        system_prompt=_PER_TICKER_PROMPT.format(
            current_date=current_date,
            web_search_cap=_WEB_SEARCH_CAP,
            mode_instructions=_PER_TICKER_INSTRUCTIONS,
        ),
        tools=[_tool_web_search, _tool_web_read],
    )

    result = await agent.run(user_prompt, deps=deps)
    output = result.output

    logger.info(
        "Trader complete (per_ticker)",
        ticker=ticker,
        ideas=len(output.trade_ideas),
        skipped=output.skipped_tickers,
    )

    # Ensure consistent metadata
    for idea in output.trade_ideas:
        if idea.analysis_date != current_date:
            idea.analysis_date = current_date

    return output


async def analyze_trade_portfolio(
    state: dict[str, Any],
    current_date: date,
    tickers: list[str],
) -> TraderOutput:
    """Run Trader for all tickers at once (portfolio mode)."""
    logger.info("Starting Trader (portfolio)", tickers=tickers)

    deps = TraderDeps(current_date=current_date)
    user_prompt = _build_portfolio_prompt(state, tickers)

    agent: Agent[TraderDeps, TraderOutput] = Agent(
        model=create_model(tier="vsmart"),
        deps_type=TraderDeps,
        output_type=TraderOutput,
        system_prompt=_PER_TICKER_PROMPT.format(
            current_date=current_date,
            web_search_cap=_WEB_SEARCH_CAP,
            mode_instructions=_PORTFOLIO_INSTRUCTIONS,
        ),
        tools=[_tool_web_search, _tool_web_read],
    )

    result = await agent.run(user_prompt, deps=deps)
    output = result.output

    logger.info(
        "Trader complete (portfolio)",
        ideas=len(output.trade_ideas),
        skipped=output.skipped_tickers,
    )

    for idea in output.trade_ideas:
        if idea.analysis_date != current_date:
            idea.analysis_date = current_date

    return output
```

- [ ] **Step 3: Lint + type check**

Run: `uv run ruff check --fix src/synesis/processing/intelligence/trader/ && uv run mypy src/synesis/processing/intelligence/trader/`
Expected: All checks passed

- [ ] **Step 4: Commit**

```bash
git add src/synesis/processing/intelligence/trader/
git commit -m "feat: add Trader agent with per-ticker and portfolio modes"
```

---

### Task 4: Wire Trader into Graph

**Files:**
- Modify: `src/synesis/processing/intelligence/graph.py`

- [ ] **Step 1: Write failing graph structure tests**

Add to `tests/unit/test_intelligence_graph.py`:

```python
class TestTraderGraphStructure:
    """Tests for Trader node presence in the graph."""

    def test_trader_gate_node_exists(self) -> None:
        graph = build_intelligence_graph(
            db=AsyncMock(), sec_edgar=AsyncMock(), yfinance=AsyncMock(),
            fred=AsyncMock(), crawler=AsyncMock(),
        )
        node_names = set(graph.get_graph().nodes)
        assert "trader_gate" in node_names

    def test_trader_node_exists(self) -> None:
        graph = build_intelligence_graph(
            db=AsyncMock(), sec_edgar=AsyncMock(), yfinance=AsyncMock(),
            fred=AsyncMock(), crawler=AsyncMock(),
        )
        node_names = set(graph.get_graph().nodes)
        assert "trader" in node_names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_intelligence_graph.py::TestTraderGraphStructure -v`
Expected: FAIL — `trader_gate` and `trader` not in node_names

- [ ] **Step 3: Add imports to graph.py**

Add to the imports section of `src/synesis/processing/intelligence/graph.py` (after the debate imports around line 37):

```python
from synesis.processing.intelligence.trader.trader import (
    analyze_trade_per_ticker,
    analyze_trade_portfolio,
)
```

- [ ] **Step 4: Add trader_gate, trader node, and trader_router to graph.py**

Add after the compiler node definition (after `compiler_node` function, before `# ── Build Graph`):

```python
    # ── Trader Gate (waits for all debate output) ────────────────

    async def trader_gate_node(state: IntelligenceState) -> dict[str, Any]:
        """Sync barrier: waits for all debate output (defer=True)."""
        return {}

    def trader_router(state: IntelligenceState) -> list[str | Send]:
        """Route to per-ticker or portfolio Trader based on config."""
        from synesis.config import get_settings

        tickers = state.get("target_tickers", [])
        if not tickers:
            logger.info("No tickers — skipping trader, going to compiler")
            return ["compiler"]

        mode = get_settings().trader_mode
        if mode == "portfolio":
            logger.info("Routing to Trader (portfolio)", tickers=tickers)
            return [Send("trader", {**state, "tickers": tickers, "mode": "portfolio"})]
        else:
            logger.info("Routing to Trader (per_ticker)", tickers=tickers)
            return [Send("trader", {**state, "ticker": t, "mode": "per_ticker"}) for t in tickers]

    # ── Trader (per-ticker or portfolio via Send) ────────────────

    async def trader_node(state: dict[str, Any]) -> dict[str, Any]:
        """Run Trader for one ticker or all tickers. Called via Send."""
        mode = state.get("mode", "per_ticker")
        try:
            current = date.fromisoformat(state["current_date"])
            if mode == "portfolio":
                tickers = state.get("tickers", [])
                result = await analyze_trade_portfolio(state, current, tickers)
            else:
                result = await analyze_trade_per_ticker(state, current)
            return {
                "trade_ideas": [idea.model_dump(mode="json") for idea in result.trade_ideas],
            }
        except Exception:
            ticker = state.get("ticker", "portfolio")
            logger.exception("Trader failed", ticker=ticker)
            return {"trade_ideas": [{"ticker": ticker, "error": True}]}
```

- [ ] **Step 5: Update graph construction — add nodes**

In the `# ── Build Graph` section, add the new nodes after `ticker_debate`:

```python
    graph.add_node("trader_gate", trader_gate_node, defer=True)
    graph.add_node("trader", trader_node)  # type: ignore[type-var]
```

- [ ] **Step 6: Update graph construction — rewire edges**

Replace the existing debate-to-compiler edges:

```python
    # OLD: debate → compiler
    # graph.add_edge("bull_researcher", "compiler")
    # graph.add_edge("bear_researcher", "compiler")
    # graph.add_edge("ticker_debate", "compiler")

    # NEW: debate → trader_gate → trader_router → trader → compiler
    graph.add_edge("bull_researcher", "trader_gate")
    graph.add_edge("bear_researcher", "trader_gate")
    graph.add_edge("ticker_debate", "trader_gate")

    graph.add_conditional_edges(
        "trader_gate",
        trader_router,
        ["trader", "compiler"],
    )

    graph.add_edge("trader", "compiler")
```

- [ ] **Step 7: Run structure tests**

Run: `uv run pytest tests/unit/test_intelligence_graph.py::TestTraderGraphStructure -v`
Expected: PASS

- [ ] **Step 8: Run full test suite**

Run: `uv run pytest tests/unit/test_intelligence_graph.py -v`
Expected: Some existing graph execution tests may need updating (they expect debate → compiler, now debate → trader_gate → compiler). Fix in next task.

- [ ] **Step 9: Lint + type check**

Run: `uv run ruff check --fix src/synesis/processing/intelligence/graph.py && uv run mypy src/synesis/processing/intelligence/graph.py`
Expected: All checks passed

- [ ] **Step 10: Commit**

```bash
git add src/synesis/processing/intelligence/graph.py
git commit -m "feat: wire Trader into graph with trader_gate barrier and trader_router"
```

---

### Task 5: Update Compiler for Trade Ideas

**Files:**
- Modify: `src/synesis/processing/intelligence/compiler.py`
- Test: `tests/unit/test_intelligence_graph.py`

- [ ] **Step 1: Write failing compiler tests**

Add to `tests/unit/test_intelligence_graph.py`:

```python
class TestCompilerTradeIdeas:
    """Tests for trade ideas in the compiled brief."""

    def test_trade_ideas_ranked_by_conviction(self) -> None:
        state = {
            "current_date": "2026-04-07",
            "social_analysis": {},
            "news_analysis": {},
            "company_analyses": [],
            "price_analyses": [],
            "bull_analyses": [],
            "bear_analyses": [],
            "trade_ideas": [
                {"ticker": "AAPL", "sentiment_score": 0.3, "thesis": "Weak buy"},
                {"ticker": "NVDA", "sentiment_score": -0.9, "thesis": "Strong sell"},
                {"ticker": "TSLA", "sentiment_score": 0.7, "thesis": "Moderate buy"},
            ],
        }
        brief = compile_brief(state)
        ideas = brief["trade_ideas"]
        assert len(ideas) == 3
        assert ideas[0]["ticker"] == "NVDA"  # abs(-0.9) = 0.9
        assert ideas[1]["ticker"] == "TSLA"  # abs(0.7) = 0.7
        assert ideas[2]["ticker"] == "AAPL"  # abs(0.3) = 0.3

    def test_trade_ideas_filters_errors(self) -> None:
        state = {
            "current_date": "2026-04-07",
            "social_analysis": {},
            "news_analysis": {},
            "company_analyses": [],
            "price_analyses": [],
            "bull_analyses": [],
            "bear_analyses": [],
            "trade_ideas": [
                {"ticker": "NVDA", "sentiment_score": 0.8, "thesis": "Buy"},
                {"ticker": "AAPL", "error": True},
            ],
        }
        brief = compile_brief(state)
        assert len(brief["trade_ideas"]) == 1
        assert brief["trade_ideas"][0]["ticker"] == "NVDA"
        assert brief["errors"]["trader_failures"] == ["AAPL"]

    def test_empty_trade_ideas(self) -> None:
        state = {
            "current_date": "2026-04-07",
            "social_analysis": {},
            "news_analysis": {},
            "company_analyses": [],
            "price_analyses": [],
            "bull_analyses": [],
            "bear_analyses": [],
            "trade_ideas": [],
        }
        brief = compile_brief(state)
        assert brief["trade_ideas"] == []
        assert brief["errors"]["trader_failures"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_intelligence_graph.py::TestCompilerTradeIdeas -v`
Expected: FAIL — `trade_ideas` key not in brief

- [ ] **Step 3: Update compiler.py**

In `src/synesis/processing/intelligence/compiler.py`, add after `macro = state.get("macro_view", {})` (around line 28):

```python
    trade_ideas_raw = state.get("trade_ideas", [])
    valid_trade_ideas = sorted(
        [t for t in trade_ideas_raw if not t.get("error")],
        key=lambda t: abs(t.get("sentiment_score", 0)),
        reverse=True,
    )
```

Add `"trade_ideas": valid_trade_ideas,` to the return dict (after `"messages_analyzed"`).

Add `"trader_failures"` to the `"errors"` dict:

```python
            "trader_failures": [
                item["ticker"] for item in trade_ideas_raw if item.get("error") and "ticker" in item
            ],
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_intelligence_graph.py::TestCompilerTradeIdeas -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run full compiler tests to check no regressions**

Run: `uv run pytest tests/unit/test_intelligence_graph.py::TestCompileBrief -v`
Expected: PASS — existing tests may need `"trade_ideas": []` added to their state dicts if they rely on exact brief keys

- [ ] **Step 6: Lint + type check**

Run: `uv run ruff check --fix src/synesis/processing/intelligence/compiler.py && uv run mypy src/synesis/processing/intelligence/compiler.py`
Expected: All checks passed

- [ ] **Step 7: Commit**

```bash
git add src/synesis/processing/intelligence/compiler.py tests/unit/test_intelligence_graph.py
git commit -m "feat: compiler includes trade_ideas ranked by conviction + trader_failures"
```

---

### Task 6: Update Existing Tests for New Graph Topology

**Files:**
- Modify: `tests/unit/test_intelligence_graph.py`

- [ ] **Step 1: Update graph execution test mocks**

The existing `TestGraphExecution` tests mock debate agents but now the graph has `trader_gate` and `trader` nodes between debate and compiler. Update the mock patches to include the Trader:

Add to the `_PATCH_PREFIX` section or wherever mocks are defined:

```python
# Mock for Trader — returns a simple TradeIdea
_MOCK_TRADER_OUTPUT = TraderOutput(
    trade_ideas=[
        TradeIdea(ticker="NVDA", sentiment_score=0.8, thesis="Buy", analysis_date=date(2026, 4, 7)),
    ],
    skipped_tickers=[],
    analysis_date=date(2026, 4, 7),
)
```

Add `TradeIdea, TraderOutput` to the imports from `models`.

In each graph execution test that runs the full pipeline, add a mock patch for the Trader:

```python
@patch(f"{_PATCH_PREFIX}.analyze_trade_per_ticker", new_callable=AsyncMock)
```

And set its return value:

```python
mock_trader.return_value = _MOCK_TRADER_OUTPUT
```

- [ ] **Step 2: Update state assertions**

Existing tests that assert on `brief` output need to account for `trade_ideas` key now being present.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/unit/test_intelligence_graph.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_intelligence_graph.py
git commit -m "test: update existing graph tests for trader_gate + trader topology"
```

---

### Task 7: Add Trader-Specific Graph Execution Tests

**Files:**
- Modify: `tests/unit/test_intelligence_graph.py`

- [ ] **Step 1: Write per-ticker mode execution test**

```python
class TestTraderGraphExecution:
    """Tests for Trader in full graph execution."""

    @pytest.mark.asyncio
    async def test_per_ticker_trader_produces_trade_ideas(self) -> None:
        """Per-ticker mode: Trader called once per ticker, ideas in brief."""
        # Mock all upstream agents + Trader
        # Build graph, invoke with 2 tickers
        # Assert: Trader called twice (once per ticker)
        # Assert: brief["trade_ideas"] has 2 entries
        ...

    @pytest.mark.asyncio
    async def test_portfolio_trader_produces_trade_ideas(self) -> None:
        """Portfolio mode: Trader called once with all tickers."""
        # Mock all upstream agents + Trader
        # Mock get_settings to return trader_mode="portfolio"
        # Build graph, invoke with 2 tickers
        # Assert: Trader called once with tickers=["AAPL", "NVDA"]
        # Assert: brief["trade_ideas"] present
        ...

    @pytest.mark.asyncio
    async def test_no_tickers_skips_trader(self) -> None:
        """No tickers → trader_router skips to compiler, no Trader call."""
        # Mock Layer 1 to return no tickers
        # Assert: Trader never called
        # Assert: brief["trade_ideas"] == []
        ...

    @pytest.mark.asyncio
    async def test_trader_partial_failure(self) -> None:
        """One ticker's Trader call fails, others succeed."""
        # Mock Trader to raise on "AAPL" but succeed on "NVDA"
        # Assert: brief has trade_ideas for NVDA
        # Assert: brief["errors"]["trader_failures"] == ["AAPL"]
        ...
```

These tests follow the same pattern as existing `TestGraphExecution` tests — mock all LLM agents, build the real graph, invoke with `ainvoke`, assert on brief output. The `...` bodies above are outlines — the implementing agent must write full test bodies following the existing `TestGraphExecution` mock pattern: `@patch` for each agent function, `AsyncMock` return values, `graph.ainvoke()` with initial state, assertions on `result["brief"]`.

**Implementation detail:** Each test needs to mock `analyze_trade_per_ticker` / `analyze_trade_portfolio` via `@patch`, plus all existing upstream agent mocks (social, news, company, price, macro, bull, bear). Follow the exact pattern used in `TestGraphExecution` and `TestDebateSubgraph`.

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/unit/test_intelligence_graph.py::TestTraderGraphExecution -v`
Expected: PASS (4 tests)

- [ ] **Step 3: Run full suite**

Run: `uv run pytest tests/unit/test_intelligence_graph.py -v`
Expected: ALL PASS

- [ ] **Step 4: Final lint + type check**

Run: `uv run ruff check --fix . && uv run ruff format . && uv run mypy src/`
Expected: All checks passed

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_intelligence_graph.py
git commit -m "test: add Trader graph execution tests (per-ticker, portfolio, skip, failure)"
```

---

### Task 8: Update PRD + Architecture Docs

**Files:**
- Modify: `docs/multi-agent-prd.md`

- [ ] **Step 1: Update Phase 3D section in PRD**

Replace the Phase 3D "Deferred" section with completion notes listing what was built: Trader agent, TradeIdea/TraderOutput models, trader_gate + trader_router graph nodes, per_ticker/portfolio modes, compiler integration.

- [ ] **Step 2: Update cost table**

Add Trader row to the cost table:

```
| Trader (per ticker, ~3-5/day) | OpenAI vsmart | ~$0.08-0.12 |
```

Update total.

- [ ] **Step 3: Commit**

```bash
git add docs/multi-agent-prd.md
git commit -m "docs: mark Phase 3D (Trader) as complete in PRD"
```
