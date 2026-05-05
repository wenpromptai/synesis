# LangGraph Patterns for Synesis Intelligence Pipeline

## Pipeline State Definition

The full state for the daily intelligence pipeline:

```python
from __future__ import annotations
from typing import Annotated, TypedDict
from operator import add

class IntelligenceState(TypedDict):
    """Shared state for the daily intelligence pipeline."""

    # --- Input (set at invocation, read-only by convention) ---
    target_tickers: list[str]
    raw_tweets: list[dict]        # from raw_tweets table
    calendar_events: list[dict]   # from calendar_events table

    # --- Layer 1 specialist reports (appended by each specialist) ---
    reports: Annotated[list[dict], add]

    # --- Layer 2 strategist outputs ---
    macro_view: dict        # from MacroStrategist
    equity_ideas: list[dict]  # from EquityStrategist

    # --- Layer 2.5 debate ---
    debate_round: int
    debate_messages: Annotated[list[str], add]
    adjudicated_ideas: Annotated[list[dict], add]

    # --- Layer 3 output ---
    brief: dict  # final DailyIntelligenceBrief
```

## Graph Topology

```
                    START
                      │
          ┌───────────┼───────────┐───────────┐
          ▼           ▼           ▼           ▼
    social_analyst  event_analyst  insider_analyst  technical_analyst
          │           │           │           │
          └───────────┼───────────┘───────────┘
                      ▼
              macro_strategist
                      │
                      ▼
              equity_strategist
                      │
                      ▼
                   gate ──────────────► compiler (no debate needed)
                      │
                      ▼ (ideas that pass gate)
                 bull_advocate
                      │
                      ▼
                 bear_advocate
                      │
                      ▼
              should_continue_debate?
                 /          \
            (loop)        (done)
               │              │
               ▼              ▼
         bull_advocate    adjudicator
                              │
                              ▼
                          compiler
                              │
                              ▼
                             END
```

## Gate Pattern (Conditional Fan-Out)

```python
def gate_for_debate(state: IntelligenceState) -> Literal["bull_advocate", "compiler"]:
    """Route high-conviction ideas to debate, rest straight to compiler."""
    gated = [
        idea for idea in state["equity_ideas"]
        if idea["conviction"] >= 0.7
        and idea["source_count"] >= 2
        and idea["direction"] in ("long", "short")
    ]
    if gated:
        return "bull_advocate"
    return "compiler"

graph.add_conditional_edges("equity_strategist", gate_for_debate)
```

## Multi-Round Debate Pattern

```python
MAX_DEBATE_ROUNDS = 2  # bull-bear-bull-bear = 2 full rounds

def should_continue_debate(state: IntelligenceState) -> Literal["bull_advocate", "adjudicator"]:
    if state["debate_round"] < MAX_DEBATE_ROUNDS:
        return "bull_advocate"
    return "adjudicator"

graph.add_edge("bull_advocate", "bear_advocate")
graph.add_conditional_edges("bear_advocate", should_continue_debate)
graph.add_edge("adjudicator", "compiler")
```

## Error Handling in Nodes

Specialists should never crash the pipeline. Wrap with graceful degradation:

```python
async def insider_flow_node(state: IntelligenceState) -> dict:
    try:
        result = await insider_agent.run(...)
        return {"reports": [{"insider_flow": result.data.model_dump()}]}
    except Exception as e:
        logger.error("InsiderFlowAnalyst failed", error=str(e), exc_info=e)
        return {"reports": [{"insider_flow": None, "error": str(e)}]}
```

Layer 2 strategists check for `None` reports and work with whatever succeeded.

## Deps via Closure Factory

```python
def build_intelligence_graph(
    sec_edgar: SECEdgarClient,
    massive: MassiveClient,
    yfinance: YFinanceClient,
    fred: FREDClient,
    db: Database,
) -> CompiledGraph:
    """Build the intelligence pipeline graph with provider deps captured via closure."""

    async def insider_flow_node(state: IntelligenceState) -> dict:
        # sec_edgar captured from outer scope
        ...

    async def technical_node(state: IntelligenceState) -> dict:
        # massive + yfinance captured from outer scope
        ...

    graph = StateGraph(IntelligenceState)
    graph.add_node("insider_flow", insider_flow_node)
    graph.add_node("technical", technical_node)
    # ... add all nodes and edges ...

    return graph.compile()
```

Call once at startup, reuse the compiled graph for each daily run.
