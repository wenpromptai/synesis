---
name: langgraph-developing
description: LangGraph v1.1+ state machine orchestration for multi-agent workflows. Use when building agent graphs, defining state, adding conditional routing, debate loops, or wiring PydanticAI agents into LangGraph nodes. Apply for files in processing/intelligence/ or any StateGraph usage. (project)
---

# LangGraph Multi-Agent Development

Patterns for LangGraph v1.1+ with PydanticAI agents and Python 3.12+ (April 2026).

**Package:** `langgraph` (v1.1.6+). MIT licensed. Pulls `langchain-core` as transitive dep but we do NOT use LangChain agents/chains/tools.

## When To Apply
- Building or editing StateGraph definitions
- Adding nodes (agent wrappers) or edges (transitions)
- Implementing conditional routing (debate loops, gates)
- Integrating PydanticAI agents as LangGraph nodes
- Debugging state flow or reducer issues

## Quick Reference

| Pattern | Current | Avoid |
|---------|---------|-------|
| State definition | `TypedDict` with `Annotated` reducers | Plain dict, no reducers |
| List accumulation | `Annotated[list[str], operator.add]` | Bare `list[str]` (overwrites!) |
| Nodes | Plain `async def(state) -> dict` | LangChain `BaseTool` / `Runnable` |
| Edges | `add_edge(START, "node")` | String-based implicit routing |
| Conditional routing | `add_conditional_edges("node", fn)` | Manual if/else in nodes |
| Invocation | `await graph.ainvoke(state)` | Sync `graph.invoke()` |
| Loop guard | `config={"recursion_limit": 25}` | No limit (infinite loop risk) |
| Agent calls | PydanticAI `agent.run()` inside node | LangChain ChatModel |

---

## Core Concepts

### 1. State (TypedDict + Reducers)

State is a `TypedDict` shared across all nodes. Each node returns a **partial dict** that gets merged.

**Critical: Reducers control merge behavior.**

```python
from __future__ import annotations
from typing import Annotated, TypedDict
from operator import add

class AgentState(TypedDict):
    # Overwrite on each update (default behavior, no reducer)
    topic: str
    round: int
    final_verdict: str

    # APPEND on each update (reducer = operator.add)
    messages: Annotated[list[str], add]
    reports: Annotated[list[dict], add]
```

Without `Annotated[list, add]`, returning `{"messages": ["new"]}` **overwrites** the entire list. This is the #1 bug.

### 2. Nodes (Plain Async Functions)

Nodes are `async def(state: State) -> dict`. No LangChain abstractions needed.

```python
from pydantic_ai import Agent

analyst_agent = Agent(
    "anthropic:claude-3-5-haiku-20241022",
    system_prompt="You are a financial analyst.",
    result_type=AnalystReport,
)

async def analyst_node(state: AgentState) -> dict:
    """Node wrapping a PydanticAI agent."""
    result = await analyst_agent.run(state["topic"])
    return {
        "reports": [{"analyst": result.data.model_dump()}],
    }
```

**Rules:**
- Always return a dict (partial state update)
- Only include keys you want to update
- Keys with reducers (Annotated) get merged; others get overwritten
- Never mutate state directly — return new values

### 3. Graph Construction

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("analyst", analyst_node)
graph.add_node("critic", critic_node)
graph.add_node("judge", judge_node)

# Fixed edges
graph.add_edge(START, "analyst")
graph.add_edge("analyst", "critic")
graph.add_edge("judge", END)

# Compile
app = graph.compile()
```

### 4. Conditional Edges (Routing + Loops)

```python
from typing import Literal

MAX_DEBATE_ROUNDS = 2

def should_continue_debate(state: AgentState) -> Literal["advocate", "judge"]:
    """Route based on debate round counter."""
    if state["round"] < MAX_DEBATE_ROUNDS:
        return "advocate"
    return "judge"

graph.add_conditional_edges("critic", should_continue_debate)
```

**Counter increment happens inside the node:**

```python
async def critic_node(state: AgentState) -> dict:
    result = await critic_agent.run(str(state["messages"]))
    return {
        "messages": [f"[CRITIC] {result.data}"],
        "round": state["round"] + 1,  # increment here
    }
```

### 5. Parallel Nodes (Fan-Out / Fan-In)

Run multiple nodes in parallel by giving them the same source edge:

```python
# Fan-out: START triggers all analysts in parallel
graph.add_edge(START, "social_analyst")
graph.add_edge(START, "insider_analyst")
graph.add_edge(START, "technical_analyst")

# Fan-in: all analysts must complete before strategist runs
graph.add_edge("social_analyst", "strategist")
graph.add_edge("insider_analyst", "strategist")
graph.add_edge("technical_analyst", "strategist")
```

LangGraph automatically waits for all incoming edges before executing a node.

### 6. Invocation (Always Async)

```python
result = await app.ainvoke(
    {
        "topic": "NVDA earnings analysis",
        "round": 0,
        "messages": [],
        "reports": [],
        "final_verdict": "",
    },
    config={"recursion_limit": 25},
)
```

**Always set `recursion_limit`.** Default is 25. Each node execution counts as 1. A 3-round debate with 2 nodes per round = 6 steps.

### 7. Streaming

```python
async for event in app.astream(initial_state):
    # event is a dict: {"node_name": {"key": "value"}}
    for node_name, update in event.items():
        print(f"{node_name}: {update}")
```

---

## Integration Pattern: PydanticAI Agent as LangGraph Node

```python
from pydantic_ai import Agent
from synesis.processing.common.llm import create_model

# Define the PydanticAI agent (with tools, deps, structured output)
insider_agent = Agent(
    create_model(),  # Haiku
    system_prompt=INSIDER_PROMPT,
    result_type=InsiderFlowAnalysis,
)

@insider_agent.tool
async def get_insider_transactions(
    ctx: RunContext[InsiderDeps], ticker: str
) -> str:
    """Fetch recent insider transactions for a ticker."""
    txns = await ctx.deps.sec_edgar.get_insider_transactions(ticker)
    return format_transactions(txns)

# Wrap as LangGraph node
async def insider_flow_node(state: PipelineState) -> dict:
    """LangGraph node that runs the insider flow analyst."""
    deps = InsiderDeps(
        sec_edgar=state["sec_edgar"],
        watchlist_tickers=state["watchlist_tickers"],
    )
    result = await insider_agent.run(
        f"Analyze insider activity for: {state['watchlist_tickers']}",
        deps=deps,
    )
    return {
        "reports": [{"insider_flow": result.data.model_dump()}],
    }
```

**Key:** The PydanticAI agent keeps its tools, deps, and structured output. LangGraph only orchestrates when and how it runs relative to other agents.

---

## Common Anti-Patterns

| Anti-Pattern | Problem | Fix |
|---|---|---|
| `list[str]` without reducer | Each node overwrites the list | Use `Annotated[list[str], add]` |
| Mutating state in-place | Undefined behavior, breaks checkpointing | Return new dict from node |
| No recursion_limit | Infinite loop on broken routing | Always set `config={"recursion_limit": N}` |
| Passing deps via state | State gets serialized; clients aren't serializable | Pass deps via closure or `config["configurable"]` |
| LangChain agent abstractions | Unnecessary complexity, we use PydanticAI | Plain async functions as nodes |
| Sync nodes | Blocks the event loop | Always use `async def` |
| Giant state dict | Context bloat, slow serialization | Keep state lean; summarize reports before adding |

## Passing Non-Serializable Deps

State must be serializable (for checkpointing). Clients (DB, Redis, httpx) are not. Two options:

**Option A: Closure (preferred for our use case)**
```python
def make_insider_node(sec_edgar: SECEdgarClient):
    async def insider_flow_node(state: PipelineState) -> dict:
        # sec_edgar captured via closure, not in state
        result = await insider_agent.run(..., deps=InsiderDeps(sec_edgar=sec_edgar))
        return {"reports": [...]}
    return insider_flow_node

graph.add_node("insider_flow", make_insider_node(sec_edgar_client))
```

**Option B: Config (LangGraph built-in)**
```python
from langchain_core.runnables import RunnableConfig

async def insider_flow_node(state: PipelineState, config: RunnableConfig) -> dict:
    sec_edgar = config["configurable"]["sec_edgar"]
    ...

result = await app.ainvoke(state, config={"configurable": {"sec_edgar": client}})
```

---

## Debugging

```python
# Compile with debug=True for step-by-step logging
app = graph.compile(debug=True)

# Visualize the graph (requires pygraphviz or mermaid)
print(app.get_graph().draw_mermaid())
```
