---
name: obsidian-kg
description: Build and maintain an LLM-compiled knowledge graph in Obsidian. Use when creating, updating, or querying the docs/kg/ wiki. Applies Karpathy-style raw→compile→query workflow with Obsidian wikilinks, atomic concept nodes, structured strategy nodes, and auto-maintained indexes.
---

# Obsidian Knowledge Graph Builder

You are an expert at building and maintaining an **LLM-compiled knowledge graph** stored as Obsidian markdown. This follows the Karpathy pattern: raw sources are indexed, then the LLM "compiles" a wiki of interlinked .md files that the user views in Obsidian.

**Key principles:**
- The LLM writes and maintains all wiki data. The user rarely edits directly.
- Every exploration adds up — queries that reveal new insights get filed back as node updates.
- Always read `_index.md` first before any KG operation to understand what exists.

---

## KG Architecture

```
docs/kg/
├── _index.md              # Master index — brief summary of every node (read this first)
├── _compile_log.md        # Append-only audit trail of compilation operations
├── raw/                   # Source documents (PDFs, articles, web clips)
│   └── synesis_briefs/    # Auto-saved pipeline briefs (YYYY-MM-DD.md)
├── sources/               # Extracted summaries from raw/ documents
├── maps/                  # Maps of Content (MOCs) — topic indexes
│   └── home.md            # Root MOC linking to all maps
├── concepts/              # Atomic concept nodes (one idea per file)
└── strategies/            # Strategy nodes (structured playbooks)
    └── options/           # Options strategies
```

## Node Types

### 1. Concept Node (`concepts/`)
One atomic idea. Title = the concept name (noun/phrase, not a sentence).
- **Purpose:** Define a single concept with links to related concepts and strategies that use it
- **Density:** ~5-10 wikilinks per node
- **Template:** See TEMPLATES.md § Concept Node

### 2. Strategy Node (`strategies/`)
A structured trading playbook. Includes mechanics, construction, risk, data requirements.
- **Purpose:** Actionable strategy with clear entry/exit rules and links to underlying concepts
- **Density:** ~8-15 wikilinks per node
- **Template:** See TEMPLATES.md § Strategy Node

### 3. Source Node (`sources/`)
Extracted summary from a raw document. Links to concepts and strategies it informs.
- **Purpose:** Bridge between raw sources and the compiled wiki
- **Density:** ~10-20 wikilinks per node (every concept/strategy mentioned gets linked)
- **Template:** See TEMPLATES.md § Source Node

### 4. Map Node (`maps/`)
MOC that gathers related nodes into a navigable index.
- **Purpose:** Topic-level entry point; gather → collide → navigate
- **Density:** Links to all nodes in its domain
- **Template:** See TEMPLATES.md § Map Node

### 5. Connection Node (`sources/connections/`)
A non-obvious relationship between 2+ existing nodes discovered during compilation or linting.
- **Purpose:** Capture insights that don't belong to any single concept or strategy — the "aha" links
- **Example:** "Calendar spreads outperform in pre-earnings + low IV rank environments"
- **Density:** Links to all nodes involved in the connection
- **Template:** See TEMPLATES.md § Connection Node

---

## Compilation Workflows

### Adding a Raw Source
1. Place document in `raw/` (PDF, .md web clip, etc.)
2. Read and extract key concepts, strategies, and insights
3. For each concept found:
   - If a concept node **already exists** → **update it** with new information from the source
   - If genuinely new → create a new concept node in `concepts/`
4. For each strategy found:
   - If a strategy node **already exists** → **update it** (add new data, refine parameters)
   - If genuinely new → create a new strategy node in `strategies/`
5. Create a source node in `sources/` summarizing the document and linking to all nodes it touches
6. Update `_index.md` with new/changed entries

### Adding a Strategy (no raw source)
1. Create strategy node from domain knowledge
2. Create any missing concept nodes it references
3. Update `_index.md`

### Query & File Back
When answering a question using KG content:
1. Read `_index.md` to identify relevant nodes
2. Read specific nodes to answer the question
3. **Check if the answer reveals new insights** not captured in any node
4. If yes → update the relevant node(s) or create a new one
5. Update `_index.md` if nodes were added/changed

This ensures every exploration compounds the knowledge base.

### Batch Compilation (`/kg-compile`)
Run `/kg-compile` to process all uncompiled raw sources.
The LLM reads the schema, current KG state, and each raw source, then decides what to extract/update/create. No hardcoded extraction rules — works for briefs, articles, PDFs, anything.

---

## Source Citations

Nodes updated or created during compilation MUST cite their sources using a collapsed callout:

```markdown
> [!quote]- Sources
> - [[sources/yartseva-2025|Yartseva (2025)]] — FCF yield as predictor
> - [[sources/brief-2026-04-07|Brief 2026-04-07]] — observed in risk_off regime
```

---

## Compilation Log

`_compile_log.md` is an append-only audit trail. Every compilation appends an entry:

```markdown
## YYYY-MM-DD
- **Source:** raw/path/to/file.md
- **Nodes updated:** node-a, node-b
- **Nodes created:** concepts/new-concept.md
- **Source node:** sources/summary-name.md
```

---

## Linting

Run `/kg-lint` to check KG health.

**Structural checks (1-6):** Broken wikilinks, orphan nodes, sparse linking, missing frontmatter, uncompiled raw files, stale index.

**Intelligence checks (7-11):** Connection discovery between unlinked nodes, missing node candidates (referenced 3+ times but don't exist), content staleness (nodes not updated despite new raw sources), impute missing data via web search, and research suggestions with concrete next steps.

---

## Index Format (`_index.md`)

The master index is the **primary retrieval mechanism**. The LLM reads this first to understand what exists and decide which nodes to read in full. At personal scale (<500 nodes), the LLM scanning a structured index outperforms vector similarity search.

Each entry is one line in a table, grouped by type:

```markdown
| Node | Type | One-liner | Updated |
|------|------|-----------|---------|
| [[covered-call]] | strategy | Long stock + short OTM call for income | 2026-04-07 |
| [[delta]] | concept | Rate of change of option price vs underlying | 2026-04-07 |
| [[yartseva-2025-multibaggers]] | source | Multibagger factors: FCF yield, small-cap, EBITDA | 2026-04-07 |
| [[iv-earnings]] | connection | IV expansion + earnings timing interaction | 2026-04-09 |
```

**Rules:**
- **One-liner:** max ~80 chars — enough for the LLM to judge relevance without reading the full node
- **Updated:** last date the node was created or significantly modified
- **Sort order:** strategies → concepts → sources → connections
- **Every node** in `docs/kg/` must appear here. `/kg-lint` check #6 enforces this.

---

## Interlinking Rules

1. **Link liberally** — every concept, strategy, or source mentioned gets a `[[wikilink]]`
2. **Bidirectional by convention** — if A links to B, B should link back to A in its Related section
3. **Use aliases** for natural reading: `[[implied-volatility|IV]]`, `[[options-chain|chain]]`
4. **Cross-type links** — strategies link to concepts they use; concepts link to strategies that demonstrate them
5. **Never orphan a node** — every node must link to at least one other node and be linked from at least one other node
6. **Prefer existing nodes** — before creating a new concept, check if it already exists under a different name

## Frontmatter Standards

All nodes MUST have:
```yaml
---
up: []                       # Hierarchical parent (if any)
related: ["[[Node]]"]         # Conceptually related nodes
created: YYYY-MM-DD
type: concept | strategy | source | connection
tags: [domain-tags]
---
```

Strategy nodes additionally have:
```yaml
category: options | equity | macro | event-driven | multi-asset
complexity: simple | medium | advanced
```

Source nodes additionally have:
```yaml
raw_file: "raw/filename.pdf"       # Path to source document
authors: ["Author Name"]
year: 2025
```

---

## Data Pipeline Awareness

When writing strategy or concept nodes, note which Synesis data providers are relevant. Available providers:

| Provider | Capabilities |
|----------|-------------|
| **yfinance** | Equity/ETF quotes, OHLCV history, options chains + Greeks, FX, realized vol |
| **massive** | Options contract reference data, historical contract listings |
| **FRED** | Economic data series (rates, unemployment, VIX, yields) |
| **SEC EDGAR** | Filings, insider transactions, Form 144, XBRL fundamentals |
| **NASDAQ** | Earnings calendar |
| **Finnhub** | Real-time prices, ticker search, company fundamentals |
| **Crawl4AI** | Web article extraction (HTML → markdown) |

Add a `Data Pipeline` section to strategy nodes **only when specific endpoints are needed**. Don't force it on strategies that are purely conceptual or don't require programmatic data.

If a strategy requires data NOT available in any Synesis provider, note it in a `[!warning]` callout.

---

## Output Guidelines

1. **Use Obsidian wikilinks** `[[note-name]]` for all internal references — never markdown links
2. **Frontmatter always present** with required fields
3. **Callouts** for structured information (see SYNTAX.md)
4. **Mermaid diagrams** for flows and decision trees
5. **LaTeX** for formulas: `$inline$` and `$$block$$`
6. **Block references** `^block-id` for reusable definitions
7. **Footer navigation** — Related section with bidirectional links
8. **No emoji** unless user explicitly requests
9. **Domain-appropriate visuals** — ASCII P&L diagrams for options, tables for screening criteria, flowcharts for decision frameworks. The LLM chooses what fits the content.
