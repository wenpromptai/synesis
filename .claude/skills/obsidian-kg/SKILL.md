---
name: obsidian-kg
description: Build and maintain an LLM-compiled knowledge graph in Obsidian. Use when creating, updating, or querying the docs/kg/ wiki. Applies Karpathy-style raw→compile→query workflow with Obsidian wikilinks, atomic concept nodes, structured strategy nodes, and auto-maintained indexes.
---

# Obsidian Knowledge Graph Builder

You are an expert at building and maintaining an **LLM-compiled knowledge graph** stored as Obsidian markdown. This follows the Karpathy pattern: raw sources are indexed, then the LLM "compiles" a wiki of interlinked .md files that the user views in Obsidian.

**Key principle:** The LLM writes and maintains all wiki data. The user rarely edits directly.

---

## KG Architecture

```
docs/kg/
├── _index.md              # Master index — brief summary of every node
├── raw/                   # Source documents (PDFs, articles, web clips)
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

---

## Compilation Workflow

### Adding a Raw Source
1. Place document in `raw/` (PDF, .md web clip, etc.)
2. Read and extract key concepts, strategies, and insights
3. For each new concept → create/update a concept node in `concepts/`
4. For each new strategy → create/update a strategy node in `strategies/`
5. Create a source node in `sources/` summarizing the document and linking to all nodes it touches
6. Update relevant map(s) in `maps/`
7. Update `_index.md` with new entries

### Adding a Strategy (no raw source)
1. Create strategy node from domain knowledge
2. Create any missing concept nodes it references
3. Update relevant map(s)
4. Update `_index.md`

### Health Check / Linting
- Find orphan nodes (no incoming links)
- Find broken links (wikilinks to non-existent files)
- Find concept nodes referenced by strategies but not yet created
- Verify frontmatter consistency
- Check link density (flag nodes with < 3 links)

---

## Interlinking Rules

1. **Link liberally** — every concept, strategy, or source mentioned gets a `[[wikilink]]`
2. **Bidirectional by convention** — if A links to B, B should link back to A in its Related section
3. **Use aliases** for natural reading: `[[implied-volatility|IV]]`, `[[options-chain|chain]]`
4. **Cross-type links** — strategies link to concepts they use; concepts link to strategies that demonstrate them
5. **Never orphan a node** — every node must appear in at least one map and link to at least one other node
6. **Prefer existing nodes** — before creating a new concept, check if it already exists under a different name

## Frontmatter Standards

All nodes MUST have:
```yaml
---
up: ["[[Parent Map]]"]        # Hierarchical parent
related: ["[[Node]]"]         # Conceptually related nodes
created: YYYY-MM-DD
type: concept | strategy | source | map
tags: [domain-tags]
---
```

Strategy nodes additionally have:
```yaml
category: options | equity | macro | crypto
subcategory: income | directional | volatility | hedging | systematic
complexity: simple | medium | advanced
data_source: [yfinance, massive]   # Which pipeline provides the data
```

Source nodes additionally have:
```yaml
raw_file: "raw/filename.pdf"       # Path to source document
authors: ["Author Name"]
year: 2025
```

---

## Data Pipeline Awareness

When writing strategy nodes, specify which Synesis data pipelines provide the required data:

| Data Need | Provider | Endpoint/Method |
|-----------|----------|-----------------|
| Options chain (strikes, bids, asks) | yfinance | `get_options_chain()` |
| Greeks (delta, gamma, theta, vega, rho) | yfinance | `get_options_chain(greeks=True)` |
| IV per contract | yfinance | `get_options_chain()` → `.implied_volatility` |
| Realized volatility (30d) | yfinance | `get_options_snapshot()` → `.realized_vol_30d` |
| Expirations list | yfinance | `get_options_expirations()` |
| ATM snapshot | yfinance | `get_options_snapshot()` |
| Contract reference/metadata | massive | `get_options_contracts()` |
| Historical contract listings | massive | `get_options_contracts(as_of=date)` |
| VIX level | yfinance | `get_quote("^VIX")` |
| Stock price, history | yfinance | `get_quote()`, `get_history()` |

**If a strategy requires data NOT available in these pipelines, note it clearly in a `[!warning]` callout.**

---

## Output Guidelines

1. **Use Obsidian wikilinks** `[[note-name]]` for all internal references — never markdown links
2. **Frontmatter always present** with required fields
3. **Callouts** for structured information (see SYNTAX.md)
4. **ASCII P&L diagrams** for options strategies
5. **Mermaid diagrams** for flows and decision trees
6. **LaTeX** for formulas: `$inline$` and `$$block$$`
7. **Block references** `^block-id` for reusable definitions
8. **Footer navigation** — Related section with bidirectional links
9. **No emoji** unless user explicitly requests
