---
name: kg-compile
description: Compile unprocessed raw sources in docs/kg/raw/ into the knowledge graph. Reads each raw file, decides what to extract, creates/updates concept and strategy nodes, creates source nodes, and updates the index. The LLM is the compiler — no hardcoded extraction rules.
user_invocable: true
---

# KG Compile

Process unprocessed raw sources into KG knowledge. You are the compiler — read the raw source, read the current KG state, and use your judgment to decide what's worth extracting and where it fits.

## Prerequisites

Before compiling, read these skill files to understand the KG schema:
1. Read `.claude/skills/obsidian-kg/SKILL.md` — node types, interlinking rules, frontmatter standards
2. Read `.claude/skills/obsidian-kg/TEMPLATES.md` — templates for each node type

## Workflow

### Step 1: Understand current KG state
- Read `docs/kg/_index.md` — master index of all existing nodes

### Step 2: Find unprocessed raw files
- Read `docs/kg/_compile_log.md` — which files have already been compiled
- Glob `docs/kg/raw/**/*` — all raw files (all subdirectories)
- Subtract already-compiled files → these are the unprocessed ones
- If no unprocessed files, report "Nothing to compile" and stop

### Step 3: Compile each unprocessed file

For each unprocessed raw file:

1. **Read it.** Understand what information it contains.

2. **Decide what to extract.** Use your judgment based on source type:

   **For pipeline briefs (`raw/synesis_briefs/`):**
   - **Ticker nodes** (`tickers/`): Create or update one node per ticker analyzed. Extract the trade decision, bull/bear thesis summaries, key financials cited, and risks. Every data point must be dated (filing period, brief date).
   - **Theme nodes** (`themes/`): Identify cross-cutting themes that link multiple tickers. These are emergent — discovered from patterns in the brief, not predefined. Examples: "AI infrastructure demand" (linking NVDA, AVGO, TSM), "customer concentration risk" (linking AAOI, AVGO), "geopolitical energy risk" (linking CVX, OIH). A ticker can belong to many themes.
   - **Source node** (`sources/`): One per brief, capturing macro regime, sector tilts, trade count, and links to all ticker/theme nodes touched.

   **For research documents, articles, PDFs:**
   - What concepts does this source discuss? (definitions, principles, factors) → `concepts/`
   - What strategies does it describe or reference? (trading approaches, frameworks) → `strategies/`
   - What connections does it reveal? (non-obvious relationships between 2+ existing nodes) → `sources/connections/`
   - What insights are worth preserving? (data points, observations, patterns)

   **For any source type:**
   - Update existing ticker/theme nodes if the source mentions tickers already in the KG
   - Create new concept/strategy nodes if genuinely new ideas are introduced

3. **Check existing nodes.** For each item you want to extract:
   - Read `_index.md` — does a node for this already exist?
   - If yes → **read the existing node**, then **update it** with new information from this source. Add a source citation callout.
   - If no → **create a new node** following TEMPLATES.md. Link it to existing related nodes.

4. **Create a source node** in `sources/` summarizing what was extracted from this raw file:
   - List every node that was updated or created
   - Cite the raw file path in frontmatter `raw_file`
   - Link to all touched nodes

5. **Update `_index.md`** with any new or significantly changed nodes.

6. **Append to `_compile_log.md`:**
   ```markdown
   ## YYYY-MM-DD
   - **Source:** raw/path/to/file
   - **Nodes updated:** node-a, node-b
   - **Nodes created:** concepts/new-thing.md
   - **Source node:** sources/summary-name.md
   ```

### Step 4: Report

After all files are processed, report:
```
Compilation complete:
- Files processed: N
- Nodes updated: M (list names)
- Nodes created: K (list names)
- Source nodes created: N
```

## Key Principles

- **You decide the structure.** Templates in TEMPLATES.md are guides, not rigid schemas. Use them as starting points but adapt the structure to fit the content. The LLM builds and maintains this KG — use your judgment about what sections, relationships, and groupings make sense.
- **Temporal provenance is mandatory.** This is the one hard rule: every data point, observation, and financial figure MUST be dated with its source (e.g., "FY2025 10-K", "2026-04-10 brief", "Q1 2026 earnings"). Relationships can change — "Customer A = 53% of revenue (FY2025 10-K)" may not be true next year. The KG must distinguish "what was true when" from "what is true now".
- **Themes are emergent.** Don't predefine theme categories. Discover them from patterns in the data. If 5 tickers share China manufacturing exposure, create a theme. If 3 tickers have customer concentration risk, create a theme. Let the data tell you what the cross-cutting relationships are.
- **You decide what to extract.** Different source types produce different outputs. A pipeline brief creates ticker nodes and themes. A research paper might create concept nodes. A web article might reveal connections. There are no hardcoded extraction rules.
- **Use web search when needed.** If you encounter unfamiliar terms, concepts, or references, search the web for context. Don't guess.
- **Prefer updating over creating.** If a node already exists, enrich it rather than creating a duplicate.
- **Always cite sources.** Every update to an existing node should add to its `> [!quote]- Sources` callout.
- **Link liberally.** Every concept, strategy, ticker, theme, or source mentioned gets a `[[wikilink]]`.
- **The KG should be richer after compilation.** If a raw file has nothing worth extracting, note it in the log and move on.
