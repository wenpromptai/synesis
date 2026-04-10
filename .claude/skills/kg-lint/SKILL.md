---
name: kg-lint
description: Run health checks on the docs/kg/ knowledge graph. Finds broken wikilinks, orphan nodes, sparse linking, missing frontmatter, and uncompiled raw files.
user_invocable: true
---

# KG Lint

Run a health check on the knowledge graph at `docs/kg/`.

## Checklist

Execute these checks in order, reporting findings for each:

### 1. Broken Wikilinks
- Glob all `.md` files in `docs/kg/` (exclude `raw/`)
- Grep for `\[\[([^\]|#]+)` to extract all wikilink targets
- For each unique target, check if a matching `.md` file exists anywhere in `docs/kg/`
- **Report:** List of broken links with the file they appear in

### 2. Orphan Nodes
- Build a set of all wikilink targets across all files (incoming links)
- Compare against all existing `.md` files
- A node is orphan if NO other file links to it (except `_index.md` and `home.md` which are entry points)
- **Report:** List of orphan files that nothing links to

### 3. Sparse Nodes
- For each `.md` file (exclude `_index.md`, maps), count outgoing wikilinks
- Flag nodes with fewer than 3 outgoing wikilinks
- **Report:** List of sparse nodes with their link count

### 4. Missing Frontmatter
- For each `.md` file (exclude `_index.md`), check YAML frontmatter exists and contains:
  - Required: `type`, `created`, `up`
  - Ticker nodes also need: `sector`, `industry`
  - Theme nodes also need: `theme_type`
  - Strategy nodes also need: `category`, `complexity`
- **Report:** List of files with missing required fields

### 5. Uncompiled Raw Files
- List all files in `raw/` (all subdirectories including `synesis_briefs/`)
- For each, check if it appears in `_compile_log.md` as already compiled
- **Report:** List of raw files without source nodes

### 6. Stale Index
- Compare nodes listed in `_index.md` against actual files in `docs/kg/`
- Flag nodes that exist on disk but are missing from the index
- Flag index entries that reference files that no longer exist
- **Report:** Missing from index / phantom index entries

## Output Format

```
## KG Health Report

**Scanned:** X nodes across Y directories
**Issues found:** N

### Broken Links (N)
- `file.md` → [[missing-target]]

### Orphan Nodes (N)
- `orphan-file.md`

### Sparse Nodes (N)
- `file.md` — 1 link (minimum: 3)

### Missing Frontmatter (N)
- `file.md` — missing: type, created

### Uncompiled Raw Files (N)
- `raw/document.pdf`

### Stale Index (N)
- Missing from index: `new-file.md`
- Phantom entry: `deleted-file.md`

---
**Suggestion:** [specific fix recommendations for critical issues]
```

## Intelligence Checks

These go beyond structural health — they discover knowledge gaps and suggest how to grow the KG.

### 7. Connection Discovery
- For each node, extract all `[[wikilinks]]` it contains
- Find clusters of nodes that share common topics/tags but don't link to each other
- Flag pairs/groups that likely should be connected or warrant a connection node
- **Report:** "These nodes all discuss X but none link to each other: [list]"
- **Suggestion:** "Consider creating a connection node in `sources/connections/`"

### 8. Missing Node Candidates
- Collect all `[[wikilinks]]` that point to non-existent files (from check #1)
- Count how many times each missing target is referenced across all nodes
- If referenced 3+ times, it's a strong candidate for a dedicated node
- **Report:** "`[[missing-concept]]` referenced by 5 nodes — strong candidate for a new concept node"

### 9. Content Staleness
- For each node, check its `created` or last `Updated` date
- Cross-reference with `_compile_log.md` — have new raw sources been compiled since this node was last updated?
- Flag nodes that haven't been updated despite relevant new raw sources mentioning related topics
- **Report:** "`delta.md` last updated 2026-04-07, but 3 briefs compiled since then mention delta-related content"

### 10. Impute Missing Data
- For nodes flagged as stale (check #9) or sparse (check #3), use **web search** to find current information that could enrich them
- For missing node candidates (check #8) that are well-known concepts, search the web for authoritative definitions and context
- **Report:** "Searched for X — found relevant info that could enrich [[node-name]]"
- Do NOT auto-update nodes. Present findings and let the user decide whether to apply them via `/kg-compile` or manual update.

### 11. Research Suggestions
- Synthesize findings from checks 7-10 into concrete, actionable next steps:
  - "Create a connection node linking `[[calendar-spread]]` and `[[earnings-options-systematic]]` — both reference pre-earnings IV but don't cross-reference"
  - "Node `[[volatility-risk-premium]]` has no source citations — consider enriching via `/kg-compile`"
  - "5 strategy nodes reference 'regime' but `[[regime-options-matrix]]` doesn't link back to all of them — add bidirectional links"
- **Report:** Numbered list of suggestions, ordered by impact

---

## Auto-Fix Option

After reporting, ask the user if they want to auto-fix:
- Add missing nodes to `_index.md`
- Remove phantom entries from `_index.md`
- Create stub concept nodes for broken links that appear 3+ times
- Add bidirectional links where missing (from check #7)

Do NOT auto-fix without asking.
