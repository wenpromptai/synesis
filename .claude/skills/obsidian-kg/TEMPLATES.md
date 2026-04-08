# KG Node Templates

## Source Citation Convention

All node types can include source citations when updated during compilation. Add as a collapsed callout at the bottom of the node, before the footer:

```markdown
> [!quote]- Sources
> - [[sources/yartseva-2025|Yartseva (2025)]] — FCF yield as predictor
> - [[sources/brief-2026-04-07|Brief 2026-04-07]] — observed in risk_off regime
```

---

## Concept Node

```markdown
---
up: ["[[Parent Map]]"]
related: ["[[Related Concept]]"]
created: {{date}}
type: concept
tags: [domain-tags]
aliases: [Alternative Name]
---

# {{Concept Name}}

> [!info] Definition
> Clear, concise definition of the concept. ^definition

## Overview

Expanded explanation. What it is, why it matters, how it works.

## Key Properties

1. **Property 1:** Description
2. **Property 2:** Description

## Formula

$$
formula here
$$

## In Practice

How this concept manifests in real trading. Link to strategies that use it:
- [[strategy-1]] uses this for...
- [[strategy-2]] depends on this when...

## Common Misconceptions

> [!warning] Watch Out
> What people often get wrong about this concept.

---
**See also:** [[Related Concept 1]] | [[Related Concept 2]]
```

---

## Strategy Node

A general-purpose strategy template. The LLM adds domain-specific sections as needed (e.g., Greeks Profile for options, screening factors for equity, macro indicators for macro strategies). Not every section is required — use what fits.

```markdown
---
up: ["[[Parent Map]]"]
related: ["[[Related Strategy]]"]
created: {{date}}
type: strategy
category: options | equity | macro | event-driven | multi-asset
complexity: simple | medium | advanced
tags: [strategy-tags]
---

# {{Strategy Name}}

> [!abstract]
> One-line description: what it does and when to use it.

## Core Mechanic

What you do, why it works. Add visuals appropriate to the domain:
- Options: ASCII P&L diagram
- Equity: screening criteria table
- Macro: decision flowchart

## When It Works

- Market conditions / regime fit
- Key signals or thresholds
- What environment makes this strategy most effective

## Construction / Implementation

How to execute. Concrete parameters, step-by-step.

## Risk Management

- Key risks and what triggers them
- Position sizing guidance
- Stop/adjustment rules

> [!danger] Key Risk
> Primary risk and what triggers it.

---
**Related strategies:** [[Strategy 1]] | [[Strategy 2]]
**Concepts:** [[Concept 1]] | [[Concept 2]]
```

---

## Source Node

```markdown
---
up: ["[[Sources Map]]"]
related: []
created: {{date}}
type: source
raw_file: "raw/filename.pdf"
authors: ["Author Name"]
year: 2025
tags: [source, domain]
---

# Source: {{Title}}

> [!info] Citation
> Authors (Year). *Title*. Publisher/Source.

> [!abstract] Key Takeaways
> 1. Finding 1
> 2. Finding 2
> 3. Finding 3

## Summary

What this source covers and its main argument.

## Extracted Concepts

- [[Concept 1]] — how this source informs it
- [[Concept 2]] — what it adds

## Extracted Strategies

- [[Strategy 1]] — connection
- [[Strategy 2]] — connection

## Key Data Points

Notable figures, tables, or statistics worth referencing.

## Limitations

What the source doesn't cover or gets wrong.

---
**See also:** [[Related Source]] | [[Related Map]]
```

---

## Connection Node

```markdown
---
up: ["[[Parent Map]]"]
related: []
created: {{date}}
type: connection
nodes: ["[[node-a]]", "[[node-b]]"]
discovered_from: compilation | linting | manual
tags: [connection]
---

# {{Relationship as statement}}

> [!abstract]
> One-line description of the non-obvious relationship.

## Nodes Involved

- [[node-a]] — role in this connection
- [[node-b]] — role in this connection

## Evidence

What data or observation revealed this connection.

## Implications

What this means for trading decisions or further research.

> [!quote]- Sources
> - [[sources/brief-YYYY-MM-DD]] — where this was observed

---
**See also:** [[Related Node 1]] | [[Related Node 2]]
```
