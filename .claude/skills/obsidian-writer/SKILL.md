---
name: obsidian-writer
description: Format and write markdown notes optimized for Obsidian. Use when creating research notes, documentation, or any markdown content for Obsidian vaults. Applies Obsidian-specific syntax including wikilinks, embeds, callouts, block references, YAML frontmatter, LYT framework patterns (MOCs, atomic notes, ACE structure), and Dataview integration. Automatically formats output for Obsidian compatibility.
---

# Obsidian Markdown Writer

You are an expert at writing well-structured, beautifully formatted markdown notes optimized for **Obsidian**. Apply the **LYT (Linking Your Thinking)** framework patterns alongside standard Obsidian syntax for maximum knowledge management effectiveness.

---

## LYT Framework Integration

The LYT framework emphasizes **linking over folders**, using Maps of Content (MOCs), and atomic/evergreen notes. Key principles:

1. **Link liberally** - Create connections even to non-existent notes (~8 links per note is good density)
2. **Structure emerges** - Don't over-organize upfront; let structure develop from links
3. **Notes are nodes** - Each note is a point in your knowledge graph, not a file in a folder

### ACE Folder Framework

Organize your vault into three core "headspaces":

| Folder | Purpose | Contains |
|--------|---------|----------|
| **Atlas** | Knowledge | Maps (MOCs) + Dots (atomic notes) |
| **Calendar** | Time | Daily notes, logs, periodic reviews |
| **Efforts** | Action | Projects with states: On, Ongoing, Simmering, Sleeping |
| **+** | Inbox | Quick captures to process later |
| **x** | Extras | Templates, images, attachments |

---

## Core Obsidian Syntax

### Internal Links (Wikilinks)

```markdown
[[Note Name]]                    # Link to another note
[[Note Name|Display Text]]       # Link with custom display text
[[Note Name#Heading]]            # Link to specific heading
[[Note Name#^block-id]]          # Link to specific block
[[Note Name#Heading|Alias]]      # Heading link with alias
```

### Embeds (Transclusion)

Prefix any wikilink with an exclamation mark to embed content inline:

```markdown
![[Note Name]]                   # Embed entire note
![[Note Name#Heading]]           # Embed specific section
![[Note Name#^block-id]]         # Embed specific block
![[image.png]]                   # Embed image from vault
![[image.png|300]]               # Embed image with width
![[image.png|300x200]]           # Embed image with dimensions
![[audio.mp3]]                   # Embed audio
![[video.mp4]]                   # Embed video
![[document.pdf]]                # Embed PDF
```

### Block References

Create linkable blocks by adding `^block-id` at end of any paragraph:

```markdown
This is a paragraph you want to reference. ^my-block-id

Link to it: [[Note#^my-block-id]]
Embed it: ![[Note#^my-block-id]]
```

**Rules for block IDs:**
- Only letters, numbers, and hyphens allowed
- Case-sensitive
- Must be unique within the note

---

## YAML Frontmatter (Properties)

Always include frontmatter at the top of notes. Use **LYT navigation properties** for better linking:

```yaml
---
up:                              # Parent/hierarchy (e.g., "[[Home]]")
  - "[[Parent Note]]"
related:                         # Related concepts
  - "[[Related Note 1]]"
  - "[[Related Note 2]]"
in:                              # Collection membership
  - "[[Maps]]"
created: 2025-12-14
tags:
  - research
  - trading
aliases:
  - Alternative Name
status: draft | in-progress | complete | review
type: research | moc | statement | effort | log
rank:                            # Priority for efforts (optional)
---
```

### LYT Navigation Properties

| Property | Purpose | Example |
|----------|---------|---------|
| `up` | Hierarchical parent(s) | `"[[Home]]"`, `"[[Habits Map]]"` |
| `related` | Conceptually related notes | `"[[Similar Topic]]"` |
| `in` | Collection membership | `"[[Maps]]"`, `"[[Sources]]"` |
| `created` | Creation date | `2025-12-14` |
| `rank` | Priority for efforts | `1-10` scale |

### Standard Obsidian Properties

- `tags` / `tag` - Categorization
- `aliases` / `alias` - Alternative names for linking
- `cssclass` - Custom CSS styling
- `publish` - For Obsidian Publish

**Custom Properties** - Add any key-value pairs:
```yaml
project: Alpha Research
priority: high
author: Your Name
source: https://example.com
```

---

## Callouts

Use callouts to highlight important information:

```markdown
> [!note]
> Default note callout

> [!tip] Custom Title
> Tip with custom title

> [!warning]- Collapsed by Default
> Click to expand

> [!danger]+ Expanded by Default
> Important warning
```

### Callout Types

| Type | Aliases | Use For |
|------|---------|---------|
| `note` | - | General information |
| `abstract` | summary, tldr | Executive summaries |
| `info` | todo | Informational content |
| `tip` | hint, important | Helpful suggestions |
| `success` | check, done | Completed items, confirmations |
| `question` | help, faq | Questions, uncertainties |
| `warning` | caution, attention | Cautions, watch-outs |
| `failure` | fail, missing | Failures, missing items |
| `danger` | error | Critical warnings |
| `bug` | - | Known issues |
| `example` | - | Examples, demonstrations |
| `quote` | cite | Citations, quotes |

### Nested Callouts

```markdown
> [!question] Research Question
> What ML model works best?
> > [!tip] Hypothesis
> > Transformer architectures may outperform LSTM
```

### Callouts with Dataview (Advanced)

Embed dynamic content inside callouts:

```markdown
> [!Box]+ ### 🔥 Active Efforts
> ```dataview
> TABLE WITHOUT ID
>   file.link as "Effort",
>   rank as "Priority"
> FROM "Efforts/On"
> SORT rank desc
> ```
```

---

## Maps of Content (MOCs)

MOCs are **index notes** that gather, develop, and navigate related ideas. They're the backbone of the LYT system.

### Three Phases of MOCs

1. **Gather** - Collect relevant notes on a topic
2. **Collide** - Develop ideas, find connections, create insights
3. **Navigate** - Use as a launching point to explore the topic

### MOC Structure

```markdown
---
up:
  - "[[Home]]"
related: []
created: {{date}}
in:
  - "[[Maps]]"
---

# Topic Map

> [!abstract] Overview
> Brief description of what this map covers.

## Core Concepts
- [[Concept 1]] - Brief description
- [[Concept 2]] - Brief description

## Key Insights
- [[Insight as statement title]]
- [[Another insight statement]]

## Related Areas
[[Related Topic 1]], [[Related Topic 2]]

---
Back to [[Home]]
```

### LYT Vision (Finding Gaps)

Use Dataview to find notes that link here but aren't linked back:

```markdown
> [!tip]+ Unrequited Notes
> Notes linking here that this note doesn't link back to:
> ```dataview
> LIST
> FROM [[]]
> AND !outgoing([[]])
> SORT file.name asc
> ```
```

---

## Statement Notes (Atomic/Evergreen Notes)

Statement notes capture **one idea per note** with the title as a complete thought.

### Naming Convention

Use **full sentences** as titles:
- ✅ "Evergreen notes compound in value over time"
- ✅ "Small wins foster a sense of control"
- ✅ "Linking your thinking encourages leaps of insight"
- ❌ "Evergreen Notes" (too vague)
- ❌ "Notes" (not specific)

### Statement Note Structure

```markdown
---
up:
  - "[[Parent MOC]]"
related:
  - "[[Related Statement]]"
created: {{date}}
---

Main content explaining the idea. Keep it focused on this single concept.

This connects to [[Another Concept]] because...

> [!note]- Supporting Evidence
> Details, quotes, or references that support this statement.

---
See also: [[Related Idea 1]], [[Related Idea 2]]
```

### Benefits of Statement Notes

- **Reusable** - Embed anywhere (prefix link with exclamation mark)
- **Linkable** - Easy to reference in other notes
- **Evergreen** - Grow in value as you add connections
- **Searchable** - Full thought in title makes finding easy

---

## Text Formatting

```markdown
**bold text**
*italic text*
***bold and italic***
~~strikethrough~~
==highlighted text==
`inline code`
```

### Comments (Hidden in Preview)

```markdown
%%
This is a block comment.
It won't appear in reading view.
Multiple lines supported.
%%

Inline comment: %%hidden%% visible
```

---

## Code Blocks

````markdown
```python
def example():
    return "Syntax highlighted"
```

```dataview
TABLE file.mtime as "Modified"
FROM #research
SORT file.mtime DESC
```

```mermaid
graph TD
    A[Start] --> B{Decision}
    B -->|Yes| C[Action]
    B -->|No| D[End]
```
````

### Supported Languages
Python, JavaScript, TypeScript, SQL, JSON, YAML, Bash, and 100+ more via Prism.js.

---

## Math (LaTeX)

**Inline math:**
```markdown
The formula $E = mc^2$ is famous.
```

**Block math:**
```markdown
$$
\frac{d}{dx}\left( \int_{a}^{x} f(u)\,du\right)=f(x)
$$
```

---

## Tables

```markdown
| Column 1 | Column 2 | Column 3 |
|----------|:--------:|---------:|
| Left     | Center   | Right    |
| aligned  | aligned  | aligned  |
```

**Alignment:**
- `:---` Left align
- `:---:` Center align
- `---:` Right align

---

## Lists

### Unordered
```markdown
- Item 1
- Item 2
  - Nested item
    - Deeper nested
```

### Ordered
```markdown
1. First item
2. Second item
   1. Nested numbered
```

### Task Lists
```markdown
- [ ] Unchecked task
- [x] Completed task
- [/] In progress (some themes)
- [-] Cancelled (some themes)
```

---

## Headings & Structure

```markdown
# H1 - Note Title (use once)
## H2 - Major Sections
### H3 - Subsections
#### H4 - Sub-subsections
##### H5 - Minor headings
###### H6 - Smallest heading
```

**Best Practice:** Use H1 only for the note title, start content sections with H2.

---

## Tags

```markdown
#tag
#nested/tag
#tag-with-dashes
#tag_with_underscores
```

Tags can appear:
- In YAML frontmatter (recommended for main tags)
- Inline in content (for contextual tagging)

---

## Footnotes

```markdown
Here is a statement[^1] that needs a citation.

Another claim[^longnote] with more detail.

[^1]: Simple footnote.
[^longnote]: Longer footnote with multiple paragraphs.

    Indent to include more content in the footnote.
```

---

## Horizontal Rules

```markdown
---
***
___
```

---

## External Links & Images

```markdown
[Link Text](https://example.com)
[Link with Title](https://example.com "Title on hover")

![Alt text](https://example.com/image.png)
![Alt text](https://example.com/image.png "Image title")
```

**Note:** External images with an exclamation mark prefix will display inline if accessible.

---

## Diagrams (Mermaid)

### Flowchart
````markdown
```mermaid
graph TD
    A[Input Data] --> B[Preprocess]
    B --> C{Valid?}
    C -->|Yes| D[Train Model]
    C -->|No| E[Clean Data]
    E --> B
    D --> F[Evaluate]
```
````

### Sequence Diagram
````markdown
```mermaid
sequenceDiagram
    User->>+API: Request data
    API->>+Database: Query
    Database-->>-API: Results
    API-->>-User: Response
```
````

### Timeline
````markdown
```mermaid
timeline
    title Project Timeline
    2025-01 : Research phase
    2025-02 : Development
    2025-03 : Testing
    2025-04 : Launch
```
````

---

## Research Note Best Practices

### Structure for Research Notes

```markdown
---
title: Research Topic
date: {{date}}
tags: [research, topic-tag]
status: in-progress
type: research
---

# Research Topic

## TL;DR
> [!abstract]
> 2-3 sentence summary of key findings

## Research Question
What are we trying to answer?

## Key Findings

### Finding 1
Details with [[links to related notes]]

> [!tip] Actionable Insight
> What to do with this finding

### Finding 2
More details...

## Methodology
How the research was conducted

## Sources
- [Source 1](url) - Brief description
- [[Internal Source Note]]

## Open Questions
- [ ] Question 1
- [ ] Question 2

## Related Notes
- [[Related Topic 1]]
- [[Related Topic 2]]
```

### Linking Conventions

1. **Use wikilinks** `[[Note]]` for internal vault links
2. **Create links liberally** - even for notes that don't exist yet
3. **Use aliases** for natural reading: `[[Technical Term|simple term]]`
4. **Embed key definitions** using exclamation mark prefix on wikilinks

### Callout Usage Guide

| Scenario | Callout Type |
|----------|--------------|
| Executive summary | `[!abstract]` or `[!tldr]` |
| Key insight or takeaway | `[!tip]` |
| Important caveat | `[!warning]` |
| Critical limitation | `[!danger]` |
| Hypothesis or question | `[!question]` |
| Code example | `[!example]` |
| Source citation | `[!quote]` |
| Action item | `[!todo]` |
| Confirmed finding | `[!success]` |

---

## Keyboard Shortcuts Reference

| Action | Windows/Linux | Mac |
|--------|---------------|-----|
| Bold | `Ctrl+B` | `Cmd+B` |
| Italic | `Ctrl+I` | `Cmd+I` |
| Link | `Ctrl+K` | `Cmd+K` |
| Code | `Ctrl+E` | `Cmd+E` |
| Toggle checkbox | `Ctrl+Enter` | `Cmd+Enter` |
| Quick switcher | `Ctrl+O` | `Cmd+O` |
| Command palette | `Ctrl+P` | `Cmd+P` |
| Search all files | `Ctrl+Shift+F` | `Cmd+Shift+F` |
| Graph view | `Ctrl+G` | `Cmd+G` |
| Split pane | `Ctrl+\` | `Cmd+\` |

---

## Dataview Integration

Dataview enables dynamic queries within your notes.

### Common Query Patterns

**List notes in a folder:**
```dataview
LIST FROM "Atlas/Maps"
SORT file.name asc
```

**Table with properties:**
```dataview
TABLE
  created as "Created",
  status as "Status"
FROM #research
WHERE status != "complete"
SORT created desc
```

**Notes by property value:**
```dataview
LIST FROM [[]]
WHERE contains(up, this.file.link)
```

**Efforts by rank:**
```dataview
TABLE WITHOUT ID
  file.link as "Effort",
  rank as "Priority"
FROM "Efforts/On"
SORT rank desc
```

### Inline Queries

```markdown
Total research notes: `= length(filter(pages(""), (p) => contains(p.tags, "research")))`

Last modified: `= this.file.mtime`
```

---

## Output Guidelines

When writing content for Obsidian:

### Frontmatter
1. **Always include YAML frontmatter** with LYT properties: `up`, `related`, `created`
2. **Use `in` property** for collection membership (e.g., `in: ["[[Maps]]"]`)
3. **Add `rank`** for effort/project prioritization

### Linking
4. **Use wikilinks** `[[Note]]` not markdown links for internal references
5. **Link liberally** - create connections even to non-existent notes (~8 per note)
6. **Use aliases** for natural reading: `[[Technical Term|simple term]]`

### Structure
7. **Apply callouts** for important information, tips, warnings
8. **Create block references** `^block-id` for reusable content
9. **Use embeds** to transclude related content (syntax: exclamation + wikilink)
10. **Structure with clear headings** (H2 for sections, H3 for subsections)
11. **Include a TL;DR** at the top of research notes using `> [!abstract]`

### Navigation
12. **Add footer navigation** - `Back to [[Home]]` or `See also: [[Related]]`
13. **Use MOCs** as index notes for topic areas
14. **Create statement notes** with full-sentence titles for atomic ideas

### Content
15. **Add task checkboxes** for action items
16. **Use code blocks** with language identifiers for syntax highlighting
17. **Embed Dataview queries** for dynamic content in callouts
