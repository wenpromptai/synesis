# Obsidian Syntax Quick Reference

## Links & Embeds

| Syntax | Result |
|--------|--------|
| `[[Note]]` | Internal link |
| `[[Note\|Alias]]` | Link with display text |
| `[[Note#Heading]]` | Link to heading |
| `[[Note#^block-id]]` | Link to block |
| `![[Note]]` | Embed entire note |
| `![[Note#^block-id]]` | Embed block |

## Frontmatter

```yaml
---
up: ["[[Parent]]"]
related: ["[[Related]]"]
created: 2026-04-07
type: concept | strategy | source | map
tags: [tag1, tag2]
aliases: [Alias]
---
```

## Callouts

```markdown
> [!type] Optional Title
> Content

> [!note]-    # Collapsed
> [!note]+    # Expanded
```

Types: note, abstract/tldr, info, tip, success, question, warning, danger, bug, example, quote

## Text

| Syntax | Result |
|--------|--------|
| `**bold**` | **bold** |
| `*italic*` | *italic* |
| `==highlight==` | highlight |
| `` `code` `` | code |
| `%%comment%%` | Hidden |

## Block References

```markdown
Paragraph to reference. ^my-id
[[Note#^my-id]]     # Link to block
```

## Math

```markdown
Inline: $E = mc^2$
Block:
$$
\sum_{i=1}^{n} x_i
$$
```

## Mermaid

````markdown
```mermaid
graph TD
    A --> B
```
````

## Dataview

```dataview
TABLE created, status
FROM #tag
WHERE status != "complete"
SORT created desc
```

## Tags

```markdown
#tag  #nested/tag  #kebab-case
```
