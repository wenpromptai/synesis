---
description: Optimize skill descriptions for better Claude triggering
---

# Improve Skill Descriptions

Analyze and optimize skill descriptions for maximum Claude triggering effectiveness.

## What This Command Does

1. Reads all skills in `.claude/skills/`
2. Analyzes each SKILL.md content deeply
3. **Optionally searches** for additional trigger keywords
4. Rewrites the `description` field for optimal triggering
5. Updates skill-rules.json with rich trigger patterns

## Why This Matters

Claude selects skills from potentially 100+ available skills based **primarily on the description field**.

> "The description is critical for skill selection: Claude uses it to choose the right Skill from potentially 100+ available Skills."
> — Anthropic Official Documentation (2025)

**Vague description = skill rarely triggers**
**Specific description = skill triggers when needed**

## Analysis Process

### 1. Content Deep Analysis

For each SKILL.md, extract:
- **Technologies** - Framework names, library names, exact terms
- **Versions** - Any version numbers mentioned (e.g., "React 18", "Pydantic v2")
- **Patterns** - What patterns does it teach?
- **File types** - What file extensions does it apply to?
- **User scenarios** - What would a user say to need this?

### 2. Identify All Trigger Scenarios

Think about WHEN Claude should use this skill:

| Trigger Type | Examples |
|--------------|----------|
| File editing | "Working on .tsx files in components/" |
| User requests | "Create a new API endpoint" |
| Code patterns | "Writing async functions" |
| Error fixing | "Fix this React hook error" |
| Questions | "How do I use useEffect?" |

### 3. Keyword Extraction Strategy

**High-value keywords:**
- Framework names (exact): `FastAPI`, `Next.js`, `Prisma`
- Version indicators: `v2`, `18+`, `2.0`
- Key concepts: `async`, `hooks`, `ORM`, `SSR`
- File types: `.tsx`, `.py`, `schema.prisma`
- Directory hints: `api/`, `components/`, `routes/`

**Avoid generic keywords:**
- "development", "best practices", "guidelines"
- "code", "application", "project"
- "various", "different", "multiple"

### 4. Optional: Web Search for Keywords

If the skill seems under-described, search for common terminology:

```
Search: "[technology] common terms developer vocabulary"
```

Extract additional trigger keywords from how developers actually talk about the technology.

## Description Formula

### Structure (Max 1024 chars)

```
[Primary tech] [version]+ [activity gerund] with [feature1], [feature2], [feature3]. 
Use when [trigger1], [trigger2], or [trigger3]. 
Apply for [file patterns].
```

### Requirements Checklist

For each description, verify:

- [ ] **Specific tech name** - "FastAPI" not "Python framework"
- [ ] **Version context** - "React 18+" or "Pydantic v2" 
- [ ] **Gerund activity** - "developing", "building", "querying"
- [ ] **3+ key features** - async, hooks, ORM, etc.
- [ ] **"Use when" phrase** - with 2-3 trigger scenarios
- [ ] **"Apply for" phrase** - with file patterns or directories
- [ ] **Action verbs** - creating, building, fixing, implementing, debugging
- [ ] **Under 300 chars** - Aim for 200-280 for best results
- [ ] **Third person** - "Develops APIs" not "You can develop APIs"
- [ ] **No marketing speak** - No "powerful", "comprehensive", "elegant"

## Example Transformations

### Example 1: Backend Skill

**Before:**
```yaml
description: Guidelines for backend development
```

**After:**
```yaml
description: FastAPI 0.115+ async API developing with Pydantic v2 validation, dependency injection, lifespan managers, and SQLAlchemy 2.0 ORM. Use when creating endpoints, models, services, or debugging backend errors. Apply for .py files in api/, routes/, services/.
```

**Why it's better:**
- Specific: "FastAPI 0.115+" not "backend"
- Versioned: "Pydantic v2", "SQLAlchemy 2.0"
- Action triggers: "creating", "debugging"
- File context: ".py files in api/, routes/"

### Example 2: Frontend Skill

**Before:**
```yaml
description: React component patterns
```

**After:**
```yaml
description: React 19+ functional component building with TypeScript, hooks (use, useActionState, useOptimistic), Server Components, and React Compiler. Use when creating components, managing state, handling forms, or fixing hook errors. Apply for .tsx files in components/, pages/, features/.
```

### Example 3: Database Skill

**Before:**
```yaml
description: Database queries and models
```

**After:**
```yaml
description: Drizzle ORM querying with TypeScript, identity columns, relations, and type-safe queries. Use when writing queries, defining schema, running migrations, or debugging database errors. Apply for drizzle.config.ts and schema/*.ts files.
```

## Also Update skill-rules.json

After improving descriptions, ensure skill-rules.json has matching rich triggers:

### Before:
```json
{
  "skills": [
    {
      "name": "fastapi-backend",
      "keywords": ["backend"]
    }
  ]
}
```

### After:
```json
{
  "skills": {
    "fastapi-developing": {
      "type": "domain",
      "enforcement": "suggest",
      "priority": "high",
      "promptTriggers": {
        "keywords": [
          "fastapi", "pydantic", "api", "endpoint", "route",
          "async", "backend", "REST", "OpenAPI", "uvicorn",
          "dependency injection", "lifespan", "middleware"
        ],
        "intentPatterns": [
          "(create|add|build|implement).*?(endpoint|route|api|controller)",
          "(fix|debug|handle).*?(error|exception|backend|api)",
          "(how to|best practice|pattern).*?(fastapi|async|pydantic)"
        ]
      },
      "fileTriggers": {
        "pathPatterns": [
          "app/**/*.py",
          "api/**/*.py",
          "routes/**/*.py",
          "services/**/*.py",
          "main.py"
        ],
        "contentPatterns": [
          "from fastapi import",
          "from pydantic import",
          "@app\\.(get|post|put|delete|patch)",
          "APIRouter"
        ]
      }
    }
  }
}
```

## Output Format

Show detailed before/after for each skill:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📁 .claude/skills/fastapi-developing/SKILL.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BEFORE (52 chars):
  description: Guidelines for backend development

AFTER (267 chars):
  description: FastAPI 0.115+ async API developing with Pydantic v2 
  validation, dependency injection, lifespan managers, and SQLAlchemy 
  2.0 ORM. Use when creating endpoints, models, services, or debugging 
  backend errors. Apply for .py files in api/, routes/, services/.

ANALYSIS:
  ✅ Specific tech name: FastAPI
  ✅ Version context: 0.115+, v2, 2.0
  ✅ Key features: 5 listed
  ✅ Trigger phrase: "Use when..."
  ✅ File context: "Apply for..."
  ✅ Length: 267 chars (under 300)

KEYWORDS ADDED TO skill-rules.json:
  + fastapi, pydantic, async, endpoint, route, api, uvicorn
  + Intent patterns: 3 added
  + Path patterns: 5 added
  + Content patterns: 4 added

✅ Updated
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Summary Report

After improving all skills, report:

```
📊 Skill Improvement Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Skills analyzed:     5
Descriptions updated: 4
Already optimal:      1

Average description length:
  Before: 47 chars
  After:  248 chars

Trigger coverage:
  Keywords added:      47
  Intent patterns:     12
  Path patterns:       18
  Content patterns:    14

⚠️ Recommendations:
  - fastapi-developing: Consider adding "swagger" keyword
  - nextjs-building: Missing "app router" in triggers
```

## Advanced: Iterative Improvement

If skills still aren't triggering after improvement:

1. **Check actual usage** - What prompts SHOULD have triggered the skill?
2. **Add those exact phrases** - User vocabulary matters
3. **Test with 3+ scenarios** - Try different phrasings
4. **Re-run /skill:improve** - With new context

```
User tried: "help me write a fastapi route"
Skill didn't trigger because: "route" wasn't in keywords

Fix: Add "route" to keywords, add intent pattern "(write|create).*?route"
```
