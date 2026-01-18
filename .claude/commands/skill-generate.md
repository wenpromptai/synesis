---
description: Scan project, research best practices, generate skills with web search
---

# Generate Project Skills

Analyze this project, research current best practices, and generate production-quality Claude Code Skills.

## What This Command Does

1. **Scan** project files to detect the tech stack + versions
2. **Research** (web search) current December 2025 best practices for each detected technology
3. **Generate** skills following progressive disclosure pattern
4. **Configure** rich trigger rules in `skill-rules.json`

## Supported Languages

This skill generator focuses on three primary languages:

- **Python** (3.13+) with **uv** package manager
- **Go** (1.22+)
- **TypeScript** (5.6+) with **pnpm** 10.x

---

## Phase 1: Detection (Scan + Extract Versions)

### Scan for Tech Stack

Look for these files/patterns AND extract version numbers:

**Python Backend (uv preferred):**
| File | Technology | Version Source |
|------|------------|----------------|
| `pyproject.toml` | Python packages | `[project.dependencies]` (PEP 621 / uv) |
| `uv.lock` | uv lockfile | Exact locked versions |
| `.python-version` | Python version | `3.13` |

Key Python setup:

- **uv** as package manager (`uv init`, `uv add`, `uv run`, `uv sync`)
- Python 3.13+ (`project.requires-python = ">=3.13"`)
- FastAPI 0.115+ (with Pydantic v2)
- Pydantic 2.x (NOT v1 patterns)
- Database driver (detected from project)
- pydantic-settings (for configuration)

**Go Backend:**
| File | Technology | Version Source |
|------|------------|----------------|
| `go.mod` | Go version | `go 1.22` or `go 1.23` |
| `go.sum` | Dependencies | Package versions |

Key Go packages to detect:

- Gin, Echo, or Fiber (web frameworks)
- log/slog (structured logging, Go 1.21+)
- Standard library http.ServeMux (Go 1.22+ with pattern routing)

**TypeScript/Node.js (pnpm 10.x preferred):**
| File | Technology | Version Source |
|------|------------|----------------|
| `package.json` | Node packages | Check dependencies |
| `pnpm-lock.yaml` | pnpm 10.x lockfile | Preferred package manager |
| `pnpm-workspace.yaml` | Monorepo config | Workspace packages |
| `tsconfig.json` | TypeScript config | `"target"`, `"module"` |

**Frontend (TypeScript):**
| File | Technology | Version Source |
|------|------------|----------------|
| `package.json` | React 19.x | `"react": "^19.0.0"` |
| `package.json` | Next.js 16.x | `"next": "^16.0.0"` |
| `package.json` | Tailwind CSS 4.x | `"tailwindcss": "^4.0.0"` |
| `package.json` | tRPC 11.x | `"@trpc/server": "^11.0.0"` |

**Database ORMs (TypeScript):**
| File | Technology | Version Source |
|------|------------|----------------|
| `drizzle.config.ts` | Drizzle ORM | package.json |
| `prisma/schema.prisma` | Prisma 6.x | package.json |

**Infrastructure:**
| File | Technology | Notes |
|------|------------|-------|
| `docker-compose.yml` | Docker | Check image versions |
| `.github/workflows/` | GitHub Actions | Check action versions |

### Output: Detection Report

```
Detected Technologies:
├── Backend: Python 3.13 + FastAPI 0.115.0 + Pydantic 2.9.0
├── Python Package Manager: uv (pyproject.toml + uv.lock)
├── Frontend: Next.js 16.0 + React 19.2 + Tailwind 4.0
├── API Layer: tRPC 11.0
├── Database: PostgreSQL + Drizzle ORM
├── Node Package Manager: pnpm 10.x
└── Infrastructure: Docker + GitHub Actions
```

---

## Phase 2: Research (Web Search)

**CRITICAL: For each detected technology, search for current best practices (December 2025).**

### Search Strategy

For each technology, perform web searches:

```
Search 1: "[Technology] best practices 2025"
Search 2: "[Technology] [detected version] patterns modern"
Search 3: "[Technology] common mistakes anti-patterns"
```

### Technology-Specific Searches

#### Python/FastAPI (with uv)

```
- "FastAPI best practices 2025 Pydantic v2 async lifespan"
- "uv python package manager pyproject.toml 2025"
- "Python 3.13 type hints patterns"
```

**Key patterns to extract:**

- Use **uv** for package management:
  - `uv init` to create project with pyproject.toml
  - `uv add fastapi` to add dependencies
  - `uv add --dev pytest mypy ruff` for dev dependencies
  - `uv run` to execute scripts (auto-syncs environment)
  - `uv.lock` for reproducible installs
- **Type safety (always run):**
  - `uv run mypy --strict .` for static type checking
  - `uv run ruff check .` for linting
- Use `lifespan` context manager (NOT `@app.on_event()` - deprecated)
- Use `model_config = ConfigDict()` (NOT `class Config:`)
- Use `model_validate()` (NOT `parse_obj()`)
- Use `model_dump()` (NOT `dict()`)
- Use `Annotated[T, Depends()]` for dependency injection
- Use `str | None` (NOT `Optional[str]`)
- Use `type` statement for type aliases (Python 3.12+)
- Use pydantic-settings for configuration

#### Go

```
- "Go 1.23 best practices 2025 patterns"
- "Go http.ServeMux pattern routing 1.22"
- "Go log/slog structured logging"
```

**Key patterns to extract:**

- Use `log/slog` for structured logging (Go 1.21+)
- Use `http.ServeMux` with pattern routing (Go 1.22+) or Gin/Echo/Fiber
- Use `fmt.Errorf` with `%w` for error wrapping
- Keep interfaces small and focused
- Use `/internal` for private code
- Go Modules are mandatory

#### TypeScript (with pnpm 10)

```
- "TypeScript 5.7 best practices 2025 strict mode"
- "pnpm 10 new features 2025"
- "pnpm workspace monorepo catalog"
```

**Key patterns to extract:**

- **Type safety (always run):**
  - `npx tsc --noEmit` for static type checking
  - Enable `strict: true` in tsconfig.json
- Use ESM (`import`/`export`, NOT `require`)
- Prefer `unknown` over `any`
- Use `satisfies` operator for type narrowing
- Use Zod for runtime validation
- Use **pnpm 10.x**:
  - `workspace:*` protocol for local packages
  - `catalog:` protocol for shared versions
  - `pnpm.onlyBuiltDependencies` for lifecycle scripts (disabled by default)
  - `inject-workspace-packages` for hard-linking

#### Next.js 16

```
- "Next.js 16 best practices 2025 App Router"
- "Next.js 16 cache components use cache"
- "React Server Components patterns"
```

**Key patterns to extract:**

- Turbopack is now default bundler
- Use `use cache` for caching (new in Next.js 16)
- React 19.2 features: View Transitions, useEffectEvent
- React Compiler is stable (auto-memoization)
- Server Components by default, Client Components for interactivity
- Use `export const dynamic = 'force-dynamic'` for SSR
- Server Actions with `"use server"` directive

#### React 19

```
- "React 19 best practices 2025 use hook"
- "React 19 useActionState useFormStatus useOptimistic"
- "React 19 Server Components patterns"
```

**Key patterns to extract:**

- New hooks: `use()`, `useActionState`, `useFormStatus`, `useOptimistic`
- `use()` can read promises and context (can be called conditionally)
- Server Actions with `"use server"` for form handling
- React Compiler eliminates need for manual `useMemo`/`useCallback`
- Avoid overusing `useEffect` - prefer derived state

#### Tailwind CSS v4

```
- "Tailwind CSS v4 best practices 2025"
- "Tailwind v4 @theme CSS variables"
- "Tailwind v4 migration from v3"
```

**Key patterns to extract:**

- Just `@import "tailwindcss"` (no `@tailwind` directives)
- Zero configuration needed
- Use `@theme` directive for CSS-first configuration
- Native cascade layers with `@layer`
- Dynamic utility values (e.g., `w-103` works without config)
- Use `color-mix()` for opacity modifiers
- Container queries built-in

#### tRPC v11

```
- "tRPC v11 best practices 2025 Next.js"
- "tRPC 11 TanStack React Query integration"
- "tRPC v11 Server Components prefetch"
```

**Key patterns to extract:**

- New TanStack React Query integration with `queryOptions()`
- Use `useTRPC()` hook with native `useQuery()`
- RSC prefetching with `createTRPCOptionsProxy`
- FormData and binary types (Blob, File) now supported
- Install: `pnpm add @trpc/server @trpc/client @trpc/tanstack-react-query @tanstack/react-query zod superjson`

#### Drizzle ORM

```
- "Drizzle ORM best practices 2025 TypeScript"
- "Drizzle PostgreSQL identity columns patterns"
```

**Key patterns to extract:**

- Use identity columns (NOT serial): `integer().primaryKey().generatedAlwaysAsIdentity()`
- Code-first schema definition in TypeScript
- Both relational and SQL-like query APIs
- Use `$onUpdateFn()` for updated_at timestamps
- Organize schema in separate `schema/` directory
- Zero dependencies, tree-shakeable (~7.4kb)

#### Prisma 6

```
- "Prisma 6 best practices 2025 TypeScript"
- "Prisma TypeScript performance optimization"
```

**Key patterns to extract:**

- Use `typeof client` for large schemas (99.9% reduction in TS instantiations)
- Prisma Accelerate for connection pooling
- Split schema into multiple files
- Use `prisma.config.ts` for configuration
- Selective field retrieval and pagination

---

## Phase 3: Generate Skills (Progressive Disclosure)

### Folder Structure

```
.claude/skills/
├── [tech-name]-[activity]/
│   ├── SKILL.md              # Main file (<500 lines)
│   └── resources/            # Progressive disclosure
│       ├── patterns.md       # Detailed patterns
│       ├── anti-patterns.md  # What to avoid
│       ├── examples.md       # Code examples
│       └── migration.md      # Version migration guide (if relevant)
```

### SKILL.md Format

```markdown
---
name: [tech-name]-[activity]
description: [Technology] [version]+ [activity] with [key features]. Use when [trigger scenarios]. Apply for [file types] and [user requests].
---

# [Technology] Development Standards

## When To Apply

- Creating/editing [file patterns]
- User mentions [keywords]
- Working with [related technologies]

## Quick Reference

### Current Patterns (December 2025)

[Most important modern patterns - keep brief]

### Key Points

- [Point 1 - actionable]
- [Point 2 - specific]
- [Point 3 - version-aware]

## Resource Files

- [patterns.md](resources/patterns.md) - Detailed implementation patterns
- [anti-patterns.md](resources/anti-patterns.md) - Common mistakes to avoid
- [examples.md](resources/examples.md) - Working code examples

## Version Notes

- Current version: [detected version]
- Minimum supported: [if applicable]
- Breaking changes from previous: [if applicable]
```

### Naming Convention

Use **gerund form** (verb + -ing) for skill names:

- ✅ `fastapi-developing`
- ✅ `nextjs-building`
- ✅ `react-components-creating`
- ✅ `drizzle-querying`
- ✅ `go-developing`
- ❌ `fastapi-backend` (noun)
- ❌ `react-guide` (noun)

### Description Formula (Max 1024 chars)

```
[Primary tech] [version]+ [activity] with [key features list].
Use when [trigger scenarios].
Apply for [file types/extensions].
```

**Good Example:**

```yaml
description: FastAPI 0.115+ async API developing with Pydantic v2 validation, dependency injection, lifespan managers, and SQLAlchemy 2.0. Use when creating endpoints, models, services, middleware, or debugging backend errors. Apply for .py files in api/, routes/, services/, or main.py.
```

```yaml
description: Next.js 16+ app building with React 19, App Router, Server Components, cache components, and Turbopack. Use when creating pages, layouts, server actions, or API routes. Apply for .tsx files in app/, components/.
```

```yaml
description: Go 1.23+ backend developing with http.ServeMux routing, slog logging, and standard library patterns. Use when creating handlers, middleware, services. Apply for .go files.
```

---

## Phase 4: Configure Triggers (skill-rules.json)

### Rich Trigger Format

```json
{
  "version": "2.0",
  "description": "Skill activation triggers with path patterns, keywords, and intent matching",
  "skills": {
    "fastapi-developing": {
      "type": "domain",
      "enforcement": "suggest",
      "priority": "high",
      "promptTriggers": {
        "keywords": [
          "fastapi",
          "api",
          "endpoint",
          "route",
          "router",
          "pydantic",
          "basemodel",
          "lifespan",
          "depends",
          "async",
          "await",
          "uvicorn",
          "backend"
        ],
        "intentPatterns": [
          "(create|add|implement|build).*?(route|endpoint|api|controller)",
          "(fix|debug|handle).*?(error|exception|backend)",
          "(how to|best practice).*?(fastapi|pydantic|async)"
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
          "@router\\.(get|post|put|delete|patch)"
        ]
      }
    },
    "nextjs-building": {
      "type": "domain",
      "enforcement": "suggest",
      "priority": "high",
      "promptTriggers": {
        "keywords": [
          "nextjs",
          "next.js",
          "next",
          "app router",
          "server component",
          "client component",
          "use server",
          "use client",
          "server action",
          "page",
          "layout",
          "loading",
          "error",
          "turbopack"
        ],
        "intentPatterns": [
          "(create|add|build).*?(page|component|layout|route)",
          "(implement|add).*?(server action|api route)",
          "(fix|debug).*?(hydration|rendering|ssr)"
        ]
      },
      "fileTriggers": {
        "pathPatterns": [
          "app/**/*.tsx",
          "app/**/*.ts",
          "components/**/*.tsx",
          "next.config.*"
        ],
        "contentPatterns": [
          "'use client'",
          "'use server'",
          "import.*from 'next",
          "export default function.*Page"
        ]
      }
    },
    "go-developing": {
      "type": "domain",
      "enforcement": "suggest",
      "priority": "high",
      "promptTriggers": {
        "keywords": [
          "go",
          "golang",
          "handler",
          "middleware",
          "goroutine",
          "channel",
          "interface",
          "struct",
          "gin",
          "echo",
          "fiber",
          "slog",
          "http.ServeMux"
        ],
        "intentPatterns": [
          "(create|add|implement).*?(handler|endpoint|middleware|service)",
          "(fix|debug|handle).*?(error|panic|goroutine)",
          "(how to|best practice).*?(go|golang|concurrency)"
        ]
      },
      "fileTriggers": {
        "pathPatterns": [
          "**/*.go",
          "cmd/**/*.go",
          "internal/**/*.go",
          "pkg/**/*.go"
        ],
        "contentPatterns": [
          "package main",
          "func \\(",
          "http\\.Handle",
          "slog\\."
        ]
      }
    }
  }
}
```

### Enforcement Levels

| Level     | Behavior                                              |
| --------- | ----------------------------------------------------- |
| `suggest` | Skill appears as recommendation                       |
| `warn`    | Yellow warning if skill not used                      |
| `block`   | Must acknowledge skill before proceeding (guardrails) |

---

## Complete Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     /skill:generate                             │
├─────────────────────────────────────────────────────────────────┤
│ Phase 1: SCAN                                                   │
│ ├── Find package files (package.json, pyproject.toml, go.mod)  │
│ ├── Check for pnpm-lock.yaml (preferred package manager)       │
│ ├── Extract technology names                                    │
│ └── Extract version numbers                                     │
├─────────────────────────────────────────────────────────────────┤
│ Phase 2: RESEARCH (Web Search)                                  │
│ ├── Search "[tech] best practices 2025"                         │
│ ├── Search "[tech] [version] patterns modern"                   │
│ ├── Search "[tech] anti-patterns mistakes"                      │
│ └── Extract: patterns, deprecations, tips                       │
├─────────────────────────────────────────────────────────────────┤
│ Phase 3: GENERATE                                               │
│ ├── Create skill folder with gerund name                        │
│ ├── Write SKILL.md (<500 lines, version-aware)                  │
│ ├── Create resources/ folder                                    │
│ │   ├── patterns.md (from research)                             │
│ │   ├── anti-patterns.md (from research)                        │
│ │   └── examples.md (from research)                             │
│ └── Write keyword-rich description                              │
├─────────────────────────────────────────────────────────────────┤
│ Phase 4: CONFIGURE                                              │
│ ├── Update skill-rules.json with rich triggers                  │
│ ├── Add pathPatterns based on project structure                 │
│ ├── Add intentPatterns for common requests                      │
│ └── Set appropriate enforcement level                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Example Output

```
🔍 Scanning project...

📦 Detected Technologies:
├── Python 3.13 + FastAPI 0.115.0 + Pydantic 2.9.0 (from pyproject.toml)
├── Python Package Manager: uv (uv.lock found)
├── TypeScript 5.7 (from tsconfig.json)
├── Next.js 16.0.0 + React 19.2.0 (from package.json)
├── Tailwind CSS 4.0.0 (from package.json)
├── tRPC 11.0.0 (from package.json)
├── Drizzle ORM 0.38.0 (from package.json)
└── Node Package Manager: pnpm 10.x

🌐 Researching best practices (December 2025)...
├── Searching "FastAPI best practices 2025 Pydantic v2 lifespan"
├── Searching "Next.js 16 App Router cache components 2025"
├── Searching "React 19 use hook server actions patterns"
├── Searching "Tailwind CSS v4 @theme CSS-first"
├── Searching "tRPC v11 TanStack React Query integration"
└── Searching "Drizzle ORM identity columns TypeScript"

📝 Generating skills...

Created:
├── .claude/skills/fastapi-developing/
│   ├── SKILL.md (478 lines)
│   └── resources/
│       ├── patterns.md (lifespan, Pydantic v2, Annotated)
│       ├── anti-patterns.md (deprecated patterns to avoid)
│       └── examples.md (working code)
├── .claude/skills/nextjs-building/
│   ├── SKILL.md (445 lines)
│   └── resources/
│       ├── app-router.md (layouts, pages, loading)
│       ├── server-components.md (RSC patterns)
│       ├── cache-components.md (use cache, new in 16)
│       └── server-actions.md
├── .claude/skills/react-developing/
│   ├── SKILL.md (389 lines)
│   └── resources/
│       ├── hooks-19.md (use, useActionState, useOptimistic)
│       └── patterns.md
├── .claude/skills/tailwind-styling/
│   ├── SKILL.md (312 lines)
│   └── resources/
│       └── v4-patterns.md (@theme, @layer, dynamic values)
├── .claude/skills/trpc-developing/
│   ├── SKILL.md (356 lines)
│   └── resources/
│       └── v11-patterns.md (queryOptions, RSC prefetch)
└── .claude/skills/drizzle-querying/
    ├── SKILL.md (298 lines)
    └── resources/
        └── patterns.md (identity columns, schema org)

⚙️ Updated .claude/skills/skill-rules.json with 6 skill triggers

✅ Skills generated with December 2025 best practices!

💡 Next steps:
   1. Review generated skills for accuracy
   2. Run /skill:improve to further optimize descriptions
   3. Test by editing relevant files
```

---

## Quick Reference: Current vs Deprecated Patterns

### Python/FastAPI

| Pattern          | ✅ Current (2025)                | ❌ Deprecated            |
| ---------------- | -------------------------------- | ------------------------ |
| Startup/shutdown | `lifespan` context manager       | `@app.on_event()`        |
| Pydantic config  | `model_config = ConfigDict(...)` | `class Config:`          |
| Parse dict       | `Model.model_validate(data)`     | `Model.parse_obj(data)`  |
| To dict          | `model.model_dump()`             | `model.dict()`           |
| Type unions      | `str \| None`                    | `Optional[str]`          |
| Dependencies     | `Annotated[T, Depends(fn)]`      | `param: T = Depends(fn)` |

### Next.js 16 / React 19

| Pattern     | ✅ Current (2025)     | ❌ Deprecated                  |
| ----------- | --------------------- | ------------------------------ |
| Bundler     | Turbopack (default)   | Webpack                        |
| Caching     | `use cache` directive | Manual fetch caching           |
| Memoization | React Compiler (auto) | Manual `useMemo`/`useCallback` |
| Form state  | `useActionState`      | `useState` + manual handling   |
| Async data  | `use()` hook          | `useEffect` + loading states   |

### Tailwind CSS v4

| Pattern | ✅ Current (2025)       | ❌ Deprecated                         |
| ------- | ----------------------- | ------------------------------------- |
| Import  | `@import "tailwindcss"` | `@tailwind base/components/utilities` |
| Config  | `@theme` in CSS         | `tailwind.config.js`                  |
| Layers  | Native `@layer`         | Plugin-based layers                   |

### tRPC v11

| Pattern     | ✅ Current (2025)                               | ❌ Deprecated       |
| ----------- | ----------------------------------------------- | ------------------- |
| React Query | `useTRPC()` + `useQuery(trpc.x.queryOptions())` | `trpc.x.useQuery()` |
| Prefetching | `createTRPCOptionsProxy`                        | Manual dehydration  |

---

## Notes

- **Always search** - Don't rely on outdated training data
- **Version-specific** - Research the exact version detected
- **Progressive disclosure** - Keep SKILL.md under 500 lines
- **Rich triggers** - Use pathPatterns + keywords + intentPatterns
- **Gerund naming** - Skill names should be activities (verb + -ing)
- **Package managers**:
  - Python: **uv** (`uv init`, `uv add`, `uv run`, `uv.lock`)
  - Node.js: **pnpm 10.x** (`pnpm add`, `workspace:*`, `catalog:`)
- **Type safety (always run before committing):**
  - Python: `uv run mypy --strict .` + `uv run ruff check .`
  - TypeScript: `npx tsc --noEmit`
