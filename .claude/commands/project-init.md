---
description: Set up product.md and tech-stack.md for a new project
---

# Initialize Project Context

Set up the persistent project context that guides all development.

**Run this once when starting a new project or adopting this system.**

## What This Command Does

1. Analyzes the existing codebase (if any)
2. Asks clarifying questions about the project
3. Creates `.claude/docs/product.md` - mission, users, goals
4. Creates `.claude/docs/tech-stack.md` - technologies, conventions
5. Optionally runs `/skill:generate` to create project-specific skills

## Files Created

### `.claude/docs/product.md`
```markdown
# [Project Name]

## Mission
[One sentence: What does this do and for whom?]

## Target Users
- **[User Type 1]**: [Their goal and context]
- **[User Type 2]**: [Their goal and context]

## Key Features
1. [Feature 1] - [Why it matters]
2. [Feature 2] - [Why it matters]
3. [Feature 3] - [Why it matters]

## Non-Goals (Out of Scope)
- [Thing we're explicitly NOT building]
- [Another thing we're NOT building]

## Success Looks Like
- [Measurable outcome 1]
- [Measurable outcome 2]
```

### `.claude/docs/tech-stack.md`
```markdown
# Tech Stack

## Languages & Frameworks
| Layer | Technology | Version | Notes |
|-------|------------|---------|-------|
| Frontend | [Framework] | [Version] | [Key plugins/libs] |
| Backend | [Framework] | [Version] | [Key plugins/libs] |
| Database | [Type] | [Version] | [ORM if any] |
| Infrastructure | [Provider] | | [Key services] |

## Development Tools
- **Package Manager**: uv (Python) / pnpm (Node.js)
- **Linting**: ruff (Python) / ESLint (TypeScript)
- **Type Checking**: mypy --strict (Python) / tsc (TypeScript)
- **Testing**: pytest (Python) / vitest (TypeScript)

## Code Conventions

### Naming
- Files: `kebab-case.ts`
- Components: `PascalCase`
- Functions: `camelCase`
- Database: `snake_case`

### Structure
```
src/
├── [folder]/ - [purpose]
├── [folder]/ - [purpose]
└── [folder]/ - [purpose]
```

## Key Patterns

### [Pattern Name]
[Brief description + example]

### [Pattern Name]
[Brief description + example]

## Avoid These
- ❌ [Anti-pattern] - [Why]
- ❌ [Anti-pattern] - [Why]
```

## Process

### For New Projects
1. Ask: "What are you building and who is it for?"
2. Ask: "What tech stack are you using or planning to use?"
3. Create both files based on answers
4. Suggest running `/skill:generate`

### For Existing Projects
1. Scan codebase for tech stack (package.json, requirements.txt, etc.)
2. Pre-fill tech-stack.md with detected technologies
3. Ask clarifying questions about mission/users
4. Create product.md based on answers
5. Run `/skill:generate` automatically

## After Initialization

Tell the user:
- What files were created
- What was detected vs. asked
- Next steps:
  - "Run `/skill:generate` to create technology-specific skills"
  - "Run `/feature:plan [feature]` when ready to build something"

## When To Re-run

This command should be run again when:
- Major tech stack changes (switching frameworks)
- Project direction changes significantly
- Starting fresh after long break

For small updates, just edit the files directly.
