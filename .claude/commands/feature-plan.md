---
description: Create PRD and tasks for a new feature. Run after /project:init.
---

# Plan a Feature

**Build**: $ARGUMENTS

## What This Command Does

1. **Read product context first** (REQUIRED):
   - `.claude/docs/product.md` → Mission, users, goals, non-goals
   - `.claude/docs/tech-stack.md` → Stack, conventions, patterns to follow
2. Creates `.claude/docs/` if it doesn't exist
3. Analyzes the codebase to understand context
4. Cross-references feature against product goals (reject if misaligned)
5. Asks clarifying questions if needed
6. Creates a PRD with user stories
7. Generates implementation tasks
8. Saves everything to `.claude/docs/feature.md`

## IMPORTANT: Context Loading

Before planning ANY feature, you MUST:

```
1. Read .claude/docs/product.md
   - Check: Does this feature align with the mission?
   - Check: Does it serve the target users?
   - Check: Is it in the "Non-Goals" section? (If yes, REJECT)

2. Read .claude/docs/tech-stack.md
   - Use the specified stack (don't introduce new frameworks)
   - Follow the code conventions listed
   - Apply the patterns, avoid the anti-patterns

3. Check .claude/docs/acontext.json (if exists)
   - Apply any corrections (highest priority rules)
   - Avoid patterns that caused losses
   - Repeat patterns that caused wins
```

If product.md doesn't exist, tell user to run `/project:init` first.

## Output Format

Create `.claude/docs/feature.md`:

```markdown
# Feature: [Feature Name]

> Created: [Date]
> Status: Planning → **Ready** → In Progress → Done

## What We're Building

[2-3 sentence description based on $ARGUMENTS]

## User Stories

### 1. [Story Title]
**As a** [user], **I want** [capability], **so that** [benefit]
- [ ] [Acceptance criterion]
- [ ] [Acceptance criterion]

### 2. [Story Title]
[Same format...]

## Tasks

### TASK-1: [Title] ⏱️ [estimate]
- [ ] [What to do]
- [ ] [What to do]
**Files**: `path/to/file.ts`

### TASK-2: [Title] ⏱️ [estimate]
[Same format...]

### TASK-3: [Title] ⏱️ [estimate]
[Same format...]

## Progress

- [x] Planning complete
- [ ] TASK-1: [Title]
- [ ] TASK-2: [Title]
- [ ] TASK-3: [Title]

---
*Run `/feature:next` to start working*
```

## Guidelines

- **Ask first**: If $ARGUMENTS is vague, ask 1-2 clarifying questions
- **Right-size tasks**: Each task should be 1-4 hours
- **Be specific**: "Add login form" not "implement auth"
- **Include files**: Which files will be created/modified

## After Creating

Tell the user:
- Summary of what was planned
- How many tasks
- "Run `/feature:next` to start the first task"
