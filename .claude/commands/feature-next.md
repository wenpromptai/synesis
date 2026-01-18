---
description: Show status and start the next task from feature.md
---

# What's Next?

Show current status and start the next task.

## What This Command Does

1. Reads `.claude/docs/feature.md`
2. Reads `.claude/docs/acontext.json` (if exists) for learned preferences
3. Finds the next incomplete task
4. Shows status overview
5. Starts working on that task

## Context Loading (Lightweight)

Before starting work, quickly scan:

```
1. .claude/docs/acontext.json → Check "corrections" array ONLY
   - These are explicit user rules (highest priority)
   - Example: "Always use httpx not requests"
   - Apply these during implementation

2. Don't re-read product.md/tech-stack.md here
   - Already incorporated into feature.md during /feature:plan
   - Saves context tokens
```

## Logic

```
IF no feature.md exists:
  → "No feature found. Run /feature:plan [what to build] first"

IF a task is marked 🔄 In Progress:
  → Show that task, continue working on it

IF all tasks complete:
  → "🎉 All done! [summary of what was built]"

ELSE:
  → Find first unchecked task
  → Mark it 🔄 In Progress in feature.md
  → Start working on it
```

## Output

### Status Header
```
📋 Feature: [Feature Name]
━━━━━━━━━━━━━━━━━━━━━━━
✅ Done:        2 tasks
🔄 Current:     TASK-3
⬚ Remaining:   1 task
━━━━━━━━━━━━━━━━━━━━━━━
Progress: [████████░░] 67%
```

### Current Task
```
## Now: TASK-3 - [Title]

What to do:
- [ ] [Step 1]
- [ ] [Step 2]

Files to modify:
- `path/to/file.ts`
```

Then **start implementing** the task.

## When Starting a Task

1. Update feature.md to mark task as 🔄 In Progress
2. Read any relevant existing code
3. Implement step by step
4. Test as you go

## When Stuck

If blocked or confused:
- Ask for clarification
- Don't guess
- Suggest breaking the task down further
