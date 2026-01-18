---
description: Mark current task as done and show next task
---

# Mark Task Complete

Mark the current task as done and show what's next.

## What This Command Does

1. Reads `.claude/docs/feature.md`
2. Finds the task marked 🔄 In Progress
3. Verifies acceptance criteria are met
4. Marks it ✅ Done
5. Shows next task (or celebrates if all done!)

## Before Marking Complete

Verify:
- [ ] All acceptance criteria checked off?
- [ ] Code works / tests pass?
- [ ] No obvious issues?

If NOT ready, say what's still needed.

## Update feature.md

Change:
```markdown
- [ ] TASK-3: [Title]
```

To:
```markdown
- [x] TASK-3: [Title] ✅
```

And update the task section:
```markdown
### TASK-3: [Title] ✅ Done
```

## Output

### If More Tasks Remain
```
✅ Completed: TASK-3 - [Title]

Progress: [████████░░] 75% (3/4 tasks)

Next up: TASK-4 - [Title]
Run /feature:next to continue
```

### If All Tasks Complete
```
🎉 Feature Complete!

Completed:
✅ TASK-1: [Title]
✅ TASK-2: [Title]
✅ TASK-3: [Title]
✅ TASK-4: [Title]

Summary: [What was built]
```

Also update feature.md status:
```markdown
> Status: Planning → Ready → In Progress → **Done**
```

## If No Task In Progress

```
No task currently in progress.
Run /feature:next to start the next task.
```
