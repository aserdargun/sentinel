---
name: phase
description: Show implementation phase details and progress
user-invocable: true
argument-hint: "phase number (1-10)"
allowed-tools: Read, Glob, Grep
context: fork
agent: Explore
---

# /phase $ARGUMENTS

## Current Source Files
!`find src/sentinel -name "*.py" -not -path "*__pycache__*" 2>/dev/null | sort || echo "No source files yet"`

## Current Configs
!`ls configs/*.yaml 2>/dev/null || echo "No configs yet"`

Show the details and progress for Phase $ARGUMENTS of the Sentinel implementation plan.

## Instructions

1. Read `PLAN.md` and find the section for Phase $ARGUMENTS
2. Extract:
   - Phase name and goal
   - All files that should be created in this phase
   - Verification criteria
3. For each file, check if it exists using Glob
4. Calculate progress percentage
5. Check if dependencies from earlier phases exist

## Output Format

```
## Phase {N}: {Name}

**Goal:** {description}

### Files
✅ src/sentinel/core/base_model.py
❌ src/sentinel/core/registry.py
...

### Progress: {done}/{total} ({percentage}%)

### Dependencies
✅ Phase 1 core abstractions
❌ Phase 2 data pipeline

### Verification Criteria
- {criterion 1}
- {criterion 2}

### Next Steps
- {what to implement next}
```
