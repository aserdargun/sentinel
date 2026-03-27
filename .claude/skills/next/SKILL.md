---
name: next
description: Show what to implement next based on current progress
user-invocable: true
allowed-tools: Glob, Grep, Read, Bash
context: fork
agent: Explore
---

# /next

Determine what should be implemented next in the Sentinel project.

## Current State
!`for pkg in core data models/statistical models/deep models/ensemble training tracking viz streaming explain api cli mcp plugins; do count=$(ls src/sentinel/$pkg/*.py 2>/dev/null | wc -l); if [ "$count" -gt 0 ]; then echo "  $pkg: $count files"; fi; done; echo "---"; echo "Models registered:"; grep -r "@register_model" src/sentinel/ 2>/dev/null | wc -l | tr -d ' '`

## Instructions

1. Using the pre-scanned state above, determine which phase the project is currently in
2. Read PLAN.md to find the NEXT incomplete phase
3. Within that phase, identify which files are missing (use Glob to check existence)
4. Prioritize by dependency order — files that other files depend on come first
5. Suggest the single most impactful next step

## Output Format

```
## Next Up

**Phase {N}: {Name}** — {X}% complete

**Implement next:** `{file_path}`
**Why:** {brief reason — dependency order, blocker for other work, etc.}
**Command:** `/implement {module_path}`

**Remaining in this phase:**
- {file_1} — {status}
- {file_2} — {status}
```
