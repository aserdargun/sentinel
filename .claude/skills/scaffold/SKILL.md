---
name: scaffold
description: Create directory structure and __init__.py files for a phase
user-invocable: true
argument-hint: "phase number (1-10)"
allowed-tools: Bash, Read, Write, Glob
context: fork
---

# /scaffold $ARGUMENTS

Create the directory structure and empty `__init__.py` files for Phase $ARGUMENTS.

## Phase Directory Mapping

| Phase | Directories to create |
|-------|----------------------|
| 1 | src/sentinel/core/, src/sentinel/data/, src/sentinel/models/statistical/, src/sentinel/training/, configs/, tests/unit/, tests/conftest.py |
| 2 | src/sentinel/tracking/, src/sentinel/cli/, tests/integration/ |
| 3 | src/sentinel/models/deep/ |
| 4 | (no new dirs — uses models/deep/ from Phase 3) |
| 5 | src/sentinel/models/ensemble/, src/sentinel/streaming/ (drift.py only) |
| 6 | src/sentinel/viz/, src/sentinel/explain/ |
| 7 | src/sentinel/streaming/ (full package) |
| 8 | src/sentinel/api/, src/sentinel/api/routes/, src/sentinel/api/dashboard/ |
| 9 | src/sentinel/mcp/ |
| 10 | src/sentinel/plugins/, docs/, tests/smoke/ |

## Instructions

1. Read PLAN.md to confirm the directories for Phase $ARGUMENTS
2. Create each directory with `mkdir -p`
3. Create `__init__.py` in each Python package directory (NOT in configs/, docs/, data/)
4. Do NOT create implementation files — just the structure
5. Report what was created

## Output Format

```
## Phase {N} Scaffold Created

Created directories:
  src/sentinel/core/
  src/sentinel/core/__init__.py
  ...

Ready for implementation. Run `/implement {first_module}` to start.
```
