---
name: status
description: Show full Sentinel implementation progress
user-invocable: true
allowed-tools: Read, Glob, Grep, Bash
context: fork
agent: Explore
---

# /status

## Pre-scanned State

### Source files by package:
!`for pkg in core data models/statistical models/deep models/ensemble training tracking viz streaming explain api cli mcp plugins; do echo "=== $pkg ==="; ls src/sentinel/$pkg/*.py 2>/dev/null || echo "(empty)"; done`

### Registered models:
!`grep -r "@register_model" src/sentinel/ 2>/dev/null | sed 's/.*@register_model("\(.*\)").*/\1/' | sort || echo "None"`

### Test files:
!`find tests -name "test_*.py" 2>/dev/null | sort || echo "No tests yet"`

### Config files:
!`ls configs/*.yaml 2>/dev/null || echo "No configs yet"`

Show the full implementation progress of the Sentinel project.

## Instructions

1. Using the pre-scanned state above, group source files by package:
   - `core/` — base abstractions
   - `data/` — data pipeline
   - `models/statistical/` — statistical models
   - `models/deep/` — deep learning models
   - `models/ensemble/` — ensemble models
   - `training/` — training engine
   - `api/` — REST API
   - `cli/` — CLI commands
   - `tracking/` — experiment tracking
   - `viz/` — plotting
   - `streaming/` — online detection
   - `explain/` — explainability
   - `mcp/` — MCP server

2. Count config files, test files, and registered models

3. Calculate phase completion (which phases have all their files)

## Output Format

```
# Sentinel Implementation Status

## Source Files
### core/ (X/Y files)
  ✅ base_model.py
  ❌ registry.py
  ...

### models/statistical/ (X/Y models)
  ✅ zscore.py
  ...

[... repeat for each package ...]

## Configs (X files)
  ✅ base.yaml
  ...

## Tests
  Unit: X files
  Integration: X files
  Smoke: X files

## Registered Models: X/16
  [list of registered model names]

## Phase Completion
  ✅ Phase 1: Foundation
  🔄 Phase 2: Training + Tracking + CLI + PI (3/5 files)
  ❌ Phase 3: Classical Deep Learning
  ...

## Overall: X% complete
```
