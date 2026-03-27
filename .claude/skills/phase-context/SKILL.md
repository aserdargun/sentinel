---
name: phase-context
description: Loads implementation phase details from PLAN.md, checks file existence, and identifies dependencies
user-invocable: false
paths:
  - PLAN.md
  - src/sentinel/**
---

# Phase Context Loader

Load the context for a specific implementation phase from PLAN.md.

## Current Source Files
!`find src/sentinel -name "*.py" -not -path "*__pycache__*" 2>/dev/null | sort || echo "No source files yet"`

## Registered Models
!`grep -r "@register_model" src/sentinel/ 2>/dev/null | sed 's/.*@register_model("\(.*\)").*/\1/' | sort || echo "No models registered yet"`

## Instructions

1. Read `PLAN.md` and extract the details for the requested phase number
2. For each file listed in that phase, check if it already exists using Glob
3. Identify dependencies on files from earlier phases (do they exist?)
4. Report:
   - **Phase goal**: what this phase accomplishes
   - **Files to create**: list with existence status (EXISTS / MISSING)
   - **Dependencies**: files from earlier phases that must exist first
   - **Verification criteria**: how to confirm this phase is complete (from PLAN.md Verification Strategy)

## Phase Overview (for quick reference)

| Phase | Name | Key Deliverables |
|-------|------|-----------------|
| 1 | Foundation | pyproject.toml, core/*, data/synthetic+loaders+ingest+validators+preprocessors, models/statistical/zscore+iforest+matrix_profile, training/evaluator+thresholds, configs, unit tests |
| 2 | Training + Tracking + CLI + PI | trainer.py, callbacks.py, tracking/*, cli/*, data/features.py, data/pi_connector.py |
| 3 | Classical Deep Learning | deep/autoencoder, rnn, lstm, gru, lstm_ae, tcn + configs + tests |
| 4 | Generative / Advanced | deep/vae, gan, tadgan, tranad, deepar, diffusion + configs + tests |
| 5 | Ensemble + Thresholding + Drift | ensemble/hybrid.py, thresholds (extend), streaming/drift.py |
| 6 | Visualization + Interpretability | viz/*, explain/*, cli/visualize.py |
| 7 | Streaming + Alerts | streaming/*, data/streaming.py |
| 8 | FastAPI + Dashboard | api/*, dashboard/*, cli/serve.py |
| 9 | MCP Server + Ollama | mcp/*, cli/mcp_serve.py |
| 10 | Polish + Docs + Tests | plugins/loader.py, README.md, docs/*, integration+smoke tests |

## Output Format

```markdown
## Phase {N}: {Name}

### Goal
{description}

### Files
- [EXISTS] src/sentinel/core/base_model.py
- [MISSING] src/sentinel/core/registry.py
- ...

### Dependencies from Earlier Phases
- Phase 1: src/sentinel/core/base_model.py [EXISTS]
- ...

### Progress: {existing}/{total} files ({percentage}%)

### Verification
{criteria from PLAN.md}
```
