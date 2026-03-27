---
name: implement
description: Implement a Sentinel module by routing to the correct specialized agent
user-invocable: true
argument-hint: "module path (e.g., models/statistical/zscore, api/routes/train, core/base_model)"
allowed-tools: Agent, Read, Glob, Grep
context: fork
agent: general-purpose
---

# /implement $ARGUMENTS

Implement the specified module by routing to the correct specialized agent.

## Routing Rules

Examine `$ARGUMENTS` and dispatch to the appropriate agent:

1. **If path contains `models/`** → use `implement-model` agent
   - e.g., `models/statistical/zscore`, `models/deep/autoencoder`
   - Agent: `implement-model`

2. **If path contains `api/routes/`** → use `build-api-route` agent
   - e.g., `api/routes/train`, `api/routes/runs`
   - Agent: `build-api-route`

3. **If path contains `tests/`** → use `write-tests` agent
   - e.g., `tests/unit/models/test_zscore`, `tests/integration/test_pipeline`
   - Agent: `write-tests`

4. **If path contains `configs/`** → use `write-config` agent
   - e.g., `configs/zscore.yaml`, `configs/autoencoder.yaml`
   - Agent: `write-config`

5. **Everything else** → use `implement-module` agent
   - e.g., `core/base_model`, `data/loaders`, `training/trainer`, `cli/main`
   - Agent: `implement-module`

## Instructions

1. Parse `$ARGUMENTS` to determine the module path
2. Determine which agent to use based on routing rules above
3. Launch the appropriate agent with a clear prompt that includes:
   - The exact module to implement
   - The full path where the file should be created (under `src/sentinel/`)
   - Any relevant context from PLAN.md
4. Wait for the agent to complete
5. Report what was created

## Example Dispatches

- `/implement models/statistical/zscore` → `implement-model` agent: "Implement the Z-Score anomaly detector at src/sentinel/models/statistical/zscore.py"
- `/implement api/routes/train` → `build-api-route` agent: "Build the /api/v1/train route at src/sentinel/api/routes/train.py"
- `/implement core/base_model` → `implement-module` agent: "Implement BaseAnomalyDetector ABC at src/sentinel/core/base_model.py"
