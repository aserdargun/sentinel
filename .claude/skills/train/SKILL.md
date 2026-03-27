---
name: train
description: Run model training with a config file
user-invocable: true
argument-hint: "config path or model name (e.g., configs/zscore.yaml or zscore)"
allowed-tools: Bash, Read, Glob
context: fork
---

# /train $ARGUMENTS

## Available Configs
!`ls configs/*.yaml 2>/dev/null || echo "No configs yet"`

Run Sentinel model training.

## Instructions

1. Determine the config file:
   - If `$ARGUMENTS` is provided, use it as the config path
   - If `$ARGUMENTS` is just a model name (e.g., "zscore"), expand to `configs/zscore.yaml`
   - If no argument, use `configs/base.yaml`

2. Verify the config file exists

3. Run training:
   ```bash
   uv run sentinel train --config {config_path}
   ```

4. After training completes, show:
   - Training duration
   - Final metrics (from stdout or experiment artifacts)
   - Path to saved artifacts
   - Any warnings or errors

5. If training fails, show the error and suggest fixes:
   - Missing dependencies → `uv add {package}`
   - Config errors → show which field is invalid
   - Data errors → suggest running data validation first
