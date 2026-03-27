---
name: run
description: Run anomaly detection on data with a trained model
user-invocable: true
argument-hint: "[model-name] [data-path] (e.g., run zscore data/raw/synthetic.parquet)"
allowed-tools: Bash, Read, Glob
context: fork
---

# /run $ARGUMENTS

Run anomaly detection with a trained model.

## Instructions

1. Parse arguments:
   - `$ARGUMENTS[0]` = model name or path to saved model
   - `$ARGUMENTS[1]` = data path (optional — uses latest dataset if omitted)

2. Check if the detect CLI command exists:
   ```bash
   uv run sentinel detect --help 2>&1
   ```

3. Find the latest trained model if a name is given:
   ```bash
   ls -t data/experiments/*/config.json 2>/dev/null | head -5
   ```

4. Run detection:
   ```bash
   uv run sentinel detect --model {model_path} --data {data_path}
   ```

5. Show results: score summary (min, max, mean, std), anomaly count, threshold used
