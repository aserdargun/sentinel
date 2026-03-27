---
name: gen
description: Generate synthetic test data with configurable parameters
user-invocable: true
argument-hint: "[features] [length] [anomaly-ratio] (e.g., gen 5 10000 0.05)"
allowed-tools: Bash, Read
context: fork
---

# /gen $ARGUMENTS

Generate synthetic multivariate time series data for testing.

## Instructions

1. Parse arguments (all optional, with defaults):
   - `$ARGUMENTS[0]` = number of features (default: 5)
   - `$ARGUMENTS[1]` = number of rows (default: 10000)
   - `$ARGUMENTS[2]` = anomaly ratio (default: 0.05)

2. Check if the `sentinel generate` CLI command exists:
   ```bash
   uv run sentinel generate --help 2>&1
   ```

3. If CLI exists, run:
   ```bash
   uv run sentinel generate --features {N} --length {L} --anomaly-ratio {R} --seed 42 --output data/raw/synthetic.parquet
   ```

4. If CLI doesn't exist yet, report:
   ```
   CLI not implemented yet. To generate data manually:
   uv run python -c "from sentinel.data.synthetic import generate_synthetic; ..."
   ```

5. Show output summary: shape, feature names, anomaly count, file path
