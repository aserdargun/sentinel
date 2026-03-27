---
name: write-config
description: Creates YAML configuration files following base.yaml inheritance pattern
model: haiku
effort: medium
maxTurns: 10
permissionMode: acceptEdits
tools:
  - Read
  - Write
  - Glob
  - Grep
---

# Write Configuration File

You are creating a YAML configuration file for the Sentinel anomaly detection platform.

## Before You Start

1. Read `configs/base.yaml` to understand the base configuration and defaults
2. Read the model implementation file to understand what parameters it expects
3. Read existing config files in `configs/` to match the pattern

## Base Config Defaults (from PLAN.md)

```yaml
# configs/base.yaml
data:
  path: "data/raw"
  processed_path: "data/processed"
  metadata_file: "data/datasets.json"
split:
  train: 0.70
  val: 0.15
  test: 0.15
seed: 42
device: "auto"
training_mode: "normal_only"    # normal_only | all_data
scheduler:
  type: "reduce_on_plateau"
  patience: 5
  factor: 0.5
  min_lr: 1.0e-6
  warmup_epochs: 0
runtime:
  max_upload_size_mb: 100
  max_features: 500
  training_timeout_s: 3600
  api_request_timeout_s: 300
  max_dataset_rows_matrix_profile: 100000
  shap_max_features: 10
llm:
  model: "nvidia/nemotron-3-nano-4b"
  ollama_url: "http://localhost:11434"
  timeout_s: 30
logging:
  level: "INFO"
  format: "json"
```

## Config File Pattern

```yaml
# configs/{model_name}.yaml
inherits: base.yaml
model: {model_name}              # Model registry name (flat key, not nested)

# Model-specific hyperparameters (flat, at top level)
window_size: 30                  # Example for zscore
threshold_sigma: 3.0

# Override base settings only when needed
# e.g., epochs, learning_rate, batch_size for deep models
```

## Rules

- ALWAYS start with `inherits: base.yaml`
- Document every parameter with a YAML comment
- Use descriptive parameter names (not abbreviations)
- Include the model name as flat key: `model: {name}` (not nested `model.name`)
- Only override base settings when the model needs different defaults
- Group related parameters under logical sections
- Use realistic default values from the model's paper or common practice

## After Creation

Verify the YAML is valid:
- No tabs (spaces only)
- Proper indentation (2 spaces)
- Quoted strings that could be misinterpreted (e.g., `"true"`, `"null"`)
