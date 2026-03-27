---
name: model-reference
description: Loads BaseAnomalyDetector ABC, registry, types, and existing model implementations as reference
user-invocable: false
paths:
  - src/sentinel/core/**
  - src/sentinel/models/**
  - configs/**
---

# Model Reference Loader

Load all reference material needed to implement a new anomaly detection model.

## Available Core Files
!`ls src/sentinel/core/*.py 2>/dev/null || echo "Core not implemented yet — run /phase 1"`

## Existing Models
!`ls src/sentinel/models/statistical/*.py src/sentinel/models/deep/*.py src/sentinel/models/ensemble/*.py 2>/dev/null || echo "No models implemented yet"`

## Existing Configs
!`ls configs/*.yaml 2>/dev/null || echo "No configs yet"`

## Instructions

1. Read the core abstractions that every model must follow:
   - `src/sentinel/core/base_model.py` — BaseAnomalyDetector ABC (the interface to implement)
   - `src/sentinel/core/registry.py` — @register_model decorator and model lookup
   - `src/sentinel/core/types.py` — type aliases and common types
   - `src/sentinel/core/exceptions.py` — error hierarchy

2. Find and read existing model implementations for reference patterns:
   - Search `src/sentinel/models/statistical/` for statistical model examples
   - Search `src/sentinel/models/deep/` for deep model examples
   - Search `src/sentinel/models/ensemble/` for ensemble examples
   - Read at least one existing model from the same category as the target model

3. Read the base config to understand config inheritance:
   - `configs/base.yaml`
   - Any existing model config in `configs/` for reference

4. Report what was found:
   - List the abstract methods that must be implemented
   - Show the `@register_model` pattern
   - Show a reference implementation from the same category (if one exists)
   - Show the config inheritance pattern

## If Core Files Don't Exist Yet

If `base_model.py` or `registry.py` don't exist, report this clearly — the model cannot be implemented until Phase 1 is complete. Suggest running `/phase 1` to check Phase 1 status.

## Output Format

```markdown
## Model Reference

### Abstract Interface
{BaseAnomalyDetector methods and signatures}

### Registry Pattern
{@register_model usage example}

### Reference Implementation ({category})
{existing model code from same category, or "No existing models in this category"}

### Config Pattern
{base.yaml + example model config}

### Checklist
- [ ] Subclass BaseAnomalyDetector
- [ ] @register_model("{name}")
- [ ] Implement: fit(), score(), save(), load(), get_params()
- [ ] Type hints on all methods
- [ ] Google docstrings
- [ ] Config file with inherits: base.yaml
- [ ] Unit test
- [ ] __init__.py updated
```
