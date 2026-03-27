---
name: new-model
description: Create a new anomaly detection model with guided setup
user-invocable: true
argument-hint: "model name (e.g., zscore, autoencoder, hybrid_ensemble)"
allowed-tools: Agent, Read, Glob, Grep
context: fork
agent: general-purpose
paths:
  - src/sentinel/models/**
  - configs/**
  - tests/unit/models/**
  - PLAN.md
---

# /new-model $ARGUMENTS

Create a new anomaly detection model named `$ARGUMENTS`.

## Existing Models
!`ls src/sentinel/models/statistical/*.py src/sentinel/models/deep/*.py src/sentinel/models/ensemble/*.py 2>/dev/null || echo "No models implemented yet"`

## Instructions

1. Ask the user which category this model belongs to:
   - **statistical** — Traditional ML/statistical methods (Z-Score, Isolation Forest, Matrix Profile)
   - **deep** — Deep learning models requiring PyTorch (Autoencoder, RNN, LSTM, GRU, LSTM-AE, TCN, VAE, GAN, TadGAN, TranAD, DeepAR, Diffusion)
   - **ensemble** — Ensemble methods combining multiple models (Hybrid Ensemble)

2. Check if this model already exists:
   - Glob for `src/sentinel/models/*/$ARGUMENTS.py`
   - If it exists, report and ask if the user wants to overwrite

3. Read `PLAN.md` to find the specification for this model (Model Zoo table)

4. Launch the `implement-model` agent with full context:
   - Model name: `$ARGUMENTS`
   - Category: (from step 1)
   - Specification: (from PLAN.md)
   - Files to create:
     - `src/sentinel/models/{category}/$ARGUMENTS.py`
     - `configs/$ARGUMENTS.yaml`
     - `tests/unit/models/test_$ARGUMENTS.py`
     - Update `src/sentinel/models/{category}/__init__.py`

5. After agent completes, report:
   - Files created
   - Test results
   - Any issues
