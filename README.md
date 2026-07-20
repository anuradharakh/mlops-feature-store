cat > README.md <<'EOF'
# ADSP 31021 Assignment 2 — Feature Store

## Project Overview

This repository implements a reproducible machine-learning workflow using:

- Feast for feature management
- MLflow for experiment tracking
- DVC for data and artifact versioning
- Scikit-learn for model development
- Git and GitHub for source control

The project predicts athlete `total_lift` and compares:

- Two feature versions
- Two hyperparameter configurations
- One algorithm
- Four total experiments

## Architecture

```text
athletes.csv
    |
    v
Ingestion and Validation
    |
    v
Preprocessing and Label Creation
    |
    +--------------------+
    |                    |
    v                    v
Feature Version 1    Feature Version 2
    |                    |
    +---------+----------+
              |
              v
        Feast Feature Store
              |
              v
      MLflow Experiments
              |
              v
      Evaluation and Reporting