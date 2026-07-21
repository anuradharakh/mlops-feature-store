# ADSP 31021 Assignment 2 — Phase-Wise Rollout

## Project Overview

**Repository:** `mlops-feature-store`

**Objective:** Build a reproducible machine-learning pipeline using Feast for feature
management, MLflow for experiment tracking, DVC for data versioning, and Scikit-learn
for model development.

---

## Assignment Requirement Mapping

| Requirement | Implementation |
|---|---|
| Dataset selection | Provided `athletes.csv` dataset |
| MLOps platform | MLflow |
| End-to-end pipeline | Modular Python pipeline |
| Feature store | Feast |
| Feature versioning | Feature Version 1 and Feature Version 2 |
| Experimentation | Four Random Forest experiments |
| Reproducibility | Git, DVC, requirements, configuration, and fixed seed |
| Documentation | README and phase-wise HTML report |

---

## Phase 1 — Repository and Environment Setup

### Objective

Create a modular VS Code repository with an isolated Python environment and
reproducible dependency management.

### Completed work

- Created the Git repository.
- Created a Python virtual environment.
- Added dependency management.
- Configured Ruff and Pytest.
- Initialized DVC.
- Created the modular source-code structure.

### Status

Completed.

---

## Phase 2 — Dataset Ingestion and Profiling

### Objective

Create an immutable, versioned source dataset and generate reproducible
raw-data validation and profiling artifacts.

### Implementation

- The original athlete ZIP archive is stored under `data/source/`.
- DVC tracks the archive without committing the large file to Git.
- The ingestion module extracts the raw CSV without modifying its contents.
- Required columns are validated automatically.
- Profiling generates schema, missing-value, numerical, and categorical reports.

### Generated artifacts

- `reports/validation/raw_profile.json`
- `reports/validation/raw_schema.csv`
- `reports/validation/raw_missing_values.csv`
- `reports/validation/raw_numeric_summary.csv`
- `reports/validation/raw_categorical_summary.csv`
- `reports/figures/raw_missing_values.png`

### Data-handling assumptions

- The source archive contains exactly one CSV file.
- The source archive is treated as immutable.
- No cleaning or feature engineering occurs during ingestion.
- `athlete_id` is retained as the candidate Feast entity key.
- Data-quality issues will be handled explicitly in Phase 3.

### Validation

- The ingestion process completed successfully.
- Required columns were validated.
- Automated ingestion tests passed.
- Ruff formatting and lint checks passed.
- DVC reports that the source dataset is up to date.

### Status

Completed.

---

## Phase 3 — Preprocessing and Label Construction

To be completed.

---

## Phase 4 — Feature Version 1

To be completed.

---

## Phase 5 — Feature Version 2

To be completed.

---

## Phase 6 — Feast Integration

To be completed.

---

## Phase 7 — Feature Retrieval

To be completed.

---

## Phase 8 — Scikit-learn Model Pipeline

To be completed.

---

## Phase 9 — MLflow Experiment Tracking

To be completed.

---

## Phase 10 — Four Required Experiments

To be completed.

---

## Phase 11 — Model Evaluation and Comparison

To be completed.

---

## Phase 12 — Automated Pipeline

To be completed.

---

## Phase 13 — Testing and Code Quality

To be completed.

---

## Final Recommendation

To be completed after experiment evaluation.
