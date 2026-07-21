# Assignment 2 Rollout Report — Feature Store and Experiment Tracking

## Executive Summary

This project implements a reproducible MLOps workflow for predicting athlete
`total_lift` using the provided athlete dataset.

The completed solution includes:

- DVC-based source-data versioning
- Modular ingestion and preprocessing
- Two managed feature versions
- Feast historical and online feature retrieval
- One persisted train/test split shared by all experiments
- A reusable Scikit-learn preprocessing and regression pipeline
- Four manually configured Random Forest experiments
- MLflow experiment tracking
- Model comparison and final recommendation
- Automated pipeline orchestration
- Testing, code-quality checks, and submission auditing

The best-performing model was `v2_hp1`, with:

- Test RMSE: `165.1166`
- Test MAE: `126.9015`
- Test R²: `0.6484`

The enhanced feature set improved generalization, while the larger HP2 model
configurations showed clear overfitting.

---

## Project Objective

The objective is to build a production-style machine-learning workflow that predicts an
athlete's combined lift total while demonstrating feature versioning and experiment
tracking.

The target is calculated as:

```text
total_lift = deadlift + candj + snatch + backsq
```

The four component columns are excluded from model features because they directly define
the target.

---

## Solution Architecture

```text
athletes.zip
    |
    v
DVC-tracked source data
    |
    v
Ingestion and schema validation
    |
    v
Preprocessing and target construction
    |
    +-------------------------+
    |                         |
    v                         v
Feature Version 1     Feature Version 2
    |                         |
    +------------+------------+
                 |
                 v
          Feast feature store
   historical + online retrieval
                 |
                 v
       Persisted train/test split
                 |
                 v
   Scikit-learn preprocessing pipeline
                 |
                 v
 Four Random Forest experiment combinations
                 |
                 v
       MLflow tracking and comparison
                 |
                 v
  Reports, HTML documentation, and audit
```

---

## Phase Status Overview

| Phase | Description | Status |
|---:|---|---|
| 1 | Repository and environment foundation | Completed |
| 2 | Source-data versioning and ingestion | Completed |
| 3 | Preprocessing and target construction | Completed |
| 4 | Feature Version 1 | Completed |
| 5 | Feature Version 2 | Completed |
| 6 | Feast feature store | Completed |
| 7 | Reproducible training split | Completed |
| 8 | Scikit-learn model pipeline | Completed |
| 9 | MLflow experiments | Completed and validated |
| 10 | Automation and submission audit | Implemented; final clean-clone validation pending |
| 11 | Model evaluation and comparison | Completed |
| 12 | Automated pipeline summary | Implemented |
| 13 | Testing and code quality | Implemented; final strict audit pending |

Phases 11–13 consolidate and document work implemented during earlier technical phases.
They are not additional model-training requirements.

---

## Phase 1 — Repository and Environment Foundation

### Objective

Create a modular, reproducible Python project rather than a notebook-only solution.

### Implementation

The repository uses:

- Python 3.11
- Editable package installation
- `src/` package layout
- YAML configuration
- Git version control
- DVC
- Pytest
- Ruff
- Make
- GitHub Actions

The repository separates:

- Configuration
- Source code
- Feature-store definitions
- Execution scripts
- Tests
- Data artifacts
- Reports
- Documentation

### Status

Completed.

---

## Phase 2 — Source-Data Versioning and Ingestion

### Objective

Track the original dataset and materialize a validated raw-data layer.

### Source data

```text
data/source/athletes.zip
```

The source ZIP is tracked using DVC metadata.

### Raw dataset profile

- Rows: `423,006`
- Columns: `27`

### Ingestion responsibilities

- Locate the source ZIP
- Extract the athlete CSV
- Validate that data is readable
- Persist a reproducible raw artifact
- Produce reviewer-facing ingestion evidence

### Status

Completed.

---

## Phase 3 — Preprocessing and Target Construction

### Objective

Create a model-ready population while preserving defensible data-cleaning decisions.

### Target construction

```text
total_lift = deadlift + candj + snatch + backsq
```

### Sentinel handling

The value `1` was explicitly treated as a sentinel in lift-component fields.

| Component | Sentinel replacements |
|---|---:|
| `deadlift` | 33 |
| `candj` | 35 |
| `snatch` | 33 |
| `backsq` | 34 |
| **Total** | **135** |

### Row counts

| Measure | Count |
|---|---:|
| Initial rows | 423,006 |
| Missing athlete IDs | 3 |
| Missing or invalid target rows | 341,296 |
| Final processed rows | 81,707 |

### Final target profile

- Minimum: `8`
- Maximum: `2,367`
- Mean: approximately `985.455`
- Median: `1,000`

### Lower-tail decision

A small number of very low target values remained after sentinel replacement. These rows
were retained because there was not enough documented evidence to classify them as
sentinels or corrupted values.

This avoids introducing an arbitrary lower cutoff solely to improve model metrics.

### Leakage prevention

The following target-component columns are excluded from the model matrix:

```text
deadlift
candj
snatch
backsq
```

### Status

Completed.

---

## Phase 4 — Feature Version 1

### Objective

Create a baseline feature set managed independently from the processed label data.

### Features

```text
age
weight
height
gender
region
```

### Characteristics

- Feature count: `5`
- Population: `81,707`
- Entity key: `athlete_id`
- Timestamp: `event_timestamp`
- No target or target-component columns

### Artifact

```text
data/features/v1/athlete_features_v1.parquet
```

### Status

Completed.

---

## Phase 5 — Feature Version 2

### Objective

Create an enhanced feature version while preserving the same entity and timestamp
population.

### Added features

#### BMI

```text
bmi = 703 × weight / height²
```

#### Squared age

```text
age_squared = age²
```

#### Weight-to-height ratio

```text
weight_height_ratio = weight / height
```

### Version comparison

| Version | Feature count | Population |
|---|---:|---:|
| v1 | 5 | 81,707 |
| v2 | 8 | 81,707 |

Version 2 is a strict extension of Version 1. No baseline feature was removed.

### Artifact

```text
data/features/v2/athlete_features_v2.parquet
```

### Status

Completed.

---

## Phase 6 — Feast Feature Store

### Objective

Register both feature versions and validate offline and online retrieval.

### Feast objects

Entity:

```text
athlete
```

Join key:

```text
athlete_id
```

Feature views:

```text
athlete_features_v1
athlete_features_v2
```

Feature services:

```text
athlete_strength_v1
athlete_strength_v2
```

### Historical retrieval

Feast point-in-time joins produced:

| Dataset | Rows |
|---|---:|
| Version 1 historical training table | 81,707 |
| Version 2 historical training table | 81,707 |

### Online retrieval

Both feature services were materialized to the local online store and queried for sample
athletes.

### Reviewer evidence

```text
reports/feast/feast_apply_output.txt
reports/feast/feature_registry_summary.json
reports/feast/historical_v1_sample.csv
reports/feast/historical_v2_sample.csv
reports/feast/online_retrieval_sample.json
reports/feast/feast_validation_summary.json
```

### Runtime state

Local registry and online-store SQLite files are ignored because they can be rebuilt from
the committed definitions and feature data.

### Status

Completed.

---

## Phase 7 — Reproducible Training Split

### Objective

Use one deterministic evaluation population across all four experiments.

### Configuration

- Test size: `0.20`
- Random state: `42`
- Shuffle: enabled
- Entity split key: `athlete_id`

### Split results

| Partition | Entities |
|---|---:|
| Train | 65,365 |
| Test | 16,342 |
| Total | 81,707 |

### Persisted artifact

```text
data/splits/athlete_split.parquet
```

### Rationale

The same membership is applied to both feature versions and both hyperparameter sets.
This prevents different split populations from confounding the model comparison.

### Evidence

```text
reports/validation/training_split_summary.json
reports/validation/training_dataset_missingness.csv
reports/figures/training_split_target_distribution.png
```

### Status

Completed.

---

## Phase 8 — Scikit-learn Model Pipeline

### Objective

Build one reusable model pipeline for both feature versions and both hyperparameter
configurations.

### Numerical preprocessing

- Median imputation

### Categorical preprocessing

- Most-frequent imputation
- One-hot encoding
- Unknown categories ignored

### Leakage protection

All preprocessing is fitted inside the Scikit-learn pipeline using training data only.

### Estimator

```text
RandomForestRegressor
```

### HP1

```yaml
n_estimators: 100
max_depth: 12
min_samples_split: 2
min_samples_leaf: 5
max_features: sqrt
bootstrap: true
```

### HP2

```yaml
n_estimators: 300
max_depth: 20
min_samples_split: 2
min_samples_leaf: 2
max_features: 1.0
bootstrap: true
```

No AutoML or automated hyperparameter tuning is used.

### Smoke test

A deterministic subset was used to validate:

- Data split loading
- Imputation
- Encoding
- Training
- Prediction
- Metric calculation
- Feature importance extraction
- Plot generation

The smoke test is not one of the four official experiments.

### Status

Completed.

---

## Phase 9 — MLflow Experiment Tracking

### Objective

Track the required two-by-two experiment matrix.

### Official runs

| Run | Feature version | Hyperparameter set |
|---|---|---|
| `v1_hp1` | v1 | hp1 |
| `v1_hp2` | v1 | hp2 |
| `v2_hp1` | v2 | hp1 |
| `v2_hp2` | v2 | hp2 |

### Tracking architecture

MLflow uses:

- SQLite for experiment and run metadata
- A local artifact directory for fitted models and run artifacts
- Explicit run names and tags
- Feature and split hashes
- Git metadata

### Logged data

Each run logs:

- Algorithm
- Feature version
- Feature names
- Hyperparameter configuration
- Train/test row counts
- Random seed
- Train RMSE, MAE, and R²
- Test RMSE, MAE, and R²
- Training and prediction time
- Generalization gaps
- Predictions
- Feature importance
- Diagnostic figures
- Model artifact
- Configuration and lineage files

### Status

Completed and validated.

---

## Phase 10 — Automation and Submission Audit

### Objective

Make the completed solution executable and reviewable with minimal manual work.

### Orchestrator

```text
scripts/run_pipeline.py
```

The orchestrator runs:

1. Ingestion
2. Preprocessing
3. Feature Version 1
4. Feature Version 2
5. Feast apply and retrieval
6. Training split creation
7. Model smoke test
8. Four MLflow experiments
9. HTML report generation
10. Submission audit
11. DVC status

Run:

```bash
python scripts/run_pipeline.py
```

Or:

```bash
make pipeline
```

### Submission audit

```text
scripts/verify_submission.py
```

The audit checks:

- Required source files
- Required generated artifacts
- Feature-version alignment
- Feature differences
- Leakage prevention
- Feast retrieval
- Split integrity
- Four-run experiment completion
- Metric completeness
- Best-run selection
- Git tracking
- Source-data portability

### Evidence

```text
reports/pipeline/pipeline_run.log
reports/pipeline/pipeline_run_summary.json
reports/submission/submission_audit.json
reports/submission/artifact_inventory.csv
```

### Validation status

The automation code is implemented. Before final submission, run the complete pipeline,
strict audit, and clean-clone reproduction test.

### Status

Implemented; final clean-clone validation pending.

---

## Phase 11 — Model Evaluation and Comparison

### Experiment results

| Rank | Run | Feature version | Hyperparameters | Train RMSE | Test RMSE | Test MAE | Test R² |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `v2_hp1` | v2 | hp1 | 158.847 | 165.117 | 126.901 | 0.6484 |
| 2 | `v1_hp1` | v1 | hp1 | 163.184 | 168.286 | 129.762 | 0.6348 |
| 3 | `v2_hp2` | v2 | hp2 | 127.456 | 169.293 | 130.229 | 0.6304 |
| 4 | `v1_hp2` | v1 | hp2 | 126.882 | 170.782 | 131.367 | 0.6239 |

### Feature-version comparison

Under HP1:

```text
RMSE improvement: 168.286 - 165.117 = 3.170
MAE improvement:  129.762 - 126.901 = 2.860
R² improvement:   0.6484 - 0.6348 = 0.0136
```

Approximate relative improvements:

- RMSE: `1.88%`
- MAE: `2.20%`

Version 2 therefore provided a modest but consistent benefit.

### Hyperparameter comparison

The HP2 models produced much lower training error but worse test error.

| Run | Train RMSE | Test RMSE | Gap |
|---|---:|---:|---:|
| `v2_hp1` | 158.847 | 165.117 | 6.270 |
| `v2_hp2` | 127.456 | 169.293 | 41.837 |
| `v1_hp1` | 163.184 | 168.286 | 5.102 |
| `v1_hp2` | 126.882 | 170.782 | 43.900 |

The large HP2 gaps are evidence of overfitting.

### Runtime comparison

HP2 also required substantially more training time without improving test performance.

The selected model balances:

- Lowest test RMSE
- Lowest test MAE
- Highest test R²
- Small generalization gap
- Lower training cost

### Status

Completed.

---

## Phase 12 — Automated Pipeline

### Purpose

Phase 12 documents the automation implemented in Phase 10.

### Entry points

```bash
python scripts/run_pipeline.py
make pipeline
```

### Useful partial-run options

```bash
python scripts/run_pipeline.py --no-reset
python scripts/run_pipeline.py --skip-experiments
python scripts/run_pipeline.py --start-at feast
python scripts/run_pipeline.py --stop-after model_smoke_test
```

### Make targets

```text
make setup
make quality
make ingestion
make preprocessing
make features
make feast
make split
make smoke
make experiments
make report
make audit
make audit-strict
make pipeline
```

### Status

Implemented.

---

## Phase 13 — Testing and Code Quality

### Test coverage

Automated tests cover:

- Ingestion
- Preprocessing
- Sentinel handling
- Target construction
- Feature definitions
- Feature Version 1
- Feature Version 2
- Feast entities
- Feast feature views and services
- Feature membership
- Leakage prevention
- Training-dataset alignment
- Deterministic splitting
- Preprocessing pipelines
- Unknown categories
- Model fitting
- Regression metrics
- Random Forest reproducibility
- MLflow matrix validation
- Best-run selection

### Quality tools

```text
pytest
ruff format
ruff check
GitHub Actions
```

Run:

```bash
make quality
```

Equivalent commands:

```bash
ruff format --check src scripts tests feature_repo
ruff check src scripts tests feature_repo
pytest -v
```

### Continuous integration

GitHub Actions performs:

- Python 3.11 environment setup
- Dependency installation
- Ruff formatting validation
- Ruff linting
- Unit tests
- Source-level submission audit

The full feature materialization and four-run experiment suite remain local because they
are more expensive than source-level CI checks.

### Status

Implemented; run the final quality and strict audit commands before submission.

---

## Final Recommendation

The recommended model is:

```text
Run:                    v2_hp1
Algorithm:              RandomForestRegressor
Feature version:        v2
Hyperparameter set:     hp1
Test RMSE:              165.1166
Test MAE:               126.9015
Test R²:                0.6484
```

### Selection rationale

`v2_hp1` was selected because it:

- Produced the lowest test RMSE
- Produced the lowest test MAE
- Produced the highest test R²
- Used the enhanced feature version
- Maintained a small train/test performance gap
- Trained much faster than HP2
- Avoided the clear overfitting observed in HP2

### Error interpretation

The `158.847` value is the training RMSE for `v2_hp1`, not MSE.

The corresponding training MSE is approximately:

```text
158.847² ≈ 25,232.36
```

The test RMSE is `165.117`, and the corresponding test MSE is approximately:

```text
165.117² ≈ 27,263.5
```

The test MAE of `126.901` is lower than RMSE because RMSE penalizes large residuals more
strongly.

---

## Reviewer Artifacts

### Feature artifacts

```text
data/features/v1/athlete_features_v1.parquet
data/features/v2/athlete_features_v2.parquet
```

### Training artifacts

```text
data/training/athlete_training_v1.parquet
data/training/athlete_training_v2.parquet
data/splits/athlete_split.parquet
```

### Feast evidence

```text
reports/feast/
```

### Experiment evidence

```text
reports/mlflow/experiment_comparison.csv
reports/mlflow/experiment_comparison.json
reports/mlflow/best_run_summary.json
reports/mlflow/runs/
```

### Figures

```text
reports/figures/
```

### Final report

```text
docs/assignment_2_rollout.html
```

---

## Reproduction Instructions

### Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Retrieve source data

```bash
dvc pull
```

### Run quality checks

```bash
make quality
```

### Run the full pipeline

```bash
make pipeline
```

### Run the final audit

```bash
python scripts/verify_submission.py
```

### Run the strict audit

```bash
make audit-strict
```