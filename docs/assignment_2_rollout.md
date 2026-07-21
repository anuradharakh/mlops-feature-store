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

### Objective

Create a deterministic processed dataset, Feast-compatible entity keys,
event timestamps, and a regression label while preventing target leakage.

### Processing decisions

- `athlete_id` is retained as the Feast entity key.
- The original retrieval timestamp is used when available.
- A deterministic fallback timestamp is generated for missing timestamps.
- Invalid survey responses are converted to missing values.
- Numerical values are converted using safe coercion.
- Implausible numerical values are replaced with missing values.
- Records missing any target component are excluded.
- Duplicate athlete records retain the latest observation.
- Missing predictor values are retained for training-only imputation.

### Target definition

`total_lift` is calculated as:

```text
deadlift + candj + snatch + backsq
```

### Target sentinel-value correction

Exploratory validation identified repeated lift-component values of `1`,
including records where all four lift components were equal to `1`. This
pattern was interpreted as a source-system placeholder rather than a valid
measurement.

The value `1` was therefore configured as a sentinel value for `deadlift`,
`candj`, `snatch`, and `backsq`. The preprocessing pipeline replaced 135
sentinel component values with missing values. Because multiple sentinel
values occurred within some records, this resulted in 37 additional athlete
records being excluded from label construction.

After correction:

- Processed rows: 81,707
- Label rows: 81,707
- Sentinel values remaining: 0
- Total-lift range: 8 to 2,367

A broad lower-target cutoff was intentionally avoided to preserve potentially
valid beginner-athlete records.
---

## Phase 4 — Feature Version 1

### Objective

Create a stable baseline feature version for athlete-strength modeling and
prepare it as the offline source for later Feast registration.

### Feature definition

Feature Version 1 contains:

- `age`
- `weight`
- `height`
- `gender`
- `region`

The table also contains the Feast-required entity and timestamp fields:

- `athlete_id`
- `event_timestamp`

### Design rationale

Version 1 intentionally uses a small, interpretable set of demographic and
physical features. Higher-cardinality survey responses are excluded from the
baseline and will be normalized or engineered in Feature Version 2.

Missing predictor values are preserved. Imputation and encoding are deferred
to the Scikit-learn training pipeline so that those transformations are fitted
only on the training partition.

### Leakage prevention

The following target-related columns are explicitly prohibited:

- `total_lift`
- `deadlift`
- `candj`
- `snatch`
- `backsq`

### Generated artifacts

- `data/features/v1/athlete_features_v1.parquet`
- `reports/validation/feature_v1_manifest.json`
- `reports/validation/feature_v1_missingness.csv`
- `reports/validation/feature_v1_schema.csv`

### Version evidence

The Version 1 manifest records the feature list, data types, entity count,
missing-value counts, schema hash, and deterministic feature-data hash.

### Status

Completed.

---

## Phase 5 — Feature Version 2

### Objective

Create an enhanced feature version that preserves the Version 1 entity
population while adding deterministic engineered features.

### Version lineage

Feature Version 2 is derived from Feature Version 1 and contains all five
baseline features:

- `age`
- `weight`
- `height`
- `gender`
- `region`

It adds:

- `bmi`
- `age_squared`
- `weight_height_ratio`

### Feature engineering

BMI was calculated using the imperial-unit formula because weight is measured
in pounds and height is measured in inches:

`bmi = 703 × weight / height²`

The remaining transformations were:

- `age_squared = age²`
- `weight_height_ratio = weight / height`

Missing source values were intentionally preserved. No feature imputation or
categorical encoding was performed during feature generation. Those operations
will be fitted only on the training partition in the model pipeline.

### Version integrity

Feature Versions 1 and 2 contain the same 81,707 athlete entities and aligned
event timestamps. Version 2 adds three features and removes none.

Both versions have separate Parquet artifacts, manifests, schemas,
missingness reports, and deterministic data hashes.

### Generated artifacts

- `data/features/v2/athlete_features_v2.parquet`
- `reports/validation/feature_v2_manifest.json`
- `reports/validation/feature_v2_missingness.csv`
- `reports/validation/feature_v2_schema.csv`
- `reports/validation/feature_v2_engineering_summary.csv`
- `reports/validation/feature_version_comparison.json`

### Status

Completed.

---

## Phase 6 — Feast Feature Store

### Objective

Register both feature versions in Feast and demonstrate historical and online
feature retrieval.

### Local architecture

The Feast deployment uses:

- Parquet files as the offline feature sources
- A local Feast registry
- SQLite as the online feature store
- `athlete_id` as the entity join key
- `event_timestamp` for point-in-time feature retrieval

### Registered objects

Entity:

- `athlete`

Feature views:

- `athlete_features_v1`
- `athlete_features_v2`

Feature services:

- `athlete_strength_v1`
- `athlete_strength_v2`

Separate feature services provide explicit model-centric version tracking.

### Historical retrieval

The Phase 3 label artifact was supplied to Feast as an entity dataframe
containing:

- `athlete_id`
- `event_timestamp`
- `total_lift`

Feast retrieved point-in-time features for all 81,707 eligible athletes for
both feature versions. The complete generated datasets are stored locally
under `data/training/`.

### Online retrieval

Both feature views were materialized from their Parquet offline sources into
the SQLite online store. Online retrieval was validated for five athlete
entities using both versioned feature services.

### Committed evidence

- `feature_repo/feature_store.yaml`
- `feature_repo/feature_definitions.py`
- `reports/feast/feast_apply_output.txt`
- `reports/feast/feature_registry_summary.json`
- `reports/feast/historical_v1_sample.csv`
- `reports/feast/historical_v2_sample.csv`
- `reports/feast/online_retrieval_sample.json`
- `reports/feast/feast_validation_summary.json`

The generated registry and online SQLite database are intentionally excluded
because they can be reproduced by running:

`python scripts/run_feast.py --reset`

### Status

Completed.

---

## Phase 7 — Reproducible Training Split

### Objective

Prepare aligned versioned training datasets and create one deterministic
train/test split that will be reused across all four experiments.

### Training datasets

The training datasets were generated through Feast historical retrieval:

- `data/training/athlete_training_v1.parquet`
- `data/training/athlete_training_v2.parquet`

Both datasets contain the same athlete entities, event timestamps, and
`total_lift` labels. Version 1 contains five model features, while Version 2
contains eight.

### Split strategy

A single 80/20 train/test split was generated using:

- Random state: `42`
- Shuffling: enabled
- Entity-level split key: `athlete_id`

The split membership is persisted as:

- `data/splits/athlete_split.parquet`

This artifact will be reused for every experiment, ensuring that differences
in model performance are attributable to feature versions or hyperparameter
configurations rather than different evaluation populations.

### Leakage prevention

The source target components were explicitly prohibited from the training
datasets:

- `deadlift`
- `candj`
- `snatch`
- `backsq`

The only label available to the model pipeline is `total_lift`.

### Generated evidence

- `reports/validation/training_split_summary.json`
- `reports/validation/training_dataset_missingness.csv`
- `reports/figures/training_split_target_distribution.png`

### Status

Completed.

---

## Phase 8 — Scikit-learn Model Pipeline

## Phase 8 — Scikit-learn Model Pipeline

### Objective

Build one reusable Scikit-learn pipeline that can be applied consistently
across both feature versions and both hyperparameter configurations.

### Preprocessing

Numerical features use median imputation. Categorical features use
most-frequent imputation followed by one-hot encoding.

All preprocessing components are contained within the fitted Scikit-learn
pipeline. The pipeline is fitted only on the training partition, preventing
test-set information from influencing imputation statistics or category
discovery.

Unknown categorical values are ignored during transformation so that
previously unseen test or serving values do not cause prediction failures.

### Estimator

The selected algorithm is `RandomForestRegressor`.

Two manually configured hyperparameter sets are defined:

- `hp1`: 100 estimators, maximum depth 12, minimum leaf size 5
- `hp2`: 300 estimators, maximum depth 20, minimum leaf size 2

No automated hyperparameter tuning or AutoML is used.

### Smoke test

A deterministic subset of 10,000 training records and 2,500 test records was
used to validate the complete model pipeline before experiment tracking.

The smoke test is an infrastructure check and is not counted as one of the
four official experiments.

### Generated evidence

- `reports/validation/model_pipeline_smoke_summary.json`
- `reports/validation/model_pipeline_smoke_predictions.csv`
- `reports/validation/model_pipeline_smoke_feature_importance.csv`
- `reports/validation/model_pipeline_structure.txt`
- `reports/figures/model_pipeline_smoke_actual_vs_predicted.png`

### Status

Completed.
---

## Phase 9 — MLflow Experiment Tracking

### Objective

Execute and track the four required model experiments using the same
Random Forest algorithm and the same persisted train/test split.

### Experiment matrix

| Run | Feature version | Hyperparameter set |
|---|---|---|
| `v1_hp1` | Version 1 | HP1 |
| `v1_hp2` | Version 1 | HP2 |
| `v2_hp1` | Version 2 | HP1 |
| `v2_hp2` | Version 2 | HP2 |

### Tracking architecture

MLflow uses:

- SQLite for experiment and run metadata
- A local filesystem artifact store for fitted models
- Explicit tags for feature version, hyperparameter configuration,
  Git commit, and split hash
- Separate metrics, predictions, feature importances, and diagnostic
  visualizations for every run

### Logged metrics

Each run records:

- Train RMSE
- Train MAE
- Train R²
- Test RMSE
- Test MAE
- Test R²
- Training duration
- Prediction duration
- Train/test RMSE gap
- Train/test R² gap

### Reproducibility

Every run references:

- The same persisted train/test membership
- The same split SHA-256 hash
- The associated feature-data hash
- The Git commit and worktree state
- The full model and feature configuration
- The locked Python environment

### Generated evidence

- `reports/mlflow/experiment_comparison.csv`
- `reports/mlflow/experiment_comparison.json`
- `reports/mlflow/best_run_summary.json`
- `reports/mlflow/runs/v1_hp1/`
- `reports/mlflow/runs/v1_hp2/`
- `reports/mlflow/runs/v2_hp1/`
- `reports/mlflow/runs/v2_hp2/`
- `reports/figures/experiment_rmse_comparison.png`
- `reports/figures/experiment_mae_comparison.png`
- `reports/figures/experiment_r2_comparison.png`
- `docs/assets/mlflow_four_experiments.png`

The local MLflow database and fitted model binaries are reproducible runtime
state and are intentionally not committed to Git.

### Status

Completed.

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
