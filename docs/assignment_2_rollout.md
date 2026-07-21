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
