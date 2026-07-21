# MLOps Feature Store — Athlete Strength Prediction

## Project Overview

This repository implements an end-to-end MLOps workflow for predicting an athlete's
combined lifting total, `total_lift`, from demographic and body-measurement features.

The project demonstrates:

- Modular data ingestion and preprocessing
- Reproducible data versioning with Git and DVC
- Two managed feature versions
- Offline and online feature retrieval with Feast
- A persisted and reproducible train/test split
- A reusable Scikit-learn model pipeline
- Four manually configured Random Forest experiments
- Experiment tracking with MLflow
- Automated testing, reporting, and submission validation

The implementation uses Python modules and command-line scripts rather than a notebook.

---

## Business and Modeling Objective

The prediction target is:

```text
total_lift = deadlift + candj + snatch + backsq
```

The four component lift columns are used to construct the label but are not used as
model features because doing so would cause direct target leakage.

The final model predicts `total_lift` from general athlete attributes such as age,
weight, height, gender, region, and engineered body-composition features.

---

## Architecture

```text
DVC-tracked athletes.zip
        |
        v
Raw-data ingestion
        |
        v
Preprocessing and target construction
        |
        +-----------------------------+
        |                             |
        v                             v
Feature Version 1             Feature Version 2
5 baseline features           5 baseline + 3 engineered
        |                             |
        +-------------+---------------+
                      |
                      v
                Feast feature store
         historical + online retrieval
                      |
                      v
          Persisted 80/20 train/test split
                      |
                      v
        Scikit-learn preprocessing pipeline
                      |
                      v
        RandomForestRegressor experiments
          v1_hp1, v1_hp2, v2_hp1, v2_hp2
                      |
                      v
             MLflow tracking and comparison
                      |
                      v
        Reports, visualizations, and audit
```

---

## Dataset Processing

The source dataset contains:

- 423,006 rows
- 27 columns

Preprocessing performs the following operations:

- Creates a stable `athlete_id`
- Creates an `event_timestamp`
- Converts lift measurements to numeric values
- Replaces the documented sentinel value `1` in lift components with missing values
- Builds `total_lift`
- Removes rows without valid target values
- Preserves low but plausible values rather than applying an arbitrary lower cutoff
- Excludes target-component columns from model features

Final model-ready population:

```text
81,707 athletes
```

Sentinel replacements:

| Lift component | Replacements |
|---|---:|
| Deadlift | 33 |
| Clean and jerk | 35 |
| Snatch | 33 |
| Back squat | 34 |
| **Total** | **135** |

---

## Feature Versions

### Feature Version 1 — Baseline

| Feature | Type |
|---|---|
| `age` | Numerical |
| `weight` | Numerical |
| `height` | Numerical |
| `gender` | Categorical |
| `region` | Categorical |

Feature count: **5**

### Feature Version 2 — Enhanced

Version 2 contains all Version 1 features plus:

| Engineered feature | Definition |
|---|---|
| `bmi` | `703 × weight / height²` |
| `age_squared` | `age²` |
| `weight_height_ratio` | `weight / height` |

Feature count: **8**

Both versions contain the same 81,707 athlete entities and timestamps.

---

## Feast Feature Store

The project uses Feast in local mode.

Registered objects include:

- Entity: `athlete`
- Join key: `athlete_id`
- Feature view: `athlete_features_v1`
- Feature view: `athlete_features_v2`
- Feature service: `athlete_strength_v1`
- Feature service: `athlete_strength_v2`

Feast is used for:

- Point-in-time historical feature retrieval
- Version-specific training datasets
- Local online-store materialization
- Online feature retrieval validation

Reviewer evidence is available under:

```text
reports/feast/
```

Local Feast registry and online-store databases are runtime state and are intentionally
excluded from Git.

---

## Reproducible Train/Test Split

One entity-level split is persisted and reused by all four experiments.

| Partition | Rows |
|---|---:|
| Train | 65,365 |
| Test | 16,342 |
| Total | 81,707 |

Configuration:

- Test fraction: `0.20`
- Random state: `42`
- Shuffle: enabled
- Split key: `athlete_id`

Persisted split:

```text
data/splits/athlete_split.parquet
```

Using the same split for every run ensures that model differences are caused by feature
versions or hyperparameters rather than different evaluation populations.

---

## Model Pipeline

Algorithm:

```text
RandomForestRegressor
```

Preprocessing is contained inside a Scikit-learn `Pipeline` and fitted only on training
data.

Numerical features:

- Median imputation

Categorical features:

- Most-frequent imputation
- One-hot encoding
- Unknown categories ignored during inference

This design prevents train/test leakage from preprocessing statistics.

---

## Hyperparameter Configurations

### HP1 — Conservative baseline

```yaml
n_estimators: 100
max_depth: 12
min_samples_split: 2
min_samples_leaf: 5
max_features: sqrt
bootstrap: true
```

### HP2 — Larger and more flexible forest

```yaml
n_estimators: 300
max_depth: 20
min_samples_split: 2
min_samples_leaf: 2
max_features: 1.0
bootstrap: true
```

No AutoML or automated hyperparameter search is used.

---

## Experiment Matrix

| Run | Feature version | Hyperparameter set |
|---|---|---|
| `v1_hp1` | v1 | hp1 |
| `v1_hp2` | v1 | hp2 |
| `v2_hp1` | v2 | hp1 |
| `v2_hp2` | v2 | hp2 |

Each MLflow run records:

- Feature version
- Hyperparameter set
- Model parameters
- Train and test RMSE
- Train and test MAE
- Train and test R²
- Training and prediction time
- Split hash
- Feature-data hash
- Git metadata
- Model artifact
- Predictions
- Feature importance
- Diagnostic visualizations

---

## Final Results

| Rank | Run | Test RMSE | Test MAE | Test R² | RMSE gap |
|---:|---|---:|---:|---:|---:|
| 1 | **`v2_hp1`** | **165.117** | **126.901** | **0.6484** | **6.270** |
| 2 | `v1_hp1` | 168.286 | 129.762 | 0.6348 | 5.102 |
| 3 | `v2_hp2` | 169.293 | 130.229 | 0.6304 | 41.837 |
| 4 | `v1_hp2` | 170.782 | 131.367 | 0.6239 | 43.900 |

### Selected model

```text
Feature version:       v2
Hyperparameter set:    hp1
Algorithm:             RandomForestRegressor
Test RMSE:             165.1166
Test MAE:              126.9015
Test R²:               0.6484
```

Feature Version 2 provided a modest but consistent improvement over Version 1 under HP1.

HP2 achieved much lower training error but worse test performance. Its train/test RMSE
gap increased substantially, indicating overfitting. HP2 also required considerably more
training time.

The final recommendation is therefore:

```text
v2_hp1
```

---

## Interpreting the Error Metrics

The best model's test RMSE is approximately `165.12`, while its test MAE is approximately
`126.90`.

This means:

- The average absolute prediction error is about 127 target units.
- RMSE is higher because it penalizes large errors more strongly.
- The model explains approximately 64.8% of the target variance.
- The corresponding RMSE should not be confused with MSE.
- The approximate test MSE is `165.1166²`, or about `27,263.5`.

The target mean is approximately 985, so the best RMSE is about 16.8% of the mean target.

---

## Repository Structure

```text
mlops-feature-store/
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   ├── pipeline.yaml
│   ├── features.yaml
│   └── model.yaml
├── data/
│   ├── source/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   │   ├── v1/
│   │   └── v2/
│   ├── training/
│   └── splits/
├── docs/
│   ├── assignment_2_rollout.md
│   ├── assignment_2_rollout.html
│   └── assets/
├── feature_repo/
│   ├── feature_store.yaml
│   └── feature_definitions.py
├── reports/
│   ├── validation/
│   ├── feast/
│   ├── mlflow/
│   ├── figures/
│   ├── pipeline/
│   └── submission/
├── scripts/
│   ├── run_ingestion.py
│   ├── run_preprocessing.py
│   ├── run_feature_v1.py
│   ├── run_feature_v2.py
│   ├── run_feast.py
│   ├── run_training_split.py
│   ├── run_model_pipeline.py
│   ├── run_experiments.py
│   ├── run_pipeline.py
│   ├── verify_submission.py
│   └── build_report.py
├── src/
│   └── athlete_mlops/
│       ├── data/
│       ├── features/
│       └── training/
├── tests/
├── Makefile
├── pyproject.toml
├── requirements.in
├── requirements.txt
└── README.md
```

---

## Environment Setup

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

When the source dataset is stored through DVC:

```bash
dvc pull
```

---

## Run Individual Stages

```bash
python scripts/run_ingestion.py
python scripts/run_preprocessing.py
python scripts/run_feature_v1.py
python scripts/run_feature_v2.py
python scripts/run_feast.py --reset
python scripts/run_training_split.py
python scripts/run_model_pipeline.py
python scripts/run_experiments.py --reset
python scripts/build_report.py
python scripts/verify_submission.py
```

---

## Run the Complete Pipeline

```bash
python scripts/run_pipeline.py
```

Or:

```bash
make pipeline
```

Useful pipeline options:

```bash
python scripts/run_pipeline.py --no-reset
python scripts/run_pipeline.py --skip-experiments
python scripts/run_pipeline.py --start-at feast
python scripts/run_pipeline.py --stop-after model_smoke_test
```

---

## MLflow UI

Start the local MLflow server:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./artifacts/mlflow \
  --host 127.0.0.1 \
  --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

Experiment name:

```text
athlete-strength-random-forest
```

---

## Code Quality

Run all quality checks:

```bash
make quality
```

Equivalent commands:

```bash
ruff format --check src scripts tests feature_repo
ruff check src scripts tests feature_repo
pytest -v
```

GitHub Actions runs formatting, linting, unit tests, and a source-level submission audit.

---

## Submission Audit

Run the standard audit:

```bash
python scripts/verify_submission.py
```

Run the strict audit after artifacts are committed and source data is portable:

```bash
python scripts/verify_submission.py \
  --require-git-tracked \
  --require-source-portable
```

Or:

```bash
make audit-strict
```

---

## Important Review Artifacts

```text
data/features/v1/athlete_features_v1.parquet
data/features/v2/athlete_features_v2.parquet
data/training/athlete_training_v1.parquet
data/training/athlete_training_v2.parquet
data/splits/athlete_split.parquet
reports/validation/
reports/feast/
reports/mlflow/
reports/figures/
reports/pipeline/
reports/submission/
docs/assignment_2_rollout.html
```

Local runtime state such as `mlflow.db`, Feast SQLite databases, and fitted MLflow model
binaries is reproducible and intentionally excluded from Git.

---

## Reproducibility Checklist

- [x] Raw source tracked with DVC metadata
- [x] Modular ingestion and preprocessing
- [x] Two feature versions
- [x] Feast historical retrieval
- [x] Feast online retrieval validation
- [x] Persisted deterministic train/test split
- [x] Reusable Scikit-learn pipeline
- [x] Four manually configured experiments
- [x] MLflow experiment tracking
- [x] Best model selected from test metrics
- [x] Reviewer-facing reports and figures
- [ ] Final full pipeline rerun in a clean clone
- [ ] Strict Git/source-portability audit after final commit

---

## Limitations

- The model uses a small set of general athlete attributes, so substantial unexplained
  variance remains.
- The dataset contains missing, implausible, and self-reported measurements.
- Low lift values were retained unless there was documented evidence that they were
  sentinel values.
- Filtering rows without valid targets may introduce selection bias.
- The local Feast and MLflow setup demonstrates reproducibility but is not a production
  deployment architecture.
- Final production use would require drift monitoring, schema monitoring, access
  controls, remote artifact storage, and scheduled retraining.

---

## Final Recommendation

Use the `v2_hp1` model configuration.

It achieved the best test RMSE, best test MAE, highest test R², a small train/test error
gap, and much lower training cost than HP2.
