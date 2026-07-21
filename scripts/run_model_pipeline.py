"""Validate the reusable Scikit-learn model pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from athlete_mlops.features.definitions import (
    build_feature_version_spec,
)
from athlete_mlops.training.modeling import (
    build_model_data_split,
    build_pipeline_metadata,
    build_random_forest_pipeline,
    fit_and_evaluate_pipeline,
    sample_model_data_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"

FEATURE_CONFIG_PATH = PROJECT_ROOT / "configs" / "features.yaml"


def resolve_project_path(path_value: str) -> Path:
    """Resolve a repository-relative path."""
    return PROJECT_ROOT / path_value


def create_actual_vs_predicted_figure(
    predictions: pd.DataFrame,
    target_column: str,
    output_path: Path,
) -> None:
    """Save actual-versus-predicted smoke-test evidence."""
    actual_column = f"actual_{target_column}"
    predicted_column = f"predicted_{target_column}"

    minimum = min(
        predictions[actual_column].min(),
        predictions[predicted_column].min(),
    )

    maximum = max(
        predictions[actual_column].max(),
        predictions[predicted_column].max(),
    )

    figure, axis = plt.subplots(figsize=(8, 7))

    axis.scatter(
        predictions[actual_column],
        predictions[predicted_column],
        alpha=0.3,
    )

    axis.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        label="Perfect prediction",
    )

    axis.set_title("Model Pipeline Smoke Test: Actual vs Predicted")
    axis.set_xlabel(f"Actual {target_column}")
    axis.set_ylabel(f"Predicted {target_column}")
    axis.legend()
    axis.grid(alpha=0.25)

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    """Run a deterministic end-to-end model pipeline smoke test."""
    with MODEL_CONFIG_PATH.open(encoding="utf-8") as config_file:
        model_config = yaml.safe_load(config_file)

    with FEATURE_CONFIG_PATH.open(encoding="utf-8") as config_file:
        feature_config = yaml.safe_load(config_file)

    training_config = model_config["training"]
    smoke_config = model_config["smoke_test"]
    artifact_config = model_config["artifacts"]
    estimator_config = model_config["model"]

    feature_version = smoke_config["feature_version"]

    hyperparameter_name = smoke_config["hyperparameter_set"]

    specification = build_feature_version_spec(
        version=feature_version,
        config=feature_config["feature_versions"][feature_version],
    )

    dataset_path = resolve_project_path(training_config["feature_datasets"][feature_version])

    split_path = resolve_project_path(artifact_config["split_membership"])

    required_inputs = [
        dataset_path,
        split_path,
    ]

    missing_inputs = [path for path in required_inputs if not path.exists()]

    if missing_inputs:
        formatted_paths = "\n".join(f"- {path}" for path in missing_inputs)

        raise FileNotFoundError(
            "Model pipeline inputs are missing:\n"
            f"{formatted_paths}\n"
            "Run `python scripts/run_training_split.py` first."
        )

    dataset = pd.read_parquet(dataset_path)

    membership = pd.read_parquet(split_path)

    full_data_split = build_model_data_split(
        dataframe=dataset,
        membership=membership,
        specification=specification,
        target_column=training_config["target_column"],
    )

    smoke_data_split = sample_model_data_split(
        data_split=full_data_split,
        max_train_rows=int(smoke_config["max_train_rows"]),
        max_test_rows=int(smoke_config["max_test_rows"]),
        random_state=int(smoke_config["random_state"]),
    )

    hyperparameters = estimator_config["hyperparameter_sets"][hyperparameter_name]

    pipeline = build_random_forest_pipeline(
        specification=specification,
        model_config=estimator_config,
        hyperparameters=hyperparameters,
    )

    fit_result = fit_and_evaluate_pipeline(
        pipeline=pipeline,
        data_split=smoke_data_split,
        target_column=training_config["target_column"],
    )

    metadata = build_pipeline_metadata(
        pipeline=fit_result.pipeline,
        specification=specification,
        hyperparameter_name=(hyperparameter_name),
        hyperparameters=hyperparameters,
    )

    summary = {
        "status": "PASS",
        "run_type": "pipeline_smoke_test",
        "official_experiment": False,
        "purpose": (
            "Validate preprocessing, fitting, prediction, "
            "and metric calculation before MLflow experiments."
        ),
        **metadata,
        "full_training_rows": int(len(full_data_split.x_train)),
        "full_test_rows": int(len(full_data_split.x_test)),
        "smoke_training_rows": int(len(smoke_data_split.x_train)),
        "smoke_test_rows": int(len(smoke_data_split.x_test)),
        "metrics": {
            metric_name: round(
                float(metric_value),
                6,
            )
            for metric_name, metric_value in fit_result.metrics.items()
        },
    }

    summary_path = resolve_project_path(artifact_config["model_smoke_summary"])

    predictions_path = resolve_project_path(artifact_config["model_smoke_predictions"])

    importance_path = resolve_project_path(artifact_config["model_smoke_feature_importance"])

    structure_path = resolve_project_path(artifact_config["model_smoke_structure"])

    figure_path = resolve_project_path(artifact_config["model_smoke_figure"])

    for output_path in [
        summary_path,
        predictions_path,
        importance_path,
        structure_path,
        figure_path,
    ]:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    fit_result.predictions.to_csv(
        predictions_path,
        index=False,
    )

    fit_result.feature_importance.to_csv(
        importance_path,
        index=False,
    )

    structure_path.write_text(
        str(fit_result.pipeline) + "\n",
        encoding="utf-8",
    )

    create_actual_vs_predicted_figure(
        predictions=fit_result.predictions,
        target_column=training_config["target_column"],
        output_path=figure_path,
    )

    print("Phase 8 model pipeline smoke test completed successfully.")
    print(f"Feature version: {feature_version}")
    print(f"Hyperparameter set: {hyperparameter_name}")
    print(f"Smoke training rows: {len(smoke_data_split.x_train):,}")
    print(f"Smoke test rows: {len(smoke_data_split.x_test):,}")
    print(f"Test RMSE: {fit_result.metrics['test_rmse']:.4f}")
    print(f"Test MAE: {fit_result.metrics['test_mae']:.4f}")
    print(f"Test R²: {fit_result.metrics['test_r2']:.4f}")
    print(f"Summary: {summary_path}")
    print("PHASE 8 STATUS: PASS")


if __name__ == "__main__":
    main()
