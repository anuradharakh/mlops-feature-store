"""Run and track the four official MLflow experiments."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
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
)
from athlete_mlops.training.tracking import (
    configure_local_tracking,
    get_git_metadata,
    log_sklearn_model_compat,
    select_best_run,
    validate_experiment_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"

FEATURE_CONFIG_PATH = PROJECT_ROOT / "configs" / "features.yaml"

REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"


def parse_args() -> argparse.Namespace:
    """Parse experiment-runner arguments."""
    parser = argparse.ArgumentParser(
        description=("Run the four official athlete-strength MLflow experiments.")
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help=("Delete the local MLflow database and model artifact store before running."),
    )

    parser.add_argument(
        "--skip-model-logging",
        action="store_true",
        help=("Track metrics and artifacts without logging the fitted model binaries."),
    )

    return parser.parse_args()


def resolve_project_path(
    path_value: str,
) -> Path:
    """Resolve a repository-relative path."""
    return PROJECT_ROOT / path_value


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load a JSON object."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def create_actual_vs_predicted_figure(
    predictions: pd.DataFrame,
    target_column: str,
    output_path: Path,
    title: str,
) -> None:
    """Save an actual-versus-predicted chart."""
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
        alpha=0.25,
        s=12,
    )

    axis.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        label="Perfect prediction",
    )

    axis.set_title(title)
    axis.set_xlabel(f"Actual {target_column}")
    axis.set_ylabel(f"Predicted {target_column}")
    axis.legend()
    axis.grid(alpha=0.25)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_residual_figure(
    predictions: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Save the test residual distribution."""
    figure, axis = plt.subplots(figsize=(9, 6))

    axis.hist(
        predictions["residual"],
        bins=60,
    )

    axis.axvline(
        0,
        linestyle="--",
        label="Zero residual",
    )

    axis.set_title(title)
    axis.set_xlabel("Actual minus predicted total lift")
    axis.set_ylabel("Frequency")
    axis.legend()
    axis.grid(alpha=0.25)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_metric_comparison_figure(
    comparison: pd.DataFrame,
    metric: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    """Create a four-run metric comparison chart."""
    ordered = comparison.sort_values("run_name")

    figure, axis = plt.subplots(figsize=(9, 6))

    axis.bar(
        ordered["run_name"],
        ordered[metric],
    )

    axis.set_title(title)
    axis.set_xlabel("Official experiment")
    axis.set_ylabel(y_label)
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    for index, value in enumerate(ordered[metric]):
        axis.text(
            index,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

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


def save_run_artifacts(
    run_directory: Path,
    run_summary: dict[str, Any],
    predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    target_column: str,
    run_name: str,
) -> None:
    """Save committed evidence for one official run."""
    if run_directory.exists():
        shutil.rmtree(run_directory)

    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (run_directory / "run_summary.json").write_text(
        json.dumps(
            run_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    predictions.to_csv(
        run_directory / "predictions.csv",
        index=False,
    )

    feature_importance.to_csv(
        run_directory / "feature_importance.csv",
        index=False,
    )

    create_actual_vs_predicted_figure(
        predictions=predictions,
        target_column=target_column,
        output_path=(run_directory / "actual_vs_predicted.png"),
        title=(f"{run_name}: Actual vs Predicted Total Lift"),
    )

    create_residual_figure(
        predictions=predictions,
        output_path=(run_directory / "residual_distribution.png"),
        title=(f"{run_name}: Test Residual Distribution"),
    )


def main() -> None:
    """Execute and track the four official experiments."""
    args = parse_args()

    with MODEL_CONFIG_PATH.open(encoding="utf-8") as config_file:
        model_config = yaml.safe_load(config_file)

    with FEATURE_CONFIG_PATH.open(encoding="utf-8") as config_file:
        feature_config = yaml.safe_load(config_file)

    training_config = model_config["training"]

    estimator_config = model_config["model"]
    tracking_config = model_config["mlflow"]
    artifact_config = model_config["artifacts"]

    official_runs = list(tracking_config["official_runs"])

    validate_experiment_matrix(
        run_definitions=official_runs,
        feature_versions={
            "v1",
            "v2",
        },
        hyperparameter_sets={
            "hp1",
            "hp2",
        },
    )

    tracking_database = resolve_project_path(tracking_config["tracking_database"])

    mlflow_artifact_root = resolve_project_path(tracking_config["artifact_root"])

    report_directory = resolve_project_path(artifact_config["experiment_report_directory"])

    comparison_csv_path = resolve_project_path(artifact_config["experiment_comparison_csv"])

    comparison_json_path = resolve_project_path(artifact_config["experiment_comparison_json"])

    best_run_path = resolve_project_path(artifact_config["best_run_summary"])

    rmse_figure_path = resolve_project_path(artifact_config["experiment_rmse_figure"])

    mae_figure_path = resolve_project_path(artifact_config["experiment_mae_figure"])

    r2_figure_path = resolve_project_path(artifact_config["experiment_r2_figure"])

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    setup = configure_local_tracking(
        database_path=tracking_database,
        artifact_root=mlflow_artifact_root,
        experiment_name=tracking_config["experiment_name"],
        reset=args.reset,
    )

    split_path = resolve_project_path(artifact_config["split_membership"])

    split_summary_path = resolve_project_path(artifact_config["split_summary"])

    required_inputs = [
        split_path,
        split_summary_path,
        MODEL_CONFIG_PATH,
        FEATURE_CONFIG_PATH,
    ]

    for feature_version in [
        "v1",
        "v2",
    ]:
        required_inputs.append(
            resolve_project_path(training_config["feature_datasets"][feature_version])
        )

        required_inputs.append(
            PROJECT_ROOT / "reports" / "validation" / (f"feature_{feature_version}_manifest.json")
        )

    missing_inputs = [path for path in required_inputs if not path.exists()]

    if missing_inputs:
        formatted = "\n".join(f"- {path}" for path in missing_inputs)

        raise FileNotFoundError(
            f"Required experiment inputs are missing:\n{formatted}\nRun Phases 6–8 first."
        )

    membership = pd.read_parquet(split_path)

    split_summary = load_json(split_summary_path)

    split_hash = split_summary["split"]["split_hash_sha256"]

    git_metadata = get_git_metadata(PROJECT_ROOT)

    run_group_id = (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )

    dataset_cache: dict[str, pd.DataFrame] = {}
    split_cache: dict[str, Any] = {}
    manifest_cache: dict[str, dict[str, Any]] = {}

    result_records: list[dict[str, Any]] = []

    for run_definition in official_runs:
        run_name = str(run_definition["run_name"])

        feature_version = str(run_definition["feature_version"])

        hyperparameter_name = str(run_definition["hyperparameter_set"])

        specification = build_feature_version_spec(
            version=feature_version,
            config=feature_config["feature_versions"][feature_version],
        )

        if feature_version not in dataset_cache:
            dataset_path = resolve_project_path(
                training_config["feature_datasets"][feature_version]
            )

            dataset_cache[feature_version] = pd.read_parquet(dataset_path)

            split_cache[feature_version] = build_model_data_split(
                dataframe=dataset_cache[feature_version],
                membership=membership,
                specification=specification,
                target_column=training_config["target_column"],
            )

            manifest_cache[feature_version] = load_json(
                PROJECT_ROOT
                / "reports"
                / "validation"
                / (f"feature_{feature_version}_manifest.json")
            )

        data_split = split_cache[feature_version]

        hyperparameters = estimator_config["hyperparameter_sets"][hyperparameter_name]

        pipeline = build_random_forest_pipeline(
            specification=specification,
            model_config=estimator_config,
            hyperparameters=hyperparameters,
        )

        tags = {
            "official_experiment": "true",
            "run_group_id": run_group_id,
            "feature_version": feature_version,
            "hyperparameter_set": (hyperparameter_name),
            "algorithm": (estimator_config["algorithm"]),
            "split_hash_sha256": split_hash,
            **git_metadata,
        }

        run_directory = report_directory / "runs" / run_name

        with mlflow.start_run(
            experiment_id=setup.experiment_id,
            run_name=run_name,
            tags=tags,
        ) as active_run:
            fit_result = fit_and_evaluate_pipeline(
                pipeline=pipeline,
                data_split=data_split,
                target_column=training_config["target_column"],
            )

            metrics = {
                **fit_result.metrics,
                "test_train_rmse_gap": (
                    fit_result.metrics["test_rmse"] - fit_result.metrics["train_rmse"]
                ),
                "train_test_r2_gap": (
                    fit_result.metrics["train_r2"] - fit_result.metrics["test_r2"]
                ),
            }

            pipeline_metadata = build_pipeline_metadata(
                pipeline=(fit_result.pipeline),
                specification=specification,
                hyperparameter_name=(hyperparameter_name),
                hyperparameters=(hyperparameters),
            )

            feature_manifest = manifest_cache[feature_version]

            parameters = {
                "algorithm": (estimator_config["algorithm"]),
                "feature_version": (feature_version),
                "feature_view_name": (specification.name),
                "hyperparameter_set": (hyperparameter_name),
                "source_feature_count": len(specification.features),
                "source_features": ",".join(specification.features),
                "transformed_feature_count": (pipeline_metadata["transformed_feature_count"]),
                "train_rows": len(data_split.x_train),
                "test_rows": len(data_split.x_test),
                "random_state": (estimator_config["random_state"]),
                "criterion": (estimator_config["criterion"]),
                "numerical_imputer": (
                    estimator_config["preprocessing"]["numerical_imputer_strategy"]
                ),
                "categorical_imputer": (
                    estimator_config["preprocessing"]["categorical_imputer_strategy"]
                ),
                "feature_data_hash_sha256": (feature_manifest["data_hash_sha256"]),
                "split_hash_sha256": (split_hash),
            }

            for key, value in hyperparameters.items():
                parameters[f"rf_{key}"] = value

            mlflow.log_params(parameters)
            mlflow.log_metrics(metrics)

            run_id = active_run.info.run_id

            model_uri = ""

            log_models = bool(tracking_config["log_models"]) and not args.skip_model_logging

            if log_models:
                input_example = data_split.x_train.head(20).copy()

                model_info = log_sklearn_model_compat(
                    pipeline=(fit_result.pipeline),
                    input_example=(input_example),
                )

                model_uri = str(
                    getattr(
                        model_info,
                        "model_uri",
                        f"runs:/{run_id}/model",
                    )
                )

            run_summary = {
                "run_id": run_id,
                "run_name": run_name,
                "run_group_id": run_group_id,
                "experiment_id": (setup.experiment_id),
                "experiment_name": (setup.experiment_name),
                "artifact_uri": (active_run.info.artifact_uri),
                "model_uri": model_uri,
                "model_logged": log_models,
                "feature_version": (feature_version),
                "hyperparameter_set": (hyperparameter_name),
                "parameters": parameters,
                "metrics": {
                    key: round(
                        float(value),
                        8,
                    )
                    for key, value in metrics.items()
                },
                "pipeline": pipeline_metadata,
                "git": git_metadata,
            }

            save_run_artifacts(
                run_directory=run_directory,
                run_summary=run_summary,
                predictions=(fit_result.predictions),
                feature_importance=(fit_result.feature_importance),
                target_column=training_config["target_column"],
                run_name=run_name,
            )

            mlflow.log_artifacts(
                str(run_directory),
                artifact_path="review",
            )

            mlflow.log_artifact(
                str(MODEL_CONFIG_PATH),
                artifact_path="configuration",
            )

            mlflow.log_artifact(
                str(FEATURE_CONFIG_PATH),
                artifact_path="configuration",
            )

            mlflow.log_artifact(
                str(split_summary_path),
                artifact_path="data_lineage",
            )

            feature_manifest_path = (
                PROJECT_ROOT
                / "reports"
                / "validation"
                / (f"feature_{feature_version}_manifest.json")
            )

            mlflow.log_artifact(
                str(feature_manifest_path),
                artifact_path="data_lineage",
            )

            if REQUIREMENTS_PATH.exists():
                mlflow.log_artifact(
                    str(REQUIREMENTS_PATH),
                    artifact_path=("environment"),
                )

            result_records.append(
                {
                    "run_id": run_id,
                    "run_name": run_name,
                    "run_group_id": (run_group_id),
                    "feature_version": (feature_version),
                    "hyperparameter_set": (hyperparameter_name),
                    "model_uri": model_uri,
                    "train_rows": len(data_split.x_train),
                    "test_rows": len(data_split.x_test),
                    **{key: float(value) for key, value in metrics.items()},
                }
            )

            print(
                f"Completed {run_name}: "
                f"RMSE="
                f"{metrics['test_rmse']:.4f}, "
                f"MAE="
                f"{metrics['test_mae']:.4f}, "
                f"R²="
                f"{metrics['test_r2']:.4f}"
            )

    comparison = (
        pd.DataFrame(result_records)
        .sort_values(
            by=[
                "test_rmse",
                "test_mae",
                "test_r2",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    comparison.insert(
        0,
        "rank",
        range(1, len(comparison) + 1),
    )

    if len(comparison) != 4:
        raise ValueError("Exactly four official experiment results were expected.")

    comparison_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        comparison_csv_path,
        index=False,
    )

    comparison_payload = {
        "status": "PASS",
        "tracking_uri": (setup.tracking_uri),
        "experiment_id": (setup.experiment_id),
        "experiment_name": (setup.experiment_name),
        "run_group_id": run_group_id,
        "official_run_count": int(len(comparison)),
        "selection_metric": ("lowest test_rmse, then lowest test_mae, then highest test_r2"),
        "split_hash_sha256": split_hash,
        "runs": comparison.to_dict(orient="records"),
    }

    comparison_json_path.write_text(
        json.dumps(
            comparison_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    best_run = select_best_run(comparison)

    best_run_payload = {
        "status": "PASS",
        "run_group_id": run_group_id,
        "run_id": str(best_run["run_id"]),
        "run_name": str(best_run["run_name"]),
        "feature_version": str(best_run["feature_version"]),
        "hyperparameter_set": str(best_run["hyperparameter_set"]),
        "model_uri": str(best_run["model_uri"]),
        "test_rmse": float(best_run["test_rmse"]),
        "test_mae": float(best_run["test_mae"]),
        "test_r2": float(best_run["test_r2"]),
        "selection_reason": ("Lowest test RMSE, with test MAE and test R² used as tie breakers."),
    }

    best_run_path.write_text(
        json.dumps(
            best_run_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    create_metric_comparison_figure(
        comparison=comparison,
        metric="test_rmse",
        title=("Official Experiment Test RMSE"),
        y_label="Test RMSE",
        output_path=rmse_figure_path,
    )

    create_metric_comparison_figure(
        comparison=comparison,
        metric="test_mae",
        title=("Official Experiment Test MAE"),
        y_label="Test MAE",
        output_path=mae_figure_path,
    )

    create_metric_comparison_figure(
        comparison=comparison,
        metric="test_r2",
        title=("Official Experiment Test R²"),
        y_label="Test R²",
        output_path=r2_figure_path,
    )

    tracked_runs = mlflow.search_runs(
        experiment_ids=[setup.experiment_id],
        filter_string=(f'tags.run_group_id = "{run_group_id}"'),
    )

    if len(tracked_runs) != 4:
        raise ValueError("MLflow did not return exactly four runs for the current run group.")

    print()
    print("Phase 9 MLflow experiments completed successfully.")
    print(f"Experiment: {setup.experiment_name}")
    print(f"Run group: {run_group_id}")
    print("Official runs tracked: 4")
    print(f"Best run: {best_run['run_name']}")
    print(f"Best test RMSE: {best_run['test_rmse']:.4f}")
    print(f"Best test MAE: {best_run['test_mae']:.4f}")
    print(f"Best test R²: {best_run['test_r2']:.4f}")
    print(f"Comparison: {comparison_csv_path}")
    print("PHASE 9 STATUS: PASS")


if __name__ == "__main__":
    main()
