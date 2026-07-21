"""MLflow configuration and experiment-tracking utilities."""

from __future__ import annotations

import inspect
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class TrackingSetup:
    """Configured local MLflow tracking environment."""

    tracking_uri: str
    experiment_id: str
    experiment_name: str
    artifact_root: Path


def build_sqlite_tracking_uri(
    database_path: Path,
) -> str:
    """Create an absolute SQLAlchemy SQLite tracking URI."""
    absolute_path = database_path.resolve()

    return f"sqlite:///{absolute_path.as_posix()}"


def reset_local_tracking_state(
    database_path: Path,
    artifact_root: Path,
) -> None:
    """Delete generated local tracking state."""
    database_files = [
        database_path,
        Path(f"{database_path}-shm"),
        Path(f"{database_path}-wal"),
        Path(f"{database_path}-journal"),
    ]

    for path in database_files:
        if path.exists():
            path.unlink()

    if artifact_root.exists():
        shutil.rmtree(artifact_root)


def configure_local_tracking(
    database_path: Path,
    artifact_root: Path,
    experiment_name: str,
    reset: bool,
) -> TrackingSetup:
    """Configure SQLite tracking and a filesystem artifact store."""
    if reset:
        reset_local_tracking_state(
            database_path=database_path,
            artifact_root=artifact_root,
        )

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracking_uri = build_sqlite_tracking_uri(database_path)

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=(artifact_root.resolve().as_uri()),
            tags={
                "project": "athlete-strength",
                "algorithm": "RandomForestRegressor",
                "run_type": "assignment-2",
            },
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    return TrackingSetup(
        tracking_uri=tracking_uri,
        experiment_id=str(experiment_id),
        experiment_name=experiment_name,
        artifact_root=artifact_root,
    )


def validate_experiment_matrix(
    run_definitions: list[dict[str, Any]],
    feature_versions: set[str],
    hyperparameter_sets: set[str],
) -> None:
    """Require exactly one run for each feature/hyperparameter pair."""
    required_combinations = {
        (
            feature_version,
            hyperparameter_set,
        )
        for feature_version in feature_versions
        for hyperparameter_set in hyperparameter_sets
    }

    actual_combinations = [
        (
            str(run["feature_version"]),
            str(run["hyperparameter_set"]),
        )
        for run in run_definitions
    ]

    if len(actual_combinations) != len(set(actual_combinations)):
        raise ValueError(
            "Official experiment matrix contains duplicate feature/hyperparameter combinations."
        )

    if set(actual_combinations) != required_combinations:
        missing = required_combinations.difference(actual_combinations)

        unexpected = set(actual_combinations).difference(required_combinations)

        raise ValueError(
            "Official experiment matrix is incomplete. "
            f"Missing: {sorted(missing)}. "
            f"Unexpected: {sorted(unexpected)}."
        )

    run_names = [str(run["run_name"]) for run in run_definitions]

    if len(run_names) != len(set(run_names)):
        raise ValueError("Official experiment run names must be unique.")


def get_git_metadata(
    project_root: Path,
) -> dict[str, str]:
    """Return the current Git commit and worktree state."""

    def run_git_command(
        arguments: list[str],
    ) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
        )

        if completed.returncode != 0:
            return "unavailable"

        return completed.stdout.strip()

    commit = run_git_command(["rev-parse", "HEAD"])

    branch = run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])

    status = run_git_command(["status", "--porcelain"])

    dirty = "unknown" if status == "unavailable" else str(bool(status)).lower()

    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": dirty,
    }


def log_sklearn_model_compat(
    pipeline: Pipeline,
    input_example: pd.DataFrame,
) -> Any:
    """Log a Scikit-learn model across MLflow API versions."""
    predictions = pipeline.predict(input_example)

    signature = infer_signature(
        input_example,
        predictions,
    )

    log_model_parameters = inspect.signature(mlflow.sklearn.log_model).parameters

    keyword_arguments: dict[str, Any] = {
        "sk_model": pipeline,
        "signature": signature,
        "input_example": input_example,
        "serialization_format": "cloudpickle",
    }

    if "name" in log_model_parameters:
        keyword_arguments["name"] = "model"
    else:
        keyword_arguments["artifact_path"] = "model"

    return mlflow.sklearn.log_model(**keyword_arguments)


def select_best_run(
    comparison: pd.DataFrame,
) -> pd.Series:
    """Select the best run using RMSE, MAE, then R²."""
    required_columns = {
        "test_rmse",
        "test_mae",
        "test_r2",
    }

    missing_columns = required_columns.difference(comparison.columns)

    if missing_columns:
        raise ValueError(f"Experiment comparison is missing metrics: {sorted(missing_columns)}")

    if comparison.empty:
        raise ValueError("Experiment comparison contains no runs.")

    ranked = comparison.sort_values(
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
    ).reset_index(drop=True)

    return ranked.iloc[0]
