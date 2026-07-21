"""Tests for MLflow experiment configuration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from athlete_mlops.training.tracking import (
    build_sqlite_tracking_uri,
    select_best_run,
    validate_experiment_matrix,
)

OFFICIAL_RUNS = [
    {
        "run_name": "v1_hp1",
        "feature_version": "v1",
        "hyperparameter_set": "hp1",
    },
    {
        "run_name": "v1_hp2",
        "feature_version": "v1",
        "hyperparameter_set": "hp2",
    },
    {
        "run_name": "v2_hp1",
        "feature_version": "v2",
        "hyperparameter_set": "hp1",
    },
    {
        "run_name": "v2_hp2",
        "feature_version": "v2",
        "hyperparameter_set": "hp2",
    },
]


def test_sqlite_tracking_uri_is_absolute(
    tmp_path: Path,
) -> None:
    """The tracking URI should resolve an absolute database path."""
    database_path = tmp_path / "mlflow.db"

    tracking_uri = build_sqlite_tracking_uri(database_path)

    assert tracking_uri.startswith("sqlite:////")

    assert tracking_uri.endswith("mlflow.db")


def test_official_experiment_matrix_is_complete() -> None:
    """All four required experiment combinations should exist."""
    validate_experiment_matrix(
        run_definitions=OFFICIAL_RUNS,
        feature_versions={
            "v1",
            "v2",
        },
        hyperparameter_sets={
            "hp1",
            "hp2",
        },
    )


def test_missing_experiment_combination_fails() -> None:
    """An incomplete matrix should fail validation."""
    with pytest.raises(
        ValueError,
        match="incomplete",
    ):
        validate_experiment_matrix(
            run_definitions=OFFICIAL_RUNS[:-1],
            feature_versions={
                "v1",
                "v2",
            },
            hyperparameter_sets={
                "hp1",
                "hp2",
            },
        )


def test_duplicate_experiment_combination_fails() -> None:
    """Duplicate feature/hyperparameter pairs should fail."""
    duplicated = [
        *OFFICIAL_RUNS,
        {
            "run_name": "duplicate",
            "feature_version": "v1",
            "hyperparameter_set": "hp1",
        },
    ]

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        validate_experiment_matrix(
            run_definitions=duplicated,
            feature_versions={
                "v1",
                "v2",
            },
            hyperparameter_sets={
                "hp1",
                "hp2",
            },
        )


def test_best_run_uses_lowest_test_rmse() -> None:
    """The primary model-selection metric should be test RMSE."""
    comparison = pd.DataFrame(
        {
            "run_name": [
                "v1_hp1",
                "v1_hp2",
                "v2_hp1",
                "v2_hp2",
            ],
            "test_rmse": [
                150.0,
                140.0,
                130.0,
                120.0,
            ],
            "test_mae": [
                100.0,
                95.0,
                90.0,
                85.0,
            ],
            "test_r2": [
                0.60,
                0.65,
                0.70,
                0.75,
            ],
        }
    )

    best = select_best_run(comparison)

    assert best["run_name"] == "v2_hp2"


def test_best_run_uses_tie_breakers() -> None:
    """MAE and R² should resolve equal-RMSE runs."""
    comparison = pd.DataFrame(
        {
            "run_name": [
                "run_a",
                "run_b",
            ],
            "test_rmse": [
                120.0,
                120.0,
            ],
            "test_mae": [
                90.0,
                85.0,
            ],
            "test_r2": [
                0.75,
                0.74,
            ],
        }
    )

    best = select_best_run(comparison)

    assert best["run_name"] == "run_b"
