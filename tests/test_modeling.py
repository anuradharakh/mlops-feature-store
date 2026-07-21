"""Tests for the reusable Scikit-learn model pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from athlete_mlops.features.definitions import (
    build_feature_version_spec,
)
from athlete_mlops.training.modeling import (
    build_model_data_split,
    build_random_forest_pipeline,
    fit_and_evaluate_pipeline,
)

FEATURE_CONFIG = {
    "name": "athlete_features_v1",
    "description": "Baseline features.",
    "entity_key": "athlete_id",
    "timestamp_column": "event_timestamp",
    "features": [
        "age",
        "weight",
        "height",
        "gender",
        "region",
    ],
    "numerical_features": [
        "age",
        "weight",
        "height",
    ],
    "categorical_features": [
        "gender",
        "region",
    ],
    "target_leakage_columns": [
        "total_lift",
        "deadlift",
        "candj",
        "snatch",
        "backsq",
    ],
}


MODEL_CONFIG = {
    "algorithm": "RandomForestRegressor",
    "random_state": 42,
    "n_jobs": 1,
    "criterion": "squared_error",
    "preprocessing": {
        "numerical_imputer_strategy": "median",
        "categorical_imputer_strategy": "most_frequent",
        "one_hot_handle_unknown": "ignore",
        "one_hot_sparse_output": False,
    },
}


HYPERPARAMETERS = {
    "description": "Test forest",
    "n_estimators": 20,
    "max_depth": 6,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
}


def build_specification():
    """Build the Version 1 test feature specification."""
    return build_feature_version_spec(
        version="v1",
        config=FEATURE_CONFIG,
    )


def build_dataset() -> pd.DataFrame:
    """Create a representative model dataset."""
    athlete_ids = list(range(1, 31))

    dataframe = pd.DataFrame(
        {
            "athlete_id": athlete_ids,
            "event_timestamp": pd.date_range(
                "2024-01-01",
                periods=30,
                freq="h",
                tz="UTC",
            ),
            "total_lift": [float(300 + athlete_id * 18) for athlete_id in athlete_ids],
            "age": [float(20 + athlete_id % 20) for athlete_id in athlete_ids],
            "weight": [float(130 + athlete_id * 2) for athlete_id in athlete_ids],
            "height": [float(64 + athlete_id % 10) for athlete_id in athlete_ids],
            "gender": [("Male" if athlete_id % 2 else "Female") for athlete_id in athlete_ids],
            "region": [("Central" if athlete_id % 3 else "West") for athlete_id in athlete_ids],
        }
    )

    dataframe.loc[2, "weight"] = np.nan
    dataframe.loc[4, "region"] = None

    return dataframe


def build_membership() -> pd.DataFrame:
    """Create deterministic train and test membership."""
    return pd.DataFrame(
        {
            "athlete_id": list(range(1, 31)),
            "split": [("train" if athlete_id <= 24 else "test") for athlete_id in range(1, 31)],
        }
    )


def test_model_split_uses_persisted_membership() -> None:
    """Model data should follow the saved split artifact."""
    result = build_model_data_split(
        dataframe=build_dataset(),
        membership=build_membership(),
        specification=build_specification(),
        target_column="total_lift",
    )

    assert len(result.x_train) == 24
    assert len(result.x_test) == 6

    assert set(result.train_entity_ids) == set(range(1, 25))

    assert set(result.test_entity_ids) == set(range(25, 31))


def test_model_features_exclude_entity_and_target() -> None:
    """The model matrix should contain only configured features."""
    result = build_model_data_split(
        dataframe=build_dataset(),
        membership=build_membership(),
        specification=build_specification(),
        target_column="total_lift",
    )

    assert list(result.x_train.columns) == [
        "age",
        "weight",
        "height",
        "gender",
        "region",
    ]

    assert "athlete_id" not in result.x_train
    assert "total_lift" not in result.x_train


def test_pipeline_contains_preprocessor_and_model() -> None:
    """The pipeline should combine preprocessing and estimation."""
    pipeline = build_random_forest_pipeline(
        specification=build_specification(),
        model_config=MODEL_CONFIG,
        hyperparameters=HYPERPARAMETERS,
    )

    assert isinstance(
        pipeline,
        Pipeline,
    )

    assert list(pipeline.named_steps) == [
        "preprocessor",
        "model",
    ]


def test_pipeline_handles_missing_values() -> None:
    """Numerical and categorical missing values should be imputed."""
    data_split = build_model_data_split(
        dataframe=build_dataset(),
        membership=build_membership(),
        specification=build_specification(),
        target_column="total_lift",
    )

    pipeline = build_random_forest_pipeline(
        specification=build_specification(),
        model_config=MODEL_CONFIG,
        hyperparameters=HYPERPARAMETERS,
    )

    result = fit_and_evaluate_pipeline(
        pipeline=pipeline,
        data_split=data_split,
        target_column="total_lift",
    )

    assert len(result.predictions) == 6

    assert result.predictions["predicted_total_lift"].notna().all()


def test_numerical_imputer_is_fitted_on_train_data() -> None:
    """Numerical imputation statistics should come from training rows."""
    data_split = build_model_data_split(
        dataframe=build_dataset(),
        membership=build_membership(),
        specification=build_specification(),
        target_column="total_lift",
    )

    pipeline = build_random_forest_pipeline(
        specification=build_specification(),
        model_config=MODEL_CONFIG,
        hyperparameters=HYPERPARAMETERS,
    )

    pipeline.fit(
        data_split.x_train,
        data_split.y_train,
    )

    numerical_imputer = (
        pipeline.named_steps["preprocessor"].named_transformers_["numerical"].named_steps["imputer"]
    )

    expected_medians = (
        data_split.x_train[
            [
                "age",
                "weight",
                "height",
            ]
        ]
        .median()
        .to_numpy()
    )

    np.testing.assert_allclose(
        numerical_imputer.statistics_,
        expected_medians,
    )


def test_unknown_test_category_is_supported() -> None:
    """An unseen test category should not cause prediction failure."""
    dataframe = build_dataset()

    dataframe.loc[
        dataframe["athlete_id"] == 30,
        "region",
    ] = "Previously unseen region"

    data_split = build_model_data_split(
        dataframe=dataframe,
        membership=build_membership(),
        specification=build_specification(),
        target_column="total_lift",
    )

    pipeline = build_random_forest_pipeline(
        specification=build_specification(),
        model_config=MODEL_CONFIG,
        hyperparameters=HYPERPARAMETERS,
    )

    pipeline.fit(
        data_split.x_train,
        data_split.y_train,
    )

    predictions = pipeline.predict(data_split.x_test)

    assert len(predictions) == 6


def test_model_metrics_are_generated() -> None:
    """Training should produce the required regression metrics."""
    data_split = build_model_data_split(
        dataframe=build_dataset(),
        membership=build_membership(),
        specification=build_specification(),
        target_column="total_lift",
    )

    pipeline = build_random_forest_pipeline(
        specification=build_specification(),
        model_config=MODEL_CONFIG,
        hyperparameters=HYPERPARAMETERS,
    )

    result = fit_and_evaluate_pipeline(
        pipeline=pipeline,
        data_split=data_split,
        target_column="total_lift",
    )

    assert {
        "train_rmse",
        "train_mae",
        "train_r2",
        "test_rmse",
        "test_mae",
        "test_r2",
        "training_seconds",
        "prediction_seconds",
    }.issubset(result.metrics)

    assert result.metrics["test_rmse"] >= 0

    assert result.metrics["test_mae"] >= 0


def test_random_forest_results_are_deterministic() -> None:
    """The fixed seed should reproduce predictions."""
    data_split = build_model_data_split(
        dataframe=build_dataset(),
        membership=build_membership(),
        specification=build_specification(),
        target_column="total_lift",
    )

    first_pipeline = build_random_forest_pipeline(
        specification=build_specification(),
        model_config=MODEL_CONFIG,
        hyperparameters=HYPERPARAMETERS,
    )

    second_pipeline = build_random_forest_pipeline(
        specification=build_specification(),
        model_config=MODEL_CONFIG,
        hyperparameters=HYPERPARAMETERS,
    )

    first_result = fit_and_evaluate_pipeline(
        pipeline=first_pipeline,
        data_split=data_split,
        target_column="total_lift",
    )

    second_result = fit_and_evaluate_pipeline(
        pipeline=second_pipeline,
        data_split=data_split,
        target_column="total_lift",
    )

    np.testing.assert_allclose(
        first_result.predictions["predicted_total_lift"],
        second_result.predictions["predicted_total_lift"],
    )
