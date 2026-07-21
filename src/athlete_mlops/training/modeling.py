"""Reusable Scikit-learn modeling pipeline for athlete regression."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from athlete_mlops.features.definitions import FeatureVersionSpec


@dataclass(frozen=True)
class ModelDataSplit:
    """Train and test data prepared for model fitting."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_entity_ids: pd.Series
    test_entity_ids: pd.Series


@dataclass(frozen=True)
class ModelFitResult:
    """Results produced by model fitting and evaluation."""

    pipeline: Pipeline
    metrics: dict[str, float]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    transformed_feature_names: tuple[str, ...]
    training_seconds: float
    prediction_seconds: float


def normalize_model_features(
    dataframe: pd.DataFrame,
    specification: FeatureVersionSpec,
) -> pd.DataFrame:
    """Normalize model input types without learning from the data."""
    missing_columns = set(specification.features).difference(dataframe.columns)

    if missing_columns:
        raise ValueError(f"Model dataset is missing required features: {sorted(missing_columns)}")

    result = dataframe[list(specification.features)].copy()

    for column in specification.numerical_features:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).astype("float64")

    for column in specification.categorical_features:
        result[column] = result[column].astype("object")

        result[column] = result[column].where(
            result[column].notna(),
            np.nan,
        )

    return result


def build_model_data_split(
    dataframe: pd.DataFrame,
    membership: pd.DataFrame,
    specification: FeatureVersionSpec,
    target_column: str,
) -> ModelDataSplit:
    """Apply persisted membership and create model matrices."""
    entity_column = specification.entity_key

    required_dataset_columns = {
        entity_column,
        target_column,
        *specification.features,
    }

    missing_dataset_columns = required_dataset_columns.difference(dataframe.columns)

    if missing_dataset_columns:
        raise ValueError(f"Training dataset is missing columns: {sorted(missing_dataset_columns)}")

    required_membership_columns = {
        entity_column,
        "split",
    }

    missing_membership_columns = required_membership_columns.difference(membership.columns)

    if missing_membership_columns:
        raise ValueError(
            f"Split membership is missing columns: {sorted(missing_membership_columns)}"
        )

    if membership[entity_column].duplicated().any():
        raise ValueError("Split membership contains duplicate entity IDs.")

    merged = dataframe.merge(
        membership[
            [
                entity_column,
                "split",
            ]
        ],
        on=entity_column,
        how="left",
        validate="one_to_one",
    )

    if merged["split"].isna().any():
        raise ValueError("Some model records do not have split membership.")

    unexpected_splits = set(merged["split"].unique()).difference(
        {
            "train",
            "test",
        }
    )

    if unexpected_splits:
        raise ValueError(f"Unexpected split values found: {sorted(unexpected_splits)}")

    train_data = (
        merged.loc[merged["split"] == "train"].sort_values(entity_column).reset_index(drop=True)
    )

    test_data = (
        merged.loc[merged["split"] == "test"].sort_values(entity_column).reset_index(drop=True)
    )

    if train_data.empty or test_data.empty:
        raise ValueError("Both train and test partitions must contain data.")

    x_train = normalize_model_features(
        dataframe=train_data,
        specification=specification,
    )

    x_test = normalize_model_features(
        dataframe=test_data,
        specification=specification,
    )

    y_train = pd.to_numeric(
        train_data[target_column],
        errors="coerce",
    ).astype("float64")

    y_test = pd.to_numeric(
        test_data[target_column],
        errors="coerce",
    ).astype("float64")

    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("Training and test targets must not contain missing values.")

    return ModelDataSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        train_entity_ids=train_data[entity_column].astype("int64"),
        test_entity_ids=test_data[entity_column].astype("int64"),
    )


def sample_model_data_split(
    data_split: ModelDataSplit,
    max_train_rows: int,
    max_test_rows: int,
    random_state: int,
) -> ModelDataSplit:
    """Create a deterministic subset for an infrastructure smoke test."""
    train_count = min(
        int(max_train_rows),
        len(data_split.x_train),
    )

    test_count = min(
        int(max_test_rows),
        len(data_split.x_test),
    )

    train_indices = (
        data_split.x_train.sample(
            n=train_count,
            random_state=random_state,
        )
        .sort_index()
        .index
    )

    test_indices = (
        data_split.x_test.sample(
            n=test_count,
            random_state=random_state,
        )
        .sort_index()
        .index
    )

    return ModelDataSplit(
        x_train=(data_split.x_train.loc[train_indices].reset_index(drop=True)),
        x_test=(data_split.x_test.loc[test_indices].reset_index(drop=True)),
        y_train=(data_split.y_train.loc[train_indices].reset_index(drop=True)),
        y_test=(data_split.y_test.loc[test_indices].reset_index(drop=True)),
        train_entity_ids=(data_split.train_entity_ids.loc[train_indices].reset_index(drop=True)),
        test_entity_ids=(data_split.test_entity_ids.loc[test_indices].reset_index(drop=True)),
    )


def build_random_forest_pipeline(
    specification: FeatureVersionSpec,
    model_config: dict[str, Any],
    hyperparameters: dict[str, Any],
) -> Pipeline:
    """Build preprocessing and Random Forest regression steps."""
    algorithm = model_config["algorithm"]

    if algorithm != "RandomForestRegressor":
        raise ValueError(f"Unsupported algorithm configured: {algorithm!r}")

    preprocessing_config = model_config["preprocessing"]

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy=preprocessing_config["numerical_imputer_strategy"]),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy=preprocessing_config["categorical_imputer_strategy"]),
            ),
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown=preprocessing_config["one_hot_handle_unknown"],
                    sparse_output=bool(preprocessing_config["one_hot_sparse_output"]),
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                list(specification.numerical_features),
            ),
            (
                "categorical",
                categorical_pipeline,
                list(specification.categorical_features),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    estimator_parameters = {
        "random_state": int(model_config["random_state"]),
        "n_jobs": int(model_config["n_jobs"]),
        "criterion": model_config["criterion"],
        **hyperparameters,
    }

    estimator_parameters.pop(
        "description",
        None,
    )

    estimator = RandomForestRegressor(**estimator_parameters)

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                estimator,
            ),
        ]
    )


def calculate_regression_metrics(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """Calculate deterministic regression evaluation metrics."""
    actual_values = np.asarray(
        actual,
        dtype="float64",
    )

    predicted_values = np.asarray(
        predicted,
        dtype="float64",
    )

    mean_squared_error_value = mean_squared_error(
        actual_values,
        predicted_values,
    )

    return {
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error_value)),
        f"{prefix}_mae": float(
            mean_absolute_error(
                actual_values,
                predicted_values,
            )
        ),
        f"{prefix}_r2": float(
            r2_score(
                actual_values,
                predicted_values,
            )
        ),
    }


def fit_and_evaluate_pipeline(
    pipeline: Pipeline,
    data_split: ModelDataSplit,
    target_column: str,
) -> ModelFitResult:
    """Fit a model pipeline and evaluate train and test performance."""
    training_start = time.perf_counter()

    pipeline.fit(
        data_split.x_train,
        data_split.y_train,
    )

    training_seconds = time.perf_counter() - training_start

    prediction_start = time.perf_counter()

    train_predictions = pipeline.predict(data_split.x_train)

    test_predictions = pipeline.predict(data_split.x_test)

    prediction_seconds = time.perf_counter() - prediction_start

    metrics = {
        **calculate_regression_metrics(
            actual=data_split.y_train,
            predicted=train_predictions,
            prefix="train",
        ),
        **calculate_regression_metrics(
            actual=data_split.y_test,
            predicted=test_predictions,
            prefix="test",
        ),
        "training_seconds": float(training_seconds),
        "prediction_seconds": float(prediction_seconds),
    }

    predictions = pd.DataFrame(
        {
            "athlete_id": (data_split.test_entity_ids.astype("int64")),
            f"actual_{target_column}": (data_split.y_test.astype("float64")),
            f"predicted_{target_column}": (test_predictions.astype("float64")),
        }
    )

    predictions["residual"] = (
        predictions[f"actual_{target_column}"] - predictions[f"predicted_{target_column}"]
    )

    preprocessor = pipeline.named_steps["preprocessor"]

    transformed_feature_names = tuple(
        str(feature_name) for feature_name in preprocessor.get_feature_names_out()
    )

    model = pipeline.named_steps["model"]

    feature_importance = (
        pd.DataFrame(
            {
                "feature": transformed_feature_names,
                "importance": model.feature_importances_,
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return ModelFitResult(
        pipeline=pipeline,
        metrics=metrics,
        predictions=predictions,
        feature_importance=feature_importance,
        transformed_feature_names=(transformed_feature_names),
        training_seconds=float(training_seconds),
        prediction_seconds=float(prediction_seconds),
    )


def build_pipeline_metadata(
    pipeline: Pipeline,
    specification: FeatureVersionSpec,
    hyperparameter_name: str,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    """Create serializable metadata for a fitted pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]

    model = pipeline.named_steps["model"]

    transformed_feature_names = [
        str(feature_name) for feature_name in preprocessor.get_feature_names_out()
    ]

    return {
        "scikit_learn_version": sklearn.__version__,
        "algorithm": type(model).__name__,
        "feature_version": specification.version,
        "feature_view_name": specification.name,
        "source_features": list(specification.features),
        "numerical_features": list(specification.numerical_features),
        "categorical_features": list(specification.categorical_features),
        "source_feature_count": len(specification.features),
        "transformed_feature_count": len(transformed_feature_names),
        "transformed_feature_names": (transformed_feature_names),
        "hyperparameter_set": hyperparameter_name,
        "hyperparameters": {
            key: value for key, value in hyperparameters.items() if key != "description"
        },
        "hyperparameter_description": (
            hyperparameters.get(
                "description",
                "",
            )
        ),
    }
