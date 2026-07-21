"""Deterministic feature-table construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from athlete_mlops.features.definitions import FeatureVersionSpec


@dataclass(frozen=True)
class FeatureBuildResult:
    """Artifacts generated for one feature version."""

    feature_data: pd.DataFrame
    manifest: dict[str, Any]
    missingness: pd.DataFrame
    schema: pd.DataFrame


def calculate_dataframe_hash(
    dataframe: pd.DataFrame,
) -> str:
    """Calculate a deterministic SHA-256 hash of a DataFrame."""
    hashed_values = pd.util.hash_pandas_object(
        dataframe,
        index=False,
    ).values

    return hashlib.sha256(hashed_values.tobytes()).hexdigest()


def calculate_schema_hash(
    dataframe: pd.DataFrame,
) -> str:
    """Calculate a deterministic hash of columns and data types."""
    schema_payload = [
        {
            "column": column,
            "data_type": str(dataframe[column].dtype),
        }
        for column in dataframe.columns
    ]

    serialized_schema = json.dumps(
        schema_payload,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(serialized_schema).hexdigest()


def build_feature_table(
    processed_data: pd.DataFrame,
    specification: FeatureVersionSpec,
) -> FeatureBuildResult:
    """Build and validate a feature table.

    Missing predictor values are intentionally preserved. Imputation and
    categorical encoding will occur inside the training-only Scikit-learn
    pipeline.

    Args:
        processed_data: Validated Phase 3 processed dataset.
        specification: Validated feature-version definition.

    Returns:
        Feature table and supporting validation artifacts.

    Raises:
        ValueError: If required columns, entity integrity, timestamps,
            or leakage protections fail.
    """
    required_columns = {
        specification.entity_key,
        specification.timestamp_column,
        *specification.features,
    }

    missing_columns = required_columns.difference(processed_data.columns)

    if missing_columns:
        raise ValueError(
            f"Processed dataset is missing required feature columns: {sorted(missing_columns)}"
        )

    leakage_features = set(specification.features) & set(specification.target_leakage_columns)

    if leakage_features:
        raise ValueError(
            f"Feature definition contains target leakage columns: {sorted(leakage_features)}"
        )

    output_columns = [
        specification.entity_key,
        specification.timestamp_column,
        *specification.features,
    ]

    feature_data = processed_data[output_columns].copy()

    feature_data = feature_data.sort_values(
        by=[
            specification.entity_key,
            specification.timestamp_column,
        ]
    ).reset_index(drop=True)

    if feature_data[specification.entity_key].isna().any():
        raise ValueError("Feature table contains missing entity keys.")

    if not feature_data[specification.entity_key].is_unique:
        raise ValueError("Feature table contains duplicate entity keys.")

    if feature_data[specification.timestamp_column].isna().any():
        raise ValueError("Feature table contains missing event timestamps.")

    missingness = pd.DataFrame(
        {
            "feature": specification.features,
            "feature_type": [
                ("numerical" if feature in specification.numerical_features else "categorical")
                for feature in specification.features
            ],
            "missing_count": [
                int(feature_data[feature].isna().sum()) for feature in specification.features
            ],
            "missing_percentage": [
                round(
                    float(feature_data[feature].isna().mean() * 100),
                    2,
                )
                for feature in specification.features
            ],
            "unique_non_null_values": [
                int(feature_data[feature].nunique(dropna=True))
                for feature in specification.features
            ],
        }
    )

    schema = pd.DataFrame(
        {
            "column": feature_data.columns,
            "data_type": [str(feature_data[column].dtype) for column in feature_data.columns],
            "nullable": [
                bool(feature_data[column].isna().any()) for column in feature_data.columns
            ],
            "unique_non_null_values": [
                int(feature_data[column].nunique(dropna=True)) for column in feature_data.columns
            ],
        }
    )

    manifest = {
        "feature_version": specification.version,
        "feature_view_name": specification.name,
        "description": specification.description,
        "entity_key": specification.entity_key,
        "timestamp_column": specification.timestamp_column,
        "features": list(specification.features),
        "numerical_features": list(specification.numerical_features),
        "categorical_features": list(specification.categorical_features),
        "feature_count": len(specification.features),
        "row_count": int(len(feature_data)),
        "entity_count": int(feature_data[specification.entity_key].nunique()),
        "duplicate_entity_count": int(feature_data[specification.entity_key].duplicated().sum()),
        "missing_feature_values": int(
            feature_data[list(specification.features)].isna().sum().sum()
        ),
        "schema_hash_sha256": calculate_schema_hash(feature_data),
        "data_hash_sha256": calculate_dataframe_hash(feature_data),
        "target_leakage_columns_excluded": list(specification.target_leakage_columns),
    }

    return FeatureBuildResult(
        feature_data=feature_data,
        manifest=manifest,
        missingness=missingness,
        schema=schema,
    )
