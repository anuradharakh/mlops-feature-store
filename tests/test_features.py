"""Tests for athlete feature-version construction."""

from __future__ import annotations

import pandas as pd
import pytest

from athlete_mlops.features.builder import build_feature_table
from athlete_mlops.features.definitions import (
    build_feature_version_spec,
)

FEATURE_CONFIG = {
    "name": "athlete_features_v1",
    "description": "Stable baseline athlete features.",
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


def build_sample_processed_data() -> pd.DataFrame:
    """Create a representative processed dataset."""
    return pd.DataFrame(
        {
            "athlete_id": [1, 2],
            "event_timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-02T00:00:00Z",
                ],
                utc=True,
            ),
            "age": [25, 30],
            "weight": [180.0, None],
            "height": [70.0, 66.0],
            "gender": ["Male", "Female"],
            "region": ["Central", None],
            "deadlift": [400.0, 300.0],
            "candj": [225.0, 175.0],
            "snatch": [175.0, 125.0],
            "backsq": [350.0, 250.0],
            "total_lift": [1150.0, 850.0],
        }
    )


def build_v1_specification():
    """Build the validated test feature specification."""
    return build_feature_version_spec(
        version="v1",
        config=FEATURE_CONFIG,
    )


def test_feature_v1_has_expected_columns() -> None:
    """Version 1 should contain entity, time, and five features."""
    result = build_feature_table(
        processed_data=build_sample_processed_data(),
        specification=build_v1_specification(),
    )

    assert list(result.feature_data.columns) == [
        "athlete_id",
        "event_timestamp",
        "age",
        "weight",
        "height",
        "gender",
        "region",
    ]


def test_feature_v1_preserves_all_entities() -> None:
    """Feature creation should preserve the eligible population."""
    processed_data = build_sample_processed_data()

    result = build_feature_table(
        processed_data=processed_data,
        specification=build_v1_specification(),
    )

    assert len(result.feature_data) == len(processed_data)

    assert result.feature_data["athlete_id"].tolist() == [
        1,
        2,
    ]


def test_missing_predictor_values_are_preserved() -> None:
    """Feature generation should not impute predictor values."""
    result = build_feature_table(
        processed_data=build_sample_processed_data(),
        specification=build_v1_specification(),
    )

    athlete_two = result.feature_data.loc[result.feature_data["athlete_id"] == 2].iloc[0]

    assert pd.isna(athlete_two["weight"])
    assert pd.isna(athlete_two["region"])


def test_target_leakage_columns_are_excluded() -> None:
    """Target and target components must not enter the table."""
    result = build_feature_table(
        processed_data=build_sample_processed_data(),
        specification=build_v1_specification(),
    )

    leakage_columns = {
        "total_lift",
        "deadlift",
        "candj",
        "snatch",
        "backsq",
    }

    assert leakage_columns.isdisjoint(result.feature_data.columns)


def test_feature_definition_rejects_leakage() -> None:
    """A feature definition containing the target should fail."""
    invalid_config = {
        **FEATURE_CONFIG,
        "features": [
            *FEATURE_CONFIG["features"],
            "total_lift",
        ],
        "numerical_features": [
            *FEATURE_CONFIG["numerical_features"],
            "total_lift",
        ],
    }

    with pytest.raises(
        ValueError,
        match="target leakage",
    ):
        build_feature_version_spec(
            version="v1",
            config=invalid_config,
        )


def test_missing_required_feature_fails() -> None:
    """A missing required source feature should fail clearly."""
    processed_data = build_sample_processed_data().drop(columns=["region"])

    with pytest.raises(
        ValueError,
        match="missing required feature columns",
    ):
        build_feature_table(
            processed_data=processed_data,
            specification=build_v1_specification(),
        )


def test_duplicate_entity_fails() -> None:
    """Duplicate entity records should not enter the feature table."""
    processed_data = build_sample_processed_data()

    duplicate_row = processed_data.iloc[[0]].copy()

    processed_data = pd.concat(
        [processed_data, duplicate_row],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate entity keys",
    ):
        build_feature_table(
            processed_data=processed_data,
            specification=build_v1_specification(),
        )


def test_manifest_contains_version_metadata() -> None:
    """The manifest should document the feature artifact."""
    result = build_feature_table(
        processed_data=build_sample_processed_data(),
        specification=build_v1_specification(),
    )

    assert result.manifest["feature_version"] == "v1"

    assert result.manifest["feature_view_name"] == "athlete_features_v1"

    assert result.manifest["feature_count"] == 5
    assert result.manifest["row_count"] == 2
    assert result.manifest["entity_count"] == 2
    assert result.manifest["duplicate_entity_count"] == 0

    assert len(result.manifest["data_hash_sha256"]) == 64
    assert len(result.manifest["schema_hash_sha256"]) == 64


def test_feature_build_is_deterministic() -> None:
    """Repeated builds should produce identical features and hashes."""
    processed_data = build_sample_processed_data()
    specification = build_v1_specification()

    first_result = build_feature_table(
        processed_data=processed_data,
        specification=specification,
    )

    second_result = build_feature_table(
        processed_data=processed_data,
        specification=specification,
    )

    pd.testing.assert_frame_equal(
        first_result.feature_data,
        second_result.feature_data,
    )

    pd.testing.assert_frame_equal(
        first_result.missingness,
        second_result.missingness,
    )

    assert first_result.manifest["data_hash_sha256"] == second_result.manifest["data_hash_sha256"]

    assert (
        first_result.manifest["schema_hash_sha256"] == second_result.manifest["schema_hash_sha256"]
    )
