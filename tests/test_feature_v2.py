"""Tests for athlete Feature Version 2."""

from __future__ import annotations

import pandas as pd
import pytest

from athlete_mlops.features.builder import (
    build_feature_table,
)
from athlete_mlops.features.definitions import (
    build_feature_version_spec,
)
from athlete_mlops.features.engineering import (
    add_v2_engineered_features,
)

FEATURE_V2_CONFIG = {
    "name": "athlete_features_v2",
    "parent_version": "v1",
    "description": "Enhanced engineered athlete features.",
    "entity_key": "athlete_id",
    "timestamp_column": "event_timestamp",
    "features": [
        "age",
        "weight",
        "height",
        "gender",
        "region",
        "bmi",
        "age_squared",
        "weight_height_ratio",
    ],
    "numerical_features": [
        "age",
        "weight",
        "height",
        "bmi",
        "age_squared",
        "weight_height_ratio",
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
    """Create representative Phase 3 data."""
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
            "age": [25.0, 30.0],
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


def build_v2_specification():
    """Build the test Version 2 specification."""
    return build_feature_version_spec(
        version="v2",
        config=FEATURE_V2_CONFIG,
    )


def test_v2_engineered_values_are_correct() -> None:
    """Engineered features should follow configured formulas."""
    result = add_v2_engineered_features(build_sample_processed_data())

    athlete_one = result.engineered_data.loc[result.engineered_data["athlete_id"] == 1].iloc[0]

    expected_bmi = 703.0 * 180.0 / (70.0**2)

    assert athlete_one["bmi"] == pytest.approx(expected_bmi)

    assert athlete_one["age_squared"] == 625.0

    assert athlete_one["weight_height_ratio"] == pytest.approx(180.0 / 70.0)


def test_missing_values_propagate_to_derived_features() -> None:
    """Missing weight should produce missing derived values."""
    result = add_v2_engineered_features(build_sample_processed_data())

    athlete_two = result.engineered_data.loc[result.engineered_data["athlete_id"] == 2].iloc[0]

    assert pd.isna(athlete_two["bmi"])

    assert pd.isna(athlete_two["weight_height_ratio"])

    assert athlete_two["age_squared"] == 900.0


def test_engineering_does_not_mutate_input() -> None:
    """Feature engineering should not modify source data."""
    source = build_sample_processed_data()
    original = source.copy(deep=True)

    add_v2_engineered_features(source)

    pd.testing.assert_frame_equal(
        source,
        original,
    )


def test_v2_has_expected_columns() -> None:
    """Version 2 should contain eight model features."""
    engineering_result = add_v2_engineered_features(build_sample_processed_data())

    result = build_feature_table(
        processed_data=(engineering_result.engineered_data),
        specification=build_v2_specification(),
    )

    assert list(result.feature_data.columns) == [
        "athlete_id",
        "event_timestamp",
        "age",
        "weight",
        "height",
        "gender",
        "region",
        "bmi",
        "age_squared",
        "weight_height_ratio",
    ]


def test_v2_preserves_entity_population() -> None:
    """Version 2 must preserve all eligible athletes."""
    source = build_sample_processed_data()

    engineering_result = add_v2_engineered_features(source)

    result = build_feature_table(
        processed_data=(engineering_result.engineered_data),
        specification=build_v2_specification(),
    )

    assert len(result.feature_data) == len(source)

    assert result.feature_data["athlete_id"].tolist() == [1, 2]


def test_v2_excludes_target_leakage() -> None:
    """Version 2 must exclude target-related fields."""
    engineering_result = add_v2_engineered_features(build_sample_processed_data())

    result = build_feature_table(
        processed_data=(engineering_result.engineered_data),
        specification=build_v2_specification(),
    )

    leakage_columns = {
        "total_lift",
        "deadlift",
        "candj",
        "snatch",
        "backsq",
    }

    assert leakage_columns.isdisjoint(result.feature_data.columns)


def test_v2_manifest_contains_version_metadata() -> None:
    """Manifest should identify the enhanced version."""
    engineering_result = add_v2_engineered_features(build_sample_processed_data())

    result = build_feature_table(
        processed_data=(engineering_result.engineered_data),
        specification=build_v2_specification(),
    )

    assert result.manifest["feature_version"] == "v2"

    assert result.manifest["feature_view_name"] == "athlete_features_v2"

    assert result.manifest["feature_count"] == 8


def test_v2_build_is_deterministic() -> None:
    """Repeated Version 2 builds should be identical."""
    source = build_sample_processed_data()
    specification = build_v2_specification()

    first_engineering = add_v2_engineered_features(source)

    second_engineering = add_v2_engineered_features(source)

    first_result = build_feature_table(
        processed_data=(first_engineering.engineered_data),
        specification=specification,
    )

    second_result = build_feature_table(
        processed_data=(second_engineering.engineered_data),
        specification=specification,
    )

    pd.testing.assert_frame_equal(
        first_result.feature_data,
        second_result.feature_data,
    )

    pd.testing.assert_frame_equal(
        first_engineering.summary,
        second_engineering.summary,
    )

    assert first_result.manifest["data_hash_sha256"] == second_result.manifest["data_hash_sha256"]
