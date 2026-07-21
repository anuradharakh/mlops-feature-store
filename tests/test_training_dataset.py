"""Tests for versioned training datasets and deterministic splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from athlete_mlops.features.definitions import (
    build_feature_version_spec,
)
from athlete_mlops.training.dataset import (
    apply_split_membership,
    create_split_membership,
    validate_training_dataset,
    validate_version_alignment,
)

V1_CONFIG = {
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


V2_CONFIG = {
    "name": "athlete_features_v2",
    "description": "Enhanced features.",
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


def build_v1_dataset() -> pd.DataFrame:
    """Build a representative Version 1 training dataset."""
    athlete_ids = list(range(1, 21))

    return pd.DataFrame(
        {
            "athlete_id": athlete_ids,
            "event_timestamp": pd.date_range(
                "2024-01-01",
                periods=20,
                freq="h",
                tz="UTC",
            ),
            "total_lift": [float(500 + value * 20) for value in athlete_ids],
            "age": [float(20 + value) for value in athlete_ids],
            "weight": [float(140 + value) for value in athlete_ids],
            "height": [68.0] * 20,
            "gender": ["Male" if value % 2 else "Female" for value in athlete_ids],
            "region": ["Central"] * 20,
        }
    )


def build_v2_dataset() -> pd.DataFrame:
    """Build an aligned Version 2 training dataset."""
    dataframe = build_v1_dataset()

    dataframe["bmi"] = 703.0 * dataframe["weight"] / dataframe["height"].pow(2)

    dataframe["age_squared"] = dataframe["age"].pow(2)

    dataframe["weight_height_ratio"] = dataframe["weight"] / dataframe["height"]

    return dataframe


def build_v1_specification():
    """Build a Version 1 feature specification."""
    return build_feature_version_spec(
        version="v1",
        config=V1_CONFIG,
    )


def build_v2_specification():
    """Build a Version 2 feature specification."""
    return build_feature_version_spec(
        version="v2",
        config=V2_CONFIG,
    )


def test_training_dataset_validation_passes() -> None:
    """A valid Feast training dataset should be accepted."""
    result = validate_training_dataset(
        dataframe=build_v1_dataset(),
        specification=build_v1_specification(),
        target_column="total_lift",
        leakage_columns=[
            "deadlift",
            "candj",
            "snatch",
            "backsq",
        ],
    )

    assert len(result) == 20
    assert result["athlete_id"].is_unique
    assert result["total_lift"].notna().all()


def test_missing_required_feature_fails() -> None:
    """A missing model feature should fail validation."""
    dataframe = build_v1_dataset().drop(columns=["region"])

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_training_dataset(
            dataframe=dataframe,
            specification=build_v1_specification(),
            target_column="total_lift",
            leakage_columns=[
                "deadlift",
                "candj",
                "snatch",
                "backsq",
            ],
        )


def test_target_leakage_column_fails() -> None:
    """Target components must not enter training datasets."""
    dataframe = build_v1_dataset()
    dataframe["deadlift"] = 400.0

    with pytest.raises(
        ValueError,
        match="target leakage columns",
    ):
        validate_training_dataset(
            dataframe=dataframe,
            specification=build_v1_specification(),
            target_column="total_lift",
            leakage_columns=[
                "deadlift",
                "candj",
                "snatch",
                "backsq",
            ],
        )


def test_feature_versions_align() -> None:
    """Version 1 and Version 2 should use identical labels."""
    validate_version_alignment(
        version_one=build_v1_dataset(),
        version_two=build_v2_dataset(),
        entity_column="athlete_id",
        timestamp_column="event_timestamp",
        target_column="total_lift",
    )


def test_mismatched_targets_fail_alignment() -> None:
    """Different targets across feature versions should fail."""
    v2_data = build_v2_dataset()
    v2_data.loc[0, "total_lift"] = 9999.0

    with pytest.raises(
        ValueError,
        match="identical entity, timestamp, and target",
    ):
        validate_version_alignment(
            version_one=build_v1_dataset(),
            version_two=v2_data,
            entity_column="athlete_id",
            timestamp_column="event_timestamp",
            target_column="total_lift",
        )


def test_split_is_deterministic() -> None:
    """The same random state should reproduce the same split."""
    dataframe = build_v1_dataset()

    first = create_split_membership(
        dataframe=dataframe,
        entity_column="athlete_id",
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    second = create_split_membership(
        dataframe=dataframe,
        entity_column="athlete_id",
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    pd.testing.assert_frame_equal(
        first.membership,
        second.membership,
    )

    assert first.summary["split_hash_sha256"] == second.summary["split_hash_sha256"]


def test_split_has_no_overlap_and_full_coverage() -> None:
    """Train and test sets should be disjoint and complete."""
    dataframe = build_v1_dataset()

    result = create_split_membership(
        dataframe=dataframe,
        entity_column="athlete_id",
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    train_ids = set(result.train_ids)
    test_ids = set(result.test_ids)

    assert train_ids.isdisjoint(test_ids)

    assert train_ids.union(test_ids) == set(dataframe["athlete_id"])

    assert len(result.train_ids) == 16
    assert len(result.test_ids) == 4


def test_split_membership_applies_to_both_versions() -> None:
    """Both feature versions should receive identical split labels."""
    v1_data = build_v1_dataset()
    v2_data = build_v2_dataset()

    split_result = create_split_membership(
        dataframe=v1_data,
        entity_column="athlete_id",
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    v1_split = apply_split_membership(
        dataframe=v1_data,
        membership=split_result.membership,
        entity_column="athlete_id",
    )

    v2_split = apply_split_membership(
        dataframe=v2_data,
        membership=split_result.membership,
        entity_column="athlete_id",
    )

    assert v1_split[["athlete_id", "split"]].equals(v2_split[["athlete_id", "split"]])
