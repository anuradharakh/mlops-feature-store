"""Validation and deterministic splitting of versioned training datasets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from athlete_mlops.features.definitions import FeatureVersionSpec


@dataclass(frozen=True)
class TrainingSplitResult:
    """Artifacts produced by the deterministic split operation."""

    membership: pd.DataFrame
    train_ids: tuple[int, ...]
    test_ids: tuple[int, ...]
    summary: dict[str, Any]


def calculate_split_hash(
    membership: pd.DataFrame,
) -> str:
    """Calculate a deterministic SHA-256 hash for split membership."""
    canonical_membership = membership.sort_values("athlete_id").reset_index(drop=True)

    hashed_values = pd.util.hash_pandas_object(
        canonical_membership,
        index=False,
    ).values

    return hashlib.sha256(hashed_values.tobytes()).hexdigest()


def validate_training_dataset(
    dataframe: pd.DataFrame,
    specification: FeatureVersionSpec,
    target_column: str,
    leakage_columns: list[str],
) -> pd.DataFrame:
    """Validate and canonicalize one Feast training dataset.

    Args:
        dataframe: Historical feature dataset retrieved from Feast.
        specification: Feature-version definition.
        target_column: Regression target column.
        leakage_columns: Target-component columns prohibited from features.

    Returns:
        Canonically ordered and validated training dataset.

    Raises:
        ValueError: If schema, entity, target, timestamp, or leakage
            validation fails.
    """
    required_columns = {
        specification.entity_key,
        specification.timestamp_column,
        target_column,
        *specification.features,
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Training dataset {specification.version} "
            "is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    present_leakage_columns = set(leakage_columns).intersection(dataframe.columns)

    if present_leakage_columns:
        raise ValueError(
            f"Training dataset {specification.version} "
            "contains target leakage columns: "
            f"{sorted(present_leakage_columns)}"
        )

    canonical_columns = [
        specification.entity_key,
        specification.timestamp_column,
        target_column,
        *specification.features,
    ]

    result = dataframe[canonical_columns].copy()

    result[specification.entity_key] = pd.to_numeric(
        result[specification.entity_key],
        errors="coerce",
    )

    result[specification.timestamp_column] = pd.to_datetime(
        result[specification.timestamp_column],
        errors="coerce",
        utc=True,
    )

    result[target_column] = pd.to_numeric(
        result[target_column],
        errors="coerce",
    )

    if result[specification.entity_key].isna().any():
        raise ValueError(f"Training dataset {specification.version} contains missing entity keys.")

    result[specification.entity_key] = result[specification.entity_key].astype("int64")

    if not result[specification.entity_key].is_unique:
        raise ValueError(
            f"Training dataset {specification.version} contains duplicate entity keys."
        )

    if result[specification.timestamp_column].isna().any():
        raise ValueError(
            f"Training dataset {specification.version} contains missing event timestamps."
        )

    if result[target_column].isna().any():
        raise ValueError(
            f"Training dataset {specification.version} contains missing target values."
        )

    result = result.sort_values(
        by=[
            specification.entity_key,
            specification.timestamp_column,
        ]
    ).reset_index(drop=True)

    return result


def validate_version_alignment(
    version_one: pd.DataFrame,
    version_two: pd.DataFrame,
    entity_column: str,
    timestamp_column: str,
    target_column: str,
) -> None:
    """Ensure both feature versions use identical entities and labels."""
    comparison_columns = [
        entity_column,
        timestamp_column,
        target_column,
    ]

    v1_comparison = (
        version_one[comparison_columns]
        .sort_values(
            by=[
                entity_column,
                timestamp_column,
            ]
        )
        .reset_index(drop=True)
    )

    v2_comparison = (
        version_two[comparison_columns]
        .sort_values(
            by=[
                entity_column,
                timestamp_column,
            ]
        )
        .reset_index(drop=True)
    )

    if not v1_comparison.equals(v2_comparison):
        raise ValueError(
            "Feature Versions 1 and 2 do not contain "
            "identical entity, timestamp, and target populations."
        )


def create_split_membership(
    dataframe: pd.DataFrame,
    entity_column: str,
    test_size: float,
    random_state: int,
    shuffle: bool,
) -> TrainingSplitResult:
    """Create one deterministic split shared by all experiments."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    entity_ids = dataframe[entity_column].astype("int64").drop_duplicates().to_numpy()

    train_ids_array, test_ids_array = train_test_split(
        entity_ids,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    train_ids = tuple(sorted(int(entity_id) for entity_id in train_ids_array))

    test_ids = tuple(sorted(int(entity_id) for entity_id in test_ids_array))

    membership = pd.concat(
        [
            pd.DataFrame(
                {
                    entity_column: train_ids,
                    "split": "train",
                }
            ),
            pd.DataFrame(
                {
                    entity_column: test_ids,
                    "split": "test",
                }
            ),
        ],
        ignore_index=True,
    )

    membership = membership.sort_values(entity_column).reset_index(drop=True)

    train_id_set = set(train_ids)
    test_id_set = set(test_ids)

    if train_id_set.intersection(test_id_set):
        raise ValueError("Train and test entity memberships overlap.")

    if train_id_set.union(test_id_set) != set(int(value) for value in entity_ids):
        raise ValueError("Train and test memberships do not cover the complete entity population.")

    if membership[entity_column].duplicated().any():
        raise ValueError("Split membership contains duplicate entity IDs.")

    summary = {
        "random_state": int(random_state),
        "test_size_requested": float(test_size),
        "shuffle": bool(shuffle),
        "total_entities": int(len(entity_ids)),
        "train_entities": int(len(train_ids)),
        "test_entities": int(len(test_ids)),
        "actual_test_fraction": round(
            len(test_ids) / len(entity_ids),
            6,
        ),
        "train_test_overlap": 0,
        "complete_population_coverage": True,
        "split_hash_sha256": calculate_split_hash(membership),
    }

    return TrainingSplitResult(
        membership=membership,
        train_ids=train_ids,
        test_ids=test_ids,
        summary=summary,
    )


def apply_split_membership(
    dataframe: pd.DataFrame,
    membership: pd.DataFrame,
    entity_column: str,
) -> pd.DataFrame:
    """Attach train/test membership to a training dataset."""
    if membership[entity_column].duplicated().any():
        raise ValueError("Split membership contains duplicate entity IDs.")

    result = dataframe.merge(
        membership,
        on=entity_column,
        how="left",
        validate="one_to_one",
    )

    if result["split"].isna().any():
        raise ValueError("Some training entities were not assigned to a train or test split.")

    unexpected_split_values = set(result["split"].unique()).difference(
        {
            "train",
            "test",
        }
    )

    if unexpected_split_values:
        raise ValueError(f"Unexpected split values were found: {sorted(unexpected_split_values)}")

    return result.sort_values(entity_column).reset_index(drop=True)


def calculate_target_statistics(
    dataframe: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    """Calculate reviewer-friendly target summary statistics."""
    target = pd.to_numeric(
        dataframe[target_column],
        errors="coerce",
    )

    return {
        "count": int(target.notna().sum()),
        "mean": round(float(target.mean()), 4),
        "median": round(float(target.median()), 4),
        "standard_deviation": round(
            float(target.std()),
            4,
        ),
        "minimum": float(target.min()),
        "maximum": float(target.max()),
    }


def build_missingness_report(
    dataframe: pd.DataFrame,
    specification: FeatureVersionSpec,
) -> pd.DataFrame:
    """Build a feature-level missingness report."""
    records = []

    for feature in specification.features:
        records.append(
            {
                "feature_version": specification.version,
                "feature": feature,
                "feature_type": (
                    "numerical" if feature in specification.numerical_features else "categorical"
                ),
                "missing_count": int(dataframe[feature].isna().sum()),
                "missing_percentage": round(
                    float(dataframe[feature].isna().mean() * 100),
                    2,
                ),
                "unique_non_null_values": int(dataframe[feature].nunique(dropna=True)),
            }
        )

    return pd.DataFrame(records)
