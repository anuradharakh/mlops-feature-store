"""Feature-version configuration and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureVersionSpec:
    """Validated definition of one feature version."""

    version: str
    name: str
    description: str
    entity_key: str
    timestamp_column: str
    features: tuple[str, ...]
    numerical_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    target_leakage_columns: tuple[str, ...]


def build_feature_version_spec(
    version: str,
    config: dict[str, Any],
) -> FeatureVersionSpec:
    """Create and validate a feature-version specification.

    Args:
        version: Logical feature version such as ``v1``.
        config: Feature-version configuration loaded from YAML.

    Returns:
        Validated immutable feature-version specification.

    Raises:
        ValueError: If the feature definition is inconsistent.
    """
    features = tuple(config["features"])
    numerical_features = tuple(config["numerical_features"])
    categorical_features = tuple(config["categorical_features"])
    leakage_columns = tuple(config["target_leakage_columns"])

    if not features:
        raise ValueError(f"Feature version {version} contains no features.")

    if len(features) != len(set(features)):
        raise ValueError(f"Feature version {version} contains duplicate features.")

    typed_features = set(numerical_features) | set(categorical_features)

    if typed_features != set(features):
        missing_type_assignments = set(features) - typed_features
        unexpected_type_assignments = typed_features - set(features)

        raise ValueError(
            "Numerical and categorical feature definitions must "
            "exactly match the feature list. "
            f"Missing assignments: {sorted(missing_type_assignments)}. "
            f"Unexpected assignments: "
            f"{sorted(unexpected_type_assignments)}."
        )

    overlapping_types = set(numerical_features) & set(categorical_features)

    if overlapping_types:
        raise ValueError(
            f"Features cannot be both numerical and categorical: {sorted(overlapping_types)}"
        )

    leakage_features = set(features) & set(leakage_columns)

    if leakage_features:
        raise ValueError(
            f"Feature definition contains target leakage columns: {sorted(leakage_features)}"
        )

    return FeatureVersionSpec(
        version=version,
        name=str(config["name"]),
        description=str(config["description"]).strip(),
        entity_key=str(config["entity_key"]),
        timestamp_column=str(config["timestamp_column"]),
        features=features,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target_leakage_columns=leakage_columns,
    )
