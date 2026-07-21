"""Version-specific athlete feature engineering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DERIVED_FEATURES_V2 = (
    "bmi",
    "age_squared",
    "weight_height_ratio",
)


@dataclass(frozen=True)
class FeatureEngineeringResult:
    """Artifacts produced by Version 2 feature engineering."""

    engineered_data: pd.DataFrame
    summary: pd.DataFrame


def add_v2_engineered_features(
    dataframe: pd.DataFrame,
) -> FeatureEngineeringResult:
    """Create deterministic Version 2 engineered features.

    Weight is interpreted as pounds and height as inches, based on the
    source dataset's documented and validated value ranges.

    Missing source values remain missing. Imputation is intentionally
    deferred to the training pipeline.

    Args:
        dataframe: Phase 3 processed athlete dataset.

    Returns:
        Dataset containing the Version 2 derived features and a summary.

    Raises:
        ValueError: If a required source column is missing.
    """
    required_columns = {
        "age",
        "weight",
        "height",
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Version 2 feature engineering is missing required "
            f"source columns: {sorted(missing_columns)}"
        )

    result = dataframe.copy()

    age = pd.to_numeric(
        result["age"],
        errors="coerce",
    )

    weight = pd.to_numeric(
        result["weight"],
        errors="coerce",
    )

    height = pd.to_numeric(
        result["height"],
        errors="coerce",
    )

    result["bmi"] = 703.0 * weight / height.pow(2)

    result["age_squared"] = age.pow(2)

    result["weight_height_ratio"] = weight / height

    for feature in DERIVED_FEATURES_V2:
        finite_mask = np.isfinite(result[feature].astype("float64"))

        result.loc[
            result[feature].notna() & ~finite_mask,
            feature,
        ] = np.nan

    summary_records = []

    for feature in DERIVED_FEATURES_V2:
        values = pd.to_numeric(
            result[feature],
            errors="coerce",
        )

        summary_records.append(
            {
                "feature": feature,
                "non_null_count": int(values.notna().sum()),
                "missing_count": int(values.isna().sum()),
                "missing_percentage": round(
                    float(values.isna().mean() * 100),
                    2,
                ),
                "minimum": (float(values.min()) if values.notna().any() else None),
                "maximum": (float(values.max()) if values.notna().any() else None),
                "mean": (round(float(values.mean()), 4) if values.notna().any() else None),
                "median": (round(float(values.median()), 4) if values.notna().any() else None),
            }
        )

    summary = pd.DataFrame(summary_records)

    return FeatureEngineeringResult(
        engineered_data=result,
        summary=summary,
    )
