"""Deterministic preprocessing and label construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreprocessingResult:
    """Artifacts produced by the preprocessing stage."""

    processed_data: pd.DataFrame
    labels: pd.DataFrame
    summary: dict[str, Any]
    invalid_value_counts: pd.DataFrame
    sentinel_value_counts: pd.DataFrame


def normalize_text_columns(
    dataframe: pd.DataFrame,
    categorical_columns: list[str],
    invalid_response_values: list[str],
) -> pd.DataFrame:
    """Normalize categorical text and convert invalid responses to missing."""
    result = dataframe.copy()

    invalid_values = {str(value).strip().lower() for value in invalid_response_values}

    for column in categorical_columns:
        if column not in result.columns:
            continue

        normalized = result[column].astype("string").str.strip()

        invalid_mask = normalized.str.lower().isin(invalid_values)

        result[column] = normalized.mask(
            invalid_mask,
            pd.NA,
        )

    return result


def coerce_numeric_columns(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
) -> pd.DataFrame:
    """Convert configured numerical columns using safe coercion."""
    result = dataframe.copy()

    for column in numeric_columns:
        if column not in result.columns:
            continue

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    return result


def replace_target_sentinel_values(
    dataframe: pd.DataFrame,
    target_components: list[str],
    sentinel_values: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace known target placeholder values with missing values.

    Args:
        dataframe: Input athlete dataset.
        target_components: Lift columns used to create the target.
        sentinel_values: Values representing invalid placeholders.

    Returns:
        The cleaned DataFrame and a replacement-count report.
    """
    result = dataframe.copy()
    records: list[dict[str, Any]] = []

    sentinel_set = set(sentinel_values)

    for column in target_components:
        if column not in result.columns:
            continue

        sentinel_mask = result[column].isin(sentinel_set)
        replacement_count = int(sentinel_mask.sum())

        records.append(
            {
                "column": column,
                "sentinel_values": sorted(sentinel_set),
                "sentinel_replacement_count": replacement_count,
            }
        )

        result.loc[sentinel_mask, column] = np.nan

    report = pd.DataFrame(
        records,
        columns=[
            "column",
            "sentinel_values",
            "sentinel_replacement_count",
        ],
    )

    return result, report


def apply_plausibility_ranges(
    dataframe: pd.DataFrame,
    plausible_ranges: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace values outside configured ranges with missing values."""
    result = dataframe.copy()
    records: list[dict[str, Any]] = []

    for column, bounds in plausible_ranges.items():
        if column not in result.columns:
            continue

        minimum = float(bounds["min"])
        maximum = float(bounds["max"])

        numeric_values = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        invalid_mask = numeric_values.notna() & ~numeric_values.between(
            minimum,
            maximum,
            inclusive="both",
        )

        records.append(
            {
                "column": column,
                "minimum_allowed": minimum,
                "maximum_allowed": maximum,
                "invalid_value_count": int(invalid_mask.sum()),
            }
        )

        result.loc[invalid_mask, column] = np.nan

    report = pd.DataFrame(
        records,
        columns=[
            "column",
            "minimum_allowed",
            "maximum_allowed",
            "invalid_value_count",
        ],
    )

    return result, report


def create_event_timestamp(
    dataframe: pd.DataFrame,
    source_timestamp_column: str,
    event_timestamp_column: str,
) -> pd.DataFrame:
    """Create a complete UTC event timestamp required by Feast."""
    result = dataframe.copy()

    fallback_timestamps = pd.date_range(
        start="2024-01-01 00:00:00",
        periods=len(result),
        freq="s",
        tz="UTC",
    )

    fallback_series = pd.Series(
        fallback_timestamps,
        index=result.index,
    )

    if source_timestamp_column in result.columns:
        parsed_timestamps = pd.to_datetime(
            result[source_timestamp_column],
            errors="coerce",
            utc=True,
        )

        result[event_timestamp_column] = parsed_timestamps.fillna(fallback_series)
    else:
        result[event_timestamp_column] = fallback_series

    return result


def build_processed_dataset(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
) -> PreprocessingResult:
    """Clean athlete data and construct the total-lift target."""
    initial_rows = len(dataframe)

    identifier_column = config["identifier_column"]
    source_timestamp_column = config["source_timestamp_column"]
    event_timestamp_column = config["event_timestamp_column"]

    target_components = list(config["target_components"])
    numeric_columns = list(config["numeric_columns"])
    categorical_columns = list(config["categorical_columns"])
    sentinel_values = list(config.get("target_sentinel_values", []))

    processed = dataframe.copy()

    processed = normalize_text_columns(
        dataframe=processed,
        categorical_columns=categorical_columns,
        invalid_response_values=list(config["invalid_response_values"]),
    )

    processed = coerce_numeric_columns(
        dataframe=processed,
        numeric_columns=numeric_columns,
    )

    processed, sentinel_value_counts = replace_target_sentinel_values(
        dataframe=processed,
        target_components=target_components,
        sentinel_values=sentinel_values,
    )

    processed, invalid_value_counts = apply_plausibility_ranges(
        dataframe=processed,
        plausible_ranges=config["plausible_ranges"],
    )

    processed = create_event_timestamp(
        dataframe=processed,
        source_timestamp_column=source_timestamp_column,
        event_timestamp_column=event_timestamp_column,
    )

    missing_identifier_rows = int(processed[identifier_column].isna().sum())

    processed = processed.dropna(subset=[identifier_column]).copy()

    processed[identifier_column] = processed[identifier_column].astype("int64")

    processed = processed.sort_values(
        by=[
            identifier_column,
            event_timestamp_column,
        ]
    )

    duplicate_entity_rows = int(
        processed.duplicated(
            subset=[identifier_column],
            keep="last",
        ).sum()
    )

    processed = processed.drop_duplicates(
        subset=[identifier_column],
        keep="last",
    ).copy()

    missing_target_mask = processed[target_components].isna().any(axis=1)

    missing_target_rows = int(missing_target_mask.sum())

    processed = processed.loc[~missing_target_mask].copy()

    processed["total_lift"] = processed[target_components].sum(axis=1)

    processed = processed.sort_values(by=identifier_column).reset_index(drop=True)

    labels = processed[
        [
            identifier_column,
            event_timestamp_column,
            "total_lift",
        ]
    ].copy()

    labels = labels.reset_index(drop=True)

    target_sentinel_values_replaced = int(sentinel_value_counts["sentinel_replacement_count"].sum())

    predictor_data = processed.drop(
        columns=[
            *target_components,
            "total_lift",
        ],
        errors="ignore",
    )

    summary = {
        "initial_rows": int(initial_rows),
        "missing_identifier_rows_removed": (missing_identifier_rows),
        "duplicate_entity_rows_removed": (duplicate_entity_rows),
        "missing_or_invalid_target_rows_removed": (missing_target_rows),
        "processed_rows": int(len(processed)),
        "label_rows": int(len(labels)),
        "processed_columns": int(processed.shape[1]),
        "target_sentinel_values": sentinel_values,
        "target_sentinel_values_replaced": (target_sentinel_values_replaced),
        "total_lift_min": float(processed["total_lift"].min()),
        "total_lift_max": float(processed["total_lift"].max()),
        "total_lift_mean": round(
            float(processed["total_lift"].mean()),
            3,
        ),
        "total_lift_median": round(
            float(processed["total_lift"].median()),
            3,
        ),
        "remaining_missing_predictor_values": int(predictor_data.isna().sum().sum()),
        "target_components_excluded_from_features": (target_components),
    }

    return PreprocessingResult(
        processed_data=processed,
        labels=labels,
        summary=summary,
        invalid_value_counts=invalid_value_counts,
        sentinel_value_counts=sentinel_value_counts,
    )
