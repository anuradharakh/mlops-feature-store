"""Schema validation and profiling for the athlete dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

REQUIRED_COLUMNS = {
    "athlete_id",
    "age",
    "weight",
    "height",
    "gender",
    "region",
    "howlong",
    "eat",
    "background",
    "experience",
    "schedule",
    "deadlift",
    "candj",
    "snatch",
    "backsq",
}


def validate_required_columns(dataframe: pd.DataFrame) -> None:
    """Validate columns required by downstream pipeline stages.

    Args:
        dataframe: Raw athlete dataset.

    Raises:
        ValueError: If required columns are missing.
    """
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")


def build_overview(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Create the high-level raw dataset profile."""
    athlete_ids = dataframe["athlete_id"].dropna()

    duplicate_athlete_ids = int(athlete_ids.duplicated(keep=False).sum())

    return {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "memory_mb": round(
            dataframe.memory_usage(deep=True).sum() / (1024**2),
            2,
        ),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "total_missing_values": int(dataframe.isna().sum().sum()),
        "missing_cell_percentage": round(
            float(dataframe.isna().mean().mean() * 100),
            2,
        ),
        "athlete_id_missing": int(dataframe["athlete_id"].isna().sum()),
        "athlete_id_unique": int(athlete_ids.nunique()),
        "duplicate_athlete_id_records": duplicate_athlete_ids,
        "required_columns": sorted(REQUIRED_COLUMNS),
    }


def build_schema_report(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create a column-level schema and completeness report."""
    row_count = len(dataframe)

    report = pd.DataFrame(
        {
            "column": dataframe.columns,
            "data_type": dataframe.dtypes.astype(str).values,
            "non_null_count": dataframe.notna().sum().values,
            "missing_count": dataframe.isna().sum().values,
            "unique_count": dataframe.nunique(dropna=True).values,
        }
    )

    report["missing_percentage"] = report["missing_count"].div(row_count).mul(100).round(2)

    return report.sort_values(
        by="missing_percentage",
        ascending=False,
    ).reset_index(drop=True)


def build_numeric_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create descriptive statistics for numeric columns."""
    numeric_data = dataframe.select_dtypes(include="number")

    if numeric_data.empty:
        return pd.DataFrame()

    summary = numeric_data.describe().transpose().reset_index()
    summary = summary.rename(columns={"index": "column"})

    return summary.round(3)


def build_categorical_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create cardinality and dominant-value statistics."""
    categorical_data = dataframe.select_dtypes(include=["object", "string", "category"])

    records: list[dict[str, Any]] = []

    for column in categorical_data.columns:
        counts = categorical_data[column].value_counts(dropna=False)

        most_frequent_value = str(counts.index[0]) if not counts.empty else ""
        most_frequent_count = int(counts.iloc[0]) if not counts.empty else 0

        records.append(
            {
                "column": column,
                "unique_count": int(categorical_data[column].nunique(dropna=True)),
                "missing_count": int(categorical_data[column].isna().sum()),
                "most_frequent_value": most_frequent_value,
                "most_frequent_count": most_frequent_count,
            }
        )

    return pd.DataFrame(records).sort_values(
        by="unique_count",
        ascending=False,
    )


def save_missing_value_plot(
    schema_report: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """Save a chart of columns with the most missing values."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_data = (
        schema_report[["column", "missing_percentage"]]
        .head(top_n)
        .sort_values(
            by="missing_percentage",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.barh(
        plot_data["column"],
        plot_data["missing_percentage"],
    )
    axis.set_title("Raw Athlete Dataset — Missing Values")
    axis.set_xlabel("Missing values (%)")
    axis.set_ylabel("Column")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_profile_artifacts(
    dataframe: pd.DataFrame,
    report_directory: Path,
    figure_directory: Path,
) -> dict[str, Any]:
    """Create and save all raw-data profiling artifacts."""
    report_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)

    overview = build_overview(dataframe)
    schema_report = build_schema_report(dataframe)
    numeric_summary = build_numeric_summary(dataframe)
    categorical_summary = build_categorical_summary(dataframe)

    overview_path = report_directory / "raw_profile.json"
    schema_path = report_directory / "raw_schema.csv"
    missing_path = report_directory / "raw_missing_values.csv"
    numeric_path = report_directory / "raw_numeric_summary.csv"
    categorical_path = report_directory / "raw_categorical_summary.csv"
    missing_figure_path = figure_directory / "raw_missing_values.png"

    overview_path.write_text(
        json.dumps(overview, indent=2),
        encoding="utf-8",
    )

    schema_report.to_csv(schema_path, index=False)

    schema_report[
        [
            "column",
            "missing_count",
            "missing_percentage",
        ]
    ].to_csv(missing_path, index=False)

    numeric_summary.to_csv(numeric_path, index=False)
    categorical_summary.to_csv(
        categorical_path,
        index=False,
    )

    save_missing_value_plot(
        schema_report,
        missing_figure_path,
    )

    return overview
