"""Run Phase 3 preprocessing and label construction."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from athlete_mlops.data.ingestion import load_raw_dataset
from athlete_mlops.data.preprocessing import (
    build_processed_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"


def resolve_project_path(path_value: str) -> Path:
    """Resolve a configured repository-relative path."""
    return PROJECT_ROOT / path_value


def save_target_distribution(
    labels: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save the processed target distribution."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.hist(
        labels["total_lift"],
        bins=40,
        edgecolor="black",
    )

    axis.set_title("Processed Athlete Dataset — Total Lift Distribution")
    axis.set_xlabel("Total lift")
    axis.set_ylabel("Athlete count")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    """Execute the Phase 3 preprocessing stage."""
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    raw_path = resolve_project_path(config["paths"]["raw_data"])
    processed_path = resolve_project_path(config["paths"]["processed_data"])
    labels_path = resolve_project_path(config["paths"]["labels"])
    report_directory = resolve_project_path(config["paths"]["validation_reports"])
    figure_directory = resolve_project_path(config["paths"]["figures"])

    raw_data = load_raw_dataset(raw_path)

    result = build_processed_dataset(
        dataframe=raw_data,
        config=config["preprocessing"],
    )

    processed_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.processed_data.to_parquet(
        processed_path,
        index=False,
    )

    result.labels.to_parquet(
        labels_path,
        index=False,
    )

    summary_path = report_directory / "preprocessing_summary.json"
    invalid_values_path = report_directory / "invalid_value_counts.csv"
    processed_schema_path = report_directory / "processed_schema.csv"
    sentinel_values_path = report_directory / "target_sentinel_value_counts.csv"

    summary_path.write_text(
        json.dumps(result.summary, indent=2),
        encoding="utf-8",
    )

    result.invalid_value_counts.to_csv(
        invalid_values_path,
        index=False,
    )

    result.sentinel_value_counts.to_csv(
        sentinel_values_path,
        index=False,
    )

    processed_schema = pd.DataFrame(
        {
            "column": result.processed_data.columns,
            "data_type": (result.processed_data.dtypes.astype(str).values),
            "missing_count": (result.processed_data.isna().sum().values),
            "unique_count": (result.processed_data.nunique(dropna=True).values),
        }
    )

    processed_schema.to_csv(
        processed_schema_path,
        index=False,
    )

    save_target_distribution(
        labels=result.labels,
        output_path=(figure_directory / "processed_total_lift_distribution.png"),
    )

    print("Phase 3 preprocessing completed successfully.")
    print(f"Initial rows: {result.summary['initial_rows']:,}")
    print(f"Processed rows: {result.summary['processed_rows']:,}")
    print(
        "Rows removed for missing or invalid target: "
        f"{result.summary['missing_or_invalid_target_rows_removed']:,}"
    )
    print(f"Duplicate athlete rows removed: {result.summary['duplicate_entity_rows_removed']:,}")
    print(f"Target sentinel values replaced: {result.summary['target_sentinel_values_replaced']:,}")
    print(
        "Total lift range: "
        f"{result.summary['total_lift_min']:.0f}–"
        f"{result.summary['total_lift_max']:.0f}"
    )
    print(f"Processed data: {processed_path}")
    print(f"Labels: {labels_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
