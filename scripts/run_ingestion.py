"""Run athlete ingestion, schema validation, and profiling."""

from __future__ import annotations

from pathlib import Path

import yaml

from athlete_mlops.data.ingestion import (
    load_raw_dataset,
    materialize_raw_csv,
)
from athlete_mlops.data.validation import (
    validate_required_columns,
    write_profile_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"


def resolve_project_path(path_value: str) -> Path:
    """Resolve a configured path relative to the repository."""
    return PROJECT_ROOT / path_value


def main() -> None:
    """Execute Phase 2 ingestion and raw-data profiling."""
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    source_path = resolve_project_path(config["paths"]["source_data"])
    raw_path = resolve_project_path(config["paths"]["raw_data"])
    report_directory = resolve_project_path(config["paths"]["validation_reports"])
    figure_directory = resolve_project_path(config["paths"]["figures"])

    materialize_raw_csv(
        source_path=source_path,
        output_path=raw_path,
    )

    raw_data = load_raw_dataset(raw_path)
    validate_required_columns(raw_data)

    overview = write_profile_artifacts(
        dataframe=raw_data,
        report_directory=report_directory,
        figure_directory=figure_directory,
    )

    print("Phase 2 ingestion completed successfully.")
    print(f"Source: {source_path}")
    print(f"Raw CSV: {raw_path}")
    print(f"Rows: {overview['rows']:,}")
    print(f"Columns: {overview['columns']}")
    print(
        "Missing values:",
        f"{overview['total_missing_values']:,}",
    )
    print(
        "Duplicate rows:",
        f"{overview['duplicate_rows']:,}",
    )
    print(
        "Unique athlete IDs:",
        f"{overview['athlete_id_unique']:,}",
    )
    print(f"Reports: {report_directory}")
    print(f"Figures: {figure_directory}")


if __name__ == "__main__":
    main()
