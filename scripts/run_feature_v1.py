"""Build Feature Version 1 for the athlete feature store."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from athlete_mlops.features.builder import build_feature_table
from athlete_mlops.features.definitions import (
    build_feature_version_spec,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"
FEATURE_CONFIG_PATH = PROJECT_ROOT / "configs" / "features.yaml"


def resolve_project_path(path_value: str) -> Path:
    """Resolve a repository-relative configured path."""
    return PROJECT_ROOT / path_value


def main() -> None:
    """Build and validate the Version 1 feature table."""
    with PIPELINE_CONFIG_PATH.open(encoding="utf-8") as config_file:
        pipeline_config = yaml.safe_load(config_file)

    with FEATURE_CONFIG_PATH.open(encoding="utf-8") as config_file:
        feature_config = yaml.safe_load(config_file)

    specification = build_feature_version_spec(
        version="v1",
        config=feature_config["feature_versions"]["v1"],
    )

    processed_path = resolve_project_path(pipeline_config["paths"]["processed_data"])
    output_path = resolve_project_path(pipeline_config["paths"]["feature_v1_data"])
    report_directory = resolve_project_path(pipeline_config["paths"]["validation_reports"])

    if not processed_path.exists():
        raise FileNotFoundError(
            "Processed dataset was not found. "
            "Run `python scripts/run_preprocessing.py` first. "
            f"Missing path: {processed_path}"
        )

    processed_data = pd.read_parquet(processed_path)

    result = build_feature_table(
        processed_data=processed_data,
        specification=specification,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.feature_data.to_parquet(
        output_path,
        index=False,
    )

    manifest_path = report_directory / "feature_v1_manifest.json"
    missingness_path = report_directory / "feature_v1_missingness.csv"
    schema_path = report_directory / "feature_v1_schema.csv"

    manifest_path.write_text(
        json.dumps(result.manifest, indent=2),
        encoding="utf-8",
    )

    result.missingness.to_csv(
        missingness_path,
        index=False,
    )

    result.schema.to_csv(
        schema_path,
        index=False,
    )

    print("Feature Version 1 built successfully.")
    print(f"Feature view name: {specification.name}")
    print(f"Rows: {result.manifest['row_count']:,}")
    print(f"Entities: {result.manifest['entity_count']:,}")
    print(f"Feature count: {result.manifest['feature_count']}")
    print(f"Missing feature values: {result.manifest['missing_feature_values']:,}")
    print(f"Feature table: {output_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Data hash: {result.manifest['data_hash_sha256']}")


if __name__ == "__main__":
    main()
