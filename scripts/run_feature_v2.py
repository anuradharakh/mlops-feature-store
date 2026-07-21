"""Build Feature Version 2 for the athlete feature store."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from athlete_mlops.features.builder import build_feature_table
from athlete_mlops.features.definitions import (
    build_feature_version_spec,
)
from athlete_mlops.features.engineering import (
    add_v2_engineered_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"

FEATURE_CONFIG_PATH = PROJECT_ROOT / "configs" / "features.yaml"


def resolve_project_path(path_value: str) -> Path:
    """Resolve a repository-relative configured path."""
    return PROJECT_ROOT / path_value


def load_json(path: Path) -> dict:
    """Load a JSON object from disk."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    """Build, compare, and validate Feature Version 2."""
    with PIPELINE_CONFIG_PATH.open(encoding="utf-8") as config_file:
        pipeline_config = yaml.safe_load(config_file)

    with FEATURE_CONFIG_PATH.open(encoding="utf-8") as config_file:
        feature_config = yaml.safe_load(config_file)

    version_config = feature_config["feature_versions"]["v2"]

    specification = build_feature_version_spec(
        version="v2",
        config=version_config,
    )

    processed_path = resolve_project_path(pipeline_config["paths"]["processed_data"])

    feature_v1_path = resolve_project_path(pipeline_config["paths"]["feature_v1_data"])

    feature_v2_path = resolve_project_path(pipeline_config["paths"]["feature_v2_data"])

    report_directory = resolve_project_path(pipeline_config["paths"]["validation_reports"])

    feature_v1_manifest_path = report_directory / "feature_v1_manifest.json"

    required_inputs = [
        processed_path,
        feature_v1_path,
        feature_v1_manifest_path,
    ]

    missing_inputs = [path for path in required_inputs if not path.exists()]

    if missing_inputs:
        formatted_paths = "\n".join(f"- {path}" for path in missing_inputs)

        raise FileNotFoundError(
            "Feature Version 2 requires current Phase 3 "
            "and Phase 4 artifacts. Missing:\n"
            f"{formatted_paths}\n"
            "Run `python scripts/run_preprocessing.py` and "
            "`python scripts/run_feature_v1.py` first."
        )

    processed_data = pd.read_parquet(processed_path)

    feature_v1_data = pd.read_parquet(feature_v1_path)

    feature_v1_manifest = load_json(feature_v1_manifest_path)

    engineering_result = add_v2_engineered_features(dataframe=processed_data)

    build_result = build_feature_table(
        processed_data=(engineering_result.engineered_data),
        specification=specification,
    )

    feature_v2_data = build_result.feature_data

    alignment_columns = [
        specification.entity_key,
        specification.timestamp_column,
    ]

    v1_alignment = feature_v1_data[alignment_columns].reset_index(drop=True)

    v2_alignment = feature_v2_data[alignment_columns].reset_index(drop=True)

    same_entity_population = v1_alignment.equals(v2_alignment)

    if not same_entity_population:
        raise ValueError(
            "Feature Version 2 does not contain the same "
            "entity and timestamp population as Version 1."
        )

    manifest = {
        **build_result.manifest,
        "parent_feature_version": version_config.get("parent_version"),
        "transformations": list(
            version_config.get(
                "transformations",
                [],
            )
        ),
        "parent_data_hash_sha256": (feature_v1_manifest["data_hash_sha256"]),
        "same_entity_population_as_v1": (same_entity_population),
    }

    v1_features = list(feature_v1_manifest["features"])

    v2_features = list(manifest["features"])

    added_features = [feature for feature in v2_features if feature not in v1_features]

    removed_features = [feature for feature in v1_features if feature not in v2_features]

    comparison = {
        "baseline_version": "v1",
        "candidate_version": "v2",
        "baseline_feature_view": (feature_v1_manifest["feature_view_name"]),
        "candidate_feature_view": (manifest["feature_view_name"]),
        "v1_feature_count": len(v1_features),
        "v2_feature_count": len(v2_features),
        "v1_features": v1_features,
        "v2_features": v2_features,
        "added_features": added_features,
        "removed_features": removed_features,
        "same_row_count": (len(feature_v1_data) == len(feature_v2_data)),
        "same_entity_population": (same_entity_population),
        "v1_row_count": int(len(feature_v1_data)),
        "v2_row_count": int(len(feature_v2_data)),
        "v1_data_hash_sha256": (feature_v1_manifest["data_hash_sha256"]),
        "v2_data_hash_sha256": (manifest["data_hash_sha256"]),
    }

    feature_v2_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_v2_data.to_parquet(
        feature_v2_path,
        index=False,
    )

    manifest_path = report_directory / "feature_v2_manifest.json"

    missingness_path = report_directory / "feature_v2_missingness.csv"

    schema_path = report_directory / "feature_v2_schema.csv"

    engineering_summary_path = report_directory / "feature_v2_engineering_summary.csv"

    comparison_path = report_directory / "feature_version_comparison.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    build_result.missingness.to_csv(
        missingness_path,
        index=False,
    )

    build_result.schema.to_csv(
        schema_path,
        index=False,
    )

    engineering_result.summary.to_csv(
        engineering_summary_path,
        index=False,
    )

    comparison_path.write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )

    print("Feature Version 2 built successfully.")
    print(f"Feature view name: {specification.name}")
    print(f"Parent version: {manifest['parent_feature_version']}")
    print(f"Rows: {manifest['row_count']:,}")
    print(f"Entities: {manifest['entity_count']:,}")
    print(f"Feature count: {manifest['feature_count']}")
    print(f"Added features: {', '.join(added_features)}")
    print(f"Same entity population as v1: {same_entity_population}")
    print(f"Feature table: {feature_v2_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Comparison: {comparison_path}")
    print(f"Data hash: {manifest['data_hash_sha256']}")


if __name__ == "__main__":
    main()
