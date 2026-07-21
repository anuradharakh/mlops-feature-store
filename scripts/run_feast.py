"""Apply, validate, materialize, and query the Feast repository."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import feast
import numpy as np
import pandas as pd
from feast import FeatureStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_REPO = PROJECT_ROOT / "feature_repo"
FEAST_DATA_DIRECTORY = FEATURE_REPO / "data"

LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "athlete_labels.parquet"

FEATURE_V1_PATH = PROJECT_ROOT / "data" / "features" / "v1" / "athlete_features_v1.parquet"

FEATURE_V2_PATH = PROJECT_ROOT / "data" / "features" / "v2" / "athlete_features_v2.parquet"

TRAINING_DIRECTORY = PROJECT_ROOT / "data" / "training"
REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "feast"

REGISTRY_PATH = FEAST_DATA_DIRECTORY / "registry.db"
ONLINE_STORE_PATH = FEAST_DATA_DIRECTORY / "online_store.db"

V1_FEATURES = [
    "age",
    "weight",
    "height",
    "gender",
    "region",
]

V2_FEATURES = [
    *V1_FEATURES,
    "bmi",
    "age_squared",
    "weight_height_ratio",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the local Feast workflow.")

    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete generated Feast registry and online-store state before applying definitions."
        ),
    )

    return parser.parse_args()


def remove_generated_state() -> None:
    """Remove local Feast registry and online SQLite files."""
    generated_paths = [
        REGISTRY_PATH,
        ONLINE_STORE_PATH,
        Path(f"{ONLINE_STORE_PATH}-shm"),
        Path(f"{ONLINE_STORE_PATH}-wal"),
    ]

    for path in generated_paths:
        if path.exists():
            path.unlink()


def run_feast_apply() -> str:
    """Apply committed Feast definitions."""
    feast_executable = shutil.which("feast")

    if feast_executable is None:
        raise RuntimeError("The Feast CLI was not found in the active Python environment.")

    completed = subprocess.run(
        [feast_executable, "apply"],
        cwd=FEATURE_REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    combined_output = "\n".join(
        section
        for section in [
            completed.stdout.strip(),
            completed.stderr.strip(),
        ]
        if section
    )

    if completed.returncode != 0:
        raise RuntimeError(f"Feast apply failed.\n{combined_output}")

    return combined_output


def get_entity_join_keys(entity: Any) -> list[str]:
    """Return entity join keys across Feast API representations."""
    plural_join_keys = getattr(
        entity,
        "join_keys",
        None,
    )

    if plural_join_keys:
        if isinstance(plural_join_keys, str):
            return [plural_join_keys]

        return [str(join_key) for join_key in plural_join_keys]

    singular_join_key = getattr(
        entity,
        "join_key",
        None,
    )

    if singular_join_key is None:
        raise AttributeError(f"Entity {entity.name!r} does not expose 'join_key' or 'join_keys'.")

    if isinstance(
        singular_join_key,
        (list, tuple, set),
    ):
        return sorted(str(join_key) for join_key in singular_join_key)

    return [str(singular_join_key)]


def convert_to_json_safe(value: Any) -> Any:
    """Convert pandas and NumPy values to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): convert_to_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [convert_to_json_safe(item) for item in value]

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return [convert_to_json_safe(item) for item in value.tolist()]

    if isinstance(value, np.generic):
        return convert_to_json_safe(value.item())

    if value is pd.NA:
        return None

    if isinstance(value, float) and np.isnan(value):
        return None

    return value


def build_registry_summary(
    store: FeatureStore,
) -> dict[str, Any]:
    """Create a deterministic summary of Feast registry objects."""
    entities = sorted(
        store.list_entities(),
        key=lambda item: item.name,
    )

    feature_views = sorted(
        store.list_feature_views(),
        key=lambda item: item.name,
    )

    feature_services = sorted(
        store.list_feature_services(),
        key=lambda item: item.name,
    )

    return {
        "project": store.project,
        "feast_version": feast.__version__,
        "entities": [
            {
                "name": entity.name,
                "join_keys": get_entity_join_keys(entity),
                "description": (entity.description or ""),
            }
            for entity in entities
        ],
        "feature_views": [
            {
                "name": feature_view.name,
                "entities": sorted(str(entity_name) for entity_name in feature_view.entities),
                "features": sorted(field.name for field in feature_view.schema),
                "online": bool(feature_view.online),
                "tags": dict(feature_view.tags or {}),
            }
            for feature_view in feature_views
        ],
        "feature_services": [
            {
                "name": feature_service.name,
                "tags": dict(feature_service.tags or {}),
            }
            for feature_service in feature_services
        ],
        "entity_count": len(entities),
        "feature_view_count": len(feature_views),
        "feature_service_count": len(feature_services),
    }


def retrieve_historical_features(
    store: FeatureStore,
    labels: pd.DataFrame,
    service_name: str,
    expected_features: list[str],
) -> pd.DataFrame:
    """Retrieve point-in-time features for all label rows."""
    entity_dataframe = labels[
        [
            "athlete_id",
            "event_timestamp",
            "total_lift",
        ]
    ].copy()

    feature_service = store.get_feature_service(service_name)

    historical_data = store.get_historical_features(
        entity_df=entity_dataframe,
        features=feature_service,
    ).to_df()

    expected_columns = {
        "athlete_id",
        "event_timestamp",
        "total_lift",
        *expected_features,
    }

    missing_columns = expected_columns.difference(historical_data.columns)

    if missing_columns:
        raise ValueError(
            f"{service_name} historical retrieval is missing columns: {sorted(missing_columns)}"
        )

    if len(historical_data) != len(labels):
        raise ValueError(
            f"{service_name} returned {len(historical_data):,} rows; expected {len(labels):,}."
        )

    expected_entity_count = labels["athlete_id"].nunique()

    actual_entity_count = historical_data["athlete_id"].nunique()

    if actual_entity_count != expected_entity_count:
        raise ValueError(
            f"{service_name} returned "
            f"{actual_entity_count:,} unique entities; "
            f"expected {expected_entity_count:,}."
        )

    historical_data = historical_data.sort_values(
        by=[
            "athlete_id",
            "event_timestamp",
        ]
    ).reset_index(drop=True)

    return historical_data


def select_online_sample(
    feature_v2_data: pd.DataFrame,
) -> list[int]:
    """Select complete entities for online retrieval evidence."""
    complete_rows = feature_v2_data.dropna(subset=V2_FEATURES)

    if len(complete_rows) < 5:
        raise ValueError(
            "At least five complete Version 2 rows are required for online validation."
        )

    sample_ids = complete_rows["athlete_id"].drop_duplicates().head(5).astype("int64").tolist()

    if len(sample_ids) != 5:
        raise ValueError("Unable to identify five distinct athletes for online validation.")

    return [int(athlete_id) for athlete_id in sample_ids]


def retrieve_online_features(
    store: FeatureStore,
    service_name: str,
    athlete_ids: list[int],
) -> dict[str, Any]:
    """Retrieve latest online features for selected athletes."""
    feature_service = store.get_feature_service(service_name)

    entity_rows = [
        {
            "athlete_id": athlete_id,
        }
        for athlete_id in athlete_ids
    ]

    response = store.get_online_features(
        features=feature_service,
        entity_rows=entity_rows,
    ).to_dict()

    return convert_to_json_safe(response)


def validate_online_response(
    response: dict[str, Any],
    expected_features: list[str],
    athlete_ids: list[int],
) -> None:
    """Validate a returned online feature vector."""
    expected_columns = {
        "athlete_id",
        *expected_features,
    }

    missing_columns = expected_columns.difference(response.keys())

    if missing_columns:
        raise ValueError(f"Online retrieval is missing fields: {sorted(missing_columns)}")

    returned_ids = [int(value) for value in response["athlete_id"]]

    if returned_ids != athlete_ids:
        raise ValueError(
            "Online retrieval returned an unexpected "
            "entity order or population. "
            f"Expected {athlete_ids}; "
            f"received {returned_ids}."
        )

    for feature in expected_features:
        values = response[feature]

        if len(values) != len(athlete_ids):
            raise ValueError(
                f"Online feature {feature!r} returned "
                f"{len(values)} values; expected "
                f"{len(athlete_ids)}."
            )


def validate_upstream_alignment(
    labels: pd.DataFrame,
    feature_v1_data: pd.DataFrame,
    feature_v2_data: pd.DataFrame,
) -> None:
    """Validate Phase 3–5 artifact population alignment."""
    expected_label_columns = {
        "athlete_id",
        "event_timestamp",
        "total_lift",
    }

    missing_label_columns = expected_label_columns.difference(labels.columns)

    if missing_label_columns:
        raise ValueError(f"Label artifact is missing columns: {sorted(missing_label_columns)}")

    for version, feature_data in [
        ("v1", feature_v1_data),
        ("v2", feature_v2_data),
    ]:
        required_columns = {
            "athlete_id",
            "event_timestamp",
        }

        missing_columns = required_columns.difference(feature_data.columns)

        if missing_columns:
            raise ValueError(f"Feature {version} is missing columns: {sorted(missing_columns)}")

        if len(feature_data) != len(labels):
            raise ValueError(
                f"Feature {version} contains "
                f"{len(feature_data):,} rows; labels "
                f"contain {len(labels):,} rows."
            )

        if not feature_data["athlete_id"].is_unique:
            raise ValueError(f"Feature {version} contains duplicate athlete IDs.")

    label_alignment = labels[
        [
            "athlete_id",
            "event_timestamp",
        ]
    ].reset_index(drop=True)

    v1_alignment = feature_v1_data[
        [
            "athlete_id",
            "event_timestamp",
        ]
    ].reset_index(drop=True)

    v2_alignment = feature_v2_data[
        [
            "athlete_id",
            "event_timestamp",
        ]
    ].reset_index(drop=True)

    if not label_alignment.equals(v1_alignment):
        raise ValueError("Version 1 entity/timestamp population is not aligned with the labels.")

    if not label_alignment.equals(v2_alignment):
        raise ValueError("Version 2 entity/timestamp population is not aligned with the labels.")


def main() -> None:
    """Run the complete local Feast workflow."""
    args = parse_args()

    required_inputs = [
        LABELS_PATH,
        FEATURE_V1_PATH,
        FEATURE_V2_PATH,
        FEATURE_REPO / "feature_store.yaml",
        FEATURE_REPO / "feature_definitions.py",
    ]

    missing_inputs = [path for path in required_inputs if not path.exists()]

    if missing_inputs:
        formatted_paths = "\n".join(f"- {path}" for path in missing_inputs)

        raise FileNotFoundError(f"Required Feast inputs are missing:\n{formatted_paths}")

    FEAST_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    TRAINING_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.reset:
        remove_generated_state()

    apply_output = run_feast_apply()

    apply_output_path = REPORT_DIRECTORY / "feast_apply_output.txt"

    apply_output_path.write_text(
        apply_output + "\n",
        encoding="utf-8",
    )

    store = FeatureStore(repo_path=str(FEATURE_REPO))

    registry_summary = build_registry_summary(store)

    registry_summary_path = REPORT_DIRECTORY / "feature_registry_summary.json"

    registry_summary_path.write_text(
        json.dumps(
            convert_to_json_safe(registry_summary),
            indent=2,
        ),
        encoding="utf-8",
    )

    labels = (
        pd.read_parquet(LABELS_PATH)
        .sort_values(
            by=[
                "athlete_id",
                "event_timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    feature_v1_data = (
        pd.read_parquet(FEATURE_V1_PATH)
        .sort_values(
            by=[
                "athlete_id",
                "event_timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    feature_v2_data = (
        pd.read_parquet(FEATURE_V2_PATH)
        .sort_values(
            by=[
                "athlete_id",
                "event_timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    validate_upstream_alignment(
        labels=labels,
        feature_v1_data=feature_v1_data,
        feature_v2_data=feature_v2_data,
    )

    historical_v1 = retrieve_historical_features(
        store=store,
        labels=labels,
        service_name="athlete_strength_v1",
        expected_features=V1_FEATURES,
    )

    historical_v2 = retrieve_historical_features(
        store=store,
        labels=labels,
        service_name="athlete_strength_v2",
        expected_features=V2_FEATURES,
    )

    historical_v1_path = TRAINING_DIRECTORY / "athlete_training_v1.parquet"

    historical_v2_path = TRAINING_DIRECTORY / "athlete_training_v2.parquet"

    historical_v1.to_parquet(
        historical_v1_path,
        index=False,
    )

    historical_v2.to_parquet(
        historical_v2_path,
        index=False,
    )

    historical_v1.head(25).to_csv(
        REPORT_DIRECTORY / "historical_v1_sample.csv",
        index=False,
    )

    historical_v2.head(25).to_csv(
        REPORT_DIRECTORY / "historical_v2_sample.csv",
        index=False,
    )

    all_timestamps = pd.concat(
        [
            feature_v1_data["event_timestamp"],
            feature_v2_data["event_timestamp"],
        ],
        ignore_index=True,
    )

    all_timestamps = pd.to_datetime(
        all_timestamps,
        utc=True,
    )

    materialization_start = (all_timestamps.min() - pd.Timedelta(seconds=1)).to_pydatetime()

    materialization_end = (all_timestamps.max() + pd.Timedelta(seconds=1)).to_pydatetime()

    store.materialize(
        materialization_start,
        materialization_end,
    )

    online_athlete_ids = select_online_sample(feature_v2_data)

    online_v1 = retrieve_online_features(
        store=store,
        service_name="athlete_strength_v1",
        athlete_ids=online_athlete_ids,
    )

    online_v2 = retrieve_online_features(
        store=store,
        service_name="athlete_strength_v2",
        athlete_ids=online_athlete_ids,
    )

    validate_online_response(
        response=online_v1,
        expected_features=V1_FEATURES,
        athlete_ids=online_athlete_ids,
    )

    validate_online_response(
        response=online_v2,
        expected_features=V2_FEATURES,
        athlete_ids=online_athlete_ids,
    )

    online_evidence = {
        "athlete_ids": online_athlete_ids,
        "athlete_strength_v1": online_v1,
        "athlete_strength_v2": online_v2,
    }

    online_evidence_path = REPORT_DIRECTORY / "online_retrieval_sample.json"

    online_evidence_path.write_text(
        json.dumps(
            online_evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    historical_population_aligned = bool(
        historical_v1["athlete_id"].equals(historical_v2["athlete_id"])
    )

    validation_summary = {
        "status": "PASS",
        "feast_version": feast.__version__,
        "project": store.project,
        "entity_name": "athlete",
        "entity_join_key": "athlete_id",
        "feature_views": [
            "athlete_features_v1",
            "athlete_features_v2",
        ],
        "feature_services": [
            "athlete_strength_v1",
            "athlete_strength_v2",
        ],
        "v1_historical_rows": int(len(historical_v1)),
        "v2_historical_rows": int(len(historical_v2)),
        "v1_historical_feature_count": len(V1_FEATURES),
        "v2_historical_feature_count": len(V2_FEATURES),
        "historical_population_aligned": (historical_population_aligned),
        "materialization_start": (materialization_start.isoformat()),
        "materialization_end": (materialization_end.isoformat()),
        "online_sample_entity_count": len(online_athlete_ids),
        "online_v1_retrieval_passed": True,
        "online_v2_retrieval_passed": True,
        "generated_training_artifacts": [
            str(historical_v1_path.relative_to(PROJECT_ROOT)),
            str(historical_v2_path.relative_to(PROJECT_ROOT)),
        ],
        "committed_review_artifacts": [
            str(apply_output_path.relative_to(PROJECT_ROOT)),
            str(registry_summary_path.relative_to(PROJECT_ROOT)),
            ("reports/feast/historical_v1_sample.csv"),
            ("reports/feast/historical_v2_sample.csv"),
            str(online_evidence_path.relative_to(PROJECT_ROOT)),
        ],
    }

    validation_summary_path = REPORT_DIRECTORY / "feast_validation_summary.json"

    validation_summary_path.write_text(
        json.dumps(
            validation_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Phase 6 Feast workflow completed successfully.")
    print(f"Feast version: {feast.__version__}")
    print(f"Project: {store.project}")
    print("Feature views: athlete_features_v1, athlete_features_v2")
    print("Feature services: athlete_strength_v1, athlete_strength_v2")
    print(f"Historical v1 rows: {len(historical_v1):,}")
    print(f"Historical v2 rows: {len(historical_v2):,}")
    print(f"Online sample athletes: {online_athlete_ids}")
    print(f"Validation report: {validation_summary_path}")
    print("PHASE 6 STATUS: PASS")


if __name__ == "__main__":
    main()
