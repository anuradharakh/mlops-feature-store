"""Validate Feast datasets and create one reproducible model split."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from athlete_mlops.features.definitions import (
    build_feature_version_spec,
)
from athlete_mlops.training.dataset import (
    apply_split_membership,
    build_missingness_report,
    calculate_target_statistics,
    create_split_membership,
    validate_training_dataset,
    validate_version_alignment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"

FEATURE_CONFIG_PATH = PROJECT_ROOT / "configs" / "features.yaml"


def resolve_project_path(path_value: str) -> Path:
    """Resolve a repository-relative path."""
    return PROJECT_ROOT / path_value


def create_target_distribution_figure(
    version_one: pd.DataFrame,
    target_column: str,
    output_path: Path,
) -> None:
    """Save train/test target-distribution evidence."""
    train_target = version_one.loc[
        version_one["split"] == "train",
        target_column,
    ]

    test_target = version_one.loc[
        version_one["split"] == "test",
        target_column,
    ]

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.hist(
        train_target,
        bins=50,
        density=True,
        alpha=0.6,
        label="Train",
    )

    axis.hist(
        test_target,
        bins=50,
        density=True,
        alpha=0.6,
        label="Test",
    )

    axis.set_title("Total Lift Distribution: Train vs Test")
    axis.set_xlabel("Total lift")
    axis.set_ylabel("Density")
    axis.legend()
    axis.grid(alpha=0.25)

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    """Run Phase 7 dataset and split preparation."""
    with MODEL_CONFIG_PATH.open(encoding="utf-8") as config_file:
        model_config = yaml.safe_load(config_file)

    with FEATURE_CONFIG_PATH.open(encoding="utf-8") as config_file:
        feature_config = yaml.safe_load(config_file)

    training_config = model_config["training"]
    split_config = model_config["split"]
    artifact_config = model_config["artifacts"]

    target_column = training_config["target_column"]
    entity_column = training_config["entity_column"]
    timestamp_column = training_config["timestamp_column"]
    leakage_columns = list(training_config["target_leakage_columns"])

    v1_specification = build_feature_version_spec(
        version="v1",
        config=feature_config["feature_versions"]["v1"],
    )

    v2_specification = build_feature_version_spec(
        version="v2",
        config=feature_config["feature_versions"]["v2"],
    )

    v1_path = resolve_project_path(training_config["feature_datasets"]["v1"])

    v2_path = resolve_project_path(training_config["feature_datasets"]["v2"])

    required_inputs = [
        v1_path,
        v2_path,
    ]

    missing_inputs = [path for path in required_inputs if not path.exists()]

    if missing_inputs:
        formatted_paths = "\n".join(f"- {path}" for path in missing_inputs)

        raise FileNotFoundError(
            "Feast training artifacts are missing:\n"
            f"{formatted_paths}\n"
            "Run `python scripts/run_feast.py --reset` first."
        )

    v1_raw = pd.read_parquet(v1_path)
    v2_raw = pd.read_parquet(v2_path)

    v1_data = validate_training_dataset(
        dataframe=v1_raw,
        specification=v1_specification,
        target_column=target_column,
        leakage_columns=leakage_columns,
    )

    v2_data = validate_training_dataset(
        dataframe=v2_raw,
        specification=v2_specification,
        target_column=target_column,
        leakage_columns=leakage_columns,
    )

    validate_version_alignment(
        version_one=v1_data,
        version_two=v2_data,
        entity_column=entity_column,
        timestamp_column=timestamp_column,
        target_column=target_column,
    )

    split_result = create_split_membership(
        dataframe=v1_data,
        entity_column=entity_column,
        test_size=float(split_config["test_size"]),
        random_state=int(split_config["random_state"]),
        shuffle=bool(split_config["shuffle"]),
    )

    v1_with_split = apply_split_membership(
        dataframe=v1_data,
        membership=split_result.membership,
        entity_column=entity_column,
    )

    v2_with_split = apply_split_membership(
        dataframe=v2_data,
        membership=split_result.membership,
        entity_column=entity_column,
    )

    v1_train = v1_with_split.loc[v1_with_split["split"] == "train"]

    v1_test = v1_with_split.loc[v1_with_split["split"] == "test"]

    v2_train = v2_with_split.loc[v2_with_split["split"] == "train"]

    v2_test = v2_with_split.loc[v2_with_split["split"] == "test"]

    split_membership_path = resolve_project_path(artifact_config["split_membership"])

    split_summary_path = resolve_project_path(artifact_config["split_summary"])

    missingness_report_path = resolve_project_path(artifact_config["missingness_report"])

    target_distribution_path = resolve_project_path(artifact_config["target_distribution_figure"])

    split_membership_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_result.membership.to_parquet(
        split_membership_path,
        index=False,
    )

    missingness_report = pd.concat(
        [
            build_missingness_report(
                dataframe=v1_data,
                specification=v1_specification,
            ),
            build_missingness_report(
                dataframe=v2_data,
                specification=v2_specification,
            ),
        ],
        ignore_index=True,
    )

    missingness_report.to_csv(
        missingness_report_path,
        index=False,
    )

    create_target_distribution_figure(
        version_one=v1_with_split,
        target_column=target_column,
        output_path=target_distribution_path,
    )

    summary = {
        "status": "PASS",
        "entity_column": entity_column,
        "timestamp_column": timestamp_column,
        "target_column": target_column,
        "feature_versions": {
            "v1": {
                "feature_count": len(v1_specification.features),
                "features": list(v1_specification.features),
                "row_count": int(len(v1_data)),
                "train_rows": int(len(v1_train)),
                "test_rows": int(len(v1_test)),
            },
            "v2": {
                "feature_count": len(v2_specification.features),
                "features": list(v2_specification.features),
                "row_count": int(len(v2_data)),
                "train_rows": int(len(v2_train)),
                "test_rows": int(len(v2_test)),
            },
        },
        "version_alignment": {
            "same_entity_population": True,
            "same_timestamp_population": True,
            "same_target_values": True,
        },
        "split": split_result.summary,
        "target_statistics": {
            "complete_dataset": (
                calculate_target_statistics(
                    dataframe=v1_data,
                    target_column=target_column,
                )
            ),
            "train": (
                calculate_target_statistics(
                    dataframe=v1_train,
                    target_column=target_column,
                )
            ),
            "test": (
                calculate_target_statistics(
                    dataframe=v1_test,
                    target_column=target_column,
                )
            ),
        },
        "artifacts": {
            "split_membership": str(split_membership_path.relative_to(PROJECT_ROOT)),
            "missingness_report": str(missingness_report_path.relative_to(PROJECT_ROOT)),
            "target_distribution_figure": str(target_distribution_path.relative_to(PROJECT_ROOT)),
        },
    }

    split_summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Phase 7 training dataset preparation completed successfully.")
    print(f"Total entities: {split_result.summary['total_entities']:,}")
    print(f"Training entities: {split_result.summary['train_entities']:,}")
    print(f"Test entities: {split_result.summary['test_entities']:,}")
    print(f"Random state: {split_result.summary['random_state']}")
    print(f"Split hash: {split_result.summary['split_hash_sha256']}")
    print(f"Split artifact: {split_membership_path}")
    print(f"Summary: {split_summary_path}")
    print("PHASE 7 STATUS: PASS")


if __name__ == "__main__":
    main()
