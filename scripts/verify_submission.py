"""Validate Assignment 2 source code and review artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIT_DIRECTORY = PROJECT_ROOT / "reports" / "submission"

AUDIT_JSON_PATH = AUDIT_DIRECTORY / "submission_audit.json"

INVENTORY_PATH = AUDIT_DIRECTORY / "artifact_inventory.csv"

EXPECTED_RUN_NAMES = {
    "v1_hp1",
    "v1_hp2",
    "v2_hp1",
    "v2_hp2",
}

LEAKAGE_COLUMNS = {
    "total_lift",
    "deadlift",
    "candj",
    "snatch",
    "backsq",
}


SOURCE_FILES = [
    Path("README.md"),
    Path("requirements.in"),
    Path("requirements.txt"),
    Path("pyproject.toml"),
    Path(".gitignore"),
    Path("configs/pipeline.yaml"),
    Path("configs/features.yaml"),
    Path("configs/model.yaml"),
    Path("src/athlete_mlops/data/ingestion.py"),
    Path("src/athlete_mlops/data/preprocessing.py"),
    Path("src/athlete_mlops/features/builder.py"),
    Path("src/athlete_mlops/features/definitions.py"),
    Path("src/athlete_mlops/features/engineering.py"),
    Path("src/athlete_mlops/training/dataset.py"),
    Path("src/athlete_mlops/training/modeling.py"),
    Path("src/athlete_mlops/training/tracking.py"),
    Path("feature_repo/feature_store.yaml"),
    Path("feature_repo/feature_definitions.py"),
    Path("scripts/run_ingestion.py"),
    Path("scripts/run_preprocessing.py"),
    Path("scripts/run_feature_v1.py"),
    Path("scripts/run_feature_v2.py"),
    Path("scripts/run_feast.py"),
    Path("scripts/run_training_split.py"),
    Path("scripts/run_model_pipeline.py"),
    Path("scripts/run_experiments.py"),
    Path("scripts/run_pipeline.py"),
    Path("scripts/build_report.py"),
    Path("tests/test_ingestion.py"),
    Path("tests/test_preprocessing.py"),
    Path("tests/test_features.py"),
    Path("tests/test_feature_v2.py"),
    Path("tests/test_feast_definitions.py"),
    Path("tests/test_training_dataset.py"),
    Path("tests/test_modeling.py"),
    Path("tests/test_tracking.py"),
]


ARTIFACT_FILES = [
    Path("data/features/v1/athlete_features_v1.parquet"),
    Path("data/features/v2/athlete_features_v2.parquet"),
    Path("data/training/athlete_training_v1.parquet"),
    Path("data/training/athlete_training_v2.parquet"),
    Path("data/splits/athlete_split.parquet"),
    Path("reports/validation/preprocessing_summary.json"),
    Path("reports/validation/feature_v1_manifest.json"),
    Path("reports/validation/feature_v2_manifest.json"),
    Path("reports/validation/feature_version_comparison.json"),
    Path("reports/feast/feature_registry_summary.json"),
    Path("reports/feast/feast_validation_summary.json"),
    Path("reports/feast/online_retrieval_sample.json"),
    Path("reports/validation/training_split_summary.json"),
    Path("reports/validation/model_pipeline_smoke_summary.json"),
    Path("reports/mlflow/experiment_comparison.csv"),
    Path("reports/mlflow/experiment_comparison.json"),
    Path("reports/mlflow/best_run_summary.json"),
    Path("reports/figures/experiment_rmse_comparison.png"),
    Path("reports/figures/experiment_mae_comparison.png"),
    Path("reports/figures/experiment_r2_comparison.png"),
    Path("docs/assignment_2_rollout.md"),
    Path("docs/assignment_2_rollout.html"),
]


for run_name in sorted(EXPECTED_RUN_NAMES):
    ARTIFACT_FILES.extend(
        [
            Path(f"reports/mlflow/runs/{run_name}/run_summary.json"),
            Path(f"reports/mlflow/runs/{run_name}/predictions.csv"),
            Path(f"reports/mlflow/runs/{run_name}/feature_importance.csv"),
            Path(f"reports/mlflow/runs/{run_name}/actual_vs_predicted.png"),
            Path(f"reports/mlflow/runs/{run_name}/residual_distribution.png"),
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse audit options."""
    parser = argparse.ArgumentParser(description=("Validate Assignment 2 source and artifacts."))

    parser.add_argument(
        "--source-only",
        action="store_true",
        help=("Validate source, configuration, and test files without generated artifacts."),
    )

    parser.add_argument(
        "--require-git-tracked",
        action="store_true",
        help=("Require review artifacts to already be tracked by Git."),
    )

    parser.add_argument(
        "--require-source-portable",
        action="store_true",
        help=(
            "Require the source dataset to be Git-tracked "
            "or available through a configured DVC remote."
        ),
    )

    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def run_command(
    command: list[str],
) -> tuple[int, str]:
    """Run a command and return status and output."""
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    output = "\n".join(
        part.strip()
        for part in [
            completed.stdout,
            completed.stderr,
        ]
        if part.strip()
    )

    return completed.returncode, output


def is_git_tracked(relative_path: Path) -> bool:
    """Return whether a repository file is tracked by Git."""
    return_code, _ = run_command(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            str(relative_path),
        ]
    )

    return return_code == 0


def has_configured_dvc_remote() -> bool:
    """Return whether at least one DVC remote is configured."""
    return_code, output = run_command(
        [
            "dvc",
            "remote",
            "list",
        ]
    )

    return return_code == 0 and bool(output.strip())


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
    category: str,
) -> None:
    """Add one audit check."""
    checks.append(
        {
            "name": name,
            "category": category,
            "status": ("PASS" if passed else "FAIL"),
            "detail": detail,
        }
    )


def validate_required_files(
    checks: list[dict[str, Any]],
    files: list[Path],
    category: str,
) -> None:
    """Validate required files and non-empty content."""
    for relative_path in files:
        absolute_path = PROJECT_ROOT / relative_path

        exists = absolute_path.exists()
        non_empty = exists and absolute_path.is_file() and absolute_path.stat().st_size > 0

        add_check(
            checks=checks,
            name=f"file:{relative_path}",
            passed=bool(non_empty),
            detail=(
                f"size_bytes={absolute_path.stat().st_size}" if non_empty else "missing or empty"
            ),
            category=category,
        )


def validate_feature_artifacts(
    checks: list[dict[str, Any]],
) -> None:
    """Validate feature manifests and leakage protection."""
    v1_manifest = load_json(PROJECT_ROOT / "reports/validation/feature_v1_manifest.json")

    v2_manifest = load_json(PROJECT_ROOT / "reports/validation/feature_v2_manifest.json")

    same_population = v1_manifest["row_count"] == v2_manifest["row_count"] == 81_707

    add_check(
        checks=checks,
        name="feature_population_alignment",
        passed=same_population,
        detail=(f"v1={v1_manifest['row_count']}, v2={v2_manifest['row_count']}"),
        category="features",
    )

    v1_leakage = set(v1_manifest["features"]) & LEAKAGE_COLUMNS

    v2_leakage = set(v2_manifest["features"]) & LEAKAGE_COLUMNS

    add_check(
        checks=checks,
        name="feature_leakage_protection",
        passed=not v1_leakage and not v2_leakage,
        detail=(f"v1_leakage={sorted(v1_leakage)}, v2_leakage={sorted(v2_leakage)}"),
        category="features",
    )

    added_features = set(v2_manifest["features"]) - set(v1_manifest["features"])

    add_check(
        checks=checks,
        name="feature_version_difference",
        passed=added_features
        == {
            "bmi",
            "age_squared",
            "weight_height_ratio",
        },
        detail=(f"added_features={sorted(added_features)}"),
        category="features",
    )


def validate_feast_artifacts(
    checks: list[dict[str, Any]],
) -> None:
    """Validate Feast retrieval evidence."""
    feast_summary = load_json(PROJECT_ROOT / "reports/feast/feast_validation_summary.json")

    passed = (
        feast_summary["status"] == "PASS"
        and feast_summary["v1_historical_rows"] == 81_707
        and feast_summary["v2_historical_rows"] == 81_707
        and feast_summary["online_v1_retrieval_passed"] is True
        and feast_summary["online_v2_retrieval_passed"] is True
    )

    add_check(
        checks=checks,
        name="feast_historical_and_online_retrieval",
        passed=passed,
        detail=(
            f"v1_rows="
            f"{feast_summary['v1_historical_rows']}, "
            f"v2_rows="
            f"{feast_summary['v2_historical_rows']}"
        ),
        category="feast",
    )


def validate_split_artifacts(
    checks: list[dict[str, Any]],
) -> None:
    """Validate the persisted train/test split."""
    split_summary = load_json(PROJECT_ROOT / "reports/validation/training_split_summary.json")

    split = split_summary["split"]

    passed = (
        split_summary["status"] == "PASS"
        and split["total_entities"] == 81_707
        and split["train_entities"] == 65_365
        and split["test_entities"] == 16_342
        and split["train_test_overlap"] == 0
        and split["complete_population_coverage"] is True
    )

    add_check(
        checks=checks,
        name="reproducible_train_test_split",
        passed=passed,
        detail=(
            f"train={split['train_entities']}, "
            f"test={split['test_entities']}, "
            f"hash={split['split_hash_sha256']}"
        ),
        category="training",
    )


def validate_experiment_artifacts(
    checks: list[dict[str, Any]],
) -> None:
    """Validate all four experiment results and best run."""
    comparison_path = PROJECT_ROOT / "reports/mlflow/experiment_comparison.csv"

    best_path = PROJECT_ROOT / "reports/mlflow/best_run_summary.json"

    comparison = pd.read_csv(comparison_path)

    best = load_json(best_path)

    run_names_valid = (
        len(comparison) == 4
        and comparison["run_name"].nunique() == 4
        and set(comparison["run_name"]) == EXPECTED_RUN_NAMES
    )

    add_check(
        checks=checks,
        name="official_experiment_matrix",
        passed=run_names_valid,
        detail=(f"runs={sorted(comparison['run_name'].tolist())}"),
        category="mlflow",
    )

    required_metrics = [
        "train_rmse",
        "train_mae",
        "train_r2",
        "test_rmse",
        "test_mae",
        "test_r2",
    ]

    metrics_valid = all(
        metric in comparison.columns and comparison[metric].notna().all()
        for metric in required_metrics
    )

    add_check(
        checks=checks,
        name="experiment_metrics_complete",
        passed=metrics_valid,
        detail=(f"metrics={required_metrics}"),
        category="mlflow",
    )

    ranked = comparison.sort_values(
        by=[
            "test_rmse",
            "test_mae",
            "test_r2",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    expected_best = ranked.iloc[0]

    best_matches = (
        str(best["run_id"]) == str(expected_best["run_id"])
        and best["run_name"] == expected_best["run_name"]
        and abs(float(best["test_rmse"]) - float(expected_best["test_rmse"])) < 1e-8
    )

    add_check(
        checks=checks,
        name="best_run_selection",
        passed=best_matches,
        detail=(f"selected={best['run_name']}, expected={expected_best['run_name']}"),
        category="mlflow",
    )


def validate_git_tracking(
    checks: list[dict[str, Any]],
    files: list[Path],
) -> None:
    """Require reviewer artifacts to be tracked in Git."""
    for relative_path in files:
        tracked = is_git_tracked(relative_path)

        add_check(
            checks=checks,
            name=f"git_tracked:{relative_path}",
            passed=tracked,
            detail=("tracked" if tracked else "not tracked"),
            category="git",
        )


def validate_source_portability(
    checks: list[dict[str, Any]],
) -> None:
    """Ensure a reviewer can obtain the source dataset."""
    source_path = Path("data/source/athletes.zip")

    dvc_pointer_path = Path("data/source/athletes.zip.dvc")

    source_is_git_tracked = is_git_tracked(source_path)

    dvc_pointer_exists = (PROJECT_ROOT / dvc_pointer_path).exists()

    dvc_remote_exists = has_configured_dvc_remote()

    portable = source_is_git_tracked or (dvc_pointer_exists and dvc_remote_exists)

    add_check(
        checks=checks,
        name="source_dataset_portability",
        passed=portable,
        detail=(
            f"git_tracked={source_is_git_tracked}, "
            f"dvc_pointer={dvc_pointer_exists}, "
            f"dvc_remote={dvc_remote_exists}"
        ),
        category="reproducibility",
    )


def create_artifact_inventory(
    files: list[Path],
) -> pd.DataFrame:
    """Create a reviewer-friendly artifact inventory."""
    records = []

    for relative_path in files:
        absolute_path = PROJECT_ROOT / relative_path

        records.append(
            {
                "path": str(relative_path),
                "exists": absolute_path.exists(),
                "size_bytes": (
                    absolute_path.stat().st_size
                    if absolute_path.exists() and absolute_path.is_file()
                    else 0
                ),
                "git_tracked": (is_git_tracked(relative_path)),
                "suffix": relative_path.suffix,
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    """Run the complete submission audit."""
    args = parse_args()

    AUDIT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    checks: list[dict[str, Any]] = []

    validate_required_files(
        checks=checks,
        files=SOURCE_FILES,
        category="source",
    )

    inventory_files = list(SOURCE_FILES)

    if not args.source_only:
        validate_required_files(
            checks=checks,
            files=ARTIFACT_FILES,
            category="artifacts",
        )

        validate_feature_artifacts(checks)
        validate_feast_artifacts(checks)
        validate_split_artifacts(checks)
        validate_experiment_artifacts(checks)

        inventory_files.extend(ARTIFACT_FILES)

    if args.require_git_tracked:
        tracked_files = SOURCE_FILES if args.source_only else SOURCE_FILES + ARTIFACT_FILES

        validate_git_tracking(
            checks=checks,
            files=tracked_files,
        )

    if args.require_source_portable:
        validate_source_portability(checks)

    inventory = create_artifact_inventory(inventory_files)

    inventory.to_csv(
        INVENTORY_PATH,
        index=False,
    )

    failed_checks = [check for check in checks if check["status"] == "FAIL"]

    payload = {
        "status": ("PASS" if not failed_checks else "FAIL"),
        "source_only": args.source_only,
        "require_git_tracked": (args.require_git_tracked),
        "require_source_portable": (args.require_source_portable),
        "check_count": len(checks),
        "passed_check_count": (len(checks) - len(failed_checks)),
        "failed_check_count": len(failed_checks),
        "checks": checks,
        "inventory": str(INVENTORY_PATH.relative_to(PROJECT_ROOT)),
    }

    AUDIT_JSON_PATH.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(f"Checks completed: {len(checks)}")
    print(f"Passed: {payload['passed_check_count']}")
    print(f"Failed: {payload['failed_check_count']}")
    print(f"Audit: {AUDIT_JSON_PATH}")
    print(f"Inventory: {INVENTORY_PATH}")

    if failed_checks:
        print()
        print("Failed checks:")

        for check in failed_checks:
            print(f"- {check['name']}: {check['detail']}")

        print("\nSUBMISSION AUDIT STATUS: FAIL")

        raise SystemExit(1)

    print("SUBMISSION AUDIT STATUS: PASS")


if __name__ == "__main__":
    main()
