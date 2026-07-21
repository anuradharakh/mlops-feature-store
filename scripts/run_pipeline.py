"""Run the complete Assignment 2 MLOps pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "pipeline"

PIPELINE_LOG_PATH = PIPELINE_REPORT_DIRECTORY / "pipeline_run.log"

PIPELINE_SUMMARY_PATH = PIPELINE_REPORT_DIRECTORY / "pipeline_run_summary.json"


@dataclass(frozen=True)
class PipelineStage:
    """One executable pipeline stage."""

    name: str
    description: str
    command: tuple[str, ...]


def build_pipeline_stages(
    reset_runtime_state: bool,
) -> list[PipelineStage]:
    """Create the ordered end-to-end pipeline definition."""
    python = sys.executable

    feast_command = [
        python,
        "scripts/run_feast.py",
    ]

    experiment_command = [
        python,
        "scripts/run_experiments.py",
    ]

    if reset_runtime_state:
        feast_command.append("--reset")
        experiment_command.append("--reset")

    return [
        PipelineStage(
            name="ingestion",
            description=("Materialize and validate the raw athlete dataset."),
            command=(
                python,
                "scripts/run_ingestion.py",
            ),
        ),
        PipelineStage(
            name="preprocessing",
            description=("Clean athlete data and construct labels."),
            command=(
                python,
                "scripts/run_preprocessing.py",
            ),
        ),
        PipelineStage(
            name="feature_v1",
            description=("Build baseline Feature Version 1."),
            command=(
                python,
                "scripts/run_feature_v1.py",
            ),
        ),
        PipelineStage(
            name="feature_v2",
            description=("Build enhanced Feature Version 2."),
            command=(
                python,
                "scripts/run_feature_v2.py",
            ),
        ),
        PipelineStage(
            name="feast",
            description=("Apply Feast and retrieve historical and online features."),
            command=tuple(feast_command),
        ),
        PipelineStage(
            name="training_split",
            description=("Create the persisted train/test split."),
            command=(
                python,
                "scripts/run_training_split.py",
            ),
        ),
        PipelineStage(
            name="model_smoke_test",
            description=("Validate the Scikit-learn model pipeline."),
            command=(
                python,
                "scripts/run_model_pipeline.py",
            ),
        ),
        PipelineStage(
            name="experiments",
            description=("Run the four official MLflow experiments."),
            command=tuple(experiment_command),
        ),
        PipelineStage(
            name="report",
            description=("Generate the final HTML rollout report."),
            command=(
                python,
                "scripts/build_report.py",
            ),
        ),
        PipelineStage(
            name="submission_audit",
            description=("Validate required submission artifacts."),
            command=(
                python,
                "scripts/verify_submission.py",
            ),
        ),
        PipelineStage(
            name="dvc_status",
            description=("Confirm DVC-tracked data is current."),
            command=(
                "dvc",
                "status",
            ),
        ),
    ]


def parse_args() -> argparse.Namespace:
    """Parse pipeline command-line options."""
    stage_names = [stage.name for stage in build_pipeline_stages(reset_runtime_state=True)]

    parser = argparse.ArgumentParser(description=("Run the Assignment 2 MLOps pipeline."))

    parser.add_argument(
        "--no-reset",
        action="store_true",
        help=("Preserve the existing Feast and MLflow runtime stores."),
    )

    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help=("Skip the four full MLflow experiments."),
    )

    parser.add_argument(
        "--start-at",
        choices=stage_names,
        help=("Begin execution at the selected stage."),
    )

    parser.add_argument(
        "--stop-after",
        choices=stage_names,
        help=("Stop execution after the selected stage."),
    )

    return parser.parse_args()


def select_stages(
    stages: list[PipelineStage],
    start_at: str | None,
    stop_after: str | None,
    skip_experiments: bool,
) -> list[PipelineStage]:
    """Select the requested contiguous pipeline stages."""
    selected = stages

    if start_at is not None:
        start_index = next(index for index, stage in enumerate(selected) if stage.name == start_at)

        selected = selected[start_index:]

    if stop_after is not None:
        stop_index = next(index for index, stage in enumerate(selected) if stage.name == stop_after)

        selected = selected[: stop_index + 1]

    if skip_experiments:
        selected = [stage for stage in selected if stage.name != "experiments"]

    if not selected:
        raise ValueError("No pipeline stages were selected.")

    return selected


def run_stage(
    stage: PipelineStage,
    log_file,
) -> dict[str, Any]:
    """Execute one pipeline stage while streaming output."""
    started = time.perf_counter()

    header = (
        "\n"
        + "=" * 80
        + f"\nSTAGE: {stage.name}\n"
        + f"DESCRIPTION: {stage.description}\n"
        + f"COMMAND: {' '.join(stage.command)}\n"
        + "=" * 80
        + "\n"
    )

    print(header, end="")
    log_file.write(header)
    log_file.flush()

    environment = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
    }

    process = subprocess.Popen(
        stage.command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout is None:
        raise RuntimeError(f"Unable to capture output for {stage.name}.")

    for line in process.stdout:
        print(line, end="")
        log_file.write(line)
        log_file.flush()

    return_code = process.wait()

    elapsed_seconds = time.perf_counter() - started

    result = {
        "stage": stage.name,
        "description": stage.description,
        "command": list(stage.command),
        "return_code": int(return_code),
        "elapsed_seconds": round(
            float(elapsed_seconds),
            4,
        ),
        "status": ("PASS" if return_code == 0 else "FAIL"),
    }

    footer = f"\n{stage.name}: {result['status']} ({elapsed_seconds:.2f} seconds)\n"

    print(footer, end="")
    log_file.write(footer)
    log_file.flush()

    return result


def write_pipeline_summary(
    results: list[dict[str, Any]],
    status: str,
) -> None:
    """Write reviewer-friendly pipeline execution evidence."""
    payload = {
        "status": status,
        "stage_count": len(results),
        "passed_stage_count": sum(result["status"] == "PASS" for result in results),
        "failed_stage_count": sum(result["status"] == "FAIL" for result in results),
        "total_elapsed_seconds": round(
            sum(float(result["elapsed_seconds"]) for result in results),
            4,
        ),
        "stages": results,
    }

    PIPELINE_SUMMARY_PATH.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """Run the selected end-to-end pipeline stages."""
    args = parse_args()

    PIPELINE_REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    stages = build_pipeline_stages(reset_runtime_state=not args.no_reset)

    selected_stages = select_stages(
        stages=stages,
        start_at=args.start_at,
        stop_after=args.stop_after,
        skip_experiments=args.skip_experiments,
    )

    results: list[dict[str, Any]] = []

    with PIPELINE_LOG_PATH.open(
        "w",
        encoding="utf-8",
    ) as log_file:
        for stage in selected_stages:
            result = run_stage(
                stage=stage,
                log_file=log_file,
            )

            results.append(result)

            if result["status"] == "FAIL":
                write_pipeline_summary(
                    results=results,
                    status="FAIL",
                )

                print("\nPIPELINE STATUS: FAIL")
                print(f"Failed stage: {stage.name}")
                print(f"Log: {PIPELINE_LOG_PATH}")

                raise SystemExit(result["return_code"])

    write_pipeline_summary(
        results=results,
        status="PASS",
    )

    print()
    print("Assignment 2 pipeline completed successfully.")
    print(f"Stages completed: {len(results)}")
    print(f"Log: {PIPELINE_LOG_PATH}")
    print(f"Summary: {PIPELINE_SUMMARY_PATH}")
    print("PIPELINE STATUS: PASS")


if __name__ == "__main__":
    main()
