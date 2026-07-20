"""Data-ingestion utilities for the athlete dataset."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pandas as pd


def materialize_raw_csv(
    source_path: Path,
    output_path: Path,
) -> Path:
    """Create the raw athlete CSV from a CSV or ZIP source.

    The operation copies the source data without applying cleaning,
    filtering, type conversion, or feature engineering.

    Args:
        source_path: Source CSV or ZIP archive.
        output_path: Destination path for the raw CSV.

    Returns:
        Path to the materialized raw CSV.

    Raises:
        FileNotFoundError: If the source file does not exist.
        ValueError: If the source format is unsupported or the archive
            does not contain exactly one CSV file.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Source dataset was not found: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_suffix = source_path.suffix.lower()

    if source_suffix == ".csv":
        shutil.copy2(source_path, output_path)
        return output_path

    if source_suffix != ".zip":
        raise ValueError(
            f"The source dataset must be a CSV or ZIP archive. Received: {source_path}"
        )

    if not zipfile.is_zipfile(source_path):
        raise ValueError(f"Invalid ZIP archive: {source_path}")

    with zipfile.ZipFile(source_path, mode="r") as archive:
        csv_members = [
            member
            for member in archive.infolist()
            if not member.is_dir()
            and member.filename.lower().endswith(".csv")
            and not member.filename.startswith("__MACOSX/")
        ]

        if len(csv_members) != 1:
            member_names = [member.filename for member in csv_members]
            raise ValueError(
                f"Expected exactly one CSV file inside the archive. Found: {member_names}"
            )

        with archive.open(csv_members[0], mode="r") as source_file:
            with output_path.open(mode="wb") as output_file:
                shutil.copyfileobj(source_file, output_file)

    return output_path


def load_raw_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load and minimally validate the raw athlete CSV.

    Args:
        dataset_path: Path to the materialized CSV.

    Returns:
        Loaded raw athlete DataFrame.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If the CSV contains no records or columns.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Raw dataset was not found: {dataset_path}")

    dataframe = pd.read_csv(
        dataset_path,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(f"The raw dataset contains no records: {dataset_path}")

    if dataframe.shape[1] == 0:
        raise ValueError(f"The raw dataset contains no columns: {dataset_path}")

    return dataframe
