"""Tests for athlete data ingestion and validation."""

from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from athlete_mlops.data.ingestion import (
    load_raw_dataset,
    materialize_raw_csv,
)
from athlete_mlops.data.validation import (
    REQUIRED_COLUMNS,
    validate_required_columns,
)


def test_materialize_raw_csv_from_zip(
    tmp_path: Path,
) -> None:
    """A ZIP containing one CSV should be extracted."""
    input_csv = tmp_path / "input.csv"
    source_zip = tmp_path / "athletes.zip"
    output_csv = tmp_path / "raw" / "athletes.csv"

    pd.DataFrame(
        {
            "athlete_id": [1, 2],
            "age": [25, 30],
        }
    ).to_csv(input_csv, index=False)

    with ZipFile(source_zip, mode="w") as archive:
        archive.write(input_csv, arcname="athletes.csv")

    result_path = materialize_raw_csv(
        source_path=source_zip,
        output_path=output_csv,
    )

    assert result_path == output_csv
    assert output_csv.exists()

    loaded_data = load_raw_dataset(output_csv)

    assert len(loaded_data) == 2
    assert list(loaded_data.columns) == [
        "athlete_id",
        "age",
    ]


def test_validate_required_columns_passes() -> None:
    """Validation should pass when all columns exist."""
    dataframe = pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))

    validate_required_columns(dataframe)


def test_validate_required_columns_fails() -> None:
    """Validation should report missing columns."""
    dataframe = pd.DataFrame({"athlete_id": [1]})

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_required_columns(dataframe)
