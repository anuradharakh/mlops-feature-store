"""Tests for deterministic athlete preprocessing and label construction."""

from __future__ import annotations

import pandas as pd

from athlete_mlops.data.preprocessing import build_processed_dataset

TEST_CONFIG = {
    "identifier_column": "athlete_id",
    "source_timestamp_column": "retrieved_datetime",
    "event_timestamp_column": "event_timestamp",
    "target_components": [
        "deadlift",
        "candj",
        "snatch",
        "backsq",
    ],
    "target_sentinel_values": [1],
    "numeric_columns": [
        "athlete_id",
        "age",
        "weight",
        "height",
        "deadlift",
        "candj",
        "snatch",
        "backsq",
    ],
    "categorical_columns": [
        "gender",
        "region",
        "howlong",
        "eat",
        "background",
        "experience",
        "schedule",
    ],
    "invalid_response_values": [
        "Decline to answer|",
        "Decline to answer",
        "",
        "NA",
        "N/A",
        "null",
        "None",
    ],
    "plausible_ranges": {
        "age": {"min": 14, "max": 100},
        "weight": {"min": 50, "max": 700},
        "height": {"min": 36, "max": 96},
        "deadlift": {"min": 1, "max": 1105},
        "candj": {"min": 1, "max": 395},
        "snatch": {"min": 1, "max": 496},
        "backsq": {"min": 1, "max": 1069},
    },
}


def build_sample_data() -> pd.DataFrame:
    """Create representative athlete records for preprocessing tests."""
    return pd.DataFrame(
        {
            "athlete_id": [1, 2, 3],
            "retrieved_datetime": [
                "2024-01-01T10:00:00Z",
                "2024-01-02T10:00:00Z",
                "2024-01-03T10:00:00Z",
            ],
            "age": [25, 30, 35],
            "weight": [180, 165, 190],
            "height": [70, 66, 72],
            "gender": ["Male", "Female", "Male"],
            "region": ["Central", "West", "East"],
            "howlong": [
                "1-2 years",
                "Decline to answer|",
                "2-4 years",
            ],
            "eat": [
                "Balanced",
                "Balanced",
                "Paleo",
            ],
            "background": [
                "Sports",
                "None",
                "Sports",
            ],
            "experience": [
                "Intermediate",
                "Beginner",
                "Advanced",
            ],
            "schedule": [
                "4 days",
                "3 days",
                "5 days",
            ],
            "deadlift": [400, 300, 2000],
            "candj": [225, 175, 200],
            "snatch": [175, 125, 150],
            "backsq": [350, 250, 300],
        }
    )


def test_total_lift_is_calculated_correctly() -> None:
    """The target should equal the sum of the four lift components."""
    result = build_processed_dataset(
        dataframe=build_sample_data(),
        config=TEST_CONFIG,
    )

    athlete_one = result.processed_data.loc[result.processed_data["athlete_id"] == 1].iloc[0]

    assert athlete_one["total_lift"] == 1150


def test_out_of_range_target_record_is_removed() -> None:
    """A record containing an implausible target component should be removed."""
    result = build_processed_dataset(
        dataframe=build_sample_data(),
        config=TEST_CONFIG,
    )

    athlete_ids = result.processed_data["athlete_id"].tolist()

    assert 3 not in athlete_ids
    assert len(result.processed_data) == 2


def test_target_sentinel_record_is_removed() -> None:
    """A target component equal to the sentinel value should invalidate the row."""
    dataframe = build_sample_data()

    dataframe.loc[
        dataframe["athlete_id"] == 1,
        ["deadlift", "candj", "snatch", "backsq"],
    ] = 1

    result = build_processed_dataset(
        dataframe=dataframe,
        config=TEST_CONFIG,
    )

    athlete_ids = result.processed_data["athlete_id"].tolist()

    assert 1 not in athlete_ids
    assert len(result.processed_data) == 1


def test_target_sentinel_replacements_are_reported() -> None:
    """The preprocessing result should report each sentinel replacement."""
    dataframe = build_sample_data()

    dataframe.loc[
        dataframe["athlete_id"] == 1,
        ["deadlift", "candj", "snatch", "backsq"],
    ] = 1

    result = build_processed_dataset(
        dataframe=dataframe,
        config=TEST_CONFIG,
    )

    replacement_counts = result.sentinel_value_counts.set_index("column")[
        "sentinel_replacement_count"
    ]

    assert replacement_counts["deadlift"] == 1
    assert replacement_counts["candj"] == 1
    assert replacement_counts["snatch"] == 1
    assert replacement_counts["backsq"] == 1
    assert replacement_counts.sum() == 4

    assert result.summary["target_sentinel_values"] == [1]
    assert result.summary["target_sentinel_values_replaced"] == 4


def test_single_sentinel_component_invalidates_target() -> None:
    """One sentinel component should be enough to remove the record."""
    dataframe = build_sample_data()

    dataframe.loc[
        dataframe["athlete_id"] == 2,
        "snatch",
    ] = 1

    result = build_processed_dataset(
        dataframe=dataframe,
        config=TEST_CONFIG,
    )

    athlete_ids = result.processed_data["athlete_id"].tolist()

    assert 2 not in athlete_ids


def test_invalid_survey_response_becomes_missing() -> None:
    """Configured invalid survey responses should become missing values."""
    result = build_processed_dataset(
        dataframe=build_sample_data(),
        config=TEST_CONFIG,
    )

    athlete_two = result.processed_data.loc[result.processed_data["athlete_id"] == 2].iloc[0]

    assert pd.isna(athlete_two["howlong"])


def test_missing_predictor_value_is_retained() -> None:
    """Missing predictors should remain for training-time imputation."""
    dataframe = build_sample_data()

    dataframe.loc[
        dataframe["athlete_id"] == 1,
        "weight",
    ] = None

    result = build_processed_dataset(
        dataframe=dataframe,
        config=TEST_CONFIG,
    )

    athlete_one = result.processed_data.loc[result.processed_data["athlete_id"] == 1].iloc[0]

    assert pd.isna(athlete_one["weight"])
    assert 1 in result.processed_data["athlete_id"].tolist()


def test_label_table_contains_only_entity_time_and_target() -> None:
    """The label artifact should not contain model predictors."""
    result = build_processed_dataset(
        dataframe=build_sample_data(),
        config=TEST_CONFIG,
    )

    assert list(result.labels.columns) == [
        "athlete_id",
        "event_timestamp",
        "total_lift",
    ]


def test_labels_align_with_processed_data() -> None:
    """Entity keys and targets should align across both output artifacts."""
    result = build_processed_dataset(
        dataframe=build_sample_data(),
        config=TEST_CONFIG,
    )

    assert result.processed_data["athlete_id"].equals(result.labels["athlete_id"])

    assert result.processed_data["total_lift"].equals(result.labels["total_lift"])


def test_event_timestamp_is_created() -> None:
    """The output should contain complete UTC event timestamps."""
    result = build_processed_dataset(
        dataframe=build_sample_data(),
        config=TEST_CONFIG,
    )

    timestamps = result.processed_data["event_timestamp"]

    assert timestamps.notna().all()
    assert pd.api.types.is_datetime64_any_dtype(timestamps.dtype)
    assert str(timestamps.dt.tz) == "UTC"


def test_duplicate_entity_keeps_latest_record() -> None:
    """The most recent observation should be retained for a duplicate entity."""
    dataframe = pd.DataFrame(
        {
            "athlete_id": [10, 10],
            "retrieved_datetime": [
                "2024-01-01T10:00:00Z",
                "2024-02-01T10:00:00Z",
            ],
            "age": [25, 26],
            "weight": [180, 185],
            "height": [70, 70],
            "gender": ["Male", "Male"],
            "region": ["Central", "Central"],
            "howlong": ["1-2 years", "2-4 years"],
            "eat": ["Balanced", "Balanced"],
            "background": ["Sports", "Sports"],
            "experience": ["Beginner", "Intermediate"],
            "schedule": ["3 days", "4 days"],
            "deadlift": [300, 400],
            "candj": [150, 200],
            "snatch": [100, 150],
            "backsq": [250, 350],
        }
    )

    result = build_processed_dataset(
        dataframe=dataframe,
        config=TEST_CONFIG,
    )

    assert len(result.processed_data) == 1

    retained_record = result.processed_data.iloc[0]

    assert retained_record["age"] == 26
    assert retained_record["deadlift"] == 400
    assert retained_record["total_lift"] == 1100
    assert result.summary["duplicate_entity_rows_removed"] == 1


def test_missing_identifier_record_is_removed() -> None:
    """Records without an athlete entity key should be removed."""
    dataframe = build_sample_data()

    dataframe.loc[
        dataframe["athlete_id"] == 1,
        "athlete_id",
    ] = None

    result = build_processed_dataset(
        dataframe=dataframe,
        config=TEST_CONFIG,
    )

    assert result.processed_data["athlete_id"].notna().all()
    assert result.summary["missing_identifier_rows_removed"] == 1


def test_summary_row_accounting_is_correct() -> None:
    """Summary removal counts should reconcile with final row count."""
    result = build_processed_dataset(
        dataframe=build_sample_data(),
        config=TEST_CONFIG,
    )

    summary = result.summary

    expected_rows = (
        summary["initial_rows"]
        - summary["missing_identifier_rows_removed"]
        - summary["duplicate_entity_rows_removed"]
        - summary["missing_or_invalid_target_rows_removed"]
    )

    assert expected_rows == summary["processed_rows"]
    assert summary["processed_rows"] == len(result.processed_data)
    assert summary["label_rows"] == len(result.labels)


def test_no_sentinel_values_remain_in_target_components() -> None:
    """Final target-component values should not include sentinel values."""
    dataframe = build_sample_data()

    dataframe.loc[
        dataframe["athlete_id"] == 1,
        "deadlift",
    ] = 1

    result = build_processed_dataset(
        dataframe=dataframe,
        config=TEST_CONFIG,
    )

    target_components = TEST_CONFIG["target_components"]

    assert not (result.processed_data[target_components] == 1).any().any()


def test_preprocessing_is_deterministic() -> None:
    """Repeated runs should produce identical artifacts and summaries."""
    dataframe = build_sample_data()

    first_result = build_processed_dataset(
        dataframe=dataframe,
        config=TEST_CONFIG,
    )

    second_result = build_processed_dataset(
        dataframe=dataframe,
        config=TEST_CONFIG,
    )

    pd.testing.assert_frame_equal(
        first_result.processed_data,
        second_result.processed_data,
    )

    pd.testing.assert_frame_equal(
        first_result.labels,
        second_result.labels,
    )

    pd.testing.assert_frame_equal(
        first_result.invalid_value_counts,
        second_result.invalid_value_counts,
    )

    pd.testing.assert_frame_equal(
        first_result.sentinel_value_counts,
        second_result.sentinel_value_counts,
    )

    assert first_result.summary == second_result.summary
