"""Tests for registered Feast feature definitions."""

from __future__ import annotations

from typing import Any

from feature_repo.feature_definitions import (
    V1_FEATURE_PATH,
    V2_FEATURE_PATH,
    athlete,
    athlete_features_v1_fv,
    athlete_features_v2_fv,
    athlete_strength_v1_service,
    athlete_strength_v2_service,
)

TARGET_LEAKAGE_COLUMNS = {
    "total_lift",
    "deadlift",
    "candj",
    "snatch",
    "backsq",
}

EXPECTED_V1_FEATURES = {
    "age",
    "weight",
    "height",
    "gender",
    "region",
}

EXPECTED_V2_FEATURES = {
    *EXPECTED_V1_FEATURES,
    "bmi",
    "age_squared",
    "weight_height_ratio",
}


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


def get_field_names(
    feature_view: Any,
) -> list[str]:
    """Return feature names from a Feast feature view."""
    return [field.name for field in feature_view.schema]


def assert_feature_names(
    feature_view: Any,
    expected_names: set[str],
) -> None:
    """Validate exact membership without relying on field order."""
    actual_names = get_field_names(feature_view)

    assert len(actual_names) == len(expected_names)

    assert len(actual_names) == len(set(actual_names))

    assert set(actual_names) == expected_names


def test_athlete_entity_uses_expected_join_key() -> None:
    """The athlete entity should join through athlete_id."""
    assert athlete.name == "athlete"

    assert get_entity_join_keys(athlete) == ["athlete_id"]


def test_feature_v1_definition_is_correct() -> None:
    """Feast Version 1 should expose five baseline features."""
    assert athlete_features_v1_fv.name == ("athlete_features_v1")

    assert_feature_names(
        feature_view=athlete_features_v1_fv,
        expected_names=EXPECTED_V1_FEATURES,
    )

    assert athlete_features_v1_fv.tags["feature_version"] == "v1"

    assert athlete_features_v1_fv.online is True


def test_feature_v2_definition_is_correct() -> None:
    """Feast Version 2 should expose eight enhanced features."""
    assert athlete_features_v2_fv.name == ("athlete_features_v2")

    assert_feature_names(
        feature_view=athlete_features_v2_fv,
        expected_names=EXPECTED_V2_FEATURES,
    )

    assert athlete_features_v2_fv.tags["feature_version"] == "v2"

    assert athlete_features_v2_fv.tags["parent_version"] == "v1"

    assert athlete_features_v2_fv.online is True


def test_feature_views_exclude_target_leakage() -> None:
    """Neither Feast view should expose target-related columns."""
    v1_features = set(get_field_names(athlete_features_v1_fv))

    v2_features = set(get_field_names(athlete_features_v2_fv))

    assert TARGET_LEAKAGE_COLUMNS.isdisjoint(v1_features)

    assert TARGET_LEAKAGE_COLUMNS.isdisjoint(v2_features)


def test_feature_services_are_versioned() -> None:
    """Each feature version should have a separate service."""
    assert athlete_strength_v1_service.name == ("athlete_strength_v1")

    assert athlete_strength_v2_service.name == ("athlete_strength_v2")

    assert athlete_strength_v1_service.name != athlete_strength_v2_service.name

    assert athlete_strength_v1_service.tags["feature_version"] == "v1"

    assert athlete_strength_v2_service.tags["feature_version"] == "v2"


def test_feature_source_files_exist() -> None:
    """Both committed offline feature sources should exist."""
    assert V1_FEATURE_PATH.exists()
    assert V1_FEATURE_PATH.is_file()

    assert V2_FEATURE_PATH.exists()
    assert V2_FEATURE_PATH.is_file()


def test_feature_versions_have_expected_difference() -> None:
    """Version 2 should contain exactly three added features."""
    added_features = EXPECTED_V2_FEATURES - EXPECTED_V1_FEATURES

    removed_features = EXPECTED_V1_FEATURES - EXPECTED_V2_FEATURES

    assert added_features == {
        "bmi",
        "age_squared",
        "weight_height_ratio",
    }

    assert removed_features == set()
