"""Feast definitions for athlete feature versions."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
)
from feast.types import Float64, String

PROJECT_ROOT = Path(__file__).resolve().parents[1]

V1_FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "v1" / "athlete_features_v1.parquet"

V2_FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "v2" / "athlete_features_v2.parquet"


athlete = Entity(
    name="athlete",
    join_keys=["athlete_id"],
    description="Unique athlete represented by athlete_id.",
)


athlete_features_v1_source = FileSource(
    name="athlete_features_v1_source",
    path=str(V1_FEATURE_PATH),
    timestamp_field="event_timestamp",
)


athlete_features_v2_source = FileSource(
    name="athlete_features_v2_source",
    path=str(V2_FEATURE_PATH),
    timestamp_field="event_timestamp",
)


athlete_features_v1_fv = FeatureView(
    name="athlete_features_v1",
    description=("Baseline demographic and physical athlete features."),
    entities=[athlete],
    ttl=timedelta(days=3650),
    schema=[
        Field(
            name="age",
            dtype=Float64,
            description="Athlete age.",
        ),
        Field(
            name="weight",
            dtype=Float64,
            description="Athlete weight in pounds.",
        ),
        Field(
            name="height",
            dtype=Float64,
            description="Athlete height in inches.",
        ),
        Field(
            name="gender",
            dtype=String,
            description="Athlete-reported gender.",
        ),
        Field(
            name="region",
            dtype=String,
            description="Athlete geographic region.",
        ),
    ],
    online=True,
    source=athlete_features_v1_source,
    tags={
        "feature_version": "v1",
        "team": "athlete_mlops",
        "feature_group": "baseline",
    },
)


athlete_features_v2_fv = FeatureView(
    name="athlete_features_v2",
    description=(
        "Enhanced athlete features with deterministic physical and nonlinear transformations."
    ),
    entities=[athlete],
    ttl=timedelta(days=3650),
    schema=[
        Field(
            name="age",
            dtype=Float64,
            description="Athlete age.",
        ),
        Field(
            name="weight",
            dtype=Float64,
            description="Athlete weight in pounds.",
        ),
        Field(
            name="height",
            dtype=Float64,
            description="Athlete height in inches.",
        ),
        Field(
            name="gender",
            dtype=String,
            description="Athlete-reported gender.",
        ),
        Field(
            name="region",
            dtype=String,
            description="Athlete geographic region.",
        ),
        Field(
            name="bmi",
            dtype=Float64,
            description=("BMI calculated using weight in pounds and height in inches."),
        ),
        Field(
            name="age_squared",
            dtype=Float64,
            description="Squared athlete age.",
        ),
        Field(
            name="weight_height_ratio",
            dtype=Float64,
            description="Athlete weight divided by height.",
        ),
    ],
    online=True,
    source=athlete_features_v2_source,
    tags={
        "feature_version": "v2",
        "parent_version": "v1",
        "team": "athlete_mlops",
        "feature_group": "enhanced",
    },
)


athlete_strength_v1_service = FeatureService(
    name="athlete_strength_v1",
    features=[
        athlete_features_v1_fv,
    ],
    tags={
        "feature_version": "v1",
        "model_family": "random_forest",
    },
)


athlete_strength_v2_service = FeatureService(
    name="athlete_strength_v2",
    features=[
        athlete_features_v2_fv,
    ],
    tags={
        "feature_version": "v2",
        "model_family": "random_forest",
    },
)
