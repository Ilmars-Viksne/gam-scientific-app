from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from gam_app.config import (
    CorrelationConfig,
    DuplicateGroupConfig,
    ExecutionConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    ProfilingConfig,
    SearchConfig,
    ValidationConfig,
    load_config,
)
from gam_app.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class InvalidConfigCase:
    case_id: str
    mutate: Callable[[ExperimentConfig], ExperimentConfig]
    message_pattern: str


def make_base_config(tmp_path: Path) -> ExperimentConfig:
    data_path = tmp_path / "data.csv"
    data_path.write_text(
        "row_id,group_col,time_col,x1,x2,target\n1,g1,2025-01-01,10,20,A\n2,g2,2025-01-02,11,21,B\n",
        encoding="utf-8",
    )
    return ExperimentConfig(
        name="valid_base_experiment",
        data_path=data_path,
        target="target",
        row_id="row_id",
        group_column="group_col",
        time_column="time_col",
        features={
            "x1": FeatureConfig(role="smooth"),
            "x2": FeatureConfig(role="smooth"),
            "row_id": FeatureConfig(role="exclude"),
            "group_col": FeatureConfig(role="exclude"),
            "time_col": FeatureConfig(role="exclude"),
        },
        models=(ModelConfig(id="gam_main", interactions="none"),),
        validation=ValidationConfig(
            strategy="stratified",
            outer_splits=5,
            outer_repeats=3,
            inner_splits=5,
            random_state=42,
            gap=0,
            test_size=None,
            duplicate_group_policy="report",
        ),
        profiling=ProfilingConfig(
            correlation=CorrelationConfig(
                enabled=True,
                review_threshold=0.75,
                warning_threshold=0.90,
                minimum_complete_pairs=3,
            ),
            duplicate_groups=DuplicateGroupConfig(
                enabled=True,
                rounding_decimals=8,
                near_duplicate_threshold=0.98,
                include_target_in_signature=False,
                maximum_pairwise_rows=10000,
            ),
        ),
        search=SearchConfig(),
        execution=ExecutionConfig(),
    )


INVALID_CONFIG_CASES: tuple[InvalidConfigCase, ...] = (
    # Universal split / structural validation
    InvalidConfigCase(
        "outer_splits_zero",
        lambda c: replace(c, validation=replace(c.validation, outer_splits=0)),
        "validation.outer_splits must be at least 2",
    ),
    InvalidConfigCase(
        "outer_splits_one",
        lambda c: replace(c, validation=replace(c.validation, outer_splits=1)),
        "validation.outer_splits must be at least 2",
    ),
    InvalidConfigCase(
        "outer_repeats_zero",
        lambda c: replace(c, validation=replace(c.validation, outer_repeats=0)),
        "validation.outer_repeats must be at least 1",
    ),
    InvalidConfigCase(
        "inner_splits_zero",
        lambda c: replace(c, validation=replace(c.validation, inner_splits=0)),
        "validation.inner_splits must be at least 2",
    ),
    InvalidConfigCase(
        "inner_splits_one",
        lambda c: replace(c, validation=replace(c.validation, inner_splits=1)),
        "validation.inner_splits must be at least 2",
    ),
    InvalidConfigCase(
        "negative_gap",
        lambda c: replace(
            c,
            validation=replace(
                c.validation, strategy="time", gap=-1, test_size=5, outer_repeats=1
            ),
        ),
        "validation.gap cannot be negative",
    ),
    InvalidConfigCase(
        "zero_test_size",
        lambda c: replace(
            c,
            validation=replace(
                c.validation, strategy="time", test_size=0, outer_repeats=1
            ),
        ),
        "validation.test_size must be at least 1",
    ),
    InvalidConfigCase(
        "negative_test_size",
        lambda c: replace(
            c,
            validation=replace(
                c.validation, strategy="time", test_size=-5, outer_repeats=1
            ),
        ),
        "validation.test_size must be at least 1",
    ),
    InvalidConfigCase(
        "unsupported_schema_version",
        lambda c: replace(c, schema_version="99.0"),
        "Unsupported configuration schema_version '99.0'",
    ),
    InvalidConfigCase(
        "duplicate_model_ids",
        lambda c: replace(
            c, models=(ModelConfig(id="gam_main"), ModelConfig(id="gam_main"))
        ),
        "Model IDs must be unique",
    ),
    InvalidConfigCase(
        "no_active_predictors",
        lambda c: replace(
            c,
            features={
                "x1": FeatureConfig(role="exclude"),
                "x2": FeatureConfig(role="exclude"),
            },
        ),
        "At least one active predictor is required",
    ),
    # Reserved role conflicts
    InvalidConfigCase(
        "target_equals_row_id",
        lambda c: replace(c, target="same_col", row_id="same_col"),
        "Data-role columns must be distinct",
    ),
    InvalidConfigCase(
        "target_equals_group",
        lambda c: replace(c, target="same_col", group_column="same_col"),
        "Data-role columns must be distinct",
    ),
    InvalidConfigCase(
        "target_equals_time",
        lambda c: replace(c, target="same_col", time_column="same_col"),
        "Data-role columns must be distinct",
    ),
    InvalidConfigCase(
        "row_id_equals_group",
        lambda c: replace(c, row_id="same_col", group_column="same_col"),
        "Data-role columns must be distinct",
    ),
    InvalidConfigCase(
        "row_id_equals_time",
        lambda c: replace(c, row_id="same_col", time_column="same_col"),
        "Data-role columns must be distinct",
    ),
    InvalidConfigCase(
        "group_equals_time",
        lambda c: replace(c, group_column="same_col", time_column="same_col"),
        "Data-role columns must be distinct",
    ),
    # Active predictor overlap
    InvalidConfigCase(
        "target_as_smooth_predictor",
        lambda c: replace(
            c,
            features={
                **c.features,
                "target": FeatureConfig(role="smooth"),
            },
        ),
        "Target, row-id, group, and time columns cannot be used as active model predictors",
    ),
    InvalidConfigCase(
        "row_id_as_linear_predictor",
        lambda c: replace(
            c,
            features={
                **c.features,
                "row_id": FeatureConfig(role="linear"),
            },
        ),
        "Target, row-id, group, and time columns cannot be used as active model predictors",
    ),
    InvalidConfigCase(
        "group_as_categorical_predictor",
        lambda c: replace(
            c,
            features={
                **c.features,
                "group_col": FeatureConfig(role="categorical"),
            },
        ),
        "Target, row-id, group, and time columns cannot be used as active model predictors",
    ),
    InvalidConfigCase(
        "time_as_smooth_predictor",
        lambda c: replace(
            c,
            features={
                **c.features,
                "time_col": FeatureConfig(role="smooth"),
            },
        ),
        "Target, row-id, group, and time columns cannot be used as active model predictors",
    ),
    # Strategy-specific invalid cases
    InvalidConfigCase(
        "stratified_with_nonzero_gap",
        lambda c: replace(
            c, validation=replace(c.validation, strategy="stratified", gap=5)
        ),
        "validation.gap applies only when validation.strategy='time'",
    ),
    InvalidConfigCase(
        "stratified_with_test_size",
        lambda c: replace(
            c, validation=replace(c.validation, strategy="stratified", test_size=10)
        ),
        "validation.test_size applies only when validation.strategy='time'",
    ),
    InvalidConfigCase(
        "stratified_with_group_policy",
        lambda c: replace(
            c,
            validation=replace(
                c.validation, strategy="stratified", duplicate_group_policy="group"
            ),
        ),
        "validation.duplicate_group_policy='group' requires validation.strategy='stratified_group'",
    ),
    InvalidConfigCase(
        "group_strategy_without_group_source",
        lambda c: replace(
            c,
            group_column=None,
            validation=replace(
                c.validation,
                strategy="stratified_group",
                duplicate_group_policy="report",
            ),
        ),
        "validation.strategy='stratified_group' requires either data.group or validation.duplicate_group_policy='group'",
    ),
    InvalidConfigCase(
        "group_strategy_with_nonzero_gap",
        lambda c: replace(
            c, validation=replace(c.validation, strategy="stratified_group", gap=2)
        ),
        "validation.gap applies only when validation.strategy='time'",
    ),
    InvalidConfigCase(
        "group_strategy_with_test_size",
        lambda c: replace(
            c,
            validation=replace(c.validation, strategy="stratified_group", test_size=5),
        ),
        "validation.test_size applies only when validation.strategy='time'",
    ),
    InvalidConfigCase(
        "group_policy_with_duplicate_diagnostics_disabled",
        lambda c: replace(
            c,
            validation=replace(
                c.validation,
                duplicate_group_policy="group",
                strategy="stratified_group",
            ),
            profiling=replace(
                c.profiling,
                duplicate_groups=replace(c.profiling.duplicate_groups, enabled=False),
            ),
        ),
        "requires profiling.duplicate_groups.enabled=true",
    ),
    InvalidConfigCase(
        "time_without_time_column",
        lambda c: replace(
            c,
            time_column=None,
            validation=replace(
                c.validation, strategy="time", outer_repeats=1, test_size=5
            ),
        ),
        "data.time is required when validation.strategy='time'",
    ),
    InvalidConfigCase(
        "time_with_outer_repeats_above_one",
        lambda c: replace(
            c,
            validation=replace(
                c.validation, strategy="time", outer_repeats=2, test_size=5
            ),
        ),
        "Time-aware validation requires outer_repeats=1",
    ),
    InvalidConfigCase(
        "time_with_group_policy",
        lambda c: replace(
            c,
            validation=replace(
                c.validation,
                strategy="time",
                outer_repeats=1,
                test_size=5,
                duplicate_group_policy="group",
            ),
        ),
        "validation.duplicate_group_policy='group' is not supported with validation.strategy='time'",
    ),
    # Correlation validation matrix
    InvalidConfigCase(
        "correlation_review_threshold_zero",
        lambda c: replace(
            c,
            profiling=replace(
                c.profiling,
                correlation=replace(c.profiling.correlation, review_threshold=0.0),
            ),
        ),
        "profiling.correlation.review_threshold must be between 0 \\(exclusive\\) and 1",
    ),
    InvalidConfigCase(
        "correlation_review_threshold_above_one",
        lambda c: replace(
            c,
            profiling=replace(
                c.profiling,
                correlation=replace(c.profiling.correlation, review_threshold=1.5),
            ),
        ),
        "profiling.correlation.review_threshold must be between 0 \\(exclusive\\) and 1",
    ),
    InvalidConfigCase(
        "correlation_warning_below_review",
        lambda c: replace(
            c,
            profiling=replace(
                c.profiling,
                correlation=replace(
                    c.profiling.correlation, review_threshold=0.8, warning_threshold=0.6
                ),
            ),
        ),
        "profiling.correlation.warning_threshold must be greater than or equal to",
    ),
    InvalidConfigCase(
        "correlation_minimum_complete_pairs_below_two",
        lambda c: replace(
            c,
            profiling=replace(
                c.profiling,
                correlation=replace(c.profiling.correlation, minimum_complete_pairs=1),
            ),
        ),
        "minimum_complete_pairs must be at least 2",
    ),
    # Near-duplicate validation matrix
    InvalidConfigCase(
        "near_duplicate_decimals_negative",
        lambda c: replace(
            c,
            profiling=replace(
                c.profiling,
                duplicate_groups=replace(
                    c.profiling.duplicate_groups, rounding_decimals=-1
                ),
            ),
        ),
        "profiling.duplicate_groups.rounding_decimals cannot be negative",
    ),
    InvalidConfigCase(
        "near_duplicate_threshold_zero",
        lambda c: replace(
            c,
            profiling=replace(
                c.profiling,
                duplicate_groups=replace(
                    c.profiling.duplicate_groups, near_duplicate_threshold=0.0
                ),
            ),
        ),
        "near_duplicate_threshold must satisfy 0.0 < threshold <= 1.0",
    ),
    InvalidConfigCase(
        "near_duplicate_threshold_above_one",
        lambda c: replace(
            c,
            profiling=replace(
                c.profiling,
                duplicate_groups=replace(
                    c.profiling.duplicate_groups, near_duplicate_threshold=1.1
                ),
            ),
        ),
        "near_duplicate_threshold must satisfy 0.0 < threshold <= 1.0",
    ),
    InvalidConfigCase(
        "maximum_pairwise_rows_one",
        lambda c: replace(
            c,
            profiling=replace(
                c.profiling,
                duplicate_groups=replace(
                    c.profiling.duplicate_groups, maximum_pairwise_rows=1
                ),
            ),
        ),
        "maximum_pairwise_rows must be at least 2",
    ),
    InvalidConfigCase(
        "include_target_in_signature_true",
        lambda c: replace(
            c,
            profiling=replace(
                c.profiling,
                duplicate_groups=replace(
                    c.profiling.duplicate_groups, include_target_in_signature=True
                ),
            ),
        ),
        "include_target_in_signature=true is not supported",
    ),
    # Feature metadata cases
    InvalidConfigCase(
        "declared_derivation_without_derived_from",
        lambda c: replace(
            c,
            features={
                **c.features,
                "x1": FeatureConfig(role="smooth", derived="declared", derived_from=()),
            },
        ),
        "Derived feature 'x1' must declare derived_from",
    ),
    InvalidConfigCase(
        "self_derivation",
        lambda c: replace(
            c,
            features={
                **c.features,
                "x1": FeatureConfig(
                    role="smooth", derived="declared", derived_from=("x1",)
                ),
            },
        ),
        "Feature 'x1' cannot be derived from itself",
    ),
    InvalidConfigCase(
        "unknown_derivation_source",
        lambda c: replace(
            c,
            features={
                **c.features,
                "x1": FeatureConfig(
                    role="smooth", derived="declared", derived_from=("unknown_col",)
                ),
            },
        ),
        "declares unknown derivation sources",
    ),
    InvalidConfigCase(
        "derived_none_with_derived_from",
        lambda c: replace(
            c,
            features={
                **c.features,
                "x1": FeatureConfig(
                    role="smooth", derived="none", derived_from=("x2",)
                ),
            },
        ),
        "derived='none' but declares derived_from",
    ),
    # Interaction cases
    InvalidConfigCase(
        "explicit_interaction_self_pair",
        lambda c: replace(
            c,
            models=(
                ModelConfig(
                    id="gam_pair", interactions="explicit", pairs=(("x1", "x1"),)
                ),
            ),
        ),
        "Explicit interactions require two distinct smooth predictors",
    ),
    InvalidConfigCase(
        "explicit_interaction_unknown_predictor",
        lambda c: replace(
            c,
            models=(
                ModelConfig(
                    id="gam_pair", interactions="explicit", pairs=(("x1", "ghost"),)
                ),
            ),
        ),
        "Explicit interactions require two distinct smooth predictors",
    ),
    # Tags and metadata validation
    InvalidConfigCase(
        "empty_tag",
        lambda c: replace(c, tags=("  ",)),
        "Tags cannot be empty or whitespace-only",
    ),
    InvalidConfigCase(
        "tag_above_64_chars",
        lambda c: replace(c, tags=("a" * 65,)),
        "exceeds maximum length of 64 characters",
    ),
    InvalidConfigCase(
        "more_than_32_tags",
        lambda c: replace(c, tags=tuple(f"tag_{i}" for i in range(33))),
        "At most 32 tags can be configured",
    ),
    InvalidConfigCase(
        "duplicate_tags_case_insensitive",
        lambda c: replace(c, tags=("Candidate", "candidate")),
        "Duplicate tag 'candidate' specified",
    ),
    InvalidConfigCase(
        "metadata_key_starts_with_digit",
        lambda c: replace(c, metadata={"1key": "value"}),
        "Metadata key '1key' is invalid",
    ),
    InvalidConfigCase(
        "reserved_metadata_key",
        lambda c: replace(c, metadata={"created_at_utc": "value"}),
        "Metadata key 'created_at_utc' is reserved",
    ),
    InvalidConfigCase(
        "metadata_value_above_256_chars",
        lambda c: replace(c, metadata={"project": "x" * 257}),
        "exceeds maximum length of 256 characters",
    ),
)


@pytest.mark.parametrize("case", INVALID_CONFIG_CASES, ids=lambda c: c.case_id)
def test_invalid_configuration_matrix(tmp_path: Path, case: InvalidConfigCase) -> None:
    base = make_base_config(tmp_path)
    invalid = case.mutate(base)
    with pytest.raises(ConfigurationError, match=case.message_pattern):
        invalid.validate()


@pytest.mark.parametrize(
    ("strategy", "policy", "enabled", "valid"),
    [
        ("stratified", "report", True, True),
        ("stratified", "error", True, True),
        ("stratified", "group", True, False),
        ("stratified_group", "report", True, True),
        ("stratified_group", "error", True, True),
        ("stratified_group", "group", True, True),
        ("time", "report", True, True),
        ("time", "error", True, True),
        ("time", "group", True, False),
    ],
)
def test_duplicate_policy_strategy_matrix(
    tmp_path: Path,
    strategy: str,
    policy: str,
    enabled: bool,
    valid: bool,
) -> None:
    base = make_base_config(tmp_path)
    val_config = replace(
        base.validation,
        strategy=strategy,  # type: ignore[arg-type]
        duplicate_group_policy=policy,  # type: ignore[arg-type]
        outer_repeats=1 if strategy == "time" else 3,
        test_size=5 if strategy == "time" else None,
    )
    prof_config = replace(
        base.profiling,
        duplicate_groups=replace(base.profiling.duplicate_groups, enabled=enabled),
    )
    cfg = replace(
        base,
        validation=val_config,
        profiling=prof_config,
        group_column="group_col"
        if strategy == "stratified_group"
        else base.group_column,
        time_column="time_col" if strategy == "time" else base.time_column,
    )

    if valid:
        cfg.validate()
    else:
        with pytest.raises(ConfigurationError):
            cfg.validate()


def test_valid_boundary_cases(tmp_path: Path) -> None:
    base = make_base_config(tmp_path)

    # reserved_column_with_role_exclude
    valid_exclude = replace(
        base,
        features={
            "x1": FeatureConfig(role="smooth"),
            "target": FeatureConfig(role="exclude"),
            "row_id": FeatureConfig(role="exclude"),
        },
    )
    valid_exclude.validate()

    # near_duplicate_threshold boundary cases
    boundary_low = replace(
        base,
        profiling=replace(
            base.profiling,
            duplicate_groups=replace(
                base.profiling.duplicate_groups,
                near_duplicate_threshold=0.0001,
                rounding_decimals=0,
            ),
        ),
    )
    boundary_low.validate()

    boundary_high = replace(
        base,
        profiling=replace(
            base.profiling,
            duplicate_groups=replace(
                base.profiling.duplicate_groups, near_duplicate_threshold=1.0
            ),
        ),
    )
    boundary_high.validate()


def test_yaml_loader_rejects_invalid_config(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    data_path.write_text("x1,target\n1,A\n2,B\n", encoding="utf-8")

    yaml_content = f"""
schema_version: "1.1"
experiment:
  name: test_yaml
data:
  path: {data_path.as_posix()}
  target: target
  row_id: null
features:
  x1:
    role: smooth
models:
  - id: gam_main
validation:
  strategy: stratified
  outer_splits: 1
"""
    config_file = tmp_path / "invalid_config.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(
        ConfigurationError, match="validation.outer_splits must be at least 2"
    ):
        load_config(config_file)
