from dataclasses import replace
from pathlib import Path

import pytest

from gam_app.config import (
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    ValidationConfig,
)
from gam_app.exceptions import ConfigurationError


@pytest.fixture
def base_config(tmp_path: Path) -> ExperimentConfig:
    data_file = tmp_path / "data.csv"
    data_file.write_text("a,b,c\n1,2,3")
    return ExperimentConfig(
        name="test_exp",
        data_path=data_file,
        target="c",
        row_id="id_col",
        group_column="group_col",
        time_column="time_col",
        features={
            "id_col": FeatureConfig(role="exclude"),
            "group_col": FeatureConfig(role="exclude"),
            "time_col": FeatureConfig(role="exclude"),
            "a": FeatureConfig(role="smooth"),
            "b": FeatureConfig(role="linear"),
        },
        models=(ModelConfig(id="m1"),),
        validation=ValidationConfig(
            strategy="stratified",
            outer_splits=5,
            outer_repeats=3,
            inner_splits=5,
            duplicate_group_policy="report",
        ),
    )


def test_valid_base_config(base_config: ExperimentConfig) -> None:
    base_config.validate()


def test_split_counts_validation(base_config: ExperimentConfig) -> None:
    with pytest.raises(
        ConfigurationError, match="validation.outer_splits must be at least 2"
    ):
        replace(
            base_config, validation=replace(base_config.validation, outer_splits=1)
        ).validate()

    with pytest.raises(
        ConfigurationError, match="validation.inner_splits must be at least 2"
    ):
        replace(
            base_config, validation=replace(base_config.validation, inner_splits=1)
        ).validate()

    with pytest.raises(
        ConfigurationError, match="validation.outer_repeats must be at least 1"
    ):
        replace(
            base_config, validation=replace(base_config.validation, outer_repeats=0)
        ).validate()

    with pytest.raises(
        ConfigurationError, match="validation.test_size must be at least 1"
    ):
        replace(
            base_config, validation=replace(base_config.validation, test_size=0)
        ).validate()

    with pytest.raises(ConfigurationError, match="validation.gap cannot be negative"):
        replace(
            base_config, validation=replace(base_config.validation, gap=-1)
        ).validate()


def test_data_role_uniqueness(base_config: ExperimentConfig) -> None:
    cfg = replace(base_config, group_column="c")  # c is also target
    with pytest.raises(ConfigurationError, match="Data-role columns must be distinct"):
        cfg.validate()


def test_reserved_columns_in_active_predictors(base_config: ExperimentConfig) -> None:
    cfg = replace(
        base_config,
        features={
            **base_config.features,
            "group_col": FeatureConfig(role="smooth"),
        },
    )
    with pytest.raises(
        ConfigurationError,
        match="Target, row-id, group, and time columns cannot be used as active model predictors",
    ):
        cfg.validate()


@pytest.mark.parametrize(
    ("strategy", "gap", "test_size"),
    [
        ("stratified", 1, None),
        ("stratified", 0, 20),
        ("stratified_group", 1, None),
        ("stratified_group", 0, 20),
    ],
)
def test_temporal_parameters_rejected_for_non_time_strategy(
    base_config: ExperimentConfig, strategy: str, gap: int, test_size: int | None
) -> None:
    cfg = replace(
        base_config,
        validation=replace(
            base_config.validation,
            strategy=strategy,
            gap=gap,
            test_size=test_size,
            duplicate_group_policy="report",
        ),
    )
    with pytest.raises(
        ConfigurationError, match="applies only when validation.strategy='time'"
    ):
        cfg.validate()


def test_time_strategy_validation(base_config: ExperimentConfig) -> None:
    # missing time column
    cfg_no_time = replace(
        base_config,
        time_column=None,
        validation=replace(base_config.validation, strategy="time", outer_repeats=1),
    )
    with pytest.raises(
        ConfigurationError,
        match="data.time is required when validation.strategy='time'",
    ):
        cfg_no_time.validate()

    # repeats != 1
    cfg_repeats = replace(
        base_config,
        validation=replace(base_config.validation, strategy="time", outer_repeats=2),
    )
    with pytest.raises(
        ConfigurationError, match="Time-aware validation requires outer_repeats=1"
    ):
        cfg_repeats.validate()

    # duplicate_group_policy='group' not supported for time
    cfg_policy_group = replace(
        base_config,
        validation=replace(
            base_config.validation,
            strategy="time",
            outer_repeats=1,
            duplicate_group_policy="group",
        ),
    )
    with pytest.raises(
        ConfigurationError, match="is not supported with validation.strategy='time'"
    ):
        cfg_policy_group.validate()


def test_correlation_and_duplicate_validation(base_config: ExperimentConfig) -> None:
    # Review threshold <= 0
    cfg = replace(
        base_config,
        profiling=replace(
            base_config.profiling,
            correlation=replace(
                base_config.profiling.correlation, review_threshold=0.0
            ),
        ),
    )
    with pytest.raises(ConfigurationError, match="review_threshold must be between 0"):
        cfg.validate()

    # Warning threshold < Review threshold
    cfg = replace(
        base_config,
        profiling=replace(
            base_config.profiling,
            correlation=replace(
                base_config.profiling.correlation,
                review_threshold=0.8,
                warning_threshold=0.7,
            ),
        ),
    )
    with pytest.raises(ConfigurationError, match="must be greater than or equal to"):
        cfg.validate()
