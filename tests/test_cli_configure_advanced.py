from pathlib import Path

import pytest

from gam_app.cli import build_parser
from gam_app.config import load_config
from gam_app.exceptions import ConfigurationError


@pytest.fixture
def sample_data(tmp_path: Path) -> Path:
    data_path = tmp_path / "data.csv"
    data_path.write_text(
        "id,group_id,time_col,target,feat1\n1,g1,2020-01-01,0,10\n2,g2,2020-01-02,1,20\n3,g3,2020-01-03,0,30\n"
    )
    return data_path


def test_configure_writes_advanced_validation_arguments(
    sample_data: Path, tmp_path: Path
) -> None:
    output_cfg = tmp_path / "config.yaml"
    parser = build_parser()
    args = parser.parse_args(
        [
            "configure",
            "--data",
            str(sample_data),
            "--target",
            "target",
            "--output",
            str(output_cfg),
            "--outer-splits",
            "4",
            "--outer-repeats",
            "2",
            "--inner-splits",
            "3",
            "--random-state",
            "123",
            "--non-interactive",
        ]
    )
    args.func(args)

    config = load_config(output_cfg)
    assert config.validation.outer_splits == 4
    assert config.validation.outer_repeats == 2
    assert config.validation.inner_splits == 3
    assert config.validation.random_state == 123


def test_configure_explicit_split_counts_override_preset(
    sample_data: Path, tmp_path: Path
) -> None:
    output_cfg = tmp_path / "config.yaml"
    parser = build_parser()
    args = parser.parse_args(
        [
            "configure",
            "--data",
            str(sample_data),
            "--target",
            "target",
            "--output",
            str(output_cfg),
            "--preset",
            "thorough",
            "--outer-splits",
            "2",
            "--non-interactive",
        ]
    )
    args.func(args)

    config = load_config(output_cfg)
    assert config.validation.outer_splits == 2
    assert config.validation.outer_repeats == 5  # thorough default


def test_configure_time_defaults_outer_repeats_to_one(
    sample_data: Path, tmp_path: Path
) -> None:
    output_cfg = tmp_path / "config.yaml"
    parser = build_parser()
    args = parser.parse_args(
        [
            "configure",
            "--data",
            str(sample_data),
            "--target",
            "target",
            "--output",
            str(output_cfg),
            "--time",
            "time_col",
            "--validation-strategy",
            "time",
            "--non-interactive",
        ]
    )
    args.func(args)

    config = load_config(output_cfg)
    assert config.validation.outer_repeats == 1


def test_configure_writes_duplicate_analysis_arguments(
    sample_data: Path, tmp_path: Path
) -> None:
    output_cfg = tmp_path / "config.yaml"
    parser = build_parser()
    args = parser.parse_args(
        [
            "configure",
            "--data",
            str(sample_data),
            "--target",
            "target",
            "--output",
            str(output_cfg),
            "--near-duplicate-decimals",
            "6",
            "--near-duplicate-threshold",
            "0.95",
            "--maximum-pairwise-rows",
            "5000",
            "--non-interactive",
        ]
    )
    args.func(args)

    config = load_config(output_cfg)
    assert config.profiling.duplicate_groups.rounding_decimals == 6
    assert config.profiling.duplicate_groups.near_duplicate_threshold == 0.95
    assert config.profiling.duplicate_groups.maximum_pairwise_rows == 5000


def test_configure_rejects_group_policy_with_stratified_strategy(
    sample_data: Path, tmp_path: Path
) -> None:
    output_cfg = tmp_path / "config.yaml"
    parser = build_parser()
    args = parser.parse_args(
        [
            "configure",
            "--data",
            str(sample_data),
            "--target",
            "target",
            "--output",
            str(output_cfg),
            "--validation-strategy",
            "stratified",
            "--duplicate-group-policy",
            "group",
            "--non-interactive",
        ]
    )
    with pytest.raises(
        ConfigurationError, match="validation.duplicate_group_policy='group' requires"
    ):
        args.func(args)

    assert not output_cfg.exists()
