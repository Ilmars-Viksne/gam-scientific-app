from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gam_app.config import ExperimentConfig, FeatureConfig, ValidationConfig
from gam_app.exceptions import ConfigurationError
from gam_app.splitting import (
    SplitContext,
    create_split_manifest,
)


def make_time_setup(tmp_path: Path, ties: bool = False):
    times = pd.date_range("2025-01-01", periods=30, freq="D")
    times_list = list(times)
    if ties:
        # Put duplicate timestamp across split 1 boundary
        times_list[9] = times_list[8]

    df = pd.DataFrame(
        {
            "x1": np.random.randn(30),
            "timestamp": times_list,
            "target": ["A", "B"] * 15,
        }
    )
    data_path = tmp_path / "time_data.csv"
    df.to_csv(data_path, index=False)

    config = ExperimentConfig(
        name="time_test",
        data_path=data_path,
        target="target",
        row_id=None,
        time_column="timestamp",
        features={"x1": FeatureConfig(role="smooth")},
        validation=ValidationConfig(
            strategy="time",
            outer_splits=3,
            outer_repeats=1,
            inner_splits=2,
            random_state=42,
        ),
    )

    context = SplitContext(
        X=df[["x1"]],
        y=df["target"],
        row_ids=pd.Series(np.arange(1, len(df) + 1).astype(str)),
        groups=None,
        times=df["timestamp"],
    )
    return config, context


def test_time_split_is_forward_only(tmp_path: Path) -> None:
    config, context = make_time_setup(tmp_path)
    manifest = create_split_manifest(config, context)

    manifest["time"] = pd.to_datetime(manifest["time"], utc=True)

    for (_, _), fold in manifest.groupby(["repeat", "fold"]):
        train_maximum = fold.loc[fold["partition"] == "train", "time"].max()

        test_minimum = fold.loc[fold["partition"] == "test", "time"].min()

        assert train_maximum < test_minimum


def test_time_test_partitions_do_not_overlap(tmp_path: Path) -> None:
    config, context = make_time_setup(tmp_path)
    manifest = create_split_manifest(config, context)

    tests = manifest.loc[manifest["partition"] == "test"]
    counts = tests.groupby("row_id").size()

    assert np.all(counts.to_numpy() == 1)


def test_time_validation_rejects_repeats(tmp_path: Path) -> None:
    config, _ = make_time_setup(tmp_path)
    invalid = replace(
        config,
        validation=replace(
            config.validation,
            outer_repeats=2,
        ),
    )

    with pytest.raises(
        ConfigurationError,
        match="requires outer_repeats=1",
    ):
        invalid.validate()


def test_time_split_rejects_equal_timestamp_boundary(tmp_path: Path) -> None:
    config, context = make_time_setup(tmp_path, ties=True)
    with pytest.raises(
        ValueError,
        match="equal-time boundary",
    ):
        create_split_manifest(config, context)
