from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gam_app.config import ExperimentConfig, FeatureConfig, ValidationConfig
from gam_app.splitting import (
    SplitContext,
    create_inner_splits,
    create_split_manifest,
)


def make_grouped_setup(tmp_path: Path):
    df = pd.DataFrame(
        {
            "x1": np.random.randn(30),
            "batch": ["g1", "g1", "g2", "g2", "g3", "g3"] * 5,
            "target": ["A", "B"] * 15,
        }
    )
    data_path = tmp_path / "grouped_data.csv"
    df.to_csv(data_path, index=False)

    config = ExperimentConfig(
        name="grouped_test",
        data_path=data_path,
        target="target",
        row_id=None,
        group_column="batch",
        features={"x1": FeatureConfig(role="smooth")},
        validation=ValidationConfig(
            strategy="stratified_group",
            outer_splits=3,
            outer_repeats=2,
            inner_splits=2,
            random_state=42,
        ),
    )

    context = SplitContext(
        X=df[["x1"]],
        y=df["target"],
        row_ids=pd.Series(np.arange(1, len(df) + 1).astype(str)),
        groups=df["batch"],
        times=None,
    )
    return config, context


def test_group_split_has_no_group_leakage(tmp_path: Path) -> None:
    config, context = make_grouped_setup(tmp_path)
    manifest = create_split_manifest(config, context)

    for (_, _), fold in manifest.groupby(["repeat", "fold"]):
        train_groups = set(fold.loc[fold["partition"] == "train", "group_id"].dropna())
        test_groups = set(fold.loc[fold["partition"] == "test", "group_id"].dropna())

        assert train_groups.isdisjoint(test_groups)


def test_group_split_tests_each_row_once_per_repeat(tmp_path: Path) -> None:
    config, context = make_grouped_setup(tmp_path)
    manifest = create_split_manifest(config, context)

    test_rows = manifest.loc[manifest["partition"] == "test"]
    counts = test_rows.groupby(["repeat", "row_id"]).size()

    assert np.all(counts.to_numpy() == 1)


def test_inner_group_splits_do_not_leak_groups(tmp_path: Path) -> None:
    config, context = make_grouped_setup(tmp_path)
    splits = create_inner_splits(
        config=config,
        X=context.X,
        y=context.y,
        groups=context.groups,
        times=None,
        seed=42,
    )

    for train, valid in splits:
        train_groups = set(context.groups.iloc[train])
        valid_groups = set(context.groups.iloc[valid])

        assert train_groups.isdisjoint(valid_groups)
