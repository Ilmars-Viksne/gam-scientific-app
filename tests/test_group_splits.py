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


def test_effective_duplicate_groups_do_not_leak_in_outer_and_inner_cv(
    tmp_path: Path,
) -> None:
    # rows 0 and 1 are duplicate predictors; rows 2 and 3 are duplicate predictors
    df = pd.DataFrame(
        {
            "x1": [1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "x2": [
                10.0,
                10.0,
                20.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
                90.0,
                100.0,
            ],
            "target": ["A", "B"] * 6,
        }
    )
    data_path = tmp_path / "dup_data.csv"
    df.to_csv(data_path, index=False)

    from gam_app.config import DuplicateGroupConfig
    from gam_app.diagnostics import analyze_duplicate_groups
    from gam_app.workflow import apply_duplicate_group_policy

    row_ids = pd.Series([f"r{i}" for i in range(len(df))])
    cfg = DuplicateGroupConfig()
    analysis = analyze_duplicate_groups(df[["x1", "x2"]], df["target"], row_ids, cfg)

    pol = apply_duplicate_group_policy(
        policy="group",
        configured_groups=None,
        duplicate_analysis=analysis,
        row_ids=row_ids,
        row_count=len(df),
    )

    eff_groups = pol.effective_groups
    assert eff_groups is not None

    config = ExperimentConfig(
        name="dup_test",
        data_path=data_path,
        target="target",
        row_id=None,
        features={
            "x1": FeatureConfig(role="smooth"),
            "x2": FeatureConfig(role="linear"),
        },
        validation=ValidationConfig(
            strategy="stratified_group",
            outer_splits=2,
            outer_repeats=1,
            inner_splits=2,
            random_state=42,
            duplicate_group_policy="group",
        ),
    )

    context = SplitContext(
        X=df[["x1", "x2"]],
        y=df["target"],
        row_ids=row_ids,
        groups=eff_groups,
        times=None,
    )

    manifest = create_split_manifest(config, context)

    # Assert no group leakage in outer folds
    for (_, _), fold in manifest.groupby(["repeat", "fold"]):
        train_groups = set(fold.loc[fold["partition"] == "train", "group_id"].dropna())
        test_groups = set(fold.loc[fold["partition"] == "test", "group_id"].dropna())
        assert train_groups.isdisjoint(test_groups)

    # Assert no group leakage in inner splits
    inner_splits = create_inner_splits(
        config=config,
        X=context.X,
        y=context.y,
        groups=eff_groups,
        times=None,
        seed=42,
    )
    for train_idx, valid_idx in inner_splits:
        train_groups = set(eff_groups.iloc[train_idx])
        valid_groups = set(eff_groups.iloc[valid_idx])
        assert train_groups.isdisjoint(valid_groups)
