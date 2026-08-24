from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from .config import ExperimentConfig
from .exceptions import DataValidationError


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)

        if left_root == right_root:
            return

        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root

        self.parent[right_root] = left_root

        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def merge_group_constraints(
    *,
    row_count: int,
    configured_groups: pd.Series | None,
    duplicate_signatures: pd.Series | None,
) -> pd.Series | None:
    if configured_groups is None and duplicate_signatures is None:
        return None

    union = DisjointSet(row_count)

    for labels in (configured_groups, duplicate_signatures):
        if labels is None:
            continue

        first_index_by_label: dict[str, int] = {}

        for row_index, value in enumerate(labels.astype(str)):
            if value in first_index_by_label:
                union.union(
                    first_index_by_label[value],
                    row_index,
                )
            else:
                first_index_by_label[value] = row_index

    roots = [union.find(index) for index in range(row_count)]
    sorted_unique_roots = sorted(set(roots))
    root_ids = {
        root: group_index
        for group_index, root in enumerate(
            sorted_unique_roots,
            start=1,
        )
    }

    return pd.Series(
        [f"group_{root_ids[root]:06d}" for root in roots],
        name="validation_group",
    )


@dataclass(frozen=True, slots=True)
class SplitContext:
    X: pd.DataFrame
    y: pd.Series
    row_ids: pd.Series
    groups: pd.Series | None
    times: pd.Series | None


@dataclass(frozen=True, slots=True)
class IndexedSplit:
    repeat: int
    fold: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


def stratified_outer_splits(
    config: ExperimentConfig,
    context: SplitContext,
) -> list[IndexedSplit]:
    splitter = RepeatedStratifiedKFold(
        n_splits=config.validation.outer_splits,
        n_repeats=config.validation.outer_repeats,
        random_state=config.validation.random_state,
    )

    results: list[IndexedSplit] = []

    for iteration, (train, test) in enumerate(
        splitter.split(context.X, context.y),
        start=1,
    ):
        repeat = (iteration - 1) // config.validation.outer_splits + 1
        fold = (iteration - 1) % config.validation.outer_splits + 1

        results.append(
            IndexedSplit(
                repeat=repeat,
                fold=fold,
                train_indices=tuple(int(i) for i in train),
                test_indices=tuple(int(i) for i in test),
            )
        )

    return results


def stratified_group_outer_splits(
    config: ExperimentConfig,
    context: SplitContext,
) -> list[IndexedSplit]:
    if context.groups is None:
        raise ValueError("Group-aware validation requires group labels.")

    results: list[IndexedSplit] = []

    for repeat in range(1, config.validation.outer_repeats + 1):
        seed = config.validation.random_state + repeat - 1

        splitter = StratifiedGroupKFold(
            n_splits=config.validation.outer_splits,
            shuffle=True,
            random_state=seed,
        )

        for fold, (train, test) in enumerate(
            splitter.split(
                context.X,
                context.y,
                groups=context.groups,
            ),
            start=1,
        ):
            results.append(
                IndexedSplit(
                    repeat=repeat,
                    fold=fold,
                    train_indices=tuple(int(i) for i in train),
                    test_indices=tuple(int(i) for i in test),
                )
            )

    return results


def time_outer_splits(
    config: ExperimentConfig,
    context: SplitContext,
) -> list[IndexedSplit]:
    if context.times is None:
        raise ValueError("Time-aware validation requires timestamps.")

    if context.times.isna().any():
        raise ValueError("Time-aware validation cannot use missing timestamps.")

    ordered_indices = (
        pd.DataFrame(
            {
                "row_index": range(len(context.X)),
                "time": context.times.to_numpy(),
                "row_id": context.row_ids.astype(str).to_numpy(),
            }
        )
        .sort_values(
            ["time", "row_id"],
            kind="stable",
        )["row_index"]
        .to_numpy(dtype=int)
    )

    splitter = TimeSeriesSplit(
        n_splits=config.validation.outer_splits,
        gap=config.validation.gap,
        test_size=config.validation.test_size,
    )

    results: list[IndexedSplit] = []

    for fold, (ordered_train, ordered_test) in enumerate(
        splitter.split(ordered_indices),
        start=1,
    ):
        train = ordered_indices[ordered_train]
        test = ordered_indices[ordered_test]

        train_times = context.times.iloc[train]
        test_times = context.times.iloc[test]

        if train_times.max() >= test_times.min():
            raise ValueError(
                "Temporal leakage or equal-time boundary detected "
                f"for repeat=1, fold={fold}. "
                f"maximum_train_time={train_times.max()}; "
                f"minimum_test_time={test_times.min()}."
            )

        results.append(
            IndexedSplit(
                repeat=1,
                fold=fold,
                train_indices=tuple(int(i) for i in train),
                test_indices=tuple(int(i) for i in test),
            )
        )

    return results


def create_outer_splits(
    config: ExperimentConfig,
    context: SplitContext,
) -> list[IndexedSplit]:
    strategy = config.validation.strategy

    if strategy == "stratified":
        return stratified_outer_splits(config, context)

    if strategy == "stratified_group":
        return stratified_group_outer_splits(config, context)

    if strategy == "time":
        return time_outer_splits(config, context)

    raise ValueError(f"Unsupported validation strategy: {strategy!r}.")


def validate_no_group_leakage(manifest: pd.DataFrame) -> None:
    if "group_id" not in manifest.columns:
        raise ValueError("Group-aware manifest lacks group_id.")

    for (repeat, fold), subset in manifest.groupby(["repeat", "fold"], sort=True):
        train_groups = set(
            subset.loc[subset["partition"] == "train", "group_id"].dropna()
        )
        test_groups = set(
            subset.loc[subset["partition"] == "test", "group_id"].dropna()
        )

        overlap = sorted(train_groups & test_groups)
        if overlap:
            raise ValueError(
                "Group leakage detected for "
                f"repeat={repeat}, fold={fold}. "
                f"Overlapping groups include: {overlap[:10]}."
            )


def validate_forward_time_order(manifest: pd.DataFrame) -> None:
    if "time" not in manifest.columns:
        raise ValueError("Time-aware manifest lacks time.")

    parsed = manifest.copy()
    parsed["time"] = pd.to_datetime(parsed["time"], errors="raise", utc=True)

    for (repeat, fold), subset in parsed.groupby(["repeat", "fold"], sort=True):
        train_times = subset.loc[subset["partition"] == "train", "time"]
        test_times = subset.loc[subset["partition"] == "test", "time"]

        maximum_train = train_times.max()
        minimum_test = test_times.min()

        if maximum_train >= minimum_test:
            raise ValueError(
                "Temporal leakage or equal-time boundary detected "
                f"for repeat={repeat}, fold={fold}. "
                f"maximum_train_time={maximum_train}; "
                f"minimum_test_time={minimum_test}."
            )


def validate_one_test_assignment_per_repeat(manifest: pd.DataFrame) -> None:
    test_rows = manifest.loc[manifest["partition"] == "test"]
    counts = test_rows.groupby(["repeat", "row_id"]).size()
    if not np.all(counts.to_numpy() == 1):
        raise ValueError("Every row must appear exactly once as test data per repeat.")


def validate_test_assignments_do_not_overlap(manifest: pd.DataFrame) -> None:
    test_rows = manifest.loc[manifest["partition"] == "test"]
    counts = test_rows.groupby("row_id").size()
    if not np.all(counts.to_numpy() == 1):
        raise ValueError("Test partitions contain overlapping row IDs.")


def validate_split_manifest(
    config: ExperimentConfig,
    context: SplitContext,
    manifest: pd.DataFrame,
) -> None:
    strategy = config.validation.strategy

    if strategy == "stratified":
        validate_one_test_assignment_per_repeat(manifest)

    elif strategy == "stratified_group":
        validate_one_test_assignment_per_repeat(manifest)
        validate_no_group_leakage(manifest)

    elif strategy == "time":
        validate_test_assignments_do_not_overlap(manifest)
        validate_forward_time_order(manifest)


def create_split_manifest(
    config: ExperimentConfig,
    context: SplitContext,
) -> pd.DataFrame:
    indexed_splits = create_outer_splits(config, context)
    rows: list[dict[str, Any]] = []

    for split in indexed_splits:
        for partition, indices in (
            ("train", split.train_indices),
            ("test", split.test_indices),
        ):
            for row_index in indices:
                rows.append(
                    {
                        "repeat": split.repeat,
                        "fold": split.fold,
                        "row_id": str(context.row_ids.iloc[row_index]),
                        "row_index": row_index,
                        "partition": partition,
                        "validation_strategy": config.validation.strategy,
                        "group_id": (
                            str(context.groups.iloc[row_index])
                            if context.groups is not None
                            else None
                        ),
                        "time": (
                            context.times.iloc[row_index].isoformat()
                            if context.times is not None
                            else None
                        ),
                    }
                )

    manifest = pd.DataFrame(rows)
    validate_split_manifest(config, context, manifest)
    return manifest


def build_time_inner_splits(
    *,
    X: pd.DataFrame,
    times: pd.Series,
    n_splits: int,
    gap: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if times.isna().any():
        raise ValueError("Inner time-aware CV requires non-missing timestamps.")

    ordered_indices = (
        pd.DataFrame(
            {
                "row_index": range(len(X)),
                "time": times.to_numpy(),
            }
        )
        .sort_values(["time", "row_index"], kind="stable")["row_index"]
        .to_numpy(dtype=int)
    )

    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    inner_splits: list[tuple[np.ndarray, np.ndarray]] = []

    for ordered_train, ordered_valid in splitter.split(ordered_indices):
        train = ordered_indices[ordered_train]
        valid = ordered_indices[ordered_valid]

        train_times = times.iloc[train]
        valid_times = times.iloc[valid]

        if train_times.max() >= valid_times.min():
            raise ValueError(
                "Inner temporal leakage or equal-time boundary detected. "
                f"maximum_train_time={train_times.max()}; "
                f"minimum_valid_time={valid_times.min()}."
            )

        inner_splits.append((train, valid))

    return inner_splits


def create_inner_splits(
    *,
    config: ExperimentConfig,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None,
    times: pd.Series | None,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    strategy = config.validation.strategy

    if strategy == "stratified":
        splitter = StratifiedKFold(
            n_splits=config.validation.inner_splits,
            shuffle=True,
            random_state=seed,
        )
        return [(train, valid) for train, valid in splitter.split(X, y)]

    if strategy == "stratified_group":
        if groups is None:
            raise ValueError("Inner group-aware CV requires groups.")

        splitter = StratifiedGroupKFold(
            n_splits=config.validation.inner_splits,
            shuffle=True,
            random_state=seed,
        )
        return [(train, valid) for train, valid in splitter.split(X, y, groups=groups)]

    if strategy == "time":
        if times is None:
            raise ValueError("Inner time-aware CV requires timestamps.")

        return build_time_inner_splits(
            X=X,
            times=times,
            n_splits=config.validation.inner_splits,
            gap=config.validation.gap,
        )

    raise ValueError(f"Unsupported validation strategy: {strategy!r}.")


def validate_class_coverage(
    *,
    y: pd.Series,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    context: str,
) -> None:
    train_classes = set(y.iloc[train_indices].astype(str))
    test_classes = set(y.iloc[test_indices].astype(str))

    if len(train_classes) < 2:
        raise DataValidationError(
            f"{context} training partition contains fewer than two target classes."
        )

    unseen_test_classes = sorted(test_classes - train_classes)
    if unseen_test_classes:
        raise DataValidationError(
            f"{context} test partition contains classes absent from training: "
            f"{unseen_test_classes}."
        )
