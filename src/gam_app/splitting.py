from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

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

# ---------------------------------------------------------------------------
# Split-integrity result model & constants
# ---------------------------------------------------------------------------

IntegrityScope = Literal["run", "repeat", "fold"]


@dataclass(frozen=True, slots=True)
class SplitIntegrityResult:
    strategy: str
    scope: IntegrityScope
    check: str
    passed: bool
    observed: str
    expected: str
    details: str = ""
    repeat: int | None = None
    fold: int | None = None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


SPLIT_INTEGRITY_COLUMNS = [
    "strategy",
    "scope",
    "repeat",
    "fold",
    "check",
    "passed",
    "observed",
    "expected",
    "details",
]

BASE_MANIFEST_COLUMNS = {
    "repeat",
    "fold",
    "row_id",
    "row_index",
    "partition",
    "validation_strategy",
}

ALL_INTEGRITY_CHECKS = [
    "required_columns_present",
    "valid_partition_labels",
    "no_duplicate_manifest_rows",
    "no_train_test_overlap",
    "complete_row_coverage",
    "one_test_assignment_per_repeat",
    "no_group_leakage",
    "strict_temporal_order",
    "non_overlapping_temporal_tests",
    "outer_repeats_equal_one",
    "test_classes_present_in_training",
]


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
    configured_groups: pd.Series | None = None,
    duplicate_signatures: pd.Series | None = None,
    label_constraints: Sequence[pd.Series | None] = (),
    edge_constraints: Sequence[pd.DataFrame | None] = (),
    row_ids: pd.Series | None = None,
) -> pd.Series:
    union = DisjointSet(row_count)

    # Convert row_id string lookup if edge_constraints are provided
    row_id_to_idx: dict[str, int] = {}
    if row_ids is not None:
        for idx, rid in enumerate(row_ids.astype(str)):
            row_id_to_idx[rid] = idx

    all_labels: list[pd.Series] = []
    if configured_groups is not None:
        all_labels.append(configured_groups)
    if duplicate_signatures is not None:
        all_labels.append(duplicate_signatures)
    for lbl in label_constraints:
        if lbl is not None:
            all_labels.append(lbl)

    for labels in all_labels:
        first_index_by_label: dict[str, int] = {}
        for row_index, value in enumerate(labels.astype(str)):
            if value in first_index_by_label:
                union.union(
                    first_index_by_label[value],
                    row_index,
                )
            else:
                first_index_by_label[value] = row_index

    for edge_df in edge_constraints:
        if edge_df is None or edge_df.empty:
            continue
        for row in edge_df.itertuples(index=False):
            left_id = str(row.left_row_id)
            right_id = str(row.right_row_id)
            if left_id in row_id_to_idx and right_id in row_id_to_idx:
                union.union(row_id_to_idx[left_id], row_id_to_idx[right_id])

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


# ---------------------------------------------------------------------------
# Split-generation functions
# ---------------------------------------------------------------------------


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

    manifest = pd.DataFrame(
        rows,
        columns=[
            "repeat",
            "fold",
            "row_id",
            "row_index",
            "partition",
            "validation_strategy",
            "group_id",
            "time",
        ],
    )
    return manifest


# ---------------------------------------------------------------------------
# Split-integrity result helpers
# ---------------------------------------------------------------------------


def _format_values(values: list[Any], limit: int = 10) -> str:
    rendered = [str(value) for value in values[:limit]]

    if len(values) > limit:
        rendered.append(f"... and {len(values) - limit} more")

    return ", ".join(rendered)


def _result(
    *,
    strategy: str,
    scope: IntegrityScope,
    check: str,
    passed: bool,
    observed: Any,
    expected: Any,
    details: str = "",
    repeat: int | None = None,
    fold: int | None = None,
) -> SplitIntegrityResult:
    return SplitIntegrityResult(
        strategy=strategy,
        scope=scope,
        check=check,
        passed=bool(passed),
        observed=str(observed),
        expected=str(expected),
        details=details,
        repeat=repeat,
        fold=fold,
    )


# ---------------------------------------------------------------------------
# Individual split-integrity check functions
# ---------------------------------------------------------------------------


def check_required_manifest_columns(
    *,
    strategy: str,
    manifest: pd.DataFrame,
) -> list[SplitIntegrityResult]:
    missing = sorted(BASE_MANIFEST_COLUMNS - set(manifest.columns))
    passed = not missing

    return [
        _result(
            strategy=strategy,
            scope="run",
            check="required_columns_present",
            passed=passed,
            observed=(
                "all required columns present"
                if passed
                else f"missing: {_format_values(missing)}"
            ),
            expected="all required manifest columns present",
            details=(
                "The split manifest contains the columns required "
                "for integrity validation."
                if passed
                else "The split manifest cannot be validated completely."
            ),
        )
    ]


def check_partition_labels(
    *,
    strategy: str,
    manifest: pd.DataFrame,
) -> list[SplitIntegrityResult]:
    if "partition" not in manifest.columns:
        return []

    allowed = {"train", "test"}
    observed = set(manifest["partition"].dropna().astype(str))
    invalid = sorted(observed - allowed)
    passed = not invalid

    return [
        _result(
            strategy=strategy,
            scope="run",
            check="valid_partition_labels",
            passed=passed,
            observed=(
                ", ".join(sorted(observed)) if observed else "no partition labels"
            ),
            expected="train, test",
            details=(
                "Only the supported partition labels occur."
                if passed
                else f"Invalid labels: {_format_values(invalid)}"
            ),
        )
    ]


def check_duplicate_manifest_rows(
    *,
    strategy: str,
    manifest: pd.DataFrame,
) -> list[SplitIntegrityResult]:
    required = {"repeat", "fold", "row_id", "partition"}
    if not required.issubset(manifest.columns):
        return []

    key = ["repeat", "fold", "row_id", "partition"]
    duplicate_mask = manifest.duplicated(subset=key, keep=False)
    duplicate_count = int(duplicate_mask.sum())
    passed = duplicate_count == 0

    return [
        _result(
            strategy=strategy,
            scope="run",
            check="no_duplicate_manifest_rows",
            passed=passed,
            observed=duplicate_count,
            expected=0,
            details=(
                "Every row assignment is unique."
                if passed
                else "Duplicate repeat/fold/row/partition records were found."
            ),
        )
    ]


def check_train_test_overlap(
    *,
    strategy: str,
    manifest: pd.DataFrame,
) -> list[SplitIntegrityResult]:
    required = {"repeat", "fold", "row_id", "partition"}
    if not required.issubset(manifest.columns):
        return []

    results: list[SplitIntegrityResult] = []

    groups = manifest.groupby(
        ["repeat", "fold"],
        sort=True,
        dropna=False,
    )

    for (repeat_key, fold_key), subset in groups:
        repeat_val = int(str(repeat_key))
        fold_val = int(str(fold_key))

        train_ids = set(
            subset.loc[subset["partition"] == "train", "row_id"].astype(str)
        )
        test_ids = set(subset.loc[subset["partition"] == "test", "row_id"].astype(str))

        overlap = sorted(train_ids & test_ids)
        passed = not overlap

        results.append(
            _result(
                strategy=strategy,
                scope="fold",
                repeat=repeat_val,
                fold=fold_val,
                check="no_train_test_overlap",
                passed=passed,
                observed=len(overlap),
                expected=0,
                details=(
                    "No row IDs occur in both training and test."
                    if passed
                    else f"Overlapping row IDs: {_format_values(overlap)}"
                ),
            )
        )

    return results


def check_complete_row_coverage(
    *,
    strategy: str,
    context: SplitContext,
    manifest: pd.DataFrame,
) -> list[SplitIntegrityResult]:
    required = {"repeat", "fold", "row_id"}
    if not required.issubset(manifest.columns):
        return []

    expected_ids = set(context.row_ids.astype(str))
    results: list[SplitIntegrityResult] = []

    for (repeat_key, fold_key), subset in manifest.groupby(
        ["repeat", "fold"],
        sort=True,
    ):
        repeat_val = int(str(repeat_key))
        fold_val = int(str(fold_key))

        observed_ids = set(subset["row_id"].astype(str))
        missing = sorted(expected_ids - observed_ids)
        unexpected = sorted(observed_ids - expected_ids)

        passed = not missing and not unexpected

        details: list[str] = []

        if missing:
            details.append(f"Missing row IDs: {_format_values(missing)}")

        if unexpected:
            details.append(f"Unexpected row IDs: {_format_values(unexpected)}")

        if passed:
            details.append("Every source row is assigned in this fold.")

        results.append(
            _result(
                strategy=strategy,
                scope="fold",
                repeat=repeat_val,
                fold=fold_val,
                check="complete_row_coverage",
                passed=passed,
                observed=len(observed_ids),
                expected=len(expected_ids),
                details=" ".join(details),
            )
        )

    return results


def check_one_test_assignment_per_repeat(
    *,
    strategy: str,
    context: SplitContext,
    manifest: pd.DataFrame,
) -> list[SplitIntegrityResult]:
    required = {"repeat", "row_id", "partition"}
    if not required.issubset(manifest.columns):
        return []

    expected_ids = set(context.row_ids.astype(str))
    results: list[SplitIntegrityResult] = []

    repeat_vals = [int(str(r)) for r in manifest["repeat"].dropna().tolist()]
    unique_repeats: list[int] = sorted(set(repeat_vals))

    for repeat in unique_repeats:
        test_rows = manifest.loc[
            (manifest["repeat"] == repeat) & (manifest["partition"] == "test")
        ]

        counts = test_rows.groupby("row_id").size()

        missing = sorted(expected_ids - set(counts.index.astype(str)))
        repeated = sorted(
            [str(row_id) for row_id, count in counts.items() if int(count) != 1]
        )

        passed = not missing and not repeated

        details: list[str] = []

        if missing:
            details.append(f"Rows never assigned to test: {_format_values(missing)}")

        if repeated:
            details.append(
                f"Rows assigned to test more than once: {_format_values(repeated)}"
            )

        if passed:
            details.append(
                "Every source row appears exactly once as test data in this repeat."
            )

        results.append(
            _result(
                strategy=strategy,
                scope="repeat",
                repeat=repeat,
                fold=None,
                check="one_test_assignment_per_repeat",
                passed=passed,
                observed=int(len(counts)),
                expected=int(len(expected_ids)),
                details=" ".join(details),
            )
        )

    return results


def check_no_group_leakage(
    *,
    strategy: str,
    manifest: pd.DataFrame,
) -> list[SplitIntegrityResult]:
    if "group_id" not in manifest.columns:
        return [
            _result(
                strategy=strategy,
                scope="run",
                check="group_column_present",
                passed=False,
                observed="group_id absent",
                expected="group_id present",
                details=(
                    "Group-aware validation requires group IDs in the split manifest."
                ),
            )
        ]

    results: list[SplitIntegrityResult] = []

    for (repeat_key, fold_key), subset in manifest.groupby(
        ["repeat", "fold"],
        sort=True,
    ):
        repeat_val = int(str(repeat_key))
        fold_val = int(str(fold_key))

        train_groups = set(
            subset.loc[subset["partition"] == "train", "group_id"].dropna().astype(str)
        )

        test_groups = set(
            subset.loc[subset["partition"] == "test", "group_id"].dropna().astype(str)
        )

        overlap = sorted(train_groups & test_groups)
        passed = not overlap

        results.append(
            _result(
                strategy=strategy,
                scope="fold",
                repeat=repeat_val,
                fold=fold_val,
                check="no_group_leakage",
                passed=passed,
                observed=len(overlap),
                expected=0,
                details=(
                    "No effective group crosses training and test."
                    if passed
                    else f"Overlapping groups: {_format_values(overlap)}"
                ),
            )
        )

    return results


def check_forward_time_order(
    *,
    strategy: str,
    manifest: pd.DataFrame,
) -> list[SplitIntegrityResult]:
    if "time" not in manifest.columns:
        return [
            _result(
                strategy=strategy,
                scope="run",
                check="time_column_present",
                passed=False,
                observed="time absent",
                expected="time present",
                details=(
                    "Time-aware validation requires timestamps in the split manifest."
                ),
            )
        ]

    parsed = manifest.copy()

    try:
        parsed["time"] = pd.to_datetime(
            parsed["time"],
            errors="raise",
            utc=True,
        )
    except (TypeError, ValueError) as error:
        return [
            _result(
                strategy=strategy,
                scope="run",
                check="timestamps_parseable",
                passed=False,
                observed=type(error).__name__,
                expected="all timestamps parseable as UTC datetimes",
                details=str(error),
            )
        ]

    results: list[SplitIntegrityResult] = []

    for (repeat_key, fold_key), subset in parsed.groupby(
        ["repeat", "fold"],
        sort=True,
    ):
        repeat_val = int(str(repeat_key))
        fold_val = int(str(fold_key))

        train_times = subset.loc[
            subset["partition"] == "train",
            "time",
        ]

        test_times = subset.loc[
            subset["partition"] == "test",
            "time",
        ]

        if train_times.empty or test_times.empty:
            results.append(
                _result(
                    strategy=strategy,
                    scope="fold",
                    repeat=repeat_val,
                    fold=fold_val,
                    check="strict_temporal_order",
                    passed=False,
                    observed=(
                        f"train_count={len(train_times)}, test_count={len(test_times)}"
                    ),
                    expected="nonempty train and test partitions",
                    details=(
                        "Temporal ordering cannot be evaluated for an empty partition."
                    ),
                )
            )
            continue

        maximum_train = train_times.max()
        minimum_test = test_times.min()
        passed = maximum_train < minimum_test

        results.append(
            _result(
                strategy=strategy,
                scope="fold",
                repeat=repeat_val,
                fold=fold_val,
                check="strict_temporal_order",
                passed=passed,
                observed=(
                    f"maximum_train_time={maximum_train}; "
                    f"minimum_test_time={minimum_test}"
                ),
                expected="train_max < test_min",
                details=(
                    "Training strictly precedes testing."
                    if passed
                    else ("Temporal leakage or an equal-time boundary was detected.")
                ),
            )
        )

    return results


def check_temporal_test_assignments_do_not_overlap(
    *,
    strategy: str,
    manifest: pd.DataFrame,
) -> list[SplitIntegrityResult]:
    required = {"row_id", "partition"}
    if not required.issubset(manifest.columns):
        return []

    test_rows = manifest.loc[
        manifest["partition"] == "test",
        "row_id",
    ].astype(str)

    counts = test_rows.value_counts()
    overlapping = sorted(
        [str(row_id) for row_id, count in counts.items() if int(count) > 1]
    )

    passed = not overlapping

    return [
        _result(
            strategy=strategy,
            scope="run",
            check="non_overlapping_temporal_tests",
            passed=passed,
            observed=len(overlapping),
            expected=0,
            details=(
                "No row appears in more than one temporal test partition."
                if passed
                else (
                    "Rows in multiple temporal test partitions: "
                    f"{_format_values(overlapping)}"
                )
            ),
        )
    ]


def check_time_outer_repeats(
    *,
    config: ExperimentConfig,
) -> list[SplitIntegrityResult]:
    observed = config.validation.outer_repeats
    passed = observed == 1

    return [
        _result(
            strategy=config.validation.strategy,
            scope="run",
            check="outer_repeats_equal_one",
            passed=passed,
            observed=observed,
            expected=1,
            details=(
                "Repeated temporal splitting is disabled."
                if passed
                else "Time-aware validation requires outer_repeats=1."
            ),
        )
    ]


def check_test_classes_present_in_training(
    *,
    strategy: str,
    context: SplitContext,
    manifest: pd.DataFrame,
) -> list[SplitIntegrityResult]:
    row_to_target: dict[str, str] = dict(
        zip(
            context.row_ids.astype(str).tolist(),
            context.y.astype(str).tolist(),
            strict=True,
        )
    )

    results: list[SplitIntegrityResult] = []

    for (repeat_key, fold_key), subset in manifest.groupby(
        ["repeat", "fold"],
        sort=True,
    ):
        repeat_val = int(str(repeat_key))
        fold_val = int(str(fold_key))

        train_ids = subset.loc[
            subset["partition"] == "train",
            "row_id",
        ].astype(str)

        test_ids = subset.loc[
            subset["partition"] == "test",
            "row_id",
        ].astype(str)

        train_classes = {
            row_to_target[row_id] for row_id in train_ids if row_id in row_to_target
        }

        test_classes = {
            row_to_target[row_id] for row_id in test_ids if row_id in row_to_target
        }

        missing = sorted(test_classes - train_classes)
        passed = not missing

        results.append(
            _result(
                strategy=strategy,
                scope="fold",
                repeat=repeat_val,
                fold=fold_val,
                check="test_classes_present_in_training",
                passed=passed,
                observed=(
                    "all test classes represented"
                    if passed
                    else f"missing: {_format_values(missing)}"
                ),
                expected="every test class occurs in training",
                details=(
                    "All test classes are represented in training."
                    if passed
                    else (
                        "The model cannot learn probabilities for one or "
                        "more classes appearing in the test partition."
                    )
                ),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Split-integrity orchestration
# ---------------------------------------------------------------------------


def _sort_key(r: SplitIntegrityResult) -> tuple[bool, int, bool, int, str]:
    return (
        r.repeat is None,
        r.repeat or 0,
        r.fold is None,
        r.fold or 0,
        r.check,
    )


def evaluate_split_integrity(
    config: ExperimentConfig,
    context: SplitContext,
    manifest: pd.DataFrame,
) -> list[SplitIntegrityResult]:
    strategy = config.validation.strategy
    results: list[SplitIntegrityResult] = []

    results.extend(
        check_required_manifest_columns(
            strategy=strategy,
            manifest=manifest,
        )
    )

    results.extend(
        check_partition_labels(
            strategy=strategy,
            manifest=manifest,
        )
    )

    results.extend(
        check_duplicate_manifest_rows(
            strategy=strategy,
            manifest=manifest,
        )
    )

    results.extend(
        check_train_test_overlap(
            strategy=strategy,
            manifest=manifest,
        )
    )

    results.extend(
        check_test_classes_present_in_training(
            strategy=strategy,
            context=context,
            manifest=manifest,
        )
    )

    if strategy in {"stratified", "stratified_group"}:
        results.extend(
            check_complete_row_coverage(
                strategy=strategy,
                context=context,
                manifest=manifest,
            )
        )

        results.extend(
            check_one_test_assignment_per_repeat(
                strategy=strategy,
                context=context,
                manifest=manifest,
            )
        )

    if strategy == "stratified_group":
        results.extend(
            check_no_group_leakage(
                strategy=strategy,
                manifest=manifest,
            )
        )

    if strategy == "time":
        results.extend(
            check_time_outer_repeats(
                config=config,
            )
        )

        results.extend(
            check_temporal_test_assignments_do_not_overlap(
                strategy=strategy,
                manifest=manifest,
            )
        )

        results.extend(
            check_forward_time_order(
                strategy=strategy,
                manifest=manifest,
            )
        )

    sorted_results = sorted(results, key=_sort_key)
    return sorted_results


def split_integrity_frame(
    results: list[SplitIntegrityResult],
) -> pd.DataFrame:
    records = [result.to_record() for result in results]

    return pd.DataFrame(
        records,
        columns=SPLIT_INTEGRITY_COLUMNS,
    )


def raise_for_split_integrity(
    results: list[SplitIntegrityResult],
) -> None:
    failures = [result for result in results if not result.passed]

    if not failures:
        return

    preview = "; ".join(
        (
            f"{failure.check}"
            f"[repeat={failure.repeat}, fold={failure.fold}]: "
            f"{failure.details or failure.observed}"
        )
        for failure in failures[:10]
    )

    remaining = len(failures) - 10

    if remaining > 0:
        preview += f"; and {remaining} more failure(s)"

    raise DataValidationError(f"Split-integrity validation failed: {preview}")


# ---------------------------------------------------------------------------
# Compatibility wrappers (raising ValueError as before)
# ---------------------------------------------------------------------------


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
    if counts.empty or not np.all(counts.to_numpy() == 1):
        raise ValueError("Every row must appear exactly once as test data per repeat.")


def validate_test_assignments_do_not_overlap(manifest: pd.DataFrame) -> None:
    test_rows = manifest.loc[manifest["partition"] == "test"]
    counts = test_rows.groupby("row_id").size()
    if counts.empty or not np.all(counts.to_numpy() == 1):
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


# ---------------------------------------------------------------------------
# Inner-split functions & class coverage validation
# ---------------------------------------------------------------------------


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
