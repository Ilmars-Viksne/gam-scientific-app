from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from .config import DuplicateGroupConfig, ExperimentConfig
from .data import load_table
from .diagnostics import (
    analyze_duplicate_groups,
)
from .exceptions import DataValidationError
from .splitting import (
    SplitContext,
    create_inner_splits,
    create_outer_splits,
    merge_group_constraints,
    validate_class_coverage,
)


class PlanFeasibilityError(DataValidationError):
    """Raised when a validation design cannot be applied to the dataset."""


FeasibilityLevel = Literal[
    "pass",
    "warning",
    "fail",
    "not_evaluated",
]


@dataclass(frozen=True, slots=True)
class FeasibilityCheck:
    check: str
    level: FeasibilityLevel
    observed: str
    required: str
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "level": self.level,
            "observed": self.observed,
            "required": self.required,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class ValidationFeasibility:
    strategy: str
    checks: tuple[FeasibilityCheck, ...]
    warnings: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not any(check.level == "fail" for check in self.checks)


def evaluate_plan_feasibility(
    config: ExperimentConfig,
) -> tuple[ValidationFeasibility, dict[str, Any]]:
    checks: list[FeasibilityCheck] = []
    warnings: list[str] = []
    dataset_info: dict[str, Any] = {}

    # Check 1: Load table and check required columns
    frame: pd.DataFrame | None = None
    try:
        frame = load_table(config.data_path)
        dataset_info["row_count"] = len(frame)
    except Exception as error:
        checks.append(
            FeasibilityCheck(
                check="required_columns_present",
                level="fail",
                observed="Dataset unreadable",
                required="readable tabular dataset",
                details=f"Could not load dataset at {config.data_path}: {error}",
            )
        )
        _add_not_evaluated_checks(checks, config)
        return ValidationFeasibility(
            config.validation.strategy, tuple(checks), tuple(warnings)
        ), dataset_info

    req_cols = {config.target, *config.features.keys()}
    if config.row_id:
        req_cols.add(config.row_id)
    if config.group_column:
        req_cols.add(config.group_column)
    if config.time_column:
        req_cols.add(config.time_column)

    present_cols = set(frame.columns)
    missing_cols = sorted(req_cols - present_cols)
    if missing_cols:
        checks.append(
            FeasibilityCheck(
                check="required_columns_present",
                level="fail",
                observed=f"{len(req_cols) - len(missing_cols)} of {len(req_cols)}",
                required="all required columns",
                details=f"Missing required columns: {missing_cols}",
            )
        )
        _add_not_evaluated_checks(checks, config)
        return ValidationFeasibility(
            config.validation.strategy, tuple(checks), tuple(warnings)
        ), dataset_info

    checks.append(
        FeasibilityCheck(
            check="required_columns_present",
            level="pass",
            observed=f"{len(req_cols)} of {len(req_cols)}",
            required="all required columns",
            details="All configured columns are present in the dataset.",
        )
    )

    # Check target missing values
    target_series = frame[config.target]
    missing_target = int(target_series.isna().sum())
    if missing_target > 0:
        checks.append(
            FeasibilityCheck(
                check="target_has_no_missing_values",
                level="fail",
                observed=f"{missing_target} missing values",
                required="0 missing values",
                details=f"Target column {config.target!r} contains {missing_target} missing values.",
            )
        )
    else:
        checks.append(
            FeasibilityCheck(
                check="target_has_no_missing_values",
                level="pass",
                observed="0 missing values",
                required="0 missing values",
                details=f"Target column {config.target!r} contains no missing values.",
            )
        )

    # Target class counts
    class_counts = target_series.value_counts(dropna=False)
    num_classes = len(class_counts)
    dataset_info["class_count"] = num_classes

    if num_classes < 2:
        checks.append(
            FeasibilityCheck(
                check="target_has_multiple_classes",
                level="fail",
                observed=f"{num_classes} class",
                required="at least 2 classes",
                details=f"Target column {config.target!r} has only {num_classes} distinct class.",
            )
        )
    else:
        checks.append(
            FeasibilityCheck(
                check="target_has_multiple_classes",
                level="pass",
                observed=f"{num_classes} classes",
                required="at least 2 classes",
                details=f"Target column {config.target!r} contains {num_classes} distinct classes.",
            )
        )

    # Outer row support
    if len(frame) < config.validation.outer_splits:
        checks.append(
            FeasibilityCheck(
                check="outer_row_support",
                level="fail",
                observed=f"{len(frame)} rows",
                required=f"at least {config.validation.outer_splits} rows",
                details=f"Dataset has {len(frame)} rows, which is less than outer_splits={config.validation.outer_splits}.",
            )
        )
    else:
        checks.append(
            FeasibilityCheck(
                check="outer_row_support",
                level="pass",
                observed=f"{len(frame)} rows",
                required=f"at least {config.validation.outer_splits} rows",
                details="Dataset has sufficient total rows for outer validation splits.",
            )
        )

    # Strategy-specific checks
    strategy = config.validation.strategy

    # Check warnings for unused columns
    if strategy != "stratified_group" and config.group_column is not None:
        warnings.append(
            f"Configured group column {config.group_column!r} is inactive under strategy {strategy!r}."
        )
    if strategy != "time" and config.time_column is not None:
        warnings.append(
            f"Configured time column {config.time_column!r} is inactive under strategy {strategy!r}."
        )
    if strategy == "time":
        warnings.append("Random state: not used by time validation.")

    # Pairwise duplicate scan limit check
    if config.profiling.duplicate_groups.enabled:
        if len(frame) > config.profiling.duplicate_groups.maximum_pairwise_rows:
            checks.append(
                FeasibilityCheck(
                    check="near_duplicate_scan_within_limit",
                    level="fail",
                    observed=f"{len(frame)} rows",
                    required=f"<= {config.profiling.duplicate_groups.maximum_pairwise_rows} rows",
                    details=(
                        f"Dataset size ({len(frame)}) exceeds maximum_pairwise_rows "
                        f"({config.profiling.duplicate_groups.maximum_pairwise_rows})."
                    ),
                )
            )
        else:
            checks.append(
                FeasibilityCheck(
                    check="near_duplicate_scan_within_limit",
                    level="pass",
                    observed=f"{len(frame)} rows",
                    required=f"<= {config.profiling.duplicate_groups.maximum_pairwise_rows} rows",
                    details="Dataset size is within the allowed pairwise duplicate scanning limit.",
                )
            )

    # If target quality failed completely (e.g. missing target values or <2 classes), skip deeper split checks
    if any(
        c.check in ("target_has_no_missing_values", "target_has_multiple_classes")
        and c.level == "fail"
        for c in checks
    ):
        _add_not_evaluated_checks(checks, config)
        return ValidationFeasibility(
            strategy, tuple(checks), tuple(warnings)
        ), dataset_info

    # Stratified checks
    if strategy == "stratified":
        min_class_count = int(class_counts.min())
        min_class_label = class_counts.idxmin()

        if min_class_count < config.validation.outer_splits:
            checks.append(
                FeasibilityCheck(
                    check="outer_class_support",
                    level="fail",
                    observed=f"class {min_class_label!r} has {min_class_count} observations",
                    required=f"at least {config.validation.outer_splits} observations",
                    details=f"Class {min_class_label!r} has {min_class_count} observations, which is less than outer_splits={config.validation.outer_splits}.",
                )
            )
        else:
            checks.append(
                FeasibilityCheck(
                    check="outer_class_support",
                    level="pass",
                    observed=f"minimum class count {min_class_count}",
                    required=f"at least {config.validation.outer_splits} observations",
                    details="Every target class has enough observations for outer splits.",
                )
            )

        smallest_outer_train_counts = [
            count - math.ceil(count / config.validation.outer_splits)
            for count in class_counts
        ]
        min_outer_train_class_count = min(smallest_outer_train_counts)

        if min_outer_train_class_count < config.validation.inner_splits:
            checks.append(
                FeasibilityCheck(
                    check="inner_class_support",
                    level="fail",
                    observed=f"minimum outer-train class count {min_outer_train_class_count}",
                    required=f"at least {config.validation.inner_splits} observations",
                    details=f"In the worst-case outer fold, a class has only {min_outer_train_class_count} training observations, which cannot support inner_splits={config.validation.inner_splits}.",
                )
            )
        else:
            checks.append(
                FeasibilityCheck(
                    check="inner_class_support",
                    level="pass",
                    observed=f"minimum outer-train class count {min_outer_train_class_count}",
                    required=f"at least {config.validation.inner_splits} observations",
                    details="Outer training subsets have sufficient class observations to support inner CV.",
                )
            )

    # Effective groups calculation for grouped or duplicate-policy=group
    effective_groups: pd.Series | None = None
    if (
        strategy == "stratified_group"
        or config.validation.duplicate_group_policy == "group"
    ):
        configured_groups = frame[config.group_column] if config.group_column else None
        duplicate_signatures: pd.Series | None = None
        row_ids = (
            frame[config.row_id] if config.row_id else pd.Series(range(len(frame)))
        )
        edge_constraints_list: list[pd.DataFrame] = []

        if (
            config.profiling.duplicate_groups.enabled
            and len(frame) <= config.profiling.duplicate_groups.maximum_pairwise_rows
        ):
            active_predictors = [
                n for n, s in config.features.items() if s.role != "exclude"
            ]
            X_pred = frame[active_predictors]
            dup_cfg = DuplicateGroupConfig(
                enabled=True,
                rounding_decimals=config.profiling.duplicate_groups.rounding_decimals,
                near_duplicate_threshold=config.profiling.duplicate_groups.near_duplicate_threshold,
                maximum_pairwise_rows=config.profiling.duplicate_groups.maximum_pairwise_rows,
            )
            diag_res = analyze_duplicate_groups(
                X=X_pred,
                y=frame[config.target],
                row_ids=row_ids,
                config=dup_cfg,
            )
            duplicate_signatures = diag_res.exact_signatures
            if not diag_res.near_edges.empty:
                edge_constraints_list.append(diag_res.near_edges)

        effective_groups = merge_group_constraints(
            row_count=len(frame),
            configured_groups=configured_groups,
            duplicate_signatures=duplicate_signatures,
            edge_constraints=edge_constraints_list,
            row_ids=row_ids,
        )

    if strategy == "stratified_group":
        assert effective_groups is not None
        num_effective_groups = effective_groups.nunique()
        dataset_info["effective_group_count"] = num_effective_groups

        if num_effective_groups < config.validation.outer_splits:
            checks.append(
                FeasibilityCheck(
                    check="effective_group_count",
                    level="fail",
                    observed=f"{num_effective_groups} groups",
                    required=f"at least {config.validation.outer_splits} groups",
                    details=f"Total effective group count ({num_effective_groups}) is less than outer_splits={config.validation.outer_splits}.",
                )
            )
        else:
            checks.append(
                FeasibilityCheck(
                    check="effective_group_count",
                    level="pass",
                    observed=f"{num_effective_groups} groups",
                    required=f"at least {config.validation.outer_splits} groups",
                    details="Dataset has sufficient effective groups for outer splits.",
                )
            )

        class_group_counts = (
            pd.DataFrame(
                {
                    "target": target_series,
                    "group": effective_groups,
                }
            )
            .drop_duplicates()
            .groupby("target")["group"]
            .nunique()
        )

        min_class_groups = int(class_group_counts.min())
        min_class_group_label = class_group_counts.idxmin()

        if min_class_groups < config.validation.outer_splits:
            checks.append(
                FeasibilityCheck(
                    check="outer_class_group_support",
                    level="fail",
                    observed=f"class {min_class_group_label!r} appears in {min_class_groups} groups",
                    required=f"at least {config.validation.outer_splits} groups",
                    details=f"Class {min_class_group_label!r} appears in only {min_class_groups} distinct effective groups, which cannot support outer_splits={config.validation.outer_splits}.",
                )
            )
        else:
            checks.append(
                FeasibilityCheck(
                    check="outer_class_group_support",
                    level="pass",
                    observed=f"minimum class groups {min_class_groups}",
                    required=f"at least {config.validation.outer_splits} groups",
                    details="Every target class occurs in sufficient distinct groups.",
                )
            )

        # Dry-run outer and inner group splits
        context = SplitContext(
            X=frame[[n for n, s in config.features.items() if s.role != "exclude"]],
            y=target_series,
            row_ids=frame[config.row_id]
            if config.row_id
            else pd.Series(range(len(frame))),
            groups=effective_groups,
            times=None,
        )

        inner_feasible = True
        inner_error_msg = ""
        try:
            outer_splits = create_outer_splits(config, context)
            for split in outer_splits:
                train_idx = np.array(split.train_indices)
                test_idx = np.array(split.test_indices)
                validate_class_coverage(
                    y=target_series,
                    train_indices=train_idx,
                    test_indices=test_idx,
                    context=f"outer fold (repeat={split.repeat}, fold={split.fold})",
                )

                X_train = context.X.iloc[train_idx]
                y_train = context.y.iloc[train_idx]
                g_train = effective_groups.iloc[train_idx]

                inner_splits = create_inner_splits(
                    config=config,
                    X=X_train,
                    y=y_train,
                    groups=g_train,
                    times=None,
                    seed=config.validation.random_state + split.repeat,
                )
                for in_train, in_val in inner_splits:
                    validate_class_coverage(
                        y=y_train,
                        train_indices=in_train,
                        test_indices=in_val,
                        context="inner fold",
                    )
        except Exception as error:
            inner_feasible = False
            inner_error_msg = str(error)

        if not inner_feasible:
            checks.append(
                FeasibilityCheck(
                    check="inner_group_split_feasible",
                    level="fail",
                    observed="split dry-run failed",
                    required="feasible outer & inner group splits",
                    details=f"Dry-run of grouped cross-validation failed: {inner_error_msg}",
                )
            )
        else:
            checks.append(
                FeasibilityCheck(
                    check="inner_group_split_feasible",
                    level="pass",
                    observed="all folds valid",
                    required="feasible outer & inner group splits",
                    details="Dry-run of outer and inner grouped cross-validation succeeded.",
                )
            )

    # Time strategy checks
    if strategy == "time":
        time_series = frame[config.time_column]
        missing_times = int(time_series.isna().sum())

        if missing_times > 0:
            checks.append(
                FeasibilityCheck(
                    check="time_values_complete",
                    level="fail",
                    observed=f"{missing_times} missing values",
                    required="0 missing values",
                    details=f"Time column {config.time_column!r} contains {missing_times} missing values.",
                )
            )
        else:
            checks.append(
                FeasibilityCheck(
                    check="time_values_complete",
                    level="pass",
                    observed="0 missing values",
                    required="0 missing values",
                    details=f"Time column {config.time_column!r} contains no missing values.",
                )
            )

        parseable = True
        parsed_times: pd.Series | None = None
        try:
            parsed_times = pd.to_datetime(time_series, errors="raise", utc=True)
        except Exception as error:
            parseable = False
            checks.append(
                FeasibilityCheck(
                    check="time_values_parseable",
                    level="fail",
                    observed="unparseable timestamps",
                    required="valid UTC datetimes",
                    details=f"Could not parse timestamps in column {config.time_column!r}: {error}",
                )
            )

        if parseable and parsed_times is not None:
            checks.append(
                FeasibilityCheck(
                    check="time_values_parseable",
                    level="pass",
                    observed="all timestamps parsed",
                    required="valid UTC datetimes",
                    details="All timestamp values parsed successfully as UTC datetimes.",
                )
            )

            context = SplitContext(
                X=frame[[n for n, s in config.features.items() if s.role != "exclude"]],
                y=target_series,
                row_ids=frame[config.row_id]
                if config.row_id
                else pd.Series(range(len(frame))),
                groups=None,
                times=parsed_times,
            )

            temporal_outer_feasible = True
            time_error_msg = ""

            try:
                outer_splits = create_outer_splits(config, context)
                if not outer_splits:
                    temporal_outer_feasible = False
                    time_error_msg = "No outer temporal splits could be generated."
                for split in outer_splits:
                    train_idx = np.array(split.train_indices)
                    test_idx = np.array(split.test_indices)
                    validate_class_coverage(
                        y=target_series,
                        train_indices=train_idx,
                        test_indices=test_idx,
                        context=f"outer temporal fold {split.fold}",
                    )

                    X_train = context.X.iloc[train_idx]
                    y_train = context.y.iloc[train_idx]
                    t_train = parsed_times.iloc[train_idx]

                    inner_splits = create_inner_splits(
                        config=config,
                        X=X_train,
                        y=y_train,
                        groups=None,
                        times=t_train,
                        seed=config.validation.random_state,
                    )
                    for in_train, in_val in inner_splits:
                        validate_class_coverage(
                            y=y_train,
                            train_indices=in_train,
                            test_indices=in_val,
                            context="inner temporal fold",
                        )
            except Exception as error:
                temporal_outer_feasible = False
                time_error_msg = str(error)

            if not temporal_outer_feasible:
                checks.append(
                    FeasibilityCheck(
                        check="temporal_outer_windows_feasible",
                        level="fail",
                        observed="window dry-run failed",
                        required="feasible temporal outer & inner windows",
                        details=f"Temporal window feasibility check failed: {time_error_msg}",
                    )
                )
            else:
                checks.append(
                    FeasibilityCheck(
                        check="temporal_outer_windows_feasible",
                        level="pass",
                        observed="all temporal windows valid",
                        required="feasible temporal outer & inner windows",
                        details="Dry-run of outer and inner temporal splits succeeded.",
                    )
                )

    return ValidationFeasibility(strategy, tuple(checks), tuple(warnings)), dataset_info


def _add_not_evaluated_checks(
    checks: list[FeasibilityCheck], config: ExperimentConfig
) -> None:
    evaluated_names = {c.check for c in checks}
    all_potential_checks = [
        "required_columns_present",
        "target_has_no_missing_values",
        "target_has_multiple_classes",
        "outer_row_support",
        "near_duplicate_scan_within_limit",
        "outer_class_support",
        "inner_class_support",
        "effective_group_count",
        "outer_class_group_support",
        "inner_group_split_feasible",
        "time_values_complete",
        "time_values_parseable",
        "temporal_outer_windows_feasible",
    ]

    for name in all_potential_checks:
        if name not in evaluated_names:
            checks.append(
                FeasibilityCheck(
                    check=name,
                    level="not_evaluated",
                    observed="N/A",
                    required="N/A",
                    details="Check was not evaluated due to earlier dataset or schema failures.",
                )
            )
