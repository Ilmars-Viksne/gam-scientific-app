from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from .config import DuplicateGroupConfig, ExperimentConfig
from .exceptions import DataValidationError
from .io_utils import write_json_atomic


@dataclass(frozen=True, slots=True)
class DuplicateAnalysis:
    exact_signatures: pd.Series
    exact_duplicate_groups: pd.DataFrame
    proper_near_duplicate_groups: pd.DataFrame
    near_edges: pd.DataFrame
    conflicting_targets: pd.DataFrame


@dataclass(frozen=True, slots=True)
class StandaloneDiagnosticSettings:
    correlation_review_threshold: float = 0.75
    correlation_warning_threshold: float = 0.90
    minimum_complete_pairs: int = 3
    near_duplicate_decimals: int = 8
    near_duplicate_threshold: float = 0.98
    maximum_pairwise_rows: int = 10_000

    def validate(self) -> None:
        if not (0.0 <= self.correlation_review_threshold <= 1.0):
            raise ValueError(
                "The correlation review threshold must be between 0 and 1."
            )

        if not (0.0 <= self.correlation_warning_threshold <= 1.0):
            raise ValueError(
                "The correlation warning threshold must be between 0 and 1."
            )

        if self.correlation_warning_threshold < self.correlation_review_threshold:
            raise ValueError(
                "The correlation warning threshold cannot be smaller "
                "than the review threshold."
            )

        if self.minimum_complete_pairs < 2:
            raise ValueError("minimum_complete_pairs must be at least 2.")

        if self.near_duplicate_decimals < 0:
            raise ValueError("near_duplicate_decimals cannot be negative.")

        if not math.isfinite(self.near_duplicate_threshold) or not (
            0.0 < self.near_duplicate_threshold <= 1.0
        ):
            raise ValueError(
                "near_duplicate_threshold must satisfy 0.0 < threshold <= 1.0."
            )

        if self.maximum_pairwise_rows < 2:
            raise ValueError("maximum_pairwise_rows must be at least 2.")


@dataclass(frozen=True, slots=True)
class StandaloneDiagnostics:
    pearson: pd.DataFrame
    spearman: pd.DataFrame
    high_correlation_pairs: pd.DataFrame
    numeric_predictor_dictionary: pd.DataFrame
    suspected_derived_relations: pd.DataFrame
    exact_duplicate_groups: pd.DataFrame
    near_duplicate_groups: pd.DataFrame
    conflicting_duplicate_targets: pd.DataFrame


@dataclass(frozen=True, slots=True)
class CorrelationAnalysis:
    pearson: pd.DataFrame
    spearman: pd.DataFrame
    high_pairs: pd.DataFrame
    numeric_summary: pd.DataFrame


@dataclass(frozen=True, slots=True)
class DerivedRelation:
    candidate: str
    source: str
    relation_type: str
    status: str
    complete_pair_count: int
    parameter_a: float | None
    parameter_b: float | None
    maximum_absolute_error: float
    maximum_relative_error: float
    correlation_pearson: float | None
    correlation_spearman: float | None
    evidence: str
    recommended_action: str


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            if self.rank[root_i] == self.rank[root_j]:
                self.rank[root_i] += 1


def canonicalize_predictors(
    X: pd.DataFrame,
    *,
    decimals: int | None = None,
) -> pd.DataFrame:
    canonical = pd.DataFrame(index=X.index)

    for col in X.columns:
        series = X[col]
        if pd.api.types.is_numeric_dtype(series):
            num = pd.to_numeric(series, errors="coerce")
            if decimals is not None:
                num = num.round(decimals)
            num_vals = num.to_numpy(dtype=float)
            num_vals = np.where(num_vals == -0.0, 0.0, num_vals)
            formatted = ["MISSING" if np.isnan(v) else (f"{v:.17g}") for v in num_vals]
            canonical[col] = formatted
        elif pd.api.types.is_bool_dtype(series):
            bool_vals = series.to_numpy()
            formatted = [
                "MISSING" if pd.isna(v) else ("TRUE" if bool(v) else "FALSE")
                for v in bool_vals
            ]
            canonical[col] = formatted
        else:
            str_vals = series.astype("string").str.strip()
            formatted = ["MISSING" if pd.isna(v) else str(v) for v in str_vals]
            canonical[col] = formatted

    return canonical


def build_exact_predictor_signatures(X: pd.DataFrame) -> pd.Series:
    canonical = canonicalize_predictors(X, decimals=None)
    signatures: list[str] = []

    for row in canonical.itertuples(index=False, name=None):
        raw = json.dumps(
            list(row),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        signatures.append(hashlib.sha256(raw.encode("utf-8")).hexdigest())

    return pd.Series(signatures, index=X.index, name="exact_predictor_signature")


def exact_duplicate_group_report(
    X: pd.DataFrame,
    row_ids: pd.Series,
) -> pd.DataFrame:
    signatures = build_exact_predictor_signatures(X)
    frame = pd.DataFrame(
        {
            "row_id": row_ids.astype(str).to_numpy(),
            "signature": signatures.to_numpy(),
        }
    )
    group_sizes = frame.groupby("signature", sort=False)["row_id"].transform("size")
    duplicates = frame.loc[group_sizes > 1].copy()
    if duplicates.empty:
        return pd.DataFrame(
            columns=["duplicate_group_id", "signature", "group_size", "row_id"]
        )

    duplicates["group_size"] = group_sizes.loc[group_sizes > 1].to_numpy()

    # Deterministic group ID based on SHA-256 of sorted member row IDs
    dup_ids: dict[str, str] = {}
    for sig, sub in duplicates.groupby("signature", sort=True):
        sig_str = str(sig)
        sorted_members = sorted(sub["row_id"].tolist())
        member_key = "\x1f".join(sorted_members)
        hash_id = hashlib.sha256(member_key.encode("utf-8")).hexdigest()[:12]
        dup_ids[sig_str] = f"duplicate_{hash_id}"

    duplicates["duplicate_group_id"] = duplicates["signature"].map(dup_ids)

    return (
        duplicates[
            [
                "duplicate_group_id",
                "signature",
                "group_size",
                "row_id",
            ]
        ]
        .sort_values(["duplicate_group_id", "row_id"], kind="stable")
        .reset_index(drop=True)
    )


def conflicting_duplicate_target_report(
    X: pd.DataFrame,
    y: pd.Series,
    row_ids: pd.Series,
) -> pd.DataFrame:
    signatures = build_exact_predictor_signatures(X)
    frame = pd.DataFrame(
        {
            "row_id": row_ids.astype(str).to_numpy(),
            "signature": signatures.to_numpy(),
            "target": y.astype(str).to_numpy(),
        }
    )
    target_counts = frame.groupby("signature", sort=False)["target"].transform(
        "nunique"
    )
    conflicts = frame.loc[target_counts > 1].copy()
    if conflicts.empty:
        return pd.DataFrame(
            columns=["signature", "target", "row_id", "distinct_target_count"]
        )

    conflicts["distinct_target_count"] = target_counts[target_counts > 1].to_numpy()
    return (
        conflicts[["signature", "target", "row_id", "distinct_target_count"]]
        .sort_values(["signature", "target", "row_id"], kind="stable")
        .reset_index(drop=True)
    )


def build_near_duplicate_signatures(
    X: pd.DataFrame,
    *,
    decimals: int,
) -> pd.Series:
    canonical = canonicalize_predictors(X, decimals=decimals)
    signatures: list[str] = []

    for row in canonical.itertuples(index=False, name=None):
        raw = json.dumps(
            list(row),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        signatures.append(hashlib.sha256(raw.encode("utf-8")).hexdigest())

    return pd.Series(signatures, index=X.index, name="near_duplicate_signature")


def duplicate_signature_report(
    *,
    signatures: pd.Series,
    row_ids: pd.Series,
    report_prefix: str = "near_duplicate",
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "row_id": row_ids.astype(str).to_numpy(),
            "signature": signatures.to_numpy(),
        }
    )
    group_sizes = frame.groupby("signature", sort=False)["row_id"].transform("size")
    duplicates = frame.loc[group_sizes > 1].copy()
    if duplicates.empty:
        return pd.DataFrame(
            columns=[f"{report_prefix}_group_id", "signature", "group_size", "row_id"]
        )

    duplicates["group_size"] = group_sizes.loc[group_sizes > 1].to_numpy()

    prefix_ids: dict[str, str] = {}
    for sig, sub in duplicates.groupby("signature", sort=True):
        sig_str = str(sig)
        sorted_members = sorted(sub["row_id"].tolist())
        member_key = "\x1f".join(sorted_members)
        hash_id = hashlib.sha256(member_key.encode("utf-8")).hexdigest()[:12]
        prefix_ids[sig_str] = f"{report_prefix}_{hash_id}"

    duplicates[f"{report_prefix}_group_id"] = duplicates["signature"].map(prefix_ids)

    return (
        duplicates[
            [
                f"{report_prefix}_group_id",
                "signature",
                "group_size",
                "row_id",
            ]
        ]
        .sort_values([f"{report_prefix}_group_id", "row_id"], kind="stable")
        .reset_index(drop=True)
    )


def analyze_duplicate_groups(
    X: pd.DataFrame,
    y: pd.Series,
    row_ids: pd.Series,
    config: DuplicateGroupConfig,
) -> DuplicateAnalysis:
    row_count = len(X)
    str_row_ids = row_ids.astype(str).to_numpy()

    # 1. Exact duplicates & conflicting targets
    exact_signatures = build_exact_predictor_signatures(X)
    exact_duplicate_groups = exact_duplicate_group_report(X, row_ids)
    conflicting_targets = conflicting_duplicate_target_report(X, y, row_ids)

    if not config.enabled:
        empty_near_groups = pd.DataFrame(
            columns=[
                "near_duplicate_group_id",
                "row_id",
                "exact_signature",
                "canonical_signature",
                "group_size",
                "distinct_exact_signature_count",
                "matched_column_count",
                "compared_column_count",
                "match_fraction",
                "is_exact_duplicate_member",
            ]
        )
        empty_edges = pd.DataFrame(
            columns=[
                "left_row_id",
                "right_row_id",
                "matched_column_count",
                "compared_column_count",
                "match_fraction",
                "threshold",
            ]
        )
        return DuplicateAnalysis(
            exact_signatures=exact_signatures,
            exact_duplicate_groups=exact_duplicate_groups,
            proper_near_duplicate_groups=empty_near_groups,
            near_edges=empty_edges,
            conflicting_targets=conflicting_targets,
        )

    # Check pairwise limit
    if row_count > config.maximum_pairwise_rows:
        raise DataValidationError(
            "Near-duplicate analysis requires an exact pairwise scan, "
            f"but the dataset contains {row_count:,} rows and the configured "
            f"limit is {config.maximum_pairwise_rows:,}. Increase "
            "profiling.duplicate_groups.maximum_pairwise_rows only after "
            "reviewing memory and runtime requirements, or disable "
            "near-duplicate analysis."
        )

    # 2. Canonicalize for near duplicates
    canonical_df = canonicalize_predictors(X, decimals=config.rounding_decimals)
    canonical_signatures: list[str] = []
    for row in canonical_df.itertuples(index=False, name=None):
        raw = json.dumps(
            list(row),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        canonical_signatures.append(hashlib.sha256(raw.encode("utf-8")).hexdigest())

    canonical_sig_series = pd.Series(canonical_signatures, index=X.index)

    # 3. Pairwise similarity comparison
    col_names = list(canonical_df.columns)
    compared_column_count = len(col_names)
    matrix = canonical_df.to_numpy(dtype=str)

    uf = UnionFind(row_count)
    edge_rows: list[dict[str, Any]] = []

    for i in range(row_count):
        row_i = matrix[i]
        for j in range(i + 1, row_count):
            row_j = matrix[j]
            if compared_column_count == 0:
                matches = 0
                fraction = 1.0
            else:
                matches = int(np.sum(row_i == row_j))
                fraction = matches / compared_column_count

            if fraction >= config.near_duplicate_threshold:
                uf.union(i, j)
                left_id = str_row_ids[i]
                right_id = str_row_ids[j]
                if left_id > right_id:
                    left_id, right_id = right_id, left_id
                edge_rows.append(
                    {
                        "left_row_id": left_id,
                        "right_row_id": right_id,
                        "matched_column_count": matches,
                        "compared_column_count": compared_column_count,
                        "match_fraction": fraction,
                        "threshold": config.near_duplicate_threshold,
                    }
                )

    if edge_rows:
        near_edges = (
            pd.DataFrame(edge_rows)
            .drop_duplicates(subset=["left_row_id", "right_row_id"])
            .sort_values(["left_row_id", "right_row_id"], kind="stable")
            .reset_index(drop=True)
        )
    else:
        near_edges = pd.DataFrame(
            columns=[
                "left_row_id",
                "right_row_id",
                "matched_column_count",
                "compared_column_count",
                "match_fraction",
                "threshold",
            ]
        )

    # 4. Build connected components & filter proper near duplicate groups
    component_members: dict[int, list[int]] = {}
    for idx in range(row_count):
        root = uf.find(idx)
        component_members.setdefault(root, []).append(idx)

    # Exact group memberships lookup
    exact_dups_set = (
        set(exact_duplicate_groups["row_id"].tolist())
        if not exact_duplicate_groups.empty
        else set()
    )

    proper_near_rows: list[dict[str, Any]] = []

    for root, members in component_members.items():
        if len(members) < 2:
            continue

        member_exact_sigs = [exact_signatures.iloc[m] for m in members]
        distinct_exact_cnt = len(set(member_exact_sigs))

        # A proper near group must contain at least two distinct exact predictor signatures
        if distinct_exact_cnt < 2:
            continue

        sorted_member_ids = sorted([str_row_ids[m] for m in members])
        member_key = "\x1f".join(sorted_member_ids)
        hash_id = hashlib.sha256(member_key.encode("utf-8")).hexdigest()[:12]
        group_id = f"near_duplicate_{hash_id}"
        group_size = len(members)

        for m in members:
            rid = str_row_ids[m]
            exact_sig = exact_signatures.iloc[m]
            canon_sig = canonical_sig_series.iloc[m]
            is_exact_dup_member = rid in exact_dups_set

            proper_near_rows.append(
                {
                    "near_duplicate_group_id": group_id,
                    "row_id": rid,
                    "exact_signature": exact_sig,
                    "canonical_signature": canon_sig,
                    "group_size": group_size,
                    "distinct_exact_signature_count": distinct_exact_cnt,
                    "matched_column_count": compared_column_count,  # Reference context
                    "compared_column_count": compared_column_count,
                    "match_fraction": 1.0,  # Group wide indicator
                    "is_exact_duplicate_member": is_exact_dup_member,
                }
            )

    if proper_near_rows:
        proper_near_duplicate_groups = (
            pd.DataFrame(proper_near_rows)
            .sort_values(["near_duplicate_group_id", "row_id"], kind="stable")
            .reset_index(drop=True)
        )
    else:
        proper_near_duplicate_groups = pd.DataFrame(
            columns=[
                "near_duplicate_group_id",
                "row_id",
                "exact_signature",
                "canonical_signature",
                "group_size",
                "distinct_exact_signature_count",
                "matched_column_count",
                "compared_column_count",
                "match_fraction",
                "is_exact_duplicate_member",
            ]
        )

    return DuplicateAnalysis(
        exact_signatures=exact_signatures,
        exact_duplicate_groups=exact_duplicate_groups,
        proper_near_duplicate_groups=proper_near_duplicate_groups,
        near_edges=near_edges,
        conflicting_targets=conflicting_targets,
    )


def numeric_predictor_frame(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    candidate_names = [
        name
        for name in config.features
        if name in frame.columns and pd.api.types.is_numeric_dtype(frame[name])
    ]

    numeric = frame.loc[:, candidate_names].copy()
    for name in numeric.columns:
        numeric[name] = pd.to_numeric(
            numeric[name],
            errors="coerce",
        )
    return numeric


def _matrix_value(
    matrix: pd.DataFrame,
    left: str,
    right: str,
) -> float | None:
    if matrix.empty or left not in matrix.index or right not in matrix.columns:
        return None
    value = matrix.loc[left, right]
    if pd.isna(value):
        return None
    return float(cast(Any, value))


def _correlation_action(
    *,
    declared_relation: bool,
    severity: str,
) -> str:
    if declared_relation:
        return (
            "Review whether both source and derived representations "
            "are required in the same model."
        )
    if severity == "warning":
        return (
            "Investigate redundancy, leakage, and contribution "
            "stability; compare sensitivity specifications."
        )
    return "Review scientific meaning and monitor term stability across outer folds."


def _empty_high_pair_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "left",
            "right",
            "pearson",
            "absolute_pearson",
            "spearman",
            "absolute_spearman",
            "maximum_absolute_correlation",
            "trigger_methods",
            "complete_pair_count",
            "left_role",
            "right_role",
            "left_derived",
            "right_derived",
            "declared_derivation_relation",
            "severity",
            "recommended_action",
        ]
    )


def build_high_correlation_pairs(
    *,
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    numeric: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    settings = config.profiling.correlation
    names = list(numeric.columns)
    rows: list[dict[str, Any]] = []

    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            pearson_value = _matrix_value(pearson, left, right)
            spearman_value = _matrix_value(spearman, left, right)

            available_absolute_values = [
                abs(value)
                for value in (pearson_value, spearman_value)
                if value is not None and np.isfinite(value)
            ]

            if not available_absolute_values:
                continue

            maximum = max(available_absolute_values)

            if maximum < settings.review_threshold:
                continue

            trigger_methods: list[str] = []
            if (
                pearson_value is not None
                and np.isfinite(pearson_value)
                and abs(pearson_value) >= settings.review_threshold
            ):
                trigger_methods.append("pearson")

            if (
                spearman_value is not None
                and np.isfinite(spearman_value)
                and abs(spearman_value) >= settings.review_threshold
            ):
                trigger_methods.append("spearman")

            complete_pair_count = int(numeric[[left, right]].dropna().shape[0])

            left_spec = config.features[left]
            right_spec = config.features[right]

            declared_relation = (
                left in right_spec.derived_from or right in left_spec.derived_from
            )

            severity = "warning" if maximum >= settings.warning_threshold else "review"

            rows.append(
                {
                    "left": left,
                    "right": right,
                    "pearson": pearson_value,
                    "absolute_pearson": (
                        abs(pearson_value) if pearson_value is not None else np.nan
                    ),
                    "spearman": spearman_value,
                    "absolute_spearman": (
                        abs(spearman_value) if spearman_value is not None else np.nan
                    ),
                    "maximum_absolute_correlation": maximum,
                    "trigger_methods": ",".join(trigger_methods),
                    "complete_pair_count": complete_pair_count,
                    "left_role": left_spec.role,
                    "right_role": right_spec.role,
                    "left_derived": left_spec.derived,
                    "right_derived": right_spec.derived,
                    "declared_derivation_relation": declared_relation,
                    "severity": severity,
                    "recommended_action": _correlation_action(
                        declared_relation=declared_relation,
                        severity=severity,
                    ),
                }
            )

    if not rows:
        return _empty_high_pair_frame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["maximum_absolute_correlation", "left", "right"],
            ascending=[False, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _numeric_summary(
    numeric: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in numeric.columns:
        spec = config.features[name]
        series = numeric[name]
        rows.append(
            {
                "predictor": name,
                "role": spec.role,
                "dtype": str(series.dtype),
                "non_missing": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
                "derived": spec.derived,
                "derived_from": ",".join(spec.derived_from),
                "derivation": spec.derivation,
                "description": spec.description,
                "unit": spec.unit,
            }
        )
    return pd.DataFrame(rows)


def calculate_correlation_analysis(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> CorrelationAnalysis:
    settings = config.profiling.correlation
    numeric = numeric_predictor_frame(frame, config)

    if numeric.empty:
        empty = pd.DataFrame()
        return CorrelationAnalysis(
            pearson=empty,
            spearman=empty,
            high_pairs=_empty_high_pair_frame(),
            numeric_summary=_numeric_summary(numeric, config),
        )

    pearson = (
        numeric.corr(
            method="pearson",
            min_periods=settings.minimum_complete_pairs,
        )
        if settings.pearson
        else pd.DataFrame()
    )

    spearman = (
        numeric.corr(
            method="spearman",
            min_periods=settings.minimum_complete_pairs,
        )
        if settings.spearman
        else pd.DataFrame()
    )

    high_pairs = build_high_correlation_pairs(
        pearson=pearson,
        spearman=spearman,
        numeric=numeric,
        config=config,
    )

    return CorrelationAnalysis(
        pearson=pearson,
        spearman=spearman,
        high_pairs=high_pairs,
        numeric_summary=_numeric_summary(numeric, config),
    )


def save_correlation_analysis(
    analysis: CorrelationAnalysis,
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if not analysis.pearson.empty:
        analysis.pearson.to_csv(
            directory / "correlation_pearson.csv",
            index=True,
            index_label="predictor",
            float_format="%.17g",
        )
    if not analysis.spearman.empty:
        analysis.spearman.to_csv(
            directory / "correlation_spearman.csv",
            index=True,
            index_label="predictor",
            float_format="%.17g",
        )
    analysis.high_pairs.to_csv(
        directory / "high_correlation_pairs.csv",
        index=False,
        float_format="%.17g",
    )
    analysis.numeric_summary.to_csv(
        directory / "numeric_predictor_dictionary.csv",
        index=False,
    )


# Suspected Derived Relations


def build_suspected_derived_relations(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    numeric = numeric_predictor_frame(frame, config)
    names = list(numeric.columns)
    min_pairs = config.profiling.correlation.minimum_complete_pairs

    rows: list[dict[str, Any]] = []

    atol = 1e-10
    rtol = 1e-8

    for cand in names:
        cand_spec = config.features[cand]
        for src in names:
            if cand == src:
                continue

            src_spec = config.features[src]
            pair_df = numeric[[src, cand]].dropna()
            count = len(pair_df)
            if count < min_pairs:
                continue

            x = pair_df[src].to_numpy(dtype=float)
            y = pair_df[cand].to_numpy(dtype=float)

            # Skip constant source
            if np.isclose(x.min(), x.max(), atol=1e-12):
                continue

            pearson_val = float(np.corrcoef(x, y)[0, 1]) if count > 1 else None
            try:
                from scipy.stats import spearmanr  # type: ignore[import-untyped]

                spearman_val = float(spearmanr(x, y).statistic)
            except Exception:
                spearman_val = None

            is_declared_derived = (
                src in cand_spec.derived_from or cand in src_spec.derived_from
            )
            status = "declared_match" if is_declared_derived else "suspected"

            rec_action = (
                "Verify the transformation against the data dictionary or "
                "source documentation before modifying the predictor set."
            )

            # 1. Exact equality check
            if np.allclose(y, x, rtol=rtol, atol=atol):
                max_abs_err = float(np.max(np.abs(y - x)))
                max_rel_err = float(np.max(np.abs(y - x) / (np.abs(x) + 1e-15)))
                rows.append(
                    {
                        "candidate": cand,
                        "source": src,
                        "relation_type": "exact",
                        "status": status,
                        "complete_pair_count": count,
                        "parameter_a": 0.0,
                        "parameter_b": 1.0,
                        "maximum_absolute_error": max_abs_err,
                        "maximum_relative_error": max_rel_err,
                        "correlation_pearson": pearson_val,
                        "correlation_spearman": spearman_val,
                        "evidence": f"y == x within rtol={rtol}, atol={atol}",
                        "recommended_action": rec_action,
                    }
                )
                continue

            # 2. Affine relation check y = a + b * x
            if len(x) >= 2:
                A = np.vstack([np.ones_like(x), x]).T
                a, b = np.linalg.lstsq(A, y, rcond=None)[0]
                expected = a + b * x
                if np.allclose(y, expected, rtol=rtol, atol=atol):
                    max_abs_err = float(np.max(np.abs(y - expected)))
                    max_rel_err = float(
                        np.max(np.abs(y - expected) / (np.abs(expected) + 1e-15))
                    )
                    rows.append(
                        {
                            "candidate": cand,
                            "source": src,
                            "relation_type": "affine",
                            "status": status,
                            "complete_pair_count": count,
                            "parameter_a": float(a),
                            "parameter_b": float(b),
                            "maximum_absolute_error": max_abs_err,
                            "maximum_relative_error": max_rel_err,
                            "correlation_pearson": pearson_val,
                            "correlation_spearman": spearman_val,
                            "evidence": f"y == {a:.4g} + {b:.4g}*x",
                            "recommended_action": rec_action,
                        }
                    )
                    continue

            # 3. Log Natural (x > 0)
            if (x > 0).all():
                log_x = np.log(x)
                if np.allclose(y, log_x, rtol=rtol, atol=atol):
                    max_abs_err = float(np.max(np.abs(y - log_x)))
                    max_rel_err = float(
                        np.max(np.abs(y - log_x) / (np.abs(log_x) + 1e-15))
                    )
                    rows.append(
                        {
                            "candidate": cand,
                            "source": src,
                            "relation_type": "log_natural",
                            "status": status,
                            "complete_pair_count": count,
                            "parameter_a": None,
                            "parameter_b": None,
                            "maximum_absolute_error": max_abs_err,
                            "maximum_relative_error": max_rel_err,
                            "correlation_pearson": pearson_val,
                            "correlation_spearman": spearman_val,
                            "evidence": "y == ln(x)",
                            "recommended_action": rec_action,
                        }
                    )
                    continue

                log10_x = np.log10(x)
                if np.allclose(y, log10_x, rtol=rtol, atol=atol):
                    max_abs_err = float(np.max(np.abs(y - log10_x)))
                    max_rel_err = float(
                        np.max(np.abs(y - log10_x) / (np.abs(log10_x) + 1e-15))
                    )
                    rows.append(
                        {
                            "candidate": cand,
                            "source": src,
                            "relation_type": "log10",
                            "status": status,
                            "complete_pair_count": count,
                            "parameter_a": None,
                            "parameter_b": None,
                            "maximum_absolute_error": max_abs_err,
                            "maximum_relative_error": max_rel_err,
                            "correlation_pearson": pearson_val,
                            "correlation_spearman": spearman_val,
                            "evidence": "y == log10(x)",
                            "recommended_action": rec_action,
                        }
                    )
                    continue

            # 4. Square relation y = x^2
            sq_x = x**2
            if np.allclose(y, sq_x, rtol=rtol, atol=atol):
                max_abs_err = float(np.max(np.abs(y - sq_x)))
                max_rel_err = float(np.max(np.abs(y - sq_x) / (np.abs(sq_x) + 1e-15)))
                rows.append(
                    {
                        "candidate": cand,
                        "source": src,
                        "relation_type": "square",
                        "status": status,
                        "complete_pair_count": count,
                        "parameter_a": None,
                        "parameter_b": None,
                        "maximum_absolute_error": max_abs_err,
                        "maximum_relative_error": max_rel_err,
                        "correlation_pearson": pearson_val,
                        "correlation_spearman": spearman_val,
                        "evidence": "y == x^2",
                        "recommended_action": rec_action,
                    }
                )
                continue

            # 5. Square root relation y = sqrt(x) (x >= 0)
            if (x >= 0).all():
                sqrt_x = np.sqrt(x)
                if np.allclose(y, sqrt_x, rtol=rtol, atol=atol):
                    max_abs_err = float(np.max(np.abs(y - sqrt_x)))
                    max_rel_err = float(
                        np.max(np.abs(y - sqrt_x) / (np.abs(sqrt_x) + 1e-15))
                    )
                    rows.append(
                        {
                            "candidate": cand,
                            "source": src,
                            "relation_type": "square_root",
                            "status": status,
                            "complete_pair_count": count,
                            "parameter_a": None,
                            "parameter_b": None,
                            "maximum_absolute_error": max_abs_err,
                            "maximum_relative_error": max_rel_err,
                            "correlation_pearson": pearson_val,
                            "correlation_spearman": spearman_val,
                            "evidence": "y == sqrt(x)",
                            "recommended_action": rec_action,
                        }
                    )
                    continue

    if not rows:
        return pd.DataFrame(
            columns=[
                "candidate",
                "source",
                "relation_type",
                "status",
                "complete_pair_count",
                "parameter_a",
                "parameter_b",
                "maximum_absolute_error",
                "maximum_relative_error",
                "correlation_pearson",
                "correlation_spearman",
                "evidence",
                "recommended_action",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["candidate", "source"], kind="stable")
        .reset_index(drop=True)
    )


def _empty_standalone_high_correlation_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "left",
            "right",
            "pearson",
            "absolute_pearson",
            "spearman",
            "absolute_spearman",
            "maximum_absolute_correlation",
            "trigger_methods",
            "complete_pair_count",
            "severity",
            "recommended_action",
        ]
    )


def _build_standalone_high_correlation_pairs(
    *,
    numeric: pd.DataFrame,
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    settings: StandaloneDiagnosticSettings,
) -> pd.DataFrame:
    names = list(numeric.columns)
    rows: list[dict[str, object]] = []

    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            pearson_value = _matrix_value(pearson, left, right)
            spearman_value = _matrix_value(spearman, left, right)

            available = [
                abs(value)
                for value in (pearson_value, spearman_value)
                if value is not None and np.isfinite(value)
            ]

            if not available:
                continue

            maximum = max(available)

            if maximum < settings.correlation_review_threshold:
                continue

            trigger_methods: list[str] = []

            if (
                pearson_value is not None
                and abs(pearson_value) >= settings.correlation_review_threshold
            ):
                trigger_methods.append("pearson")

            if (
                spearman_value is not None
                and abs(spearman_value) >= settings.correlation_review_threshold
            ):
                trigger_methods.append("spearman")

            complete_pair_count = int(numeric[[left, right]].dropna().shape[0])

            severity = (
                "warning"
                if maximum >= settings.correlation_warning_threshold
                else "review"
            )

            rows.append(
                {
                    "left": left,
                    "right": right,
                    "pearson": (pearson_value if pearson_value is not None else np.nan),
                    "absolute_pearson": (
                        abs(pearson_value) if pearson_value is not None else np.nan
                    ),
                    "spearman": (
                        spearman_value if spearman_value is not None else np.nan
                    ),
                    "absolute_spearman": (
                        abs(spearman_value) if spearman_value is not None else np.nan
                    ),
                    "maximum_absolute_correlation": maximum,
                    "trigger_methods": ",".join(trigger_methods),
                    "complete_pair_count": complete_pair_count,
                    "severity": severity,
                    "recommended_action": (
                        "Review redundancy, derivation, leakage, and contribution "
                        "stability. Do not remove a predictor automatically."
                    ),
                }
            )

    if not rows:
        return _empty_standalone_high_correlation_pairs()

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["maximum_absolute_correlation", "left", "right"],
            ascending=[False, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def calculate_standalone_diagnostics(
    *,
    frame: pd.DataFrame,
    target: str,
    settings: StandaloneDiagnosticSettings,
) -> StandaloneDiagnostics:
    settings.validate()

    if target not in frame.columns:
        raise ValueError(f"Target column does not exist: {target!r}.")

    row_ids = pd.Series(
        np.arange(1, len(frame) + 1, dtype=np.int64),
        index=frame.index,
        name="row_id",
    )

    predictors = frame.drop(columns=[target]).copy()

    numeric_columns = [
        column
        for column in predictors.columns
        if pd.api.types.is_numeric_dtype(predictors[column])
    ]

    numeric = predictors.loc[:, numeric_columns].copy()

    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    pearson = numeric.corr(
        method="pearson",
        min_periods=settings.minimum_complete_pairs,
    )

    spearman = numeric.corr(
        method="spearman",
        min_periods=settings.minimum_complete_pairs,
    )

    high_pairs = _build_standalone_high_correlation_pairs(
        numeric=numeric,
        pearson=pearson,
        spearman=spearman,
        settings=settings,
    )

    dictionary_rows: list[dict[str, object]] = []

    for column in predictors.columns:
        series = predictors[column]
        dictionary_rows.append(
            {
                "predictor": str(column),
                "dtype": str(series.dtype),
                "numeric": pd.api.types.is_numeric_dtype(series),
                "non_missing": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
                "derived_status": "not_declared",
                "derived_from": "",
                "derivation": "",
            }
        )

    numeric_dictionary = pd.DataFrame(dictionary_rows)

    duplicate_config = DuplicateGroupConfig(
        enabled=True,
        rounding_decimals=settings.near_duplicate_decimals,
        near_duplicate_threshold=settings.near_duplicate_threshold,
        maximum_pairwise_rows=settings.maximum_pairwise_rows,
    )

    analysis = analyze_duplicate_groups(
        X=predictors,
        y=frame[target],
        row_ids=row_ids,
        config=duplicate_config,
    )

    suspected_relations = pd.DataFrame(
        columns=[
            "candidate",
            "source",
            "relation_type",
            "status",
            "complete_pair_count",
            "parameter_a",
            "parameter_b",
            "maximum_absolute_error",
            "maximum_relative_error",
            "pearson",
            "spearman",
            "evidence",
            "recommended_action",
        ]
    )

    return StandaloneDiagnostics(
        pearson=pearson,
        spearman=spearman,
        high_correlation_pairs=high_pairs,
        numeric_predictor_dictionary=numeric_dictionary,
        suspected_derived_relations=suspected_relations,
        exact_duplicate_groups=analysis.exact_duplicate_groups,
        near_duplicate_groups=analysis.proper_near_duplicate_groups,
        conflicting_duplicate_targets=analysis.conflicting_targets,
    )


def write_standalone_diagnostics(
    *,
    diagnostics: StandaloneDiagnostics,
    output_directory: Path,
    settings: StandaloneDiagnosticSettings,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    diagnostics.pearson.to_csv(
        output_directory / "correlation_pearson.csv",
        index=True,
        index_label="predictor",
        encoding="utf-8",
        float_format="%.17g",
    )

    diagnostics.spearman.to_csv(
        output_directory / "correlation_spearman.csv",
        index=True,
        index_label="predictor",
        encoding="utf-8",
        float_format="%.17g",
    )

    diagnostics.high_correlation_pairs.to_csv(
        output_directory / "high_correlation_pairs.csv",
        index=False,
        encoding="utf-8",
        float_format="%.17g",
    )

    diagnostics.numeric_predictor_dictionary.to_csv(
        output_directory / "numeric_predictor_dictionary.csv",
        index=False,
        encoding="utf-8",
    )

    diagnostics.suspected_derived_relations.to_csv(
        output_directory / "suspected_derived_relations.csv",
        index=False,
        encoding="utf-8",
        float_format="%.17g",
    )

    diagnostics.exact_duplicate_groups.to_csv(
        output_directory / "exact_duplicate_groups.csv",
        index=False,
        encoding="utf-8",
    )

    diagnostics.near_duplicate_groups.to_csv(
        output_directory / "near_duplicate_groups.csv",
        index=False,
        encoding="utf-8",
    )

    diagnostics.conflicting_duplicate_targets.to_csv(
        output_directory / "conflicting_duplicate_targets.csv",
        index=False,
        encoding="utf-8",
    )

    write_json_atomic(
        output_directory / "diagnostics_manifest.json",
        {
            "correlation": {
                "pearson_enabled": True,
                "spearman_enabled": True,
                "review_threshold": settings.correlation_review_threshold,
                "warning_threshold": settings.correlation_warning_threshold,
                "minimum_complete_pairs": settings.minimum_complete_pairs,
                "numeric_predictor_count": len(diagnostics.pearson.columns),
                "high_correlation_pair_count": len(diagnostics.high_correlation_pairs),
            },
            "derived_relations": {
                "enabled": False,
                "implementation_status": "deferred",
                "suspected_relation_count": 0,
            },
            "duplicate_groups": {
                "near_duplicate_decimals": settings.near_duplicate_decimals,
                "exact_group_count": (
                    int(
                        diagnostics.exact_duplicate_groups[
                            "duplicate_group_id"
                        ].nunique()
                    )
                    if not diagnostics.exact_duplicate_groups.empty
                    else 0
                ),
                "near_group_count": (
                    int(
                        diagnostics.near_duplicate_groups[
                            "near_duplicate_group_id"
                        ].nunique()
                    )
                    if not diagnostics.near_duplicate_groups.empty
                    else 0
                ),
                "conflicting_target_group_count": (
                    int(
                        diagnostics.conflicting_duplicate_targets["signature"].nunique()
                    )
                    if not diagnostics.conflicting_duplicate_targets.empty
                    else 0
                ),
            },
        },
    )
