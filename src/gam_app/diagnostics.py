from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from . import __version__
from .config import DuplicateGroupConfig, ExperimentConfig
from .diagnostic_schema import (
    CONFIG_TO_DIAGNOSTIC_DERIVED_STATUS,
    CONFLICTING_DUPLICATE_TARGET_COLUMNS,
    DIAGNOSTICS_SCHEMA_NAME,
    DIAGNOSTICS_SCHEMA_VERSION,
    DOMINANT_CORRELATION_TOLERANCE,
    EXACT_DUPLICATE_GROUP_COLUMNS,
    HIGH_CORRELATION_PAIR_COLUMNS,
    NEAR_DUPLICATE_GROUP_COLUMNS,
    PREDICTOR_DICTIONARY_COLUMNS,
    SEVERITY_ORDER,
    SUSPECTED_DERIVED_RELATION_COLUMNS,
    DeclaredDerivationRelation,
    DiagnosticArtifacts,
    DiagnosticContextKind,
    DiagnosticFeatureMetadata,
    DiagnosticSeverity,
    DominantMethod,
    build_artifact_manifest_entry,
    format_logical_dataset_path,
)
from .exceptions import DataValidationError
from .io_utils import utc_now, write_csv_atomic, write_json_atomic


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

    @property
    def rounding_decimals(self) -> int:
        return self.near_duplicate_decimals

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

    @property
    def high_pairs(self) -> pd.DataFrame:
        return self.high_correlation_pairs

    @property
    def numeric_summary(self) -> pd.DataFrame:
        return self.numeric_predictor_dictionary


@dataclass(frozen=True, slots=True)
class CorrelationAnalysis:
    pearson: pd.DataFrame
    spearman: pd.DataFrame
    high_pairs: pd.DataFrame
    numeric_summary: pd.DataFrame

    @property
    def high_correlation_pairs(self) -> pd.DataFrame:
        return self.high_pairs

    @property
    def numeric_predictor_dictionary(self) -> pd.DataFrame:
        return self.numeric_summary


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

    if canonical.columns.empty:
        empty_hash = hashlib.sha256(b"[]").hexdigest()
        signatures = [empty_hash] * len(X)
    else:
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
        return pd.DataFrame(columns=list(EXACT_DUPLICATE_GROUP_COLUMNS))

    duplicates["group_size"] = group_sizes.loc[group_sizes > 1].to_numpy()

    dup_ids: dict[str, str] = {}
    for sig, sub in duplicates.groupby("signature", sort=True):
        sig_str = str(sig)
        sorted_members = sorted(sub["row_id"].tolist())
        member_key = "\x1f".join(sorted_members)
        hash_id = hashlib.sha256(member_key.encode("utf-8")).hexdigest()[:12]
        dup_ids[sig_str] = f"duplicate_{hash_id}"

    duplicates["duplicate_group_id"] = duplicates["signature"].map(dup_ids)

    res = (
        duplicates[list(EXACT_DUPLICATE_GROUP_COLUMNS)]
        .sort_values(["duplicate_group_id", "row_id"], kind="stable")
        .reset_index(drop=True)
    )
    return res


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
        return pd.DataFrame(columns=list(CONFLICTING_DUPLICATE_TARGET_COLUMNS))

    conflicts["distinct_target_count"] = target_counts[target_counts > 1].to_numpy()
    res = (
        conflicts[list(CONFLICTING_DUPLICATE_TARGET_COLUMNS)]
        .sort_values(["signature", "target", "row_id"], kind="stable")
        .reset_index(drop=True)
    )
    return res


def build_near_duplicate_signatures(
    X: pd.DataFrame,
    *,
    decimals: int,
) -> pd.Series:
    canonical = canonicalize_predictors(X, decimals=decimals)
    signatures: list[str] = []

    if canonical.columns.empty:
        empty_hash = hashlib.sha256(b"[]").hexdigest()
        signatures = [empty_hash] * len(X)
    else:
        for row in canonical.itertuples(index=False, name=None):
            raw = json.dumps(
                list(row),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            signatures.append(hashlib.sha256(raw.encode("utf-8")).hexdigest())

    return pd.Series(signatures, index=X.index, name="near_duplicate_signature")


def analyze_duplicate_groups(
    X: pd.DataFrame,
    y: pd.Series,
    row_ids: pd.Series,
    config: DuplicateGroupConfig,
) -> DuplicateAnalysis:
    row_count = len(X)
    str_row_ids = row_ids.astype(str).to_numpy()

    exact_signatures = build_exact_predictor_signatures(X)
    exact_duplicate_groups = exact_duplicate_group_report(X, row_ids)
    conflicting_targets = conflicting_duplicate_target_report(X, y, row_ids)

    if not config.enabled:
        empty_near_groups = pd.DataFrame(columns=list(NEAR_DUPLICATE_GROUP_COLUMNS))
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

    if row_count > config.maximum_pairwise_rows:
        raise DataValidationError(
            "Near-duplicate analysis requires an exact pairwise scan, "
            f"but the dataset contains {row_count:,} rows and the configured "
            f"limit is {config.maximum_pairwise_rows:,}. Increase "
            "profiling.duplicate_groups.maximum_pairwise_rows only after "
            "reviewing memory and runtime requirements, or disable "
            "near-duplicate analysis."
        )

    canonical_df = canonicalize_predictors(X, decimals=config.rounding_decimals)
    canonical_signatures: list[str] = []
    if canonical_df.columns.empty:
        empty_hash = hashlib.sha256(b"[]").hexdigest()
        canonical_signatures = [empty_hash] * len(X)
    else:
        for row in canonical_df.itertuples(index=False, name=None):
            raw = json.dumps(
                list(row),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            canonical_signatures.append(hashlib.sha256(raw.encode("utf-8")).hexdigest())

    canonical_sig_series = pd.Series(canonical_signatures, index=X.index)

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

    component_members: dict[int, list[int]] = {}
    for idx in range(row_count):
        root_idx = uf.find(idx)
        component_members.setdefault(root_idx, []).append(idx)

    exact_dups_set = (
        set(exact_duplicate_groups["row_id"].tolist())
        if not exact_duplicate_groups.empty
        else set()
    )

    proper_near_rows: list[dict[str, Any]] = []

    for _root, members in component_members.items():
        if len(members) < 2:
            continue

        member_exact_sigs = [exact_signatures.iloc[m] for m in members]
        distinct_exact_cnt = len(set(member_exact_sigs))

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
                    "matched_column_count": compared_column_count,
                    "compared_column_count": compared_column_count,
                    "match_fraction": 1.0,
                    "is_exact_duplicate_member": is_exact_dup_member,
                }
            )

    if proper_near_rows:
        proper_near_duplicate_groups = (
            pd.DataFrame(proper_near_rows)[list(NEAR_DUPLICATE_GROUP_COLUMNS)]
            .sort_values(["near_duplicate_group_id", "row_id"], kind="stable")
            .reset_index(drop=True)
        )
    else:
        proper_near_duplicate_groups = pd.DataFrame(
            columns=list(NEAR_DUPLICATE_GROUP_COLUMNS)
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


def feature_metadata_from_config(
    config: ExperimentConfig,
    predictors: Sequence[str] | None = None,
) -> dict[str, DiagnosticFeatureMetadata]:
    names = predictors if predictors is not None else list(config.features.keys())
    result: dict[str, DiagnosticFeatureMetadata] = {}
    for name in names:
        if name in config.features:
            spec = config.features[name]
            derived_status = CONFIG_TO_DIAGNOSTIC_DERIVED_STATUS.get(
                spec.derived,
                "not_declared",
            )
            result[name] = DiagnosticFeatureMetadata(
                role=spec.role,
                derived_status=derived_status,
                derived_from=spec.derived_from,
                derivation=spec.derivation,
                description=spec.description,
                unit=spec.unit,
                metadata_status="provided",
            )
        else:
            result[name] = DiagnosticFeatureMetadata(
                role=None,
                derived_status="not_evaluated",
                derived_from=(),
                derivation=None,
                description=None,
                unit=None,
                metadata_status="not_provided",
            )
    return result


def standalone_feature_metadata(
    predictors: Sequence[str],
) -> dict[str, DiagnosticFeatureMetadata]:
    return {
        str(name): DiagnosticFeatureMetadata(
            role=None,
            derived_status="not_evaluated",
            derived_from=(),
            derivation=None,
            description=None,
            unit=None,
            metadata_status="not_provided",
        )
        for name in predictors
    }


def build_empty_high_correlation_pairs() -> pd.DataFrame:
    return pd.DataFrame(columns=list(HIGH_CORRELATION_PAIR_COLUMNS))


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


def build_high_correlation_pairs(
    *,
    numeric: pd.DataFrame,
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    review_threshold: float,
    warning_threshold: float,
    feature_metadata: Mapping[str, DiagnosticFeatureMetadata],
) -> pd.DataFrame:
    names = list(numeric.columns)
    rows: list[dict[str, Any]] = []
    row_count = len(numeric)

    default_meta = DiagnosticFeatureMetadata(
        role=None,
        derived_status="not_evaluated",
        derived_from=(),
        derivation=None,
        description=None,
        unit=None,
        metadata_status="not_provided",
    )

    for left_index, left in enumerate(names):
        left_meta = feature_metadata.get(left, default_meta)
        for right in names[left_index + 1 :]:
            right_meta = feature_metadata.get(right, default_meta)

            pearson_value = _matrix_value(pearson, left, right)
            spearman_value = _matrix_value(spearman, left, right)

            pearson_finite = pearson_value is not None and np.isfinite(pearson_value)
            spearman_finite = spearman_value is not None and np.isfinite(spearman_value)

            available_absolute_values = []
            if pearson_finite and pearson_value is not None:
                available_absolute_values.append(abs(pearson_value))
            if spearman_finite and spearman_value is not None:
                available_absolute_values.append(abs(spearman_value))

            if not available_absolute_values:
                continue

            maximum = max(available_absolute_values)

            if maximum < review_threshold:
                continue

            if (
                pearson_finite
                and spearman_finite
                and pearson_value is not None
                and spearman_value is not None
            ):
                diff = abs(abs(pearson_value) - abs(spearman_value))
                if diff <= DOMINANT_CORRELATION_TOLERANCE:
                    dominant_method: DominantMethod = "tie"
                    dominant_correlation: float | None = None
                elif abs(pearson_value) > abs(spearman_value):
                    dominant_method = "pearson"
                    dominant_correlation = pearson_value
                else:
                    dominant_method = "spearman"
                    dominant_correlation = spearman_value
            elif pearson_finite:
                dominant_method = "pearson"
                dominant_correlation = pearson_value
            elif spearman_finite:
                dominant_method = "spearman"
                dominant_correlation = spearman_value
            else:
                dominant_method = "none"
                dominant_correlation = None

            triggers: list[str] = []
            if (
                pearson_finite
                and pearson_value is not None
                and abs(pearson_value) >= review_threshold
            ):
                triggers.append("pearson")
            if (
                spearman_finite
                and spearman_value is not None
                and abs(spearman_value) >= review_threshold
            ):
                triggers.append("spearman")
            trigger_methods_str = "|".join(triggers)

            complete_pair_count = int(numeric[[left, right]].dropna().shape[0])
            complete_pair_fraction = (
                complete_pair_count / row_count if row_count > 0 else None
            )

            if (
                left_meta.metadata_status == "provided"
                and right_meta.metadata_status == "provided"
            ):
                declared_relation: DeclaredDerivationRelation = (
                    "yes"
                    if (
                        left in right_meta.derived_from
                        or right in left_meta.derived_from
                    )
                    else "no"
                )
            else:
                declared_relation = "unknown"

            severity: DiagnosticSeverity = (
                "warning" if maximum >= warning_threshold else "review"
            )

            if declared_relation == "yes":
                recommended_action = (
                    "Review the declared derivation and avoid using both predictors "
                    "without a documented modelling rationale."
                )
            elif severity == "warning":
                recommended_action = (
                    "Review for redundancy, unstable attribution, or derived-variable "
                    "dependence before modelling."
                )
            else:
                recommended_action = (
                    "Review scientific meaning and monitor term "
                    "stability across outer folds."
                )

            rows.append(
                {
                    "left": left,
                    "right": right,
                    "pearson": pearson_value if pearson_finite else np.nan,
                    "absolute_pearson": (
                        abs(pearson_value)
                        if pearson_finite and pearson_value is not None
                        else np.nan
                    ),
                    "spearman": spearman_value if spearman_finite else np.nan,
                    "absolute_spearman": (
                        abs(spearman_value)
                        if spearman_finite and spearman_value is not None
                        else np.nan
                    ),
                    "maximum_absolute_correlation": maximum,
                    "dominant_method": dominant_method,
                    "dominant_correlation": dominant_correlation,
                    "trigger_methods": trigger_methods_str,
                    "complete_pair_count": complete_pair_count,
                    "row_count": row_count,
                    "complete_pair_fraction": complete_pair_fraction,
                    "left_role": left_meta.role or "",
                    "right_role": right_meta.role or "",
                    "left_derived_status": left_meta.derived_status,
                    "right_derived_status": right_meta.derived_status,
                    "left_derived_from": ",".join(left_meta.derived_from),
                    "right_derived_from": ",".join(right_meta.derived_from),
                    "declared_derivation_relation": declared_relation,
                    "severity": severity,
                    "recommended_action": recommended_action,
                }
            )

    if not rows:
        return build_empty_high_correlation_pairs()

    df = pd.DataFrame(rows)
    df["_severity_order"] = df["severity"].map(SEVERITY_ORDER)

    df = (
        df.sort_values(
            [
                "_severity_order",
                "maximum_absolute_correlation",
                "complete_pair_count",
                "left",
                "right",
            ],
            ascending=[True, False, False, True, True],
            kind="stable",
        )
        .drop(columns="_severity_order")
        .reset_index(drop=True)
    )

    df.insert(0, "rank", np.arange(1, len(df) + 1, dtype=np.int64))

    return df[list(HIGH_CORRELATION_PAIR_COLUMNS)]


def build_predictor_dictionary(
    *,
    predictors: pd.DataFrame,
    feature_metadata: Mapping[str, DiagnosticFeatureMetadata],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    default_meta = DiagnosticFeatureMetadata(
        role=None,
        derived_status="not_evaluated",
        derived_from=(),
        derivation=None,
        description=None,
        unit=None,
        metadata_status="not_provided",
    )

    for name in predictors.columns:
        series = predictors[name]
        meta = feature_metadata.get(str(name), default_meta)

        rows.append(
            {
                "predictor": str(name),
                "role": meta.role if meta.role is not None else "",
                "dtype": str(series.dtype),
                "numeric": bool(pd.api.types.is_numeric_dtype(series)),
                "non_missing": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
                "metadata_status": meta.metadata_status,
                "derived_status": meta.derived_status,
                "derived_from": ",".join(meta.derived_from),
                "derivation": meta.derivation or "",
                "description": meta.description or "",
                "unit": meta.unit or "",
            }
        )

    if not rows:
        return pd.DataFrame(columns=list(PREDICTOR_DICTIONARY_COLUMNS))

    df = pd.DataFrame(rows)
    return df[list(PREDICTOR_DICTIONARY_COLUMNS)]


def calculate_correlation_analysis(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> CorrelationAnalysis:
    settings = config.profiling.correlation
    numeric = numeric_predictor_frame(frame, config)

    feature_metadata = feature_metadata_from_config(
        config,
        predictors=list(frame.columns),
    )

    predictor_dictionary = build_predictor_dictionary(
        predictors=frame,
        feature_metadata=feature_metadata,
    )

    if numeric.empty:
        empty = pd.DataFrame()
        return CorrelationAnalysis(
            pearson=empty,
            spearman=empty,
            high_pairs=build_empty_high_correlation_pairs(),
            numeric_summary=predictor_dictionary,
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
        numeric=numeric,
        pearson=pearson,
        spearman=spearman,
        review_threshold=settings.review_threshold,
        warning_threshold=settings.warning_threshold,
        feature_metadata=feature_metadata,
    )

    return CorrelationAnalysis(
        pearson=pearson,
        spearman=spearman,
        high_pairs=high_pairs,
        numeric_summary=predictor_dictionary,
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


def build_suspected_derived_relations(
    frame: pd.DataFrame,
    config: ExperimentConfig | None = None,
    *,
    minimum_complete_pairs: int = 3,
    feature_metadata: Mapping[str, DiagnosticFeatureMetadata] | None = None,
) -> pd.DataFrame:
    if config is not None:
        numeric = numeric_predictor_frame(frame, config)
        min_pairs = config.profiling.correlation.minimum_complete_pairs
        meta = feature_metadata_from_config(config, list(frame.columns))
    else:
        numeric_cols = [
            col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])
        ]
        numeric = frame.loc[:, numeric_cols].copy()
        for col in numeric.columns:
            numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
        min_pairs = minimum_complete_pairs
        meta = (
            dict(feature_metadata)
            if feature_metadata is not None
            else standalone_feature_metadata(list(frame.columns))
        )

    names = list(numeric.columns)
    rows: list[dict[str, Any]] = []

    atol = 1e-10
    rtol = 1e-8

    for cand in names:
        cand_meta = meta.get(cand, standalone_feature_metadata([cand])[cand])
        for src in names:
            if cand == src:
                continue

            src_meta = meta.get(src, standalone_feature_metadata([src])[src])
            pair_df = numeric[[src, cand]].dropna()
            count = len(pair_df)
            if count < min_pairs:
                continue

            x = pair_df[src].to_numpy(dtype=float)
            y = pair_df[cand].to_numpy(dtype=float)

            if np.isclose(x.min(), x.max(), atol=1e-12):
                continue

            pearson_val = float(np.corrcoef(x, y)[0, 1]) if count > 1 else None
            try:
                from scipy.stats import spearmanr  # type: ignore[import-untyped]

                spearman_val = float(spearmanr(x, y).statistic)
            except Exception:
                spearman_val = None

            is_declared_derived = (
                src in cand_meta.derived_from or cand in src_meta.derived_from
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
        return pd.DataFrame(columns=list(SUSPECTED_DERIVED_RELATION_COLUMNS))

    return (
        pd.DataFrame(rows)[list(SUSPECTED_DERIVED_RELATION_COLUMNS)]
        .sort_values(["candidate", "source"], kind="stable")
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

    feature_meta = standalone_feature_metadata(list(predictors.columns))

    high_pairs = build_high_correlation_pairs(
        numeric=numeric,
        pearson=pearson,
        spearman=spearman,
        review_threshold=settings.correlation_review_threshold,
        warning_threshold=settings.correlation_warning_threshold,
        feature_metadata=feature_meta,
    )

    predictor_dictionary = build_predictor_dictionary(
        predictors=predictors,
        feature_metadata=feature_meta,
    )

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

    suspected_relations = build_suspected_derived_relations(
        predictors,
        minimum_complete_pairs=settings.minimum_complete_pairs,
        feature_metadata=feature_meta,
    )

    return StandaloneDiagnostics(
        pearson=pearson,
        spearman=spearman,
        high_correlation_pairs=high_pairs,
        numeric_predictor_dictionary=predictor_dictionary,
        suspected_derived_relations=suspected_relations,
        exact_duplicate_groups=analysis.exact_duplicate_groups,
        near_duplicate_groups=analysis.proper_near_duplicate_groups,
        conflicting_duplicate_targets=analysis.conflicting_targets,
    )


def to_diagnostic_artifacts(
    obj: DiagnosticArtifacts | CorrelationAnalysis | StandaloneDiagnostics,
) -> DiagnosticArtifacts:
    if isinstance(obj, DiagnosticArtifacts):
        return obj

    if isinstance(obj, StandaloneDiagnostics):
        return DiagnosticArtifacts(
            pearson=obj.pearson,
            spearman=obj.spearman,
            high_correlation_pairs=obj.high_correlation_pairs,
            numeric_predictor_dictionary=obj.numeric_predictor_dictionary,
            exact_duplicate_groups=obj.exact_duplicate_groups,
            near_duplicate_groups=obj.near_duplicate_groups,
            conflicting_duplicate_targets=obj.conflicting_duplicate_targets,
            suspected_derived_relations=obj.suspected_derived_relations,
        )

    if isinstance(obj, CorrelationAnalysis):
        return DiagnosticArtifacts(
            pearson=obj.pearson,
            spearman=obj.spearman,
            high_correlation_pairs=obj.high_pairs,
            numeric_predictor_dictionary=obj.numeric_summary,
            exact_duplicate_groups=pd.DataFrame(
                columns=list(EXACT_DUPLICATE_GROUP_COLUMNS)
            ),
            near_duplicate_groups=pd.DataFrame(
                columns=list(NEAR_DUPLICATE_GROUP_COLUMNS)
            ),
            conflicting_duplicate_targets=pd.DataFrame(
                columns=list(CONFLICTING_DUPLICATE_TARGET_COLUMNS)
            ),
            suspected_derived_relations=pd.DataFrame(
                columns=list(SUSPECTED_DERIVED_RELATION_COLUMNS)
            ),
        )

    raise TypeError(
        f"Cannot convert object of type {type(obj).__name__} to DiagnosticArtifacts."
    )


def write_diagnostics(
    *,
    artifacts: DiagnosticArtifacts | CorrelationAnalysis | StandaloneDiagnostics,
    output_directory: Path,
    context_kind: DiagnosticContextKind,
    settings: Any = None,
    data_path: Path | None = None,
    target: str | None = None,
    run_id: str | None = None,
    model_id: str | None = None,
    command: str | None = None,
    row_count: int | None = None,
    column_count: int | None = None,
    predictor_count: int | None = None,
    data_hash: str | None = None,
    validation_info: dict[str, Any] | None = None,
    split_integrity_info: dict[str, Any] | None = None,
) -> Path:
    arts = to_diagnostic_artifacts(artifacts)
    output_directory.mkdir(parents=True, exist_ok=True)

    # 1. Write CSV files atomically
    p_path = output_directory / "correlation_pearson.csv"
    if not arts.pearson.empty:
        arts.pearson.to_csv(
            p_path,
            index=True,
            index_label="predictor",
            encoding="utf-8",
            float_format="%.17g",
        )
    else:
        pd.DataFrame().to_csv(
            p_path, index=True, index_label="predictor", encoding="utf-8"
        )

    s_path = output_directory / "correlation_spearman.csv"
    if not arts.spearman.empty:
        arts.spearman.to_csv(
            s_path,
            index=True,
            index_label="predictor",
            encoding="utf-8",
            float_format="%.17g",
        )
    else:
        pd.DataFrame().to_csv(
            s_path, index=True, index_label="predictor", encoding="utf-8"
        )

    h_path = output_directory / "high_correlation_pairs.csv"
    write_csv_atomic(
        arts.high_correlation_pairs, h_path, index=False, float_format="%.17g"
    )

    d_path = output_directory / "numeric_predictor_dictionary.csv"
    write_csv_atomic(arts.numeric_predictor_dictionary, d_path, index=False)

    s_rel_path = output_directory / "suspected_derived_relations.csv"
    write_csv_atomic(
        arts.suspected_derived_relations, s_rel_path, index=False, float_format="%.17g"
    )

    ex_path = output_directory / "exact_duplicate_groups.csv"
    write_csv_atomic(arts.exact_duplicate_groups, ex_path, index=False)

    nr_path = output_directory / "near_duplicate_groups.csv"
    write_csv_atomic(arts.near_duplicate_groups, nr_path, index=False)

    c_path = output_directory / "conflicting_duplicate_targets.csv"
    write_csv_atomic(arts.conflicting_duplicate_targets, c_path, index=False)

    # 2. Build Artifact Inventory
    artifact_entries = [
        build_artifact_manifest_entry(
            artifact_id="pearson_matrix",
            relative_path="correlation_pearson.csv",
            media_type="text/csv",
            schema_id="correlation_matrix/1.0",
            file_path=p_path,
            row_count=len(arts.pearson),
        ),
        build_artifact_manifest_entry(
            artifact_id="spearman_matrix",
            relative_path="correlation_spearman.csv",
            media_type="text/csv",
            schema_id="correlation_matrix/1.0",
            file_path=s_path,
            row_count=len(arts.spearman),
        ),
        build_artifact_manifest_entry(
            artifact_id="high_correlation_pairs",
            relative_path="high_correlation_pairs.csv",
            media_type="text/csv",
            schema_id="high_correlation_pairs/1.0",
            file_path=h_path,
            row_count=len(arts.high_correlation_pairs),
        ),
        build_artifact_manifest_entry(
            artifact_id="predictor_dictionary",
            relative_path="numeric_predictor_dictionary.csv",
            media_type="text/csv",
            schema_id="predictor_dictionary/1.0",
            file_path=d_path,
            row_count=len(arts.numeric_predictor_dictionary),
        ),
        build_artifact_manifest_entry(
            artifact_id="suspected_derived_relations",
            relative_path="suspected_derived_relations.csv",
            media_type="text/csv",
            schema_id="suspected_derived_relations/1.0",
            file_path=s_rel_path,
            row_count=len(arts.suspected_derived_relations),
        ),
        build_artifact_manifest_entry(
            artifact_id="exact_duplicate_groups",
            relative_path="exact_duplicate_groups.csv",
            media_type="text/csv",
            schema_id="exact_duplicate_groups/1.0",
            file_path=ex_path,
            row_count=len(arts.exact_duplicate_groups),
        ),
        build_artifact_manifest_entry(
            artifact_id="near_duplicate_groups",
            relative_path="near_duplicate_groups.csv",
            media_type="text/csv",
            schema_id="near_duplicate_groups/1.0",
            file_path=nr_path,
            row_count=len(arts.near_duplicate_groups),
        ),
        build_artifact_manifest_entry(
            artifact_id="conflicting_duplicate_targets",
            relative_path="conflicting_duplicate_targets.csv",
            media_type="text/csv",
            schema_id="conflicting_duplicate_targets/1.0",
            file_path=c_path,
            row_count=len(arts.conflicting_duplicate_targets),
        ),
    ]

    # 3. Assemble Manifest Sections
    high_pairs_df = arts.high_correlation_pairs
    warning_cnt = (
        int((high_pairs_df["severity"] == "warning").sum())
        if not high_pairs_df.empty and "severity" in high_pairs_df.columns
        else 0
    )
    review_cnt = (
        int((high_pairs_df["severity"] == "review").sum())
        if not high_pairs_df.empty and "severity" in high_pairs_df.columns
        else 0
    )

    # Correlation parameters
    review_thresh = getattr(
        settings,
        "review_threshold",
        getattr(settings, "correlation_review_threshold", 0.75),
    )
    warn_thresh = getattr(
        settings,
        "warning_threshold",
        getattr(settings, "correlation_warning_threshold", 0.90),
    )
    min_pairs = getattr(settings, "minimum_complete_pairs", 3)
    corr_enabled = (
        getattr(settings, "enabled", True) if hasattr(settings, "enabled") else True
    )
    pearson_enabled = getattr(settings, "pearson", True)
    spearman_enabled = getattr(settings, "spearman", True)

    num_pred_cnt = len(arts.pearson.columns) if not arts.pearson.empty else 0
    evaluated_pairs = (num_pred_cnt * (num_pred_cnt - 1)) // 2

    corr_analysis_payload = {
        "status": "completed" if corr_enabled else "disabled",
        "methods": {
            "pearson": pearson_enabled,
            "spearman": spearman_enabled,
        },
        "parameters": {
            "review_threshold": review_thresh,
            "warning_threshold": warn_thresh,
            "minimum_complete_pairs": min_pairs,
            "threshold_inclusive": True,
        },
        "ordering": [
            {"field": "severity_rank", "direction": "ascending"},
            {"field": "maximum_absolute_correlation", "direction": "descending"},
            {"field": "complete_pair_count", "direction": "descending"},
            {"field": "left", "direction": "ascending"},
            {"field": "right", "direction": "ascending"},
        ],
        "results": {
            "numeric_predictor_count": num_pred_cnt,
            "evaluated_pair_count": evaluated_pairs,
            "high_correlation_pair_count": len(arts.high_correlation_pairs),
            "warning_pair_count": warning_cnt,
            "review_pair_count": review_cnt,
        },
    }

    # Duplicate parameters
    dup_rounding = getattr(
        settings, "rounding_decimals", getattr(settings, "near_duplicate_decimals", 8)
    )
    dup_thresh = getattr(settings, "near_duplicate_threshold", 0.98)
    dup_max_rows = getattr(settings, "maximum_pairwise_rows", 10000)
    dup_enabled = (
        getattr(settings, "enabled", True) if hasattr(settings, "enabled") else True
    )

    exact_grp_cnt = (
        int(arts.exact_duplicate_groups["duplicate_group_id"].nunique())
        if not arts.exact_duplicate_groups.empty
        and "duplicate_group_id" in arts.exact_duplicate_groups.columns
        else 0
    )
    exact_row_cnt = len(arts.exact_duplicate_groups)
    near_grp_cnt = (
        int(arts.near_duplicate_groups["near_duplicate_group_id"].nunique())
        if not arts.near_duplicate_groups.empty
        and "near_duplicate_group_id" in arts.near_duplicate_groups.columns
        else 0
    )
    near_row_cnt = len(arts.near_duplicate_groups)
    conflicting_cnt = (
        int(arts.conflicting_duplicate_targets["signature"].nunique())
        if not arts.conflicting_duplicate_targets.empty
        and "signature" in arts.conflicting_duplicate_targets.columns
        else 0
    )

    dup_analysis_payload = {
        "status": "completed" if dup_enabled else "disabled",
        "parameters": {
            "rounding_decimals": dup_rounding,
            "near_duplicate_threshold": dup_thresh,
            "threshold_inclusive": True,
            "maximum_pairwise_rows": dup_max_rows,
            "include_target_in_signature": False,
            "missing_equals_missing": True,
            "transitive_closure": True,
        },
        "algorithm": {
            "name": "canonical_match_fraction_connected_components",
            "component_algorithm": "union_find",
        },
        "results": {
            "exact_group_count": exact_grp_cnt,
            "exact_row_count": exact_row_cnt,
            "proper_near_group_count": near_grp_cnt,
            "proper_near_row_count": near_row_cnt,
            "conflicting_target_group_count": conflicting_cnt,
        },
    }

    # Data dictionary section
    dict_df = arts.numeric_predictor_dictionary
    dict_total = len(dict_df)
    dict_provided = (
        int((dict_df["metadata_status"] == "provided").sum())
        if not dict_df.empty and "metadata_status" in dict_df.columns
        else 0
    )
    dict_not_provided = (
        int((dict_df["metadata_status"] == "not_provided").sum())
        if not dict_df.empty and "metadata_status" in dict_df.columns
        else 0
    )
    derived_declared = (
        int((dict_df["derived_status"] == "declared").sum())
        if not dict_df.empty and "derived_status" in dict_df.columns
        else 0
    )
    derived_not_declared = (
        int((dict_df["derived_status"] == "not_declared").sum())
        if not dict_df.empty and "derived_status" in dict_df.columns
        else 0
    )
    derived_suspected = (
        int((dict_df["derived_status"] == "suspected").sum())
        if not dict_df.empty and "derived_status" in dict_df.columns
        else 0
    )
    derived_not_eval = (
        int((dict_df["derived_status"] == "not_evaluated").sum())
        if not dict_df.empty and "derived_status" in dict_df.columns
        else 0
    )

    data_dict_payload = {
        "status": "completed",
        "metadata_status_values": {
            "provided": "Dictionary metadata was supplied for the predictor.",
            "not_provided": "No dictionary metadata was supplied.",
        },
        "derived_status_values": {
            "declared": "Metadata explicitly declares the predictor as derived.",
            "not_declared": "Metadata was evaluated and does not declare derivation.",
            "suspected": "Diagnostics identified a possible undeclared relation.",
            "not_evaluated": "No derivation determination was made.",
        },
        "results": {
            "predictor_count": dict_total,
            "metadata_provided_count": dict_provided,
            "metadata_not_provided_count": dict_not_provided,
            "derived_declared_count": derived_declared,
            "derived_not_declared_count": derived_not_declared,
            "derived_suspected_count": derived_suspected,
            "derived_not_evaluated_count": derived_not_eval,
        },
    }

    # Derived relations section
    derived_relations_payload = {
        "status": "completed",
        "results": {
            "suspected_relation_count": len(arts.suspected_derived_relations),
        },
    }

    # Context section
    context_payload = {
        "kind": context_kind,
        "run_id": run_id,
        "model_id": model_id,
        "command": command
        or ("diagnostics" if context_kind == "standalone" else "run"),
    }

    # Dataset section
    if data_path is not None:
        dataset_path_info = format_logical_dataset_path(
            data_path, base_dir=output_directory
        )
    else:
        dataset_path_info = {
            "path": None,
            "path_kind": "unknown",
            "configured_path": None,
        }

    num_cols = (
        int((dict_df["numeric"] == True).sum())  # noqa: E712
        if not dict_df.empty and "numeric" in dict_df.columns
        else 0
    )

    dataset_payload = {
        "path": dataset_path_info.get("path"),
        "path_kind": dataset_path_info.get("path_kind"),
        "configured_path": dataset_path_info.get("configured_path"),
        "sha256": data_hash,
        "row_count": row_count,
        "column_count": column_count,
        "target": target,
        "predictor_count": predictor_count
        if predictor_count is not None
        else dict_total,
        "numeric_predictor_count": num_cols,
    }

    # Validation and Split Integrity top-level & analysis section
    if validation_info is not None:
        validation_payload = validation_info
    else:
        validation_payload = (
            {"status": "not_applicable"} if context_kind == "standalone" else {}
        )

    if split_integrity_info is not None:
        split_integrity_payload = split_integrity_info
    else:
        split_integrity_payload = (
            {"status": "not_applicable"}
            if context_kind == "standalone"
            else {"status": "pending"}
        )

    manifest = {
        "schema_name": DIAGNOSTICS_SCHEMA_NAME,
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "generator": {
            "application": "gam-app",
            "application_version": __version__,
        },
        "context": context_payload,
        "dataset": dataset_payload,
        "analyses": {
            "correlation": corr_analysis_payload,
            "derived_relations": derived_relations_payload,
            "duplicate_groups": dup_analysis_payload,
            "data_dictionary": data_dict_payload,
            "split_integrity": split_integrity_payload,
        },
        "validation": validation_payload,
        "split_integrity": split_integrity_payload,
        "artifacts": [entry.to_dict() for entry in artifact_entries],
    }

    manifest_path = output_directory / "diagnostics_manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest_path


def write_standalone_diagnostics(
    *,
    diagnostics: StandaloneDiagnostics | DiagnosticArtifacts,
    output_directory: Path,
    settings: StandaloneDiagnosticSettings,
    data_path: Path | None = None,
    target: str | None = None,
    row_count: int | None = None,
    column_count: int | None = None,
    data_hash: str | None = None,
) -> Path:
    return write_diagnostics(
        artifacts=diagnostics,
        output_directory=output_directory,
        context_kind="standalone",
        settings=settings,
        data_path=data_path,
        target=target,
        row_count=row_count,
        column_count=column_count,
        data_hash=data_hash,
        command="profile",
    )
