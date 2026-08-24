from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from .config import ExperimentConfig


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


# Exact & Near Duplicate Diagnostics


def build_exact_predictor_signatures(X: pd.DataFrame) -> pd.Series:
    normalized = X.copy()
    for name in normalized.columns:
        series = normalized[name]
        if pd.api.types.is_numeric_dtype(series):
            normalized[name] = pd.to_numeric(series, errors="coerce").astype("Float64")
        else:
            normalized[name] = series.astype("string")

    signatures: list[str] = []
    for row in normalized.itertuples(index=False, name=None):
        payload = [None if pd.isna(value) else value for value in row]
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
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
    duplicates["duplicate_group_id"] = "duplicate_" + duplicates.groupby(
        "signature", sort=True
    ).ngroup().add(1).astype(str).str.zfill(6)

    return duplicates[
        [
            "duplicate_group_id",
            "signature",
            "group_size",
            "row_id",
        ]
    ].sort_values(["duplicate_group_id", "row_id"], kind="stable")


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
    return conflicts[
        ["signature", "target", "row_id", "distinct_target_count"]
    ].sort_values(["signature", "target", "row_id"], kind="stable")


def build_near_duplicate_signatures(
    X: pd.DataFrame,
    *,
    decimals: int,
) -> pd.Series:
    normalized = X.copy()
    for name in normalized.columns:
        if pd.api.types.is_numeric_dtype(normalized[name]):
            normalized[name] = (
                pd.to_numeric(normalized[name], errors="coerce")
                .round(decimals)
                .astype("Float64")
            )
        else:
            normalized[name] = normalized[name].astype("string").str.strip()

    return build_exact_predictor_signatures(normalized).rename(
        "near_duplicate_signature"
    )


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
    duplicates[f"{report_prefix}_group_id"] = f"{report_prefix}_" + duplicates.groupby(
        "signature", sort=True
    ).ngroup().add(1).astype(str).str.zfill(6)

    return duplicates[
        [
            f"{report_prefix}_group_id",
            "signature",
            "group_size",
            "row_id",
        ]
    ].sort_values([f"{report_prefix}_group_id", "row_id"], kind="stable")


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
