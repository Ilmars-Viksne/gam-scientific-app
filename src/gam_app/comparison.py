from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from .exceptions import RunComparabilityError
from .io_utils import read_json, sha256_file, write_json_atomic, write_text_atomic

COMPARABILITY_SCHEMA_NAME = "gam_run_comparability"
COMPARABILITY_SCHEMA_VERSION = "1.0"

COMPARISON_MANIFEST_SCHEMA_NAME = "gam_comparison_manifest"
COMPARISON_MANIFEST_SCHEMA_VERSION = "1.0"

ComparabilityLevel = Literal["pass", "warning", "fail", "not_evaluated"]


@dataclass(frozen=True, slots=True)
class ComparabilityCheck:
    check: str
    level: ComparabilityLevel
    left: Any
    right: Any
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "level": self.level,
            "left": self.left,
            "right": self.right,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class ComparabilityAssessment:
    left_run: Path
    right_run: Path
    left_model: str
    right_model: str
    checks: tuple[ComparabilityCheck, ...]

    @property
    def comparable(self) -> bool:
        return not any(check.level == "fail" for check in self.checks)

    @property
    def warnings(self) -> tuple[ComparabilityCheck, ...]:
        return tuple(check for check in self.checks if check.level == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_run": str(self.left_run),
            "right_run": str(self.right_run),
            "left_model": self.left_model,
            "right_model": self.right_model,
            "comparable": self.comparable,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True, slots=True)
class PairedComparisonResult:
    assessment: ComparabilityAssessment
    comparison: pd.DataFrame | None
    summary: pd.DataFrame | None
    sensitivity_context: dict[str, Any] | None = None


def canonical_test_membership(split_manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"repeat", "fold", "row_id", "partition"}
    if not required.issubset(split_manifest.columns):
        missing = sorted(required - set(split_manifest.columns))
        raise ValueError(f"Split manifest is missing required columns: {missing}")

    return (
        split_manifest.loc[
            split_manifest["partition"] == "test",
            ["repeat", "fold", "row_id"],
        ]
        .sort_values(["repeat", "fold", "row_id"], kind="stable")
        .reset_index(drop=True)
    )


def canonical_split_assignments(split_manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"repeat", "fold", "row_id", "partition"}
    if not required.issubset(split_manifest.columns):
        missing = sorted(required - set(split_manifest.columns))
        raise ValueError(f"Split manifest is missing required columns: {missing}")

    return (
        split_manifest.loc[
            :,
            ["repeat", "fold", "row_id", "partition"],
        ]
        .sort_values(["repeat", "fold", "partition", "row_id"], kind="stable")
        .reset_index(drop=True)
    )


def split_manifest_sha256(split_manifest: pd.DataFrame) -> str:
    canonical = canonical_split_assignments(split_manifest)
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def model_result_fingerprint(fold_metrics: pd.DataFrame) -> str:
    cols = [
        c
        for c in fold_metrics.columns
        if c
        in ["repeat", "fold", "log_loss", "accuracy", "balanced_accuracy", "macro_f1"]
    ]
    sorted_df = fold_metrics.sort_values(["repeat", "fold"], kind="stable").reset_index(
        drop=True
    )
    payload = sorted_df[cols].to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def find_shared_sensitivity_context(
    left_run: Path,
    right_run: Path,
    requested_sensitivity_id: str | None = None,
) -> dict[str, Any] | None:
    left_sens_dir = left_run / "sensitivity"
    right_sens_dir = right_run / "sensitivity"

    left_memberships: dict[str, dict[str, Any]] = {}
    if left_sens_dir.is_dir():
        for p in left_sens_dir.glob("*.json"):
            try:
                data = read_json(p)
                sid = data.get("sensitivity_id")
                if sid:
                    left_memberships[sid] = data
            except Exception:
                pass

    right_memberships: dict[str, dict[str, Any]] = {}
    if right_sens_dir.is_dir():
        for p in right_sens_dir.glob("*.json"):
            try:
                data = read_json(p)
                sid = data.get("sensitivity_id")
                if sid:
                    right_memberships[sid] = data
            except Exception:
                pass

    shared_ids = sorted(set(left_memberships.keys()) & set(right_memberships.keys()))

    if requested_sensitivity_id:
        if requested_sensitivity_id not in shared_ids:
            return {
                "shared": False,
                "requested_sensitivity_id": requested_sensitivity_id,
                "details": f"Requested sensitivity ID {requested_sensitivity_id!r} is not shared by both runs.",
            }
        selected_id = requested_sensitivity_id
    elif len(shared_ids) == 1:
        selected_id = shared_ids[0]
    elif len(shared_ids) > 1:
        return {
            "shared": True,
            "sensitivity_ids": shared_ids,
            "selected_sensitivity_id": None,
            "details": "Multiple shared sensitivity studies found; specify --sensitivity to target one.",
        }
    else:
        return {"shared": False}

    left_m = left_memberships[selected_id]
    right_m = right_memberships[selected_id]

    manifest_path_str = left_m.get("manifest_path") or right_m.get("manifest_path")
    declared_varied: list[str] = []
    if manifest_path_str:
        manifest_p = left_run / "sensitivity" / manifest_path_str
        if not manifest_p.exists():
            manifest_p = Path(manifest_path_str)
        if manifest_p.is_file():
            try:
                mdata = read_json(manifest_p)
                declared_varied = mdata.get("declared_varied_paths") or []
            except Exception:
                pass

    return {
        "shared": True,
        "sensitivity_ids": shared_ids,
        "selected_sensitivity_id": selected_id,
        "left_role": left_m.get("role"),
        "right_role": right_m.get("role"),
        "declared_varied_paths": declared_varied,
    }


def _load_run_components(run_path: Path, model_id: str) -> dict[str, Any]:
    res: dict[str, Any] = {"run_path": run_path, "model_id": model_id, "errors": []}

    run_json_path = run_path / "run.json"
    status_json_path = run_path / "status.json"
    config_yaml_path = run_path / "config.yaml"
    split_manifest_path = run_path / "split_manifest.csv"
    fold_metrics_path = run_path / "results" / model_id / "fold_metrics.csv"

    for p, name in [
        (run_json_path, "run.json"),
        (status_json_path, "status.json"),
        (config_yaml_path, "config.yaml"),
        (split_manifest_path, "split_manifest.csv"),
        (fold_metrics_path, f"results/{model_id}/fold_metrics.csv"),
    ]:
        if not p.is_file():
            res["errors"].append(f"Missing required file: {name}")

    if res["errors"]:
        return res

    try:
        res["run_json"] = read_json(run_json_path)
    except Exception as e:
        res["errors"].append(f"Failed to read run.json: {e}")

    try:
        res["status_json"] = read_json(status_json_path)
    except Exception as e:
        res["errors"].append(f"Failed to read status.json: {e}")

    try:
        res["config"] = yaml.safe_load(config_yaml_path.read_text(encoding="utf-8"))
    except Exception as e:
        res["errors"].append(f"Failed to read config.yaml: {e}")

    try:
        res["split_manifest"] = pd.read_csv(split_manifest_path)
    except Exception as e:
        res["errors"].append(f"Failed to read split_manifest.csv: {e}")

    try:
        res["fold_metrics"] = pd.read_csv(fold_metrics_path)
    except Exception as e:
        res["errors"].append(f"Failed to read fold_metrics.csv: {e}")

    return res


def assess_run_comparability(
    left_run: Path,
    left_model: str,
    right_run: Path,
    right_model: str,
) -> ComparabilityAssessment:
    checks: list[ComparabilityCheck] = []

    left_comp = _load_run_components(left_run, left_model)
    right_comp = _load_run_components(right_run, right_model)

    if left_comp["errors"] or right_comp["errors"]:
        left_errs = "; ".join(left_comp["errors"]) if left_comp["errors"] else "OK"
        right_errs = "; ".join(right_comp["errors"]) if right_comp["errors"] else "OK"
        checks.append(
            ComparabilityCheck(
                check="required_files_exist",
                level="fail",
                left=left_errs,
                right=right_errs,
                details="Missing or unreadable required run files.",
            )
        )
        return ComparabilityAssessment(
            left_run=left_run,
            right_run=right_run,
            left_model=left_model,
            right_model=right_model,
            checks=tuple(checks),
        )

    checks.append(
        ComparabilityCheck(
            check="required_files_exist",
            level="pass",
            left="present",
            right="present",
            details="All required run files are present and readable.",
        )
    )

    # Completed model check
    left_state = left_comp["status_json"].get("state")
    right_state = right_comp["status_json"].get("state")
    if left_state != "completed" or right_state != "completed":
        checks.append(
            ComparabilityCheck(
                check="run_completed",
                level="fail",
                left=left_state,
                right=right_state,
                details="Both runs must be completed.",
            )
        )
    else:
        checks.append(
            ComparabilityCheck(
                check="run_completed",
                level="pass",
                left=left_state,
                right=right_state,
                details="Both runs completed successfully.",
            )
        )

    # Dataset hash equality
    left_data_hash = left_comp["run_json"].get("data", {}).get("sha256") or left_comp[
        "run_json"
    ].get("dataset_hash")
    right_data_hash = right_comp["run_json"].get("data", {}).get(
        "sha256"
    ) or right_comp["run_json"].get("dataset_hash")

    if not left_data_hash or not right_data_hash:
        checks.append(
            ComparabilityCheck(
                check="dataset_hash_equal",
                level="fail",
                left=left_data_hash or "missing",
                right=right_data_hash or "missing",
                details="Dataset SHA-256 hash is missing from run metadata.",
            )
        )
    elif left_data_hash != right_data_hash:
        checks.append(
            ComparabilityCheck(
                check="dataset_hash_equal",
                level="fail",
                left=left_data_hash,
                right=right_data_hash,
                details="Dataset SHA-256 hashes differ.",
            )
        )
    else:
        checks.append(
            ComparabilityCheck(
                check="dataset_hash_equal",
                level="pass",
                left=left_data_hash,
                right=right_data_hash,
                details="Dataset SHA-256 hashes are identical.",
            )
        )

    # Target equality
    left_target = left_comp["config"].get("data", {}).get("target")
    right_target = right_comp["config"].get("data", {}).get("target")

    if left_target != right_target:
        checks.append(
            ComparabilityCheck(
                check="target_equal",
                level="fail",
                left=left_target,
                right=right_target,
                details="Target columns differ.",
            )
        )
    else:
        checks.append(
            ComparabilityCheck(
                check="target_equal",
                level="pass",
                left=left_target,
                right=right_target,
                details="Target columns match.",
            )
        )

    # Validation strategy equality
    left_strat = left_comp["config"].get("validation", {}).get("strategy")
    right_strat = right_comp["config"].get("validation", {}).get("strategy")

    if left_strat != right_strat:
        checks.append(
            ComparabilityCheck(
                check="validation_strategy_equal",
                level="fail",
                left=left_strat,
                right=right_strat,
                details="Validation strategies differ.",
            )
        )
    else:
        checks.append(
            ComparabilityCheck(
                check="validation_strategy_equal",
                level="pass",
                left=left_strat,
                right=right_strat,
                details="Validation strategies match.",
            )
        )

    # Outer fold design equality
    left_val = left_comp["config"].get("validation", {})
    right_val = right_comp["config"].get("validation", {})
    left_outer = (left_val.get("outer_splits"), left_val.get("outer_repeats"))
    right_outer = (right_val.get("outer_splits"), right_val.get("outer_repeats"))

    if left_outer != right_outer:
        checks.append(
            ComparabilityCheck(
                check="outer_fold_design_equal",
                level="fail",
                left=f"{left_outer[0]} splits x {left_outer[1]} repeats",
                right=f"{right_outer[0]} splits x {right_outer[1]} repeats",
                details="Outer fold designs differ.",
            )
        )
    else:
        checks.append(
            ComparabilityCheck(
                check="outer_fold_design_equal",
                level="pass",
                left=f"{left_outer[0]} splits x {left_outer[1]} repeats",
                right=f"{right_outer[0]} splits x {right_outer[1]} repeats",
                details="Outer fold designs match.",
            )
        )

    # Inner fold design
    left_inner = left_val.get("inner_splits")
    right_inner = right_val.get("inner_splits")
    if left_inner != right_inner:
        checks.append(
            ComparabilityCheck(
                check="inner_splits_equal",
                level="warning",
                left=left_inner,
                right=right_inner,
                details="Inner fold counts differ (affects hyperparameter selection).",
            )
        )

    # Split manifest equality
    left_sm = left_comp["split_manifest"]
    right_sm = right_comp["split_manifest"]

    try:
        left_canonical = canonical_split_assignments(left_sm)
        right_canonical = canonical_split_assignments(right_sm)
        if not left_canonical.equals(right_canonical):
            checks.append(
                ComparabilityCheck(
                    check="outer_split_assignments_equal",
                    level="fail",
                    left=f"sha256:{split_manifest_sha256(left_sm)[:8]}",
                    right=f"sha256:{split_manifest_sha256(right_sm)[:8]}",
                    details="Outer split train/test assignments differ across folds.",
                )
            )
        else:
            checks.append(
                ComparabilityCheck(
                    check="outer_split_assignments_equal",
                    level="pass",
                    left=f"sha256:{split_manifest_sha256(left_sm)[:8]}",
                    right=f"sha256:{split_manifest_sha256(right_sm)[:8]}",
                    details="Outer split assignments match exactly across all folds.",
                )
            )
    except Exception as e:
        checks.append(
            ComparabilityCheck(
                check="outer_split_assignments_equal",
                level="fail",
                left="invalid",
                right="invalid",
                details=f"Failed to evaluate canonical split assignments: {e}",
            )
        )

    # Check strategy-specific extra details
    if left_strat == "time" and right_strat == "time":
        left_time = (
            left_val.get("gap"),
            left_val.get("test_size"),
            left_comp["config"].get("data", {}).get("time"),
        )
        right_time = (
            right_val.get("gap"),
            right_val.get("test_size"),
            right_comp["config"].get("data", {}).get("time"),
        )
        if left_time != right_time:
            checks.append(
                ComparabilityCheck(
                    check="temporal_design_equal",
                    level="fail",
                    left=f"gap={left_time[0]}, test_size={left_time[1]}, col={left_time[2]}",
                    right=f"gap={right_time[0]}, test_size={right_time[1]}, col={right_time[2]}",
                    details="Temporal validation parameters differ.",
                )
            )

    # Metric schema & fold key check
    left_fm = left_comp["fold_metrics"]
    right_fm = right_comp["fold_metrics"]
    required_metrics = ["log_loss", "accuracy", "balanced_accuracy", "macro_f1"]

    missing_left = [m for m in required_metrics if m not in left_fm.columns]
    missing_right = [m for m in required_metrics if m not in right_fm.columns]

    if missing_left or missing_right:
        checks.append(
            ComparabilityCheck(
                check="metric_schema_equal",
                level="fail",
                left=f"missing: {missing_left}",
                right=f"missing: {missing_right}",
                details="Required metric columns are missing from fold metrics.",
            )
        )
    else:
        # Check nonfinite
        left_nonfinite = not np.isfinite(left_fm[required_metrics].to_numpy()).all()
        right_nonfinite = not np.isfinite(right_fm[required_metrics].to_numpy()).all()

        if left_nonfinite or right_nonfinite:
            checks.append(
                ComparabilityCheck(
                    check="metrics_finite",
                    level="fail",
                    left="nonfinite present" if left_nonfinite else "finite",
                    right="nonfinite present" if right_nonfinite else "finite",
                    details="Fold metrics contain nonfinite (NaN or Inf) values.",
                )
            )

        # Check duplicates in fold metrics
        left_dups = left_fm.duplicated(subset=["repeat", "fold"]).any()
        right_dups = right_fm.duplicated(subset=["repeat", "fold"]).any()

        if left_dups or right_dups:
            checks.append(
                ComparabilityCheck(
                    check="fold_keys_unique",
                    level="fail",
                    left="duplicates present" if left_dups else "unique",
                    right="duplicates present" if right_dups else "unique",
                    details="Fold metrics contain duplicate (repeat, fold) keys.",
                )
            )

        # Fold key set equality
        left_keys = set(zip(left_fm["repeat"], left_fm["fold"], strict=False))
        right_keys = set(zip(right_fm["repeat"], right_fm["fold"], strict=False))

        if left_keys != right_keys:
            checks.append(
                ComparabilityCheck(
                    check="fold_keys_equal",
                    level="fail",
                    left=f"{len(left_keys)} keys",
                    right=f"{len(right_keys)} keys",
                    details="Fold key sets (repeat, fold) differ between runs.",
                )
            )
        else:
            checks.append(
                ComparabilityCheck(
                    check="fold_keys_equal",
                    level="pass",
                    left=f"{len(left_keys)} keys",
                    right=f"{len(right_keys)} keys",
                    details="Fold key sets match.",
                )
            )

    # Informational / Warnings check
    if left_model != right_model:
        checks.append(
            ComparabilityCheck(
                check="model_id_equal",
                level="warning",
                left=left_model,
                right=right_model,
                details="Different model IDs selected (expected when comparing models).",
            )
        )

    # Compare search grids
    left_search = left_comp["config"].get("search", {})
    right_search = right_comp["config"].get("search", {})
    if left_search != right_search:
        checks.append(
            ComparabilityCheck(
                check="search_grid_equal",
                level="warning",
                left=str(left_search),
                right=str(right_search),
                details="Hyperparameter search grids differ.",
            )
        )

    return ComparabilityAssessment(
        left_run=left_run,
        right_run=right_run,
        left_model=left_model,
        right_model=right_model,
        checks=tuple(checks),
    )


def compare_paired_run_results(
    *,
    left_run: Path,
    left_model: str,
    right_run: Path,
    right_model: str,
    sensitivity_context: dict[str, Any] | None = None,
) -> PairedComparisonResult:
    assessment = assess_run_comparability(
        left_run=left_run,
        left_model=left_model,
        right_run=right_run,
        right_model=right_model,
    )

    if not assessment.comparable:
        failed_msgs = [
            f"- [{c.check}] {c.details} (left={c.left}, right={c.right})"
            for c in assessment.checks
            if c.level == "fail"
        ]
        msg = (
            "The selected run results are not comparable as paired folds:\n"
            + "\n".join(failed_msgs)
        )
        raise RunComparabilityError(msg, assessment=assessment)

    left_fm = pd.read_csv(left_run / "results" / left_model / "fold_metrics.csv")
    right_fm = pd.read_csv(right_run / "results" / right_model / "fold_metrics.csv")

    merged = left_fm.merge(
        right_fm,
        on=["repeat", "fold"],
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )

    metrics = ["log_loss", "accuracy", "balanced_accuracy", "macro_f1"]
    for metric in metrics:
        if metric in left_fm.columns and metric in right_fm.columns:
            merged[f"{metric}_difference"] = (
                merged[f"{metric}_right"] - merged[f"{metric}_left"]
            )

    diff_cols = [
        f"{metric}_difference"
        for metric in metrics
        if f"{metric}_difference" in merged.columns
    ]
    summary = merged[diff_cols].agg(["mean", "std", "median"]).T

    return PairedComparisonResult(
        assessment=assessment,
        comparison=merged,
        summary=summary,
        sensitivity_context=sensitivity_context,
    )


def render_comparison_text(
    assessment: ComparabilityAssessment,
    result: PairedComparisonResult | None = None,
    sensitivity_context: dict[str, Any] | None = None,
) -> str:
    lines = []
    lines.append("Paired-run comparability")
    lines.append("========================")
    lines.append("")
    lines.append(f"{'Check':<32} {'Status':<15} {'Left':<18} {'Right':<18}")
    lines.append("-" * 83)

    for c in assessment.checks:
        status_str = c.level.upper()
        left_str = str(c.left)[:17]
        right_str = str(c.right)[:17]
        lines.append(f"{c.check:<32} {status_str:<15} {left_str:<18} {right_str:<18}")

    lines.append("")

    if not assessment.comparable:
        lines.append("The selected results are not comparable as paired folds.")
        lines.append("")
        for c in assessment.checks:
            if c.level == "fail":
                lines.append(f"- {c.details}")
    else:
        lines.append("The selected run results are comparable as paired folds.")
        if assessment.warnings:
            lines.append("")
            lines.append("Warnings:")
            for c in assessment.warnings:
                lines.append(f"- [{c.check}] {c.details}")

    ctx = sensitivity_context or (result.sensitivity_context if result else None)
    if ctx:
        lines.append("")
        lines.append("Sensitivity provenance:")
        if ctx.get("shared"):
            lines.append(
                f"- Shared sensitivity study: {ctx.get('sensitivity_id') or ctx.get('selected_sensitivity_id')}"
            )
            lines.append(
                f"- Roles: left={ctx.get('left_role')}, right={ctx.get('right_role')}"
            )
        else:
            lines.append(
                "- Runs are not explicitly linked by a shared sensitivity manifest."
            )

    if result and result.summary is not None:
        lines.append("")
        lines.append("Metric differences (right minus left)")
        lines.append("------------------------------------")
        lines.append(result.summary.to_string())

    return "\n".join(lines)


def render_comparison_json(
    assessment: ComparabilityAssessment,
    result: PairedComparisonResult | None = None,
    output_directory: Path | None = None,
    check_only: bool = False,
    sensitivity_context: dict[str, Any] | None = None,
) -> str:
    left_json_path = assessment.left_run / "run.json"
    right_json_path = assessment.right_run / "run.json"

    left_run_id = (
        read_json(left_json_path).get("run_id")
        if left_json_path.is_file()
        else assessment.left_run.name
    )
    right_run_id = (
        read_json(right_json_path).get("run_id")
        if right_json_path.is_file()
        else assessment.right_run.name
    )

    metric_diffs = None
    if result and result.summary is not None and not check_only:
        metric_diffs = {}
        for row_label, row in result.summary.iterrows():
            metric_name = str(row_label).replace("_difference", "")
            metric_diffs[metric_name] = {
                "mean": float(row["mean"]) if pd.notna(row["mean"]) else None,
                "standard_deviation": float(row["std"])
                if pd.notna(row["std"])
                else None,
                "median": float(row["median"]) if pd.notna(row["median"]) else None,
            }

    ctx = sensitivity_context or (result.sensitivity_context if result else None)

    payload = {
        "schema_name": "gam_comparison",
        "schema_version": "1.0",
        "comparable": assessment.comparable,
        "check_only": check_only,
        "left": {
            "run_id": left_run_id,
            "model_id": assessment.left_model,
            "run_path": str(assessment.left_run.resolve()),
        },
        "right": {
            "run_id": right_run_id,
            "model_id": assessment.right_model,
            "run_path": str(assessment.right_run.resolve()),
        },
        "comparability_checks": [c.to_dict() for c in assessment.checks],
        "sensitivity": ctx,
        "metric_differences": metric_diffs,
        "output_directory": str(output_directory.resolve())
        if output_directory
        else None,
        "artifacts_written": bool(result and not check_only and output_directory),
    }

    return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)


def write_comparison_manifest(
    result: PairedComparisonResult,
    output_directory: Path,
) -> None:
    left_run = result.assessment.left_run
    right_run = result.assessment.right_run
    left_model = result.assessment.left_model
    right_model = result.assessment.right_model

    left_json = read_json(left_run / "run.json")
    right_json = read_json(right_run / "run.json")

    left_sm = pd.read_csv(left_run / "split_manifest.csv")
    right_sm = pd.read_csv(right_run / "split_manifest.csv")

    left_fm = pd.read_csv(left_run / "results" / left_model / "fold_metrics.csv")
    right_fm = pd.read_csv(right_run / "results" / right_model / "fold_metrics.csv")

    comp_csv = output_directory / "comparison.csv"
    summ_csv = output_directory / "summary.csv"

    manifest_payload = {
        "schema_name": COMPARISON_MANIFEST_SCHEMA_NAME,
        "schema_version": COMPARISON_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "left": {
            "run_id": left_json.get("run_id", left_run.name),
            "run_path": str(left_run.resolve()),
            "model_id": left_model,
            "data_hash": left_json.get("data", {}).get("sha256")
            or left_json.get("dataset_hash"),
            "config_hash": left_json.get("config_hash"),
            "split_fingerprint": split_manifest_sha256(left_sm),
            "model_result_fingerprint": model_result_fingerprint(left_fm),
        },
        "right": {
            "run_id": right_json.get("run_id", right_run.name),
            "run_path": str(right_run.resolve()),
            "model_id": right_model,
            "data_hash": right_json.get("data", {}).get("sha256")
            or right_json.get("dataset_hash"),
            "config_hash": right_json.get("config_hash"),
            "split_fingerprint": split_manifest_sha256(right_sm),
            "model_result_fingerprint": model_result_fingerprint(right_fm),
        },
        "comparability": {
            "comparable": result.assessment.comparable,
            "checks": [c.to_dict() for c in result.assessment.checks],
        },
        "sensitivity": result.sensitivity_context,
        "difference_direction": "right_minus_left",
        "artifacts": {
            "comparison": {
                "path": "comparison.csv",
                "sha256": sha256_file(comp_csv) if comp_csv.is_file() else None,
                "row_count": len(result.comparison)
                if result.comparison is not None
                else 0,
            },
            "summary": {
                "path": "summary.csv",
                "sha256": sha256_file(summ_csv) if summ_csv.is_file() else None,
                "row_count": len(result.summary) if result.summary is not None else 0,
            },
        },
    }

    write_json_atomic(output_directory / "comparison_manifest.json", manifest_payload)


def write_comparison_outputs(
    result: PairedComparisonResult,
    output_directory: Path,
    overwrite: bool = False,
) -> None:
    managed_files = [
        output_directory / "comparison.csv",
        output_directory / "summary.csv",
        output_directory / "comparison_manifest.json",
    ]

    if not overwrite:
        for f in managed_files:
            if f.exists():
                raise FileExistsError(
                    f"Managed output file already exists: {f}. Use --overwrite to replace it."
                )

    output_directory.mkdir(parents=True, exist_ok=True)

    if result.comparison is not None:
        comp_csv_str = result.comparison.to_csv(index=False)
        write_text_atomic(output_directory / "comparison.csv", comp_csv_str)

    if result.summary is not None:
        summ_csv_str = result.summary.to_csv()
        write_text_atomic(output_directory / "summary.csv", summ_csv_str)

    write_comparison_manifest(result, output_directory)
