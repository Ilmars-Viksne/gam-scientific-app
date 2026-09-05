from __future__ import annotations

import json
import platform
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import pandas as pd
import sklearn

from . import __version__
from .config import dump_config_dict, load_config
from .data import validate_training_data
from .diagnostic_schema import (
    DiagnosticArtifacts,
    build_artifact_manifest_entry,
    update_diagnostics_manifest,
)
from .diagnostics import (
    DuplicateAnalysis,
    analyze_duplicate_groups,
    build_suspected_derived_relations,
    calculate_correlation_analysis,
    write_diagnostics,
)
from .evaluation import fit_final_model, run_model
from .exceptions import DataValidationError
from .io_utils import (
    format_duration,
    sha256_file,
    stable_hash,
    utc_now,
    write_csv_atomic,
    write_json_atomic,
    write_yaml_atomic,
)
from .reporting import create_reports
from .run_store import FileRunStore
from .splitting import (
    ALL_INTEGRITY_CHECKS,
    SplitContext,
    SplitIntegrityResult,
    create_split_manifest,
    evaluate_split_integrity,
    merge_group_constraints,
    raise_for_split_integrity,
    split_integrity_frame,
)


@dataclass(frozen=True, slots=True)
class CreatedRun:
    run_id: str
    run_directory: Path
    metadata_path: Path
    status_path: Path


@dataclass(frozen=True, slots=True)
class DuplicatePolicyResult:
    policy: Literal["report", "group", "error"]
    effective_groups: pd.Series | None
    exact_group_count: int
    proper_near_group_count: int
    grouped_row_count: int
    configured_group_count: int
    effective_group_count: int


def apply_duplicate_group_policy(
    *,
    policy: Literal["report", "group", "error"],
    configured_groups: pd.Series | None,
    duplicate_analysis: DuplicateAnalysis,
    row_ids: pd.Series,
    row_count: int,
) -> DuplicatePolicyResult:
    exact_dups = duplicate_analysis.exact_duplicate_groups
    proper_near_dups = duplicate_analysis.proper_near_duplicate_groups

    exact_group_count = (
        int(exact_dups["duplicate_group_id"].nunique()) if not exact_dups.empty else 0
    )
    proper_near_group_count = (
        int(proper_near_dups["near_duplicate_group_id"].nunique())
        if not proper_near_dups.empty
        else 0
    )

    configured_group_count = (
        int(configured_groups.nunique()) if configured_groups is not None else 0
    )

    if policy == "report":
        return DuplicatePolicyResult(
            policy="report",
            effective_groups=configured_groups,
            exact_group_count=exact_group_count,
            proper_near_group_count=proper_near_group_count,
            grouped_row_count=0,
            configured_group_count=configured_group_count,
            effective_group_count=configured_group_count,
        )

    if policy == "error":
        if exact_group_count > 0 or proper_near_group_count > 0:
            exact_rows = (
                set(exact_dups["row_id"].tolist()) if not exact_dups.empty else set()
            )
            near_rows = (
                set(proper_near_dups["row_id"].tolist())
                if not proper_near_dups.empty
                else set()
            )
            affected_row_count = len(exact_rows | near_rows)

            raise DataValidationError(
                "Duplicate observations were found while "
                "validation.duplicate_group_policy='error': "
                f"{exact_group_count} exact duplicate group(s), "
                f"{proper_near_group_count} proper near-duplicate group(s), "
                f"affecting {affected_row_count} row(s). "
                "Review the duplicate diagnostics before continuing."
            )
        return DuplicatePolicyResult(
            policy="error",
            effective_groups=configured_groups,
            exact_group_count=0,
            proper_near_group_count=0,
            grouped_row_count=0,
            configured_group_count=configured_group_count,
            effective_group_count=configured_group_count,
        )

    if policy == "group":
        exact_sig_series = duplicate_analysis.exact_signatures
        near_edges = duplicate_analysis.near_edges

        effective = merge_group_constraints(
            row_count=row_count,
            configured_groups=configured_groups,
            duplicate_signatures=exact_sig_series,
            edge_constraints=[near_edges],
            row_ids=row_ids,
        )

        effective_group_count = int(effective.nunique())

        group_sizes = effective.value_counts()
        large_groups = set(group_sizes[group_sizes > 1].index)
        grouped_row_count = int(effective.isin(large_groups).sum())

        return DuplicatePolicyResult(
            policy="group",
            effective_groups=effective,
            exact_group_count=exact_group_count,
            proper_near_group_count=proper_near_group_count,
            grouped_row_count=grouped_row_count,
            configured_group_count=configured_group_count,
            effective_group_count=effective_group_count,
        )

    raise ValueError(f"Unsupported duplicate group policy: {policy!r}")


@dataclass(frozen=True, slots=True)
class PersistedIntegritySummary:
    artifact: str
    passed: bool
    result_count: int
    distinct_check_count: int
    failed_result_count: int


def _persist_split_integrity(
    *,
    run_directory: Path,
    diagnostics_manifest_path: Path,
    results: list[SplitIntegrityResult],
    store: FileRunStore,
    strategy: str,
) -> PersistedIntegritySummary:
    integrity_path = run_directory / "diagnostics" / "split_integrity.csv"
    integrity_frame = split_integrity_frame(results)

    write_csv_atomic(
        integrity_frame,
        integrity_path,
        index=False,
        encoding="utf-8",
    )

    failed_results = [result for result in results if not result.passed]
    failed_result_count = len(failed_results)
    integrity_passed = failed_result_count == 0

    evaluated_checks = sorted({result.check for result in results})
    distinct_check_count = len(evaluated_checks)

    not_applicable_checks = sorted(set(ALL_INTEGRITY_CHECKS) - set(evaluated_checks))

    result_count = len(results)
    artifact_relative = "diagnostics/split_integrity.csv"

    split_integrity_payload = {
        "status": "completed",
        "artifact": artifact_relative,
        "result_count": result_count,
        "distinct_check_count": distinct_check_count,
        "failed_result_count": failed_result_count,
        "passed": integrity_passed,
        "evaluated_checks": evaluated_checks,
        "not_applicable_checks": not_applicable_checks,
    }

    update_diagnostics_manifest(
        diagnostics_manifest_path,
        split_integrity_updates=split_integrity_payload,
        validation_updates={
            "strategy": strategy,
            "split_integrity_passed": integrity_passed,
        },
        artifact_entries=[
            build_artifact_manifest_entry(
                artifact_id="split_integrity",
                relative_path="split_integrity.csv",
                media_type="text/csv",
                schema_id="split_integrity/1.0",
                file_path=integrity_path,
                row_count=len(integrity_frame),
            )
        ],
    )

    store.event(
        "split_integrity_evaluated",
        strategy=strategy,
        passed=integrity_passed,
        result_count=result_count,
        distinct_check_count=distinct_check_count,
        failed_result_count=failed_result_count,
        artifact="diagnostics/split_integrity.csv",
    )

    return PersistedIntegritySummary(
        artifact="diagnostics/split_integrity.csv",
        passed=integrity_passed,
        result_count=result_count,
        distinct_check_count=distinct_check_count,
        failed_result_count=failed_result_count,
    )


def create_run_result(config_path: Path, workspace: Path) -> CreatedRun:
    config = load_config(config_path)

    timestamp = utc_now().replace(":", "").replace("+00:00", "Z")
    identifier = uuid.uuid4().hex[:8]
    run_id = f"run-{timestamp}-{identifier}"

    run_directory = workspace / "runs" / run_id

    try:
        store = FileRunStore(run_directory)
        store.initialize()
        resolved = dump_config_dict(config)
        data_hash = sha256_file(config.data_path)
        config_hash = stable_hash(resolved)

        created_at = utc_now()
        created_run_path = run_directory.resolve().as_posix()
        created_workspace_path = workspace.resolve().as_posix()

        run_metadata = {
            "schema_name": "gam_run_metadata",
            "schema_version": "1.0",
            "run_id": run_id,
            "created_at_utc": created_at,
            "created_run_path": created_run_path,
            "created_workspace_path": created_workspace_path,
            "application_version": __version__,
            "configuration_schema_version": config.schema_version,
            "experiment": {
                "name": config.name,
                "primary_metric": config.primary_metric,
            },
            "dataset": {
                "basename": config.data_path.name,
                "target": config.target,
                "data_hash": data_hash,
                "configured_path": str(config.data_path),
            },
            "validation": {
                "strategy": config.validation.strategy,
                "outer_splits": config.validation.outer_splits,
                "outer_repeats": config.validation.outer_repeats,
                "inner_splits": config.validation.inner_splits,
                "duplicate_group_policy": config.validation.duplicate_group_policy,
                "group_column": config.group_column,
                "time_column": config.time_column,
            },
            "models": [model.id for model in config.models],
            "config_hash": config_hash,
            "data_hash": data_hash,
            "tags": list(config.tags),
            "metadata": {k: config.metadata[k] for k in sorted(config.metadata)},
        }

        write_json_atomic(
            store.root / "run.json",
            run_metadata,
        )
        write_yaml_atomic(store.root / "config.yaml", resolved)
        write_json_atomic(
            store.root / "environment.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "scikit_learn": sklearn.__version__,
                "pandas": pd.__version__,
            },
        )
        write_json_atomic(store.status_path, {"state": "created", "run_id": run_id})

        store.event(
            "run_created",
            run_id=run_id,
            run_path=created_run_path,
            config_hash=config_hash,
            data_hash=data_hash,
        )

        return CreatedRun(
            run_id=run_id,
            run_directory=run_directory,
            metadata_path=store.root / "run.json",
            status_path=store.status_path,
        )
    except Exception:
        if run_directory.exists():
            shutil.rmtree(run_directory, ignore_errors=True)
        raise


def create_run(config_path: Path, workspace: Path) -> Path:
    return create_run_result(config_path, workspace).run_directory


def execute_run(run_directory: Path) -> None:
    run_started_at = perf_counter()
    store = FileRunStore(run_directory)
    store.acquire_lock()
    try:
        config = load_config(run_directory / "config.yaml")
        run = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
        dataset = validate_training_data(config)

        X = dataset.X
        y = dataset.y
        row_ids = dataset.row_ids
        groups = dataset.groups
        times = dataset.times
        source_frame = dataset.source_frame

        diagnostics_directory = run_directory / "diagnostics"
        diagnostics_directory.mkdir(parents=True, exist_ok=True)

        corr_analysis = calculate_correlation_analysis(source_frame, config)

        duplicate_analysis = analyze_duplicate_groups(
            X=X,
            y=y,
            row_ids=row_ids,
            config=config.profiling.duplicate_groups,
        )

        duplicate_analysis.near_edges.to_csv(
            diagnostics_directory / "near_duplicate_edges.csv",
            index=False,
        )

        suspected_relations = build_suspected_derived_relations(source_frame, config)

        artifacts = DiagnosticArtifacts(
            pearson=corr_analysis.pearson,
            spearman=corr_analysis.spearman,
            high_correlation_pairs=corr_analysis.high_correlation_pairs,
            numeric_predictor_dictionary=corr_analysis.numeric_predictor_dictionary,
            exact_duplicate_groups=duplicate_analysis.exact_duplicate_groups,
            near_duplicate_groups=duplicate_analysis.proper_near_duplicate_groups,
            conflicting_duplicate_targets=duplicate_analysis.conflicting_targets,
            suspected_derived_relations=suspected_relations,
        )

        diagnostics_manifest_path = write_diagnostics(
            artifacts=artifacts,
            output_directory=diagnostics_directory,
            context_kind="run",
            settings=config.profiling.correlation,
            data_path=config.data_path,
            target=config.target,
            run_id=run.get("run_id"),
            row_count=len(X),
            column_count=len(source_frame.columns),
            predictor_count=len(X.columns),
            data_hash=run["data_hash"],
            validation_info={
                "strategy": config.validation.strategy,
                "outer_splits": config.validation.outer_splits,
                "outer_repeats": config.validation.outer_repeats,
                "inner_splits": config.validation.inner_splits,
                "duplicate_group_policy": config.validation.duplicate_group_policy,
                "split_integrity_passed": True,
            },
        )

        policy_result = apply_duplicate_group_policy(
            policy=config.validation.duplicate_group_policy,
            configured_groups=groups,
            duplicate_analysis=duplicate_analysis,
            row_ids=row_ids,
            row_count=len(X),
        )

        effective_groups = policy_result.effective_groups

        if effective_groups is not None:
            exact_dup_map = (
                dict(
                    zip(
                        duplicate_analysis.exact_duplicate_groups["row_id"].astype(str),
                        duplicate_analysis.exact_duplicate_groups[
                            "duplicate_group_id"
                        ].astype(str),
                        strict=False,
                    )
                )
                if not duplicate_analysis.exact_duplicate_groups.empty
                else {}
            )
            near_dup_map = (
                dict(
                    zip(
                        duplicate_analysis.proper_near_duplicate_groups[
                            "row_id"
                        ].astype(str),
                        duplicate_analysis.proper_near_duplicate_groups[
                            "near_duplicate_group_id"
                        ].astype(str),
                        strict=False,
                    )
                )
                if not duplicate_analysis.proper_near_duplicate_groups.empty
                else {}
            )
            cfg_group_map = (
                dict(zip(row_ids.astype(str), groups.astype(str), strict=False))
                if groups is not None
                else {}
            )

            eff_rows: list[dict[str, Any]] = []
            for idx, rid in enumerate(row_ids.astype(str)):
                cfg_g = cfg_group_map.get(rid)
                ex_g = exact_dup_map.get(rid)
                nr_g = near_dup_map.get(rid)
                eff_g = str(effective_groups.iloc[idx])

                sources: list[str] = []
                if cfg_g is not None:
                    sources.append("configured")
                if ex_g is not None:
                    sources.append("exact")
                if nr_g is not None:
                    sources.append("near")
                if not sources:
                    sources.append("singleton")

                eff_rows.append(
                    {
                        "row_id": rid,
                        "configured_group": cfg_g,
                        "exact_duplicate_group_id": ex_g,
                        "near_duplicate_group_id": nr_g,
                        "effective_group_id": eff_g,
                        "group_sources": "|".join(sources),
                    }
                )

            eff_frame = pd.DataFrame(eff_rows)
            write_csv_atomic(
                eff_frame,
                diagnostics_directory / "effective_validation_groups.csv",
                index=False,
                encoding="utf-8",
            )

        exact_group_cnt = policy_result.exact_group_count
        near_group_cnt = policy_result.proper_near_group_count
        conflicting_group_cnt = (
            int(duplicate_analysis.conflicting_targets["signature"].nunique())
            if not duplicate_analysis.conflicting_targets.empty
            else 0
        )

        write_json_atomic(
            run_directory / "data_manifest.json",
            {
                "path": str(config.data_path),
                "sha256": run["data_hash"],
                "rows": len(X),
                "predictors": list(X.columns),
                "target": config.target,
                "row_id": config.row_id,
                "group_column": config.group_column,
                "time_column": config.time_column,
                "validation_strategy": config.validation.strategy,
                "class_counts": y.value_counts().to_dict(),
                "duplicate_group_policy": config.validation.duplicate_group_policy,
                "group_count": (
                    int(effective_groups.nunique())
                    if effective_groups is not None
                    else None
                ),
                "time_minimum": (
                    times.min().isoformat() if times is not None else None
                ),
                "time_maximum": (
                    times.max().isoformat() if times is not None else None
                ),
                "exact_duplicate_group_count": exact_group_cnt,
                "near_duplicate_group_count": near_group_cnt,
                "conflicting_duplicate_group_count": conflicting_group_cnt,
                "configured_group_count": policy_result.configured_group_count,
                "effective_validation_group_count": policy_result.effective_group_count,
                "duplicate_group_enforcement_applied": (
                    config.validation.duplicate_group_policy == "group"
                ),
            },
        )

        context = SplitContext(
            X=X,
            y=y,
            row_ids=row_ids,
            groups=effective_groups,
            times=times,
        )

        split_path = run_directory / "split_manifest.csv"
        if split_path.exists():
            splits = pd.read_csv(
                split_path,
                dtype={
                    "row_id": "string",
                    "group_id": "string",
                },
            )
            splits["row_id"] = splits["row_id"].astype(str)
            if "group_id" in splits.columns:
                splits["group_id"] = splits["group_id"].astype("string")
        else:
            splits = create_split_manifest(config, context)
            splits.to_csv(
                split_path,
                index=False,
                encoding="utf-8",
            )

        store.update_status(
            state="running",
            phase="split_integrity",
        )

        integrity_results = evaluate_split_integrity(
            config,
            context,
            splits,
        )

        _persist_split_integrity(
            run_directory=run_directory,
            diagnostics_manifest_path=diagnostics_manifest_path,
            results=integrity_results,
            store=store,
            strategy=config.validation.strategy,
        )

        raise_for_split_integrity(integrity_results)

        store.update_status(
            state="running", phase="evaluation", started_at_utc=utc_now()
        )
        store.event("run_started")
        for model in config.models:
            run_model(
                config,
                model,
                X,
                y,
                row_ids,
                effective_groups,
                times,
                splits,
                store,
                run["data_hash"],
                run["config_hash"],
            )
            if store.requested("PAUSE") or store.requested("CANCEL"):
                return
            print(
                f"[{model.id}] Fitting final model on {len(X)} observations",
                flush=True,
            )
            final_fit_started_at = perf_counter()
            fit_final_model(
                config,
                model,
                X,
                y,
                store,
                groups=effective_groups,
                times=times,
            )
            print(
                f"[{model.id}] Final model completed in "
                f"{format_duration(perf_counter() - final_fit_started_at)}",
                flush=True,
            )
        create_reports(config, store)
        store.update_status(
            state="completed", phase="reporting", completed_at_utc=utc_now()
        )
        store.event("run_completed")
        print(
            f"Run completed in {format_duration(perf_counter() - run_started_at)}",
            flush=True,
        )
    except Exception as error:
        store.update_status(state="failed", error=repr(error))
        store.event("run_failed", level="ERROR", error=repr(error))
        raise
    finally:
        store.release_lock()
