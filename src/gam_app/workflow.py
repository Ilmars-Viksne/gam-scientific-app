from __future__ import annotations

import json
import platform
import sys
import uuid
from pathlib import Path
from time import perf_counter

import pandas as pd
import sklearn

from . import __version__
from .config import dump_config_dict, load_config
from .data import validate_training_data
from .diagnostics import (
    build_exact_predictor_signatures,
    build_near_duplicate_signatures,
    build_suspected_derived_relations,
    calculate_correlation_analysis,
    conflicting_duplicate_target_report,
    duplicate_signature_report,
    exact_duplicate_group_report,
    save_correlation_analysis,
)
from .evaluation import fit_final_model, run_model
from .exceptions import DataValidationError
from .io_utils import (
    format_duration,
    sha256_file,
    stable_hash,
    utc_now,
    write_json_atomic,
    write_yaml_atomic,
)
from .reporting import create_reports
from .run_store import FileRunStore
from .splitting import (
    ALL_INTEGRITY_CHECKS,
    SplitContext,
    create_split_manifest,
    evaluate_split_integrity,
    merge_group_constraints,
    raise_for_split_integrity,
    split_integrity_frame,
)


def create_run(config_path: Path, workspace: Path) -> Path:
    config = load_config(config_path)

    timestamp = utc_now().replace(":", "").replace("+00:00", "Z")
    identifier = uuid.uuid4().hex[:8]
    run_id = f"run-{timestamp}-{identifier}"

    run_directory = workspace / "runs" / run_id
    store = FileRunStore(run_directory)
    store.initialize()
    resolved = dump_config_dict(config)
    data_hash = sha256_file(config.data_path)
    config_hash = stable_hash(resolved)
    write_json_atomic(
        store.root / "run.json",
        {
            "schema_version": config.schema_version,
            "run_id": run_id,
            "created_at_utc": utc_now(),
            "data_hash": data_hash,
            "config_hash": config_hash,
            "application_version": __version__,
        },
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
    return run_directory


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

        # Correlation Diagnostics
        high_corr_count = 0
        if config.profiling.correlation.enabled:
            corr_analysis = calculate_correlation_analysis(source_frame, config)
            save_correlation_analysis(corr_analysis, diagnostics_directory)
            high_corr_count = len(corr_analysis.high_pairs)

        # Duplicate Diagnostics
        exact_duplicates = pd.DataFrame()
        conflicting_duplicates = pd.DataFrame()
        near_duplicate_rows = pd.DataFrame()

        if config.profiling.duplicate_groups.enabled:
            exact_duplicates = exact_duplicate_group_report(X, row_ids)
            exact_duplicates.to_csv(
                diagnostics_directory / "exact_duplicate_groups.csv",
                index=False,
            )

            conflicting_duplicates = conflicting_duplicate_target_report(X, y, row_ids)
            conflicting_duplicates.to_csv(
                diagnostics_directory / "conflicting_duplicate_targets.csv",
                index=False,
            )

            near_signatures = build_near_duplicate_signatures(
                X,
                decimals=config.profiling.duplicate_groups.rounding_decimals,
            )
            near_duplicate_rows = duplicate_signature_report(
                signatures=near_signatures,
                row_ids=row_ids,
                report_prefix="near_duplicate",
            )
            near_duplicate_rows.to_csv(
                diagnostics_directory / "near_duplicate_groups.csv",
                index=False,
            )

        # Suspected Derived Relations
        suspected_relations = build_suspected_derived_relations(source_frame, config)
        suspected_relations.to_csv(
            diagnostics_directory / "suspected_derived_relations.csv",
            index=False,
        )

        # Write diagnostics manifest
        numeric_pred_count = sum(
            1
            for name, spec in config.features.items()
            if name in source_frame.columns
            and pd.api.types.is_numeric_dtype(source_frame[name])
        )
        exact_group_cnt = (
            int(exact_duplicates["duplicate_group_id"].nunique())
            if not exact_duplicates.empty
            else 0
        )
        conflicting_group_cnt = (
            int(conflicting_duplicates["signature"].nunique())
            if not conflicting_duplicates.empty
            else 0
        )
        near_group_cnt = (
            int(near_duplicate_rows["near_duplicate_group_id"].nunique())
            if not near_duplicate_rows.empty
            else 0
        )

        diagnostics_manifest_path = diagnostics_directory / "diagnostics_manifest.json"
        write_json_atomic(
            diagnostics_manifest_path,
            {
                "correlation": {
                    "enabled": config.profiling.correlation.enabled,
                    "pearson_enabled": config.profiling.correlation.pearson,
                    "spearman_enabled": config.profiling.correlation.spearman,
                    "review_threshold": config.profiling.correlation.review_threshold,
                    "warning_threshold": config.profiling.correlation.warning_threshold,
                    "numeric_predictor_count": numeric_pred_count,
                    "high_correlation_pair_count": high_corr_count,
                },
                "duplicate_groups": {
                    "enabled": config.profiling.duplicate_groups.enabled,
                    "rounding_decimals": (
                        config.profiling.duplicate_groups.rounding_decimals
                    ),
                    "exact_group_count": exact_group_cnt,
                    "near_group_count": near_group_cnt,
                    "conflicting_target_group_count": conflicting_group_cnt,
                },
                "derived_relations": {
                    "enabled": True,
                    "suspected_relation_count": len(suspected_relations),
                },
                "validation": {
                    "strategy": config.validation.strategy,
                    "outer_splits": config.validation.outer_splits,
                    "outer_repeats": config.validation.outer_repeats,
                    "inner_splits": config.validation.inner_splits,
                    "duplicate_group_policy": config.validation.duplicate_group_policy,
                    "split_integrity_passed": True,
                },
            },
        )

        # Effective groups resolution & policy enforcement
        effective_groups = groups
        if config.validation.duplicate_group_policy == "error":
            if not exact_duplicates.empty:
                raise DataValidationError(
                    "Exact duplicate predictor groups were found and "
                    "validation.duplicate_group_policy is 'error'. "
                    "Review diagnostics/exact_duplicate_groups.csv."
                )

        if config.validation.duplicate_group_policy == "group":
            dup_sigs = build_exact_predictor_signatures(X)
            effective_groups = merge_group_constraints(
                row_count=len(X),
                configured_groups=groups,
                duplicate_signatures=dup_sigs,
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
                "conflicting_duplicate_group_count": conflicting_group_cnt,
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

        integrity_frame = split_integrity_frame(integrity_results)
        integrity_path = diagnostics_directory / "split_integrity.csv"
        integrity_frame.to_csv(
            integrity_path,
            index=False,
            encoding="utf-8",
        )

        integrity_passed = (
            bool(integrity_frame["passed"].all())
            if not integrity_frame.empty
            else False
        )
        failed_result_count = (
            int((~integrity_frame["passed"]).sum()) if not integrity_frame.empty else 0
        )
        evaluated_checks = (
            sorted(set(integrity_frame["check"])) if not integrity_frame.empty else []
        )
        not_applicable_checks = sorted(
            set(ALL_INTEGRITY_CHECKS) - set(evaluated_checks)
        )

        diagnostics_manifest = json.loads(
            diagnostics_manifest_path.read_text(encoding="utf-8")
        )
        diagnostics_manifest["split_integrity"] = {
            "artifact": str(integrity_path.relative_to(run_directory)),
            "result_count": int(len(integrity_frame)),
            "distinct_check_count": int(len(evaluated_checks)),
            "failed_result_count": failed_result_count,
            "passed": integrity_passed,
            "evaluated_checks": evaluated_checks,
            "not_applicable_checks": not_applicable_checks,
        }
        diagnostics_manifest["validation"]["split_integrity_passed"] = integrity_passed
        write_json_atomic(diagnostics_manifest_path, diagnostics_manifest)

        store.event(
            "split_integrity_evaluated",
            strategy=config.validation.strategy,
            passed=integrity_passed,
            result_count=int(len(integrity_frame)),
            failed_result_count=failed_result_count,
            artifact=str(integrity_path.relative_to(run_directory)),
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
