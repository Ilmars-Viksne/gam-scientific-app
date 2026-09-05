from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .diagnostic_schema import DIAGNOSTICS_SCHEMA_NAME
from .exceptions import DiagnosticReviewError
from .io_utils import read_json, sha256_file, write_json_atomic

DIAGNOSTIC_REVIEW_SCHEMA_NAME = "gam_diagnostic_review"
DIAGNOSTIC_REVIEW_SCHEMA_VERSION = "1.0"

ArtifactValidationStatus = Literal[
    "valid", "missing", "inconsistent", "not_written", "legacy"
]
ReviewPriority = Literal["none", "information", "review", "warning", "unavailable"]

PRIORITY_ORDER: dict[ReviewPriority, int] = {
    "none": 0,
    "information": 1,
    "review": 2,
    "warning": 3,
    "unavailable": 4,
}


@dataclass(frozen=True, slots=True)
class DiagnosticReviewItem:
    check: str
    priority: ReviewPriority
    status: str
    count: int | None
    artifact: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "priority": self.priority,
            "status": self.status,
            "count": self.count,
            "artifact": self.artifact,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticArtifactValidation:
    artifact_id: str
    path: str
    status: ArtifactValidationStatus
    expected_sha256: str | None
    actual_sha256: str | None
    expected_row_count: int | None
    actual_row_count: int | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "path": self.path,
            "status": self.status,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
            "expected_row_count": self.expected_row_count,
            "actual_row_count": self.actual_row_count,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticReview:
    run_id: str | None
    run_path: Path
    diagnostics_manifest_schema: str | None
    package_status: Literal["valid", "incomplete", "invalid"]
    scientific_priority: ReviewPriority
    verification_status: str
    validation_strategy: str | None
    duplicate_group_policy: str | None
    items: tuple[DiagnosticReviewItem, ...]
    artifacts: tuple[DiagnosticArtifactValidation, ...]

    def to_dict(self) -> dict[str, Any]:
        review_item_count = sum(
            1 for item in self.items if item.priority in {"review", "warning"}
        )
        warning_item_count = sum(1 for item in self.items if item.priority == "warning")
        unavailable_item_count = sum(
            1 for item in self.items if item.priority == "unavailable"
        )

        return {
            "schema_name": DIAGNOSTIC_REVIEW_SCHEMA_NAME,
            "schema_version": DIAGNOSTIC_REVIEW_SCHEMA_VERSION,
            "run_id": self.run_id,
            "run_path": str(self.run_path.resolve()),
            "diagnostics_manifest_schema": self.diagnostics_manifest_schema,
            "package_status": self.package_status,
            "scientific_priority": self.scientific_priority,
            "verification_status": self.verification_status,
            "validation": {
                "strategy": self.validation_strategy,
                "duplicate_group_policy": self.duplicate_group_policy,
            },
            "summary": {
                "review_item_count": review_item_count,
                "warning_item_count": warning_item_count,
                "unavailable_item_count": unavailable_item_count,
            },
            "items": [i.to_dict() for i in self.items],
            "artifacts": [a.to_dict() for a in self.artifacts],
        }


def review_diagnostics(
    *,
    diagnostics_directory: Path,
    run_directory: Path | None = None,
    verify_artifacts: bool = True,
) -> DiagnosticReview:
    run_path = (
        run_directory if run_directory is not None else diagnostics_directory.parent
    )

    run_id = None
    val_strategy = None
    dup_policy = None

    run_json_p = run_path / "run.json"
    if run_json_p.is_file():
        try:
            rdata = read_json(run_json_p)
            run_id = rdata.get("run_id")
            val_info = rdata.get("validation") or {}
            val_strategy = val_info.get("strategy")
            dup_policy = val_info.get("duplicate_group_policy")
        except Exception:
            pass

    manifest_p = diagnostics_directory / "diagnostics_manifest.json"

    items: list[DiagnosticReviewItem] = []
    artifact_validations: list[DiagnosticArtifactValidation] = []
    package_status: Literal["valid", "incomplete", "invalid"] = "valid"

    if not manifest_p.is_file():
        package_status = "incomplete"
        # Check for legacy diagnostic files
        items.append(
            DiagnosticReviewItem(
                check="diagnostic_package_integrity",
                priority="unavailable",
                status="missing",
                count=None,
                artifact="diagnostics_manifest.json",
                message="diagnostics_manifest.json is missing.",
            )
        )
        return DiagnosticReview(
            run_id=run_id,
            run_path=run_path,
            diagnostics_manifest_schema=None,
            package_status="incomplete",
            scientific_priority="unavailable",
            verification_status="not_performed",
            validation_strategy=val_strategy,
            duplicate_group_policy=dup_policy,
            items=tuple(items),
            artifacts=(),
        )

    try:
        manifest_data = read_json(manifest_p)
    except Exception as e:
        items.append(
            DiagnosticReviewItem(
                check="diagnostic_package_integrity",
                priority="unavailable",
                status="invalid",
                count=None,
                artifact="diagnostics_manifest.json",
                message=f"Failed to parse diagnostics_manifest.json: {e}",
            )
        )
        return DiagnosticReview(
            run_id=run_id,
            run_path=run_path,
            diagnostics_manifest_schema=None,
            package_status="invalid",
            scientific_priority="unavailable",
            verification_status="not_performed",
            validation_strategy=val_strategy,
            duplicate_group_policy=dup_policy,
            items=tuple(items),
            artifacts=(),
        )

    manifest_schema = manifest_data.get("schema_name")
    if manifest_schema != DIAGNOSTICS_SCHEMA_NAME:
        package_status = "invalid"

    analyses = manifest_data.get("analyses") or {}
    artifacts_dict = manifest_data.get("artifacts") or {}

    # Verify artifacts if requested
    verification_status = "performed" if verify_artifacts else "not_performed"

    for art_id, art_info in artifacts_dict.items():
        art_path_str = art_info.get("path") or ""
        art_status = art_info.get("status")

        if art_status == "not_written":
            artifact_validations.append(
                DiagnosticArtifactValidation(
                    artifact_id=art_id,
                    path=art_path_str,
                    status="not_written",
                    expected_sha256=None,
                    actual_sha256=None,
                    expected_row_count=None,
                    actual_row_count=None,
                    message="Artifact not written by design.",
                )
            )
            continue

        if not verify_artifacts:
            artifact_validations.append(
                DiagnosticArtifactValidation(
                    artifact_id=art_id,
                    path=art_path_str,
                    status="valid",
                    expected_sha256=art_info.get("sha256"),
                    actual_sha256=None,
                    expected_row_count=art_info.get("row_count"),
                    actual_row_count=None,
                    message="Verification skipped as requested.",
                )
            )
            continue

        # Safe path check
        art_p = diagnostics_directory / art_path_str
        try:
            art_p_resolved = art_p.resolve()
            diag_p_resolved = diagnostics_directory.resolve()
            if not art_p_resolved.is_relative_to(diag_p_resolved):
                package_status = "invalid"
                artifact_validations.append(
                    DiagnosticArtifactValidation(
                        artifact_id=art_id,
                        path=art_path_str,
                        status="inconsistent",
                        expected_sha256=art_info.get("sha256"),
                        actual_sha256=None,
                        expected_row_count=art_info.get("row_count"),
                        actual_row_count=None,
                        message="Path escapes diagnostics directory.",
                    )
                )
                continue
        except Exception:
            package_status = "invalid"
            artifact_validations.append(
                DiagnosticArtifactValidation(
                    artifact_id=art_id,
                    path=art_path_str,
                    status="inconsistent",
                    expected_sha256=art_info.get("sha256"),
                    actual_sha256=None,
                    expected_row_count=art_info.get("row_count"),
                    actual_row_count=None,
                    message="Invalid artifact path.",
                )
            )
            continue

        if not art_p.is_file():
            package_status = "invalid"
            artifact_validations.append(
                DiagnosticArtifactValidation(
                    artifact_id=art_id,
                    path=art_path_str,
                    status="missing",
                    expected_sha256=art_info.get("sha256"),
                    actual_sha256=None,
                    expected_row_count=art_info.get("row_count"),
                    actual_row_count=None,
                    message="Artifact file is missing on disk.",
                )
            )
            continue

        actual_sha = sha256_file(art_p)
        expected_sha = art_info.get("sha256")

        actual_row_count = None
        if art_p.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(art_p)
                actual_row_count = len(df)
            except Exception:
                pass

        expected_row_count = art_info.get("row_count")

        sha_match = expected_sha is None or actual_sha == expected_sha
        row_match = (
            expected_row_count is None
            or actual_row_count is None
            or actual_row_count == expected_row_count
        )

        if sha_match and row_match:
            artifact_validations.append(
                DiagnosticArtifactValidation(
                    artifact_id=art_id,
                    path=art_path_str,
                    status="valid",
                    expected_sha256=expected_sha,
                    actual_sha256=actual_sha,
                    expected_row_count=expected_row_count,
                    actual_row_count=actual_row_count,
                    message="Artifact checksum and row count match.",
                )
            )
        else:
            package_status = "invalid"
            msg = "Artifact checksum or row count mismatch."
            artifact_validations.append(
                DiagnosticArtifactValidation(
                    artifact_id=art_id,
                    path=art_path_str,
                    status="inconsistent",
                    expected_sha256=expected_sha,
                    actual_sha256=actual_sha,
                    expected_row_count=expected_row_count,
                    actual_row_count=actual_row_count,
                    message=msg,
                )
            )

    # Scientific Review Items
    # 1. Correlation
    corr = analyses.get("correlation") or {}
    corr_status = corr.get("status", "not_evaluated")
    if corr_status == "completed":
        rev_cnt = corr.get("review_pair_count", 0)
        warn_cnt = corr.get("warning_pair_count", 0)
        prio: ReviewPriority = "none"
        if warn_cnt > 0:
            prio = "warning"
        elif rev_cnt > 0:
            prio = "review"

        items.append(
            DiagnosticReviewItem(
                check="correlation_review_pairs",
                priority=prio,
                status="completed",
                count=rev_cnt,
                artifact="correlation/high_correlation_pairs.csv",
                message=f"Found {rev_cnt} correlation review pairs and {warn_cnt} warning pairs.",
            )
        )
    elif corr_status == "disabled":
        items.append(
            DiagnosticReviewItem(
                check="correlation_review_pairs",
                priority="information",
                status="disabled",
                count=None,
                artifact=None,
                message="Correlation analysis was disabled.",
            )
        )
    elif corr_status == "failed":
        items.append(
            DiagnosticReviewItem(
                check="correlation_review_pairs",
                priority="unavailable",
                status="failed",
                count=None,
                artifact=None,
                message="Correlation analysis failed during execution.",
            )
        )

    # 2. Declared derived predictors
    feat_meta = analyses.get("feature_metadata") or {}
    decl_count = feat_meta.get("declared_derived_count", 0)
    items.append(
        DiagnosticReviewItem(
            check="declared_derived_predictors",
            priority="information",
            status="completed",
            count=decl_count,
            artifact="feature_dictionary.csv",
            message=f"{decl_count} declared derived predictor(s) present.",
        )
    )

    # 3. Suspected derived relations
    derived = analyses.get("derived_relationships") or {}
    der_status = derived.get("status", "not_evaluated")
    if der_status == "completed":
        susp_cnt = derived.get("suspected_count", 0)
        items.append(
            DiagnosticReviewItem(
                check="suspected_derived_relations",
                priority="review" if susp_cnt > 0 else "none",
                status="completed",
                count=susp_cnt,
                artifact="suspected_derived_relationships.csv",
                message="Review whether the relationship reflects intended feature engineering, measurement definitions, or redundant representations.",
            )
        )

    # 4. Duplicate groups
    dups = analyses.get("duplicate_groups") or {}
    dup_status = dups.get("status", "not_evaluated")
    if dup_status == "completed":
        exact_cnt = dups.get("exact_group_count", 0)
        near_cnt = dups.get("proper_near_group_count", 0)
        conf_cnt = dups.get("conflicting_target_group_count", 0)

        exact_prio: ReviewPriority = "none"
        if exact_cnt > 0:
            exact_prio = "warning" if dup_policy == "error" else "review"

        items.append(
            DiagnosticReviewItem(
                check="exact_duplicate_groups",
                priority=exact_prio,
                status="completed",
                count=exact_cnt,
                artifact="exact_duplicate_groups.csv",
                message=f"Found {exact_cnt} exact duplicate predictor group(s).",
            )
        )

        items.append(
            DiagnosticReviewItem(
                check="proper_near_duplicate_groups",
                priority="review" if near_cnt > 0 else "none",
                status="completed",
                count=near_cnt,
                artifact="proper_near_duplicate_groups.csv",
                message=f"Found {near_cnt} proper near-duplicate predictor group(s).",
            )
        )

        conf_prio: ReviewPriority = "warning" if conf_cnt > 0 else "none"
        items.append(
            DiagnosticReviewItem(
                check="conflicting_duplicate_targets",
                priority=conf_prio,
                status="completed",
                count=conf_cnt,
                artifact="conflicting_duplicate_targets.csv",
                message=(
                    "Predictor-identical records contain different target labels. "
                    "A deterministic classifier using only the available predictors cannot distinguish records within those groups. "
                    "Review the records and their data provenance before interpreting model performance."
                ),
            )
        )

    # 5. Split integrity
    split_int = analyses.get("split_integrity") or {}
    sp_status = split_int.get("status", "not_evaluated")
    if sp_status == "completed":
        failed_splits = split_int.get("failed_checks_count", 0)
        if failed_splits > 0:
            package_status = "invalid"
            items.append(
                DiagnosticReviewItem(
                    check="split_integrity",
                    priority="warning",
                    status="completed",
                    count=failed_splits,
                    artifact="split_integrity.csv",
                    message=f"{failed_splits} split integrity check(s) failed.",
                )
            )
        else:
            items.append(
                DiagnosticReviewItem(
                    check="split_integrity",
                    priority="none",
                    status="completed",
                    count=0,
                    artifact="split_integrity.csv",
                    message="All split integrity checks passed.",
                )
            )

    # Determine highest scientific priority
    highest_priority: ReviewPriority = "none"
    for item in items:
        if PRIORITY_ORDER[item.priority] > PRIORITY_ORDER[highest_priority]:
            highest_priority = item.priority

    return DiagnosticReview(
        run_id=run_id,
        run_path=run_path,
        diagnostics_manifest_schema=manifest_schema,
        package_status=package_status,
        scientific_priority=highest_priority,
        verification_status=verification_status,
        validation_strategy=val_strategy,
        duplicate_group_policy=dup_policy,
        items=tuple(items),
        artifacts=tuple(artifact_validations),
    )


def render_review_text(review: DiagnosticReview) -> str:
    lines = []
    lines.append("Diagnostic review")
    lines.append("=================")
    lines.append(f"Run: {review.run_id or review.run_path.name}")
    lines.append(f"Package status: {review.package_status.upper()}")
    lines.append(f"Scientific priority: {review.scientific_priority.upper()}")
    lines.append(f"Validation strategy: {review.validation_strategy or 'unknown'}")
    lines.append(f"Duplicate policy: {review.duplicate_group_policy or 'unknown'}")
    lines.append("")

    lines.append(f"{'Review item':<32} {'Status':<15} {'Count':<8} {'Priority':<12}")
    lines.append("-" * 70)

    for item in review.items:
        cnt_str = str(item.count) if item.count is not None else "-"
        lines.append(
            f"{item.check:<32} {item.status:<15} {cnt_str:<8} {item.priority.upper():<12}"
        )

    lines.append("")
    lines.append("Recommended review")
    lines.append("------------------")
    for item in review.items:
        if item.priority in {"review", "warning"}:
            lines.append(f"- [{item.priority.upper()}] {item.check}: {item.message}")

    if not any(item.priority in {"review", "warning"} for item in review.items):
        lines.append("- No warning or review priority findings detected.")

    return "\n".join(lines)


def render_review_json(review: DiagnosticReview) -> str:
    return json.dumps(review.to_dict(), indent=2, ensure_ascii=False, allow_nan=False)


def write_review_output(
    review: DiagnosticReview,
    output_path: Path,
    overwrite: bool = False,
) -> None:
    if output_path.exists() and not overwrite:
        raise DiagnosticReviewError(
            f"Review output path already exists: {output_path}. Use --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, review.to_dict())
