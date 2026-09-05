from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd

from .io_utils import read_json, sha256_file, write_json_atomic

DIAGNOSTICS_SCHEMA_VERSION: Final[str] = "1.0"
DIAGNOSTICS_SCHEMA_NAME: Final[str] = "gam_diagnostics_manifest"

HIGH_CORRELATION_PAIR_COLUMNS: Final[tuple[str, ...]] = (
    "rank",
    "left",
    "right",
    "pearson",
    "absolute_pearson",
    "spearman",
    "absolute_spearman",
    "maximum_absolute_correlation",
    "dominant_method",
    "dominant_correlation",
    "trigger_methods",
    "complete_pair_count",
    "row_count",
    "complete_pair_fraction",
    "left_role",
    "right_role",
    "left_derived_status",
    "right_derived_status",
    "left_derived_from",
    "right_derived_from",
    "declared_derivation_relation",
    "severity",
    "recommended_action",
)

PREDICTOR_DICTIONARY_COLUMNS: Final[tuple[str, ...]] = (
    "predictor",
    "role",
    "dtype",
    "numeric",
    "non_missing",
    "missing",
    "unique",
    "metadata_status",
    "derived_status",
    "derived_from",
    "derivation",
    "description",
    "unit",
)

EXACT_DUPLICATE_GROUP_COLUMNS: Final[tuple[str, ...]] = (
    "duplicate_group_id",
    "signature",
    "group_size",
    "row_id",
)

NEAR_DUPLICATE_GROUP_COLUMNS: Final[tuple[str, ...]] = (
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
)

CONFLICTING_DUPLICATE_TARGET_COLUMNS: Final[tuple[str, ...]] = (
    "signature",
    "target",
    "row_id",
    "distinct_target_count",
)

SUSPECTED_DERIVED_RELATION_COLUMNS: Final[tuple[str, ...]] = (
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
)

SPLIT_INTEGRITY_COLUMNS: Final[tuple[str, ...]] = (
    "check",
    "passed",
    "split_count",
    "distinct_split_types",
    "total_evaluation_rows",
    "failed_split_count",
    "details",
)

AnalysisStatus = Literal[
    "completed",
    "disabled",
    "not_applicable",
    "deferred",
    "failed",
]

MetadataStatus = Literal["provided", "not_provided", "partial"]

DiagnosticDerivedStatus = Literal[
    "declared",
    "not_declared",
    "suspected",
    "not_evaluated",
]

DeclaredDerivationRelation = Literal["yes", "no", "unknown"]

DominantMethod = Literal["pearson", "spearman", "tie", "none"]

DiagnosticSeverity = Literal["warning", "review"]

DiagnosticContextKind = Literal["standalone", "run"]

CONFIG_TO_DIAGNOSTIC_DERIVED_STATUS: Final[dict[str, DiagnosticDerivedStatus]] = {
    "none": "not_declared",
    "declared": "declared",
    "suspected": "suspected",
}

SEVERITY_ORDER: Final[dict[str, int]] = {
    "warning": 0,
    "review": 1,
}

DOMINANT_CORRELATION_TOLERANCE: Final[float] = 1e-12


@dataclass(frozen=True, slots=True)
class DiagnosticFeatureMetadata:
    role: str | None
    derived_status: DiagnosticDerivedStatus
    derived_from: tuple[str, ...]
    derivation: str | None
    description: str | None
    unit: str | None
    metadata_status: MetadataStatus = "provided"


@dataclass(frozen=True, slots=True)
class DiagnosticArtifacts:
    pearson: pd.DataFrame
    spearman: pd.DataFrame
    high_correlation_pairs: pd.DataFrame
    numeric_predictor_dictionary: pd.DataFrame
    exact_duplicate_groups: pd.DataFrame
    near_duplicate_groups: pd.DataFrame
    conflicting_duplicate_targets: pd.DataFrame
    suspected_derived_relations: pd.DataFrame


@dataclass(frozen=True, slots=True)
class ArtifactManifestEntry:
    id: str
    path: str
    media_type: str
    schema: str
    status: Literal["written", "not_written", "deferred", "failed"] = "written"
    row_count: int | None = None
    byte_count: int | None = None
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "media_type": self.media_type,
            "schema": self.schema,
            "status": self.status,
            "row_count": self.row_count,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
        }


def format_logical_dataset_path(
    path: Path,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    configured_path_str = str(path)
    resolved_path = path.resolve()

    if base_dir is not None:
        try:
            rel = resolved_path.relative_to(base_dir.resolve())
            return {
                "path": rel.as_posix(),
                "path_kind": "project_relative",
                "configured_path": configured_path_str,
            }
        except ValueError:
            pass

    cwd = Path.cwd().resolve()
    try:
        rel = resolved_path.relative_to(cwd)
        return {
            "path": rel.as_posix(),
            "path_kind": "project_relative",
            "configured_path": configured_path_str,
        }
    except ValueError:
        return {
            "path": None,
            "path_kind": "absolute_external",
            "configured_path": configured_path_str,
            "basename": path.name,
        }


def build_artifact_manifest_entry(
    *,
    artifact_id: str,
    relative_path: str,
    media_type: str,
    schema_id: str,
    file_path: Path | None = None,
    row_count: int | None = None,
    status: Literal["written", "not_written", "deferred", "failed"] = "written",
) -> ArtifactManifestEntry:
    if status == "written" and file_path is not None and file_path.exists():
        size = file_path.stat().st_size
        digest = sha256_file(file_path)
        return ArtifactManifestEntry(
            id=artifact_id,
            path=relative_path,
            media_type=media_type,
            schema=schema_id,
            status="written",
            row_count=row_count if row_count is not None else 0,
            byte_count=size,
            sha256=digest,
        )

    return ArtifactManifestEntry(
        id=artifact_id,
        path=relative_path,
        media_type=media_type,
        schema=schema_id,
        status=status,
        row_count=row_count,
        byte_count=None,
        sha256=None,
    )


def update_diagnostics_manifest(
    manifest_path: Path,
    *,
    analysis_updates: dict[str, Any] | None = None,
    validation_updates: dict[str, Any] | None = None,
    split_integrity_updates: dict[str, Any] | None = None,
    artifact_entries: list[ArtifactManifestEntry] | None = None,
) -> None:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest path does not exist: {manifest_path}")

    manifest = read_json(manifest_path)

    if analysis_updates:
        analyses = manifest.setdefault("analyses", {})
        for name, payload in analysis_updates.items():
            analyses[name] = payload

    if validation_updates:
        validation = manifest.setdefault("validation", {})
        validation.update(validation_updates)

    if split_integrity_updates:
        manifest["split_integrity"] = split_integrity_updates
        analyses = manifest.setdefault("analyses", {})
        analyses["split_integrity"] = split_integrity_updates

    if artifact_entries:
        existing_artifacts: list[dict[str, Any]] = manifest.get("artifacts", [])
        artifact_map = {a["id"]: a for a in existing_artifacts}
        for entry in artifact_entries:
            artifact_map[entry.id] = entry.to_dict()
        manifest["artifacts"] = list(artifact_map.values())

    write_json_atomic(manifest_path, manifest)
