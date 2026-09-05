from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    CURRENT_CONFIG_SCHEMA_VERSION,
    LEGACY_CONFIG_SCHEMA_VERSIONS,
    SUPPORTED_CONFIG_SCHEMA_VERSIONS,
)
from .exceptions import ConfigurationError

CONFIG_MIGRATIONS: dict[tuple[str, str], tuple[str, ...]] = {
    ("1.0", "1.1"): (
        "Add validation.strategy; the legacy behavior maps to 'stratified'.",
        "Add data.group and data.time with null values when unused.",
        "Add validation.gap with value 0.",
        "Add validation.test_size with a null value.",
        "Add validation.duplicate_group_policy with value 'report'.",
        "Add profiling.correlation settings or accept current defaults.",
        "Add profiling.duplicate_groups settings or accept current defaults.",
    ),
}


@dataclass(frozen=True, slots=True)
class ConfigSchemaAssessment:
    detected_version: str
    current_version: str
    supported: bool
    migration_recommended: bool
    guidance: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConfigMigrationResult:
    source_version: str
    target_version: str
    payload: dict[str, Any]
    changes: tuple[str, ...]


def assess_config_schema(payload: Mapping[str, Any]) -> ConfigSchemaAssessment:
    detected_version = str(payload.get("schema_version", "1.0"))
    current_version = CURRENT_CONFIG_SCHEMA_VERSION
    supported = detected_version in SUPPORTED_CONFIG_SCHEMA_VERSIONS
    migration_recommended = detected_version in LEGACY_CONFIG_SCHEMA_VERSIONS

    guidance_list: list[str] = []
    if "schema_version" not in payload:
        guidance_list.append(
            "No schema_version was declared. The file was interpreted as schema 1.0."
        )

    if (detected_version, current_version) in CONFIG_MIGRATIONS:
        guidance_list.extend(CONFIG_MIGRATIONS[(detected_version, current_version)])

    return ConfigSchemaAssessment(
        detected_version=detected_version,
        current_version=current_version,
        supported=supported,
        migration_recommended=migration_recommended,
        guidance=tuple(guidance_list),
    )


def _rebase_path(
    path_str: str,
    *,
    input_dir: Path,
    output_dir: Path,
) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return p.as_posix()

    abs_data_path = (input_dir / p).resolve()
    try:
        rel_to_out = os.path.relpath(abs_data_path, output_dir)
        return Path(rel_to_out).as_posix()
    except ValueError:
        return abs_data_path.as_posix()


def migrate_config_payload(
    payload: Mapping[str, Any],
    *,
    input_directory: Path = Path("."),
    output_directory: Path = Path("."),
    target_version: str = CURRENT_CONFIG_SCHEMA_VERSION,
) -> ConfigMigrationResult:
    assessment = assess_config_schema(payload)
    if not assessment.supported:
        raise ConfigurationError(
            f"Unsupported configuration schema_version {assessment.detected_version!r}. "
            f"This installation supports: {', '.join(SUPPORTED_CONFIG_SCHEMA_VERSIONS)}. "
            f"The current schema is {CURRENT_CONFIG_SCHEMA_VERSION!r}. "
            "If this file was created by an older version, use "
            "'gam-app migrate-config'. If it was created by a newer version, "
            "upgrade gam-app rather than removing or changing schema_version manually."
        )

    migrated: dict[str, Any] = {
        key: (value.copy() if isinstance(value, dict) else value)
        for key, value in payload.items()
    }

    changes: list[str] = []
    source_version = assessment.detected_version

    if source_version == target_version and "schema_version" in payload:
        # Check if rebase is needed
        data_sec = migrated.get("data", {})
        if "path" in data_sec and not Path(data_sec["path"]).is_absolute():
            old_path = data_sec["path"]
            new_path = _rebase_path(
                old_path,
                input_dir=input_directory,
                output_dir=output_directory,
            )
            if old_path != new_path:
                migrated["data"] = {**data_sec, "path": new_path}
                changes.append(f"Rebased data.path from {old_path!r} to {new_path!r}.")

        return ConfigMigrationResult(
            source_version=source_version,
            target_version=target_version,
            payload=migrated,
            changes=tuple(changes),
        )

    # Migrating 1.0 -> 1.1
    data = migrated.setdefault("data", {})
    if "path" in data:
        old_path = data["path"]
        new_path = _rebase_path(
            old_path,
            input_dir=input_directory,
            output_dir=output_directory,
        )
        data["path"] = new_path
        if old_path != new_path:
            changes.append(f"Rebased data.path from {old_path!r} to {new_path!r}.")

    data.setdefault("group", None)
    data.setdefault("time", None)

    validation = migrated.setdefault("validation", {})
    if "strategy" not in validation:
        validation["strategy"] = "stratified"
        changes.append("Added validation.strategy='stratified'.")

    if "gap" not in validation:
        validation["gap"] = 0
        changes.append("Added validation.gap=0.")

    if "test_size" not in validation:
        validation["test_size"] = None
        changes.append("Added validation.test_size=null.")

    if "duplicate_group_policy" not in validation:
        validation["duplicate_group_policy"] = "report"
        changes.append("Added validation.duplicate_group_policy='report'.")

    profiling = migrated.setdefault("profiling", {})
    correlation = profiling.setdefault("correlation", {})
    correlation.setdefault("enabled", True)
    correlation.setdefault("pearson", True)
    correlation.setdefault("spearman", True)
    correlation.setdefault("review_threshold", 0.75)
    correlation.setdefault("warning_threshold", 0.90)
    correlation.setdefault("minimum_complete_pairs", 3)

    duplicate_groups = profiling.setdefault("duplicate_groups", {})
    duplicate_groups.setdefault("enabled", True)
    duplicate_groups.setdefault("rounding_decimals", 8)
    duplicate_groups.setdefault("near_duplicate_threshold", 0.98)
    duplicate_groups.setdefault("maximum_pairwise_rows", 10000)
    duplicate_groups.setdefault("include_target_in_signature", False)

    migrated["schema_version"] = target_version
    changes.append(f"Updated schema_version to {target_version!r}.")

    return ConfigMigrationResult(
        source_version=source_version,
        target_version=target_version,
        payload=migrated,
        changes=tuple(changes),
    )
