from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from .comparison import (
    assess_run_comparability,
    split_manifest_sha256,
)
from .exceptions import SensitivityManifestError
from .io_utils import read_json, write_json_atomic

SENSITIVITY_MANIFEST_SCHEMA_NAME = "gam_sensitivity_manifest"
SENSITIVITY_MANIFEST_SCHEMA_VERSION = "1.0"

SENSITIVITY_MEMBERSHIP_SCHEMA_NAME = "gam_sensitivity_membership"
SENSITIVITY_MEMBERSHIP_SCHEMA_VERSION = "1.0"

SENSITIVITY_INVARIANTS = {
    "dataset",
    "target",
    "class_labels",
    "predictor_set",
    "feature_roles",
    "validation_strategy",
    "outer_split_assignments",
    "metric_schema",
    "application_version",
}

SensitivityStatus = Literal["draft", "ready", "incomplete"]
SensitivityMemberRole = Literal["reference", "variant"]


@dataclass(frozen=True, slots=True)
class SensitivityMember:
    run_id: str
    run_path: Path
    role: SensitivityMemberRole
    label: str
    varied_settings: Mapping[str, Any]
    diagnostic_review: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {
            "run_id": self.run_id,
            "path": str(self.run_path),
            "role": self.role,
            "label": self.label,
            "varied_settings": dict(self.varied_settings),
        }
        if self.diagnostic_review:
            res["diagnostic_review"] = dict(self.diagnostic_review)
        return res


@dataclass(frozen=True, slots=True)
class SensitivityDesignCheck:
    check: str
    level: Literal["pass", "warning", "fail"]
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "level": self.level,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class SensitivityManifest:
    sensitivity_id: str
    name: str
    description: str | None
    created_at_utc: str
    reference_run_id: str
    members: tuple[SensitivityMember, ...]
    declared_varied_paths: tuple[str, ...]
    expected_invariants: tuple[str, ...]
    status: SensitivityStatus
    checks: tuple[SensitivityDesignCheck, ...]
    pairwise_comparability: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": SENSITIVITY_MANIFEST_SCHEMA_NAME,
            "schema_version": SENSITIVITY_MANIFEST_SCHEMA_VERSION,
            "sensitivity_id": self.sensitivity_id,
            "name": self.name,
            "description": self.description,
            "created_at_utc": self.created_at_utc,
            "reference_run_id": self.reference_run_id,
            "declared_varied_paths": list(self.declared_varied_paths),
            "expected_invariants": list(self.expected_invariants),
            "members": [m.to_dict() for m in self.members],
            "checks": [c.to_dict() for c in self.checks],
            "pairwise_comparability": list(self.pairwise_comparability),
            "status": self.status,
        }


def flatten_mapping(
    payload: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_mapping(value, prefix=full_key))
        else:
            flat[full_key] = value
    return flat


def _load_member_run_info(run_path: Path) -> dict[str, Any]:
    run_json_p = run_path / "run.json"
    status_p = run_path / "status.json"
    config_p = run_path / "config.yaml"
    split_p = run_path / "split_manifest.csv"

    if not run_json_p.is_file():
        raise SensitivityManifestError(f"Run directory {run_path} is missing run.json.")
    if not status_p.is_file():
        raise SensitivityManifestError(
            f"Run directory {run_path} is missing status.json."
        )
    if not config_p.is_file():
        raise SensitivityManifestError(
            f"Run directory {run_path} is missing config.yaml."
        )

    run_json = read_json(run_json_p)
    status_json = read_json(status_p)
    config = yaml.safe_load(config_p.read_text(encoding="utf-8"))

    split_manifest = pd.read_csv(split_p) if split_p.is_file() else None

    diag_rev = None
    rev_json_p = run_path / "reviews" / "diagnostic_review.json"
    if rev_json_p.is_file():
        try:
            rev_data = read_json(rev_json_p)
            summary = rev_data.get("summary", {})
            diag_rev = {
                "package_status": rev_data.get("package_status"),
                "scientific_priority": rev_data.get("scientific_priority"),
                "warning_item_count": summary.get("warning_item_count", 0),
                "review_item_count": summary.get("review_item_count", 0),
            }
        except Exception:
            pass

    return {
        "run_path": run_path,
        "run_id": run_json.get("run_id", run_path.name),
        "run_json": run_json,
        "status_json": status_json,
        "config": config,
        "flat_config": flatten_mapping(config) if isinstance(config, dict) else {},
        "split_manifest": split_manifest,
        "diagnostic_review": diag_rev,
    }


def create_sensitivity_manifest(
    *,
    workspace: Path,
    sensitivity_id: str,
    name: str,
    description: str | None = None,
    reference_run: Path,
    variant_runs: list[Path],
    vary: list[str],
    invariants: list[str],
    output: Path | None = None,
    overwrite: bool = False,
) -> SensitivityManifest:
    if not sensitivity_id or not sensitivity_id.strip():
        raise SensitivityManifestError("Sensitivity ID cannot be empty.")

    for inv in invariants:
        if inv not in SENSITIVITY_INVARIANTS:
            raise SensitivityManifestError(
                f"Unknown invariant identifier {inv!r}. Allowed invariants: {sorted(SENSITIVITY_INVARIANTS)}"
            )

    # 1. Load reference and variants
    ref_info = _load_member_run_info(reference_run)
    variant_infos = [_load_member_run_info(v) for v in variant_runs]

    all_infos = [ref_info] + variant_infos
    run_ids = [info["run_id"] for info in all_infos]

    if len(run_ids) != len(set(run_ids)):
        raise SensitivityManifestError(
            f"Duplicate run IDs found in sensitivity members: {run_ids}"
        )

    checks: list[SensitivityDesignCheck] = []
    status: SensitivityStatus = "ready"

    # Check completed status
    for info in all_infos:
        st = info["status_json"].get("state")
        if st != "completed":
            checks.append(
                SensitivityDesignCheck(
                    check="member_run_completed",
                    level="warning",
                    details=f"Member run {info['run_id']} state is {st!r} (expected 'completed').",
                )
            )
            status = "incomplete"

    # Check declared vary paths exist and differ
    ref_flat = ref_info["flat_config"]
    members_list: list[SensitivityMember] = []

    # Add reference member
    ref_varied_settings = {
        path: ref_flat.get(path) for path in vary if path in ref_flat
    }
    members_list.append(
        SensitivityMember(
            run_id=ref_info["run_id"],
            run_path=ref_info["run_path"],
            role="reference",
            label="reference",
            varied_settings=ref_varied_settings,
            diagnostic_review=ref_info.get("diagnostic_review"),
        )
    )

    for v_info in variant_infos:
        v_flat = v_info["flat_config"]
        v_varied_settings: dict[str, Any] = {}
        has_difference_on_vary = False

        for path in vary:
            if path not in ref_flat or path not in v_flat:
                checks.append(
                    SensitivityDesignCheck(
                        check="declared_vary_path_exists",
                        level="fail",
                        details=f"Declared vary path {path!r} missing in configuration of {v_info['run_id']}.",
                    )
                )
                raise SensitivityManifestError(
                    f"Declared vary path {path!r} is missing from configuration in member runs."
                )

            ref_val = ref_flat[path]
            v_val = v_flat[path]
            v_varied_settings[path] = v_val

            if ref_val != v_val:
                has_difference_on_vary = True

        if vary and not has_difference_on_vary:
            checks.append(
                SensitivityDesignCheck(
                    check="declared_vary_path_differs",
                    level="warning",
                    details=f"Variant run {v_info['run_id']} has identical values to reference on all declared vary paths.",
                )
            )
            status = "incomplete"

        # Check undeclared configuration differences
        all_keys = set(ref_flat.keys()) | set(v_flat.keys())
        undeclared_diffs = []
        for k in all_keys:
            if k in vary:
                continue
            if ref_flat.get(k) != v_flat.get(k):
                undeclared_diffs.append(k)

        if undeclared_diffs:
            checks.append(
                SensitivityDesignCheck(
                    check="undeclared_configuration_differences",
                    level="warning",
                    details=f"Variant run {v_info['run_id']} has undeclared configuration differences in: {sorted(undeclared_diffs)}",
                )
            )
            status = "incomplete"

        members_list.append(
            SensitivityMember(
                run_id=v_info["run_id"],
                run_path=v_info["run_path"],
                role="variant",
                label=f"variant-{v_info['run_id']}",
                varied_settings=v_varied_settings,
                diagnostic_review=v_info.get("diagnostic_review"),
            )
        )

    # Check invariants
    for inv in invariants:
        inv_failed = False
        for v_info in variant_infos:
            if inv == "dataset":
                ref_h = ref_info["run_json"].get("data", {}).get("sha256") or ref_info[
                    "run_json"
                ].get("dataset_hash")
                v_h = v_info["run_json"].get("data", {}).get("sha256") or v_info[
                    "run_json"
                ].get("dataset_hash")
                if not ref_h or not v_h or ref_h != v_h:
                    inv_failed = True
            elif inv == "target":
                if ref_info["config"].get("data", {}).get("target") != v_info[
                    "config"
                ].get("data", {}).get("target"):
                    inv_failed = True
            elif inv == "validation_strategy":
                if ref_info["config"].get("validation", {}).get("strategy") != v_info[
                    "config"
                ].get("validation", {}).get("strategy"):
                    inv_failed = True
            elif inv == "outer_split_assignments":
                if (
                    ref_info["split_manifest"] is None
                    or v_info["split_manifest"] is None
                ):
                    inv_failed = True
                else:
                    ref_fingerprint = split_manifest_sha256(ref_info["split_manifest"])
                    v_fingerprint = split_manifest_sha256(v_info["split_manifest"])
                    if ref_fingerprint != v_fingerprint:
                        inv_failed = True

        if inv_failed:
            checks.append(
                SensitivityDesignCheck(
                    check=f"invariant_{inv}",
                    level="fail",
                    details=f"Expected invariant {inv!r} failed between reference and variant runs.",
                )
            )
            raise SensitivityManifestError(
                f"Expected invariant {inv!r} violated between reference and variant run."
            )

        checks.append(
            SensitivityDesignCheck(
                check=f"invariant_{inv}",
                level="pass",
                details=f"Expected invariant {inv!r} holds across all members.",
            )
        )

    # Assess pairwise comparability
    pairwise_list: list[dict[str, Any]] = []
    ref_model = ref_info["config"].get("models", [{}])[0].get("id", "gam_main")

    for v_info in variant_infos:
        v_model = v_info["config"].get("models", [{}])[0].get("id", "gam_main")
        assessment = assess_run_comparability(
            left_run=ref_info["run_path"],
            left_model=ref_model,
            right_run=v_info["run_path"],
            right_model=v_model,
        )

        failed_checks = [c.check for c in assessment.checks if c.level == "fail"]
        warning_checks = [c.check for c in assessment.checks if c.level == "warning"]

        pairwise_list.append(
            {
                "reference_run_id": ref_info["run_id"],
                "variant_run_id": v_info["run_id"],
                "comparable": assessment.comparable,
                "failed_checks": failed_checks,
                "warning_checks": warning_checks,
            }
        )

    # Construct manifest object
    manifest = SensitivityManifest(
        sensitivity_id=sensitivity_id,
        name=name,
        description=description,
        created_at_utc=datetime.now(UTC).isoformat(),
        reference_run_id=ref_info["run_id"],
        members=tuple(members_list),
        declared_varied_paths=tuple(vary),
        expected_invariants=tuple(invariants),
        status=status,
        checks=tuple(checks),
        pairwise_comparability=tuple(pairwise_list),
    )

    # Save manifest and member references
    default_manifest_path = (
        workspace / "sensitivity" / sensitivity_id / "sensitivity_manifest.json"
    )
    manifest_out_path = output if output is not None else default_manifest_path

    if manifest_out_path.exists() and not overwrite:
        raise SensitivityManifestError(
            f"Sensitivity manifest output path already exists: {manifest_out_path}. Use --overwrite to replace it."
        )

    manifest_out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(manifest_out_path, manifest.to_dict())

    # Write local membership reference to all member runs
    for m in members_list:
        mem_file = m.run_path / "sensitivity" / f"{sensitivity_id}.json"

        if mem_file.exists() and not overwrite:
            pass  # Do not overwrite if exists unless overwrite, or overwrite it safely
        mem_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            rel_manifest_path = manifest_out_path.relative_to(
                m.run_path / "sensitivity"
            ).as_posix()
        except Exception:
            rel_manifest_path = str(manifest_out_path.resolve())

        mem_payload = {
            "schema_name": SENSITIVITY_MEMBERSHIP_SCHEMA_NAME,
            "schema_version": SENSITIVITY_MEMBERSHIP_SCHEMA_VERSION,
            "sensitivity_id": sensitivity_id,
            "manifest_path": rel_manifest_path,
            "role": m.role,
            "reference_run_id": ref_info["run_id"],
            "member_run_id": m.run_id,
        }
        write_json_atomic(mem_file, mem_payload)

    return manifest


def render_sensitivity_text(manifest: SensitivityManifest) -> str:
    lines = []
    lines.append("Sensitivity Study Manifest")
    lines.append("==========================")
    lines.append(f"Sensitivity ID: {manifest.sensitivity_id}")
    lines.append(f"Name: {manifest.name}")
    if manifest.description:
        lines.append(f"Description: {manifest.description}")
    lines.append(f"Created at: {manifest.created_at_utc}")
    lines.append(f"Status: {manifest.status.upper()}")
    lines.append(f"Reference Run: {manifest.reference_run_id}")
    lines.append("")

    lines.append("Declared Varied Paths:")
    if manifest.declared_varied_paths:
        for p in manifest.declared_varied_paths:
            lines.append(f"- {p}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("Expected Invariants:")
    if manifest.expected_invariants:
        for inv in manifest.expected_invariants:
            lines.append(f"- {inv}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("Members:")
    for m in manifest.members:
        lines.append(f"- [{m.role.upper()}] {m.run_id} ({m.run_path})")
        if m.varied_settings:
            for k, v in m.varied_settings.items():
                lines.append(f"    {k} = {v}")
    lines.append("")

    lines.append("Design Checks:")
    for c in manifest.checks:
        lines.append(f"- [{c.level.upper()}] {c.check}: {c.details}")
    lines.append("")

    lines.append("Pairwise Comparability:")
    for p in manifest.pairwise_comparability:
        comp_str = "PASS" if p["comparable"] else "FAIL"
        lines.append(f"- {p['reference_run_id']} vs {p['variant_run_id']}: {comp_str}")
        if p["failed_checks"]:
            lines.append(f"    Failed: {p['failed_checks']}")
        if p["warning_checks"]:
            lines.append(f"    Warnings: {p['warning_checks']}")

    return "\n".join(lines)
