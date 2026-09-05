from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from test_run_comparability import create_mock_run

from gam_app.diagnostic_review import review_diagnostics
from gam_app.io_utils import sha256_file


def create_mock_diagnostics(
    run_dir: Path,
    *,
    review_pair_count: int = 0,
    warning_pair_count: int = 0,
    exact_duplicate_groups: int = 0,
    conflicting_targets: int = 0,
    failed_split_checks: int = 0,
) -> Path:
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dict = {}

    # Split integrity file
    sp_df = pd.DataFrame(
        [{"check": "check_1", "status": "FAIL" if failed_split_checks > 0 else "PASS"}]
    )
    sp_path = diag_dir / "split_integrity.csv"
    sp_df.to_csv(sp_path, index=False)
    artifacts_dict["split_integrity"] = {
        "path": "split_integrity.csv",
        "status": "written",
        "sha256": sha256_file(sp_path),
        "row_count": len(sp_df),
    }

    manifest_payload = {
        "schema_name": "gam_diagnostics_manifest",
        "schema_version": "1.0",
        "created_at_utc": "2026-01-01T00:00:00Z",
        "analyses": {
            "correlation": {
                "status": "completed",
                "review_pair_count": review_pair_count,
                "warning_pair_count": warning_pair_count,
            },
            "feature_metadata": {
                "status": "completed",
                "declared_derived_count": 0,
            },
            "derived_relationships": {
                "status": "completed",
                "suspected_count": 0,
            },
            "duplicate_groups": {
                "status": "completed",
                "exact_group_count": exact_duplicate_groups,
                "proper_near_group_count": 0,
                "conflicting_target_group_count": conflicting_targets,
            },
            "split_integrity": {
                "status": "completed",
                "failed_checks_count": failed_split_checks,
            },
        },
        "artifacts": artifacts_dict,
    }

    (diag_dir / "diagnostics_manifest.json").write_text(
        json.dumps(manifest_payload), encoding="utf-8"
    )
    return diag_dir


def test_valid_diagnostic_package_no_findings(tmp_path: Path) -> None:
    run_dir = create_mock_run(tmp_path, "run-1")
    diag_dir = create_mock_diagnostics(run_dir)

    review = review_diagnostics(diagnostics_directory=diag_dir, run_directory=run_dir)
    assert review.package_status == "valid"
    assert review.scientific_priority in {"none", "information"}


def test_correlation_warning_priority(tmp_path: Path) -> None:
    run_dir = create_mock_run(tmp_path, "run-1")
    diag_dir = create_mock_diagnostics(run_dir, warning_pair_count=2)

    review = review_diagnostics(diagnostics_directory=diag_dir, run_directory=run_dir)
    assert review.package_status == "valid"
    assert review.scientific_priority == "warning"


def test_conflicting_targets_warning_priority(tmp_path: Path) -> None:
    run_dir = create_mock_run(tmp_path, "run-1")
    diag_dir = create_mock_diagnostics(run_dir, conflicting_targets=1)

    review = review_diagnostics(diagnostics_directory=diag_dir, run_directory=run_dir)
    assert review.package_status == "valid"
    assert review.scientific_priority == "warning"


def test_split_integrity_failure_invalid_package(tmp_path: Path) -> None:
    run_dir = create_mock_run(tmp_path, "run-1")
    diag_dir = create_mock_diagnostics(run_dir, failed_split_checks=1)

    review = review_diagnostics(diagnostics_directory=diag_dir, run_directory=run_dir)
    assert review.package_status == "invalid"
    assert review.scientific_priority == "warning"


def test_missing_artifact_invalid_package(tmp_path: Path) -> None:
    run_dir = create_mock_run(tmp_path, "run-1")
    diag_dir = create_mock_diagnostics(run_dir)

    (diag_dir / "split_integrity.csv").unlink()

    review = review_diagnostics(diagnostics_directory=diag_dir, run_directory=run_dir)
    assert review.package_status == "invalid"


def test_checksum_mismatch_invalid_package(tmp_path: Path) -> None:
    run_dir = create_mock_run(tmp_path, "run-1")
    diag_dir = create_mock_diagnostics(run_dir)

    (diag_dir / "split_integrity.csv").write_text(
        "corrupted,content\n1,2\n", encoding="utf-8"
    )

    review = review_diagnostics(diagnostics_directory=diag_dir, run_directory=run_dir)
    assert review.package_status == "invalid"
