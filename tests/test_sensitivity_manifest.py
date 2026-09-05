from __future__ import annotations

from pathlib import Path

import pytest

# Import mock helper from test_run_comparability
from test_run_comparability import create_mock_run

from gam_app.exceptions import SensitivityManifestError
from gam_app.sensitivity import (
    create_sensitivity_manifest,
)


def test_create_sensitivity_manifest_success(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ref_run = create_mock_run(workspace / "runs", "run-a", search_grid={"C": [0.1]})
    var_run = create_mock_run(workspace / "runs", "run-b", search_grid={"C": [1.0]})

    manifest = create_sensitivity_manifest(
        workspace=workspace,
        sensitivity_id="grid-study",
        name="Search grid sensitivity",
        description="Testing search grid variation",
        reference_run=ref_run,
        variant_runs=[var_run],
        vary=["search.C"],
        invariants=["dataset", "target"],
    )

    assert manifest.sensitivity_id == "grid-study"
    assert manifest.status in {"ready", "incomplete"}
    assert len(manifest.members) == 2

    # Check local membership references
    ref_mem_file = ref_run / "sensitivity" / "grid-study.json"
    var_mem_file = var_run / "sensitivity" / "grid-study.json"
    assert ref_mem_file.is_file()
    assert var_mem_file.is_file()

    # Check study manifest file
    study_manifest_p = (
        workspace / "sensitivity" / "grid-study" / "sensitivity_manifest.json"
    )
    assert study_manifest_p.is_file()


def test_create_sensitivity_duplicate_run_ids_fail(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ref_run = create_mock_run(workspace / "runs", "run-a")

    with pytest.raises(SensitivityManifestError, match="Duplicate run IDs"):
        create_sensitivity_manifest(
            workspace=workspace,
            sensitivity_id="study-1",
            name="Study",
            reference_run=ref_run,
            variant_runs=[ref_run],
            vary=[],
            invariants=[],
        )


def test_create_sensitivity_invariant_violation_fails(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ref_run = create_mock_run(workspace / "runs", "run-a", target="target_A")
    var_run = create_mock_run(workspace / "runs", "run-b", target="target_B")

    with pytest.raises(SensitivityManifestError, match="invariant 'target' violated"):
        create_sensitivity_manifest(
            workspace=workspace,
            sensitivity_id="study-inv",
            name="Invariant study",
            reference_run=ref_run,
            variant_runs=[var_run],
            vary=[],
            invariants=["target"],
        )


def test_create_sensitivity_unknown_invariant_fails(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ref_run = create_mock_run(workspace / "runs", "run-a")
    var_run = create_mock_run(workspace / "runs", "run-b")

    with pytest.raises(SensitivityManifestError, match="Unknown invariant identifier"):
        create_sensitivity_manifest(
            workspace=workspace,
            sensitivity_id="study-inv",
            name="Invariant study",
            reference_run=ref_run,
            variant_runs=[var_run],
            vary=[],
            invariants=["invalid_invariant_name"],
        )
