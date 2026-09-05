from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from e2e_helpers import (
    assert_run_completed,
    create_and_execute,
    load_json_object,
    write_config,
    write_dataset,
)


def make_grouped_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for class_index, target in enumerate(("A", "B", "C")):
        for group_index in range(6):
            group_id = f"{target}-g{group_index}"

            for member in range(2):
                base = class_index * 100 + group_index * 10 + member
                rows.append(
                    {
                        "row_id": f"{group_id}-{member}",
                        "batch_id": group_id,
                        "x1": float(base),
                        "x2": float(base % 7),
                        "x3": float(member),
                        "x4": float(class_index),
                        "target": target,
                    }
                )

    frame = pd.DataFrame(rows)

    # Exact duplicate bridge: A-g0 and A-g1 receive identical predictors for member 0.
    # row A-g0-0 (idx 0) and row A-g1-0 (idx 2)
    idx_g0 = frame.index[frame["row_id"] == "A-g0-0"][0]
    idx_g1 = frame.index[frame["row_id"] == "A-g1-0"][0]
    frame.loc[idx_g1, ["x1", "x2", "x3", "x4"]] = frame.loc[
        idx_g0, ["x1", "x2", "x3", "x4"]
    ].to_numpy()

    # Near-duplicate bridge: A-g1-1 and A-g2-0 match on 3 of 4 predictors (75% match threshold).
    idx_g1_1 = frame.index[frame["row_id"] == "A-g1-1"][0]
    idx_g2_0 = frame.index[frame["row_id"] == "A-g2-0"][0]
    frame.loc[idx_g2_0, ["x1", "x2", "x4"]] = frame.loc[
        idx_g1_1, ["x1", "x2", "x4"]
    ].to_numpy()
    frame.loc[idx_g2_0, "x3"] = float(frame.loc[idx_g1_1, "x3"]) + 10.0

    return frame


@pytest.mark.e2e
def test_group_aware_run_merges_transitive_groups(tmp_path: Path) -> None:
    frame = make_grouped_frame()
    data_path = tmp_path / "data.csv"
    write_dataset(frame, data_path)

    config_payload = {
        "schema_version": "1.1",
        "experiment": {
            "name": "e2e_group_aware",
            "primary_metric": "log_loss",
        },
        "data": {
            "path": str(data_path),
            "target": "target",
            "row_id": "row_id",
            "group": "batch_id",
            "time": None,
        },
        "features": {
            "x1": {"role": "smooth"},
            "x2": {"role": "smooth"},
            "x3": {"role": "smooth"},
            "x4": {"role": "smooth"},
            "batch_id": {"role": "exclude"},
            "row_id": {"role": "exclude"},
        },
        "models": [
            {
                "id": "gam_main",
                "interactions": "none",
            }
        ],
        "validation": {
            "strategy": "stratified_group",
            "outer_splits": 3,
            "outer_repeats": 1,
            "inner_splits": 2,
            "random_state": 42,
            "gap": 0,
            "test_size": None,
            "duplicate_group_policy": "group",
        },
        "profiling": {
            "duplicate_groups": {
                "enabled": True,
                "rounding_decimals": 8,
                "near_duplicate_threshold": 0.75,
                "maximum_pairwise_rows": 10000,
            },
        },
        "search": {
            "n_knots": [3],
            "degree": [2],
            "C": [1.0],
            "interaction_scale": [1.0],
        },
        "execution": {
            "workers": 1,
            "checkpoint_unit": "outer_fold",
            "stop_on_convergence_warning": False,
        },
    }

    config_path = tmp_path / "config.yaml"
    write_config(config_payload, config_path)

    workspace = tmp_path / "workspace"
    run_directory = create_and_execute(
        config_path=config_path,
        workspace=workspace,
    )

    # 1. Assert run completion
    assert_run_completed(run_directory)

    # 2. Check policy metadata in data_manifest.json
    data_manifest = load_json_object(run_directory / "data_manifest.json")
    assert data_manifest["duplicate_group_policy"] == "group"
    assert data_manifest["duplicate_group_enforcement_applied"] is True

    # 3. Read effective_validation_groups.csv and assert transitive closure
    effective_groups = pd.read_csv(
        run_directory / "diagnostics/effective_validation_groups.csv"
    )
    effective_by_row = effective_groups.set_index("row_id")["effective_group_id"]

    assert effective_by_row.loc["A-g0-0"] == effective_by_row.loc["A-g1-0"]
    assert effective_by_row.loc["A-g1-1"] == effective_by_row.loc["A-g2-0"]
    # Transitive link between g0 and g2 via g1
    assert effective_by_row.loc["A-g0-0"] == effective_by_row.loc["A-g2-0"]

    # 4. Configured group preservation: every batch_id maps to exactly 1 effective group
    joined = frame.merge(
        effective_groups,
        on="row_id",
        validate="one_to_one",
    )
    configured_group_mapping = joined.groupby("batch_id")[
        "effective_group_id"
    ].nunique()
    assert (configured_group_mapping == 1).all()

    # 5. No outer-fold leakage
    split_manifest = pd.read_csv(run_directory / "split_manifest.csv")
    membership = (
        split_manifest.merge(
            effective_groups[["row_id", "effective_group_id"]],
            on="row_id",
            validate="many_to_one",
        )
        .groupby(["repeat", "fold", "effective_group_id"])["partition"]
        .nunique()
    )
    assert (membership == 1).all()

    # 6. Verify split-integrity manifest and evidence
    integrity = pd.read_csv(run_directory / "diagnostics/split_integrity.csv")
    assert integrity["passed"].all()

    group_checks = integrity.loc[integrity["check"].eq("no_group_leakage")]
    assert not group_checks.empty
    assert group_checks["passed"].all()

    diag_manifest = load_json_object(
        run_directory / "diagnostics/diagnostics_manifest.json"
    )
    split_integrity_info = diag_manifest["split_integrity"]
    assert split_integrity_info["passed"] is True
    assert split_integrity_info["failed_result_count"] == 0

    # 7. Report assertions
    report_text = (run_directory / "reports/report.html").read_text(encoding="utf-8")
    assert "stratified_group" in report_text
    assert "batch_id" in report_text
    assert "group" in report_text
