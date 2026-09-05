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


def make_stratified_diagnostic_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for class_index, target in enumerate(("A", "B", "C")):
        for observation in range(12):
            x1 = class_index * 10.0 + observation
            rows.append(
                {
                    "row_id": f"{target}-{observation:02d}",
                    "x1": x1,
                    "x2": 2.0 * x1,
                    "x3": float(observation % 4),
                    "category": "low" if observation % 2 == 0 else "high",
                    "target": target,
                }
            )

    frame = pd.DataFrame(rows)

    # Exact duplicate with a consistent target.
    frame.loc[1, ["x1", "x2", "x3", "category"]] = frame.loc[
        0, ["x1", "x2", "x3", "category"]
    ].to_numpy()

    # Predictor-identical rows with conflicting targets.
    frame.loc[13, ["x1", "x2", "x3", "category"]] = frame.loc[
        12, ["x1", "x2", "x3", "category"]
    ].to_numpy()
    frame.loc[13, "target"] = "C"

    # Near duplicate, but not exact.
    frame.loc[25, "x3"] = float(frame.loc[24, "x3"]) + 1.0

    return frame


@pytest.mark.e2e
def test_stratified_run_writes_complete_diagnostics(tmp_path: Path) -> None:
    frame = make_stratified_diagnostic_frame()
    data_path = tmp_path / "data.csv"
    write_dataset(frame, data_path)

    config_payload = {
        "schema_version": "1.1",
        "experiment": {
            "name": "e2e_stratified",
            "primary_metric": "log_loss",
        },
        "data": {
            "path": str(data_path),
            "target": "target",
            "row_id": "row_id",
            "group": None,
            "time": None,
        },
        "features": {
            "x1": {"role": "smooth"},
            "x2": {"role": "smooth"},
            "x3": {"role": "smooth"},
            "category": {"role": "categorical", "categories": ["low", "high"]},
            "row_id": {"role": "exclude"},
        },
        "models": [
            {
                "id": "gam_main",
                "interactions": "none",
            }
        ],
        "validation": {
            "strategy": "stratified",
            "outer_splits": 3,
            "outer_repeats": 2,
            "inner_splits": 2,
            "random_state": 42,
            "gap": 0,
            "test_size": None,
            "duplicate_group_policy": "report",
        },
        "profiling": {
            "correlation": {
                "enabled": True,
                "review_threshold": 0.75,
                "warning_threshold": 0.90,
            },
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

    # 1. Assert run completion and file existence
    assert_run_completed(run_directory)

    for relative_path in (
        "run.json",
        "config.yaml",
        "data_manifest.json",
        "environment.json",
        "split_manifest.csv",
        "status.json",
        "events.jsonl",
        "diagnostics/diagnostics_manifest.json",
        "reports/report.html",
        "results/gam_main/fold_metrics.csv",
        "models/gam_main/model.joblib",
    ):
        assert (run_directory / relative_path).is_file(), (
            f"Missing file: {relative_path}"
        )

    # 2. Verify split manifest structure
    split_manifest = pd.read_csv(run_directory / "split_manifest.csv")

    assert set(split_manifest["repeat"]) == {1, 2}
    assert set(split_manifest["fold"]) == {1, 2, 3}
    assert set(split_manifest["partition"]) == {"train", "test"}

    test_rows = split_manifest.loc[split_manifest["partition"] == "test"]
    test_counts = test_rows.groupby(["repeat", "row_id"]).size()
    assert (test_counts == 1).all()

    for _, fold_frame in split_manifest.groupby(["repeat", "fold"]):
        train_ids = set(fold_frame.loc[fold_frame["partition"] == "train", "row_id"])
        test_ids = set(fold_frame.loc[fold_frame["partition"] == "test", "row_id"])
        assert train_ids.isdisjoint(test_ids)

    # 3. Verify ordinary stratification
    row_targets = frame.set_index("row_id")["target"]
    for _, fold_frame in test_rows.groupby(["repeat", "fold"]):
        fold_targets = row_targets.loc[fold_frame["row_id"]]
        assert set(fold_targets) == {"A", "B", "C"}

    # 4. Confirm report policy metadata
    manifest = load_json_object(run_directory / "diagnostics/diagnostics_manifest.json")
    dup_policy_info = manifest["analyses"]["duplicate_groups"]
    assert dup_policy_info["status"] == "completed"

    data_manifest = load_json_object(run_directory / "data_manifest.json")
    assert data_manifest["duplicate_group_policy"] == "report"
    assert data_manifest["duplicate_group_enforcement_applied"] is False

    # 5. Verify diagnostic findings
    dup_results = dup_policy_info["results"]
    assert dup_results["exact_group_count"] >= 1
    assert dup_results["proper_near_group_count"] >= 1
    assert dup_results["conflicting_target_group_count"] >= 1

    high_pairs = pd.read_csv(run_directory / "diagnostics/high_correlation_pairs.csv")
    assert (
        ((high_pairs["left"] == "x1") & (high_pairs["right"] == "x2"))
        | ((high_pairs["left"] == "x2") & (high_pairs["right"] == "x1"))
    ).any()

    # 6. Verify HTML report
    report_text = (run_directory / "reports/report.html").read_text(encoding="utf-8")
    assert "Validation design" in report_text
    assert "Predictor diagnostics" in report_text
    assert "stratified" in report_text
    assert "Conflicting duplicate targets detected." in report_text
    assert "High predictor correlations" in report_text
    assert "Predictor dictionary" in report_text
