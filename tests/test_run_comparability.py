from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from gam_app.comparison import (
    assess_run_comparability,
    compare_paired_run_results,
)
from gam_app.exceptions import RunComparabilityError


def create_mock_run(
    tmp_path: Path,
    run_name: str,
    *,
    data_hash: str = "abc123sha256",
    target: str = "target",
    strategy: str = "stratified",
    outer_splits: int = 3,
    outer_repeats: int = 1,
    inner_splits: int = 3,
    model_ids: list[str] | None = None,
    state: str = "completed",
    split_manifest_df: pd.DataFrame | None = None,
    fold_metrics_df: pd.DataFrame | None = None,
    search_grid: dict[str, Any] | None = None,
) -> Path:
    run_dir = tmp_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if model_ids is None:
        model_ids = ["gam_main", "gam_pairwise"]

    # 1. run.json
    run_json = {
        "schema_name": "gam_run_metadata",
        "schema_version": "1.0",
        "run_id": run_name,
        "created_at_utc": "2026-01-01T00:00:00Z",
        "dataset_hash": data_hash,
        "config_hash": "cfg123",
        "data": {"sha256": data_hash, "target": target},
    }
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")

    # 2. status.json
    status_json = {"state": state, "run_id": run_name}
    (run_dir / "status.json").write_text(json.dumps(status_json), encoding="utf-8")

    # 3. config.yaml
    cfg = {
        "schema_version": "1.1",
        "data": {"target": target, "path": "data.csv"},
        "validation": {
            "strategy": strategy,
            "outer_splits": outer_splits,
            "outer_repeats": outer_repeats,
            "inner_splits": inner_splits,
        },
        "search": search_grid or {"C": [0.1, 1.0], "n_knots": [3], "degree": [2]},
        "models": [{"id": m} for m in model_ids],
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # 4. split_manifest.csv
    if split_manifest_df is None:
        rows = []
        row_idx = 0
        for r in range(1, outer_repeats + 1):
            for f in range(1, outer_splits + 1):
                for partition in ["train", "test"]:
                    for _ in range(5):
                        rows.append(
                            {
                                "repeat": r,
                                "fold": f,
                                "row_id": row_idx,
                                "partition": partition,
                            }
                        )
                        row_idx += 1
        split_manifest_df = pd.DataFrame(rows)

    split_manifest_df.to_csv(run_dir / "split_manifest.csv", index=False)

    # 5. results/<model>/fold_metrics.csv
    if fold_metrics_df is None:
        fm_rows = []
        for r in range(1, outer_repeats + 1):
            for f in range(1, outer_splits + 1):
                fm_rows.append(
                    {
                        "repeat": r,
                        "fold": f,
                        "log_loss": 0.45,
                        "accuracy": 0.85,
                        "balanced_accuracy": 0.82,
                        "macro_f1": 0.83,
                    }
                )
        fold_metrics_df = pd.DataFrame(fm_rows)

    for m in model_ids:
        res_dir = run_dir / "results" / m
        res_dir.mkdir(parents=True, exist_ok=True)
        fold_metrics_df.to_csv(res_dir / "fold_metrics.csv", index=False)

    return run_dir


def test_same_run_two_models_comparable(tmp_path: Path) -> None:
    run_dir = create_mock_run(tmp_path, "run-1")
    assessment = assess_run_comparability(
        left_run=run_dir,
        left_model="gam_main",
        right_run=run_dir,
        right_model="gam_pairwise",
    )
    assert assessment.comparable
    res = compare_paired_run_results(
        left_run=run_dir,
        left_model="gam_main",
        right_run=run_dir,
        right_model="gam_pairwise",
    )
    assert res.summary is not None
    assert len(res.summary) == 4


def test_two_runs_identical_splits_comparable(tmp_path: Path) -> None:
    run1 = create_mock_run(tmp_path, "run-1")
    run2 = create_mock_run(tmp_path, "run-2")
    assessment = assess_run_comparability(
        left_run=run1,
        left_model="gam_main",
        right_run=run2,
        right_model="gam_main",
    )
    assert assessment.comparable


def test_different_dataset_hashes_fail(tmp_path: Path) -> None:
    run1 = create_mock_run(tmp_path, "run-1", data_hash="hash-A")
    run2 = create_mock_run(tmp_path, "run-2", data_hash="hash-B")
    assessment = assess_run_comparability(
        left_run=run1,
        left_model="gam_main",
        right_run=run2,
        right_model="gam_main",
    )
    assert not assessment.comparable
    with pytest.raises(RunComparabilityError):
        compare_paired_run_results(
            left_run=run1,
            left_model="gam_main",
            right_run=run2,
            right_model="gam_main",
        )


def test_different_targets_fail(tmp_path: Path) -> None:
    run1 = create_mock_run(tmp_path, "run-1", target="target_A")
    run2 = create_mock_run(tmp_path, "run-2", target="target_B")
    assessment = assess_run_comparability(
        left_run=run1,
        left_model="gam_main",
        right_run=run2,
        right_model="gam_main",
    )
    assert not assessment.comparable


def test_different_validation_strategies_fail(tmp_path: Path) -> None:
    run1 = create_mock_run(tmp_path, "run-1", strategy="stratified")
    run2 = create_mock_run(tmp_path, "run-2", strategy="time")
    assessment = assess_run_comparability(
        left_run=run1,
        left_model="gam_main",
        right_run=run2,
        right_model="gam_main",
    )
    assert not assessment.comparable


def test_different_outer_splits_fail(tmp_path: Path) -> None:
    run1 = create_mock_run(tmp_path, "run-1", outer_splits=3)
    run2 = create_mock_run(tmp_path, "run-2", outer_splits=5)
    assessment = assess_run_comparability(
        left_run=run1,
        left_model="gam_main",
        right_run=run2,
        right_model="gam_main",
    )
    assert not assessment.comparable


def test_reordered_split_manifest_rows_pass(tmp_path: Path) -> None:
    run1 = create_mock_run(tmp_path, "run-1")

    sm1 = pd.read_csv(run1 / "split_manifest.csv")
    sm2 = sm1.sample(frac=1.0, random_state=42).reset_index(drop=True)

    run2 = create_mock_run(tmp_path, "run-2", split_manifest_df=sm2)

    assessment = assess_run_comparability(
        left_run=run1,
        left_model="gam_main",
        right_run=run2,
        right_model="gam_main",
    )
    assert assessment.comparable


def test_different_test_rows_fail(tmp_path: Path) -> None:
    run1 = create_mock_run(tmp_path, "run-1")

    sm1 = pd.read_csv(run1 / "split_manifest.csv")
    sm2 = sm1.copy()
    # Modify a row_id in sm2 for test partition
    mask = sm2["partition"] == "test"
    sm2.loc[mask, "row_id"] = sm2.loc[mask, "row_id"] + 1000

    run2 = create_mock_run(tmp_path, "run-2", split_manifest_df=sm2)

    assessment = assess_run_comparability(
        left_run=run1,
        left_model="gam_main",
        right_run=run2,
        right_model="gam_main",
    )
    assert not assessment.comparable


def test_missing_split_manifest_fails(tmp_path: Path) -> None:
    run1 = create_mock_run(tmp_path, "run-1")
    run2 = create_mock_run(tmp_path, "run-2")
    (run2 / "split_manifest.csv").unlink()

    assessment = assess_run_comparability(
        left_run=run1,
        left_model="gam_main",
        right_run=run2,
        right_model="gam_main",
    )
    assert not assessment.comparable


def test_selected_model_absent_fails(tmp_path: Path) -> None:
    run1 = create_mock_run(tmp_path, "run-1")
    run2 = create_mock_run(tmp_path, "run-2")

    assessment = assess_run_comparability(
        left_run=run1,
        left_model="gam_main",
        right_run=run2,
        right_model="nonexistent_model",
    )
    assert not assessment.comparable


def test_nonfinite_metric_value_fails(tmp_path: Path) -> None:
    run1 = create_mock_run(tmp_path, "run-1")

    fm = pd.read_csv(run1 / "results" / "gam_main" / "fold_metrics.csv")
    fm.loc[0, "log_loss"] = np.nan

    run2 = create_mock_run(tmp_path, "run-2", fold_metrics_df=fm)

    assessment = assess_run_comparability(
        left_run=run1,
        left_model="gam_main",
        right_run=run2,
        right_model="gam_main",
    )
    assert not assessment.comparable


def test_different_search_grid_warns(tmp_path: Path) -> None:
    run1 = create_mock_run(tmp_path, "run-1", search_grid={"C": [0.1]})
    run2 = create_mock_run(tmp_path, "run-2", search_grid={"C": [1.0]})

    assessment = assess_run_comparability(
        left_run=run1,
        left_model="gam_main",
        right_run=run2,
        right_model="gam_main",
    )
    assert assessment.comparable
    assert any(
        c.check == "search_grid_equal" and c.level == "warning"
        for c in assessment.checks
    )
