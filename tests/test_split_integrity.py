from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gam_app.config import (
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    ValidationConfig,
)
from gam_app.exceptions import DataValidationError
from gam_app.splitting import (
    SplitContext,
    evaluate_split_integrity,
    split_integrity_frame,
    validate_forward_time_order,
    validate_no_group_leakage,
    validate_one_test_assignment_per_repeat,
)
from gam_app.workflow import create_run, execute_run


def make_test_config(data_path: Path, strategy: str = "stratified") -> ExperimentConfig:
    return ExperimentConfig(
        name="test_experiment",
        data_path=data_path,
        target="target",
        row_id="row_id",
        group_column="group_col" if strategy == "stratified_group" else None,
        time_column="time_col" if strategy == "time" else None,
        features={
            "x": FeatureConfig(role="smooth"),
        },
        validation=ValidationConfig(
            strategy=strategy,
            outer_splits=2,
            outer_repeats=1 if strategy == "time" else 1,
            inner_splits=2,
            random_state=42,
            test_size=1 if strategy == "time" else None,
        ),
        models=(
            ModelConfig(
                id="m1",
            ),
        ),
    )


def test_validate_no_group_leakage_raises_on_leak() -> None:
    manifest = pd.DataFrame(
        {
            "repeat": [1, 1, 1, 1],
            "fold": [1, 1, 1, 1],
            "partition": ["train", "test", "train", "test"],
            "group_id": ["g1", "g1", "g2", "g3"],
        }
    )
    with pytest.raises(ValueError, match="Group leakage detected"):
        validate_no_group_leakage(manifest)


def test_validate_forward_time_order_raises_on_leak() -> None:
    manifest = pd.DataFrame(
        {
            "repeat": [1, 1],
            "fold": [1, 1],
            "partition": ["train", "test"],
            "time": ["2025-01-02T00:00:00Z", "2025-01-01T00:00:00Z"],
        }
    )
    with pytest.raises(ValueError, match="Temporal leakage"):
        validate_forward_time_order(manifest)


def test_validate_one_test_assignment_per_repeat_raises_on_duplicate() -> None:
    manifest = pd.DataFrame(
        {
            "repeat": [1, 1],
            "row_id": ["r1", "r1"],
            "partition": ["test", "test"],
        }
    )
    with pytest.raises(ValueError, match="Every row must appear exactly once"):
        validate_one_test_assignment_per_repeat(manifest)


def test_valid_stratified_integrity_results_pass(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    config = make_test_config(csv_path, strategy="stratified")

    context = SplitContext(
        X=pd.DataFrame({"x": range(6)}),
        y=pd.Series(["a", "a", "a", "b", "b", "b"]),
        row_ids=pd.Series(["r1", "r2", "r3", "r4", "r5", "r6"]),
        groups=None,
        times=None,
    )

    manifest = pd.DataFrame(
        {
            "repeat": [1] * 12,
            "fold": [1] * 6 + [2] * 6,
            "row_id": [
                "r3",
                "r4",
                "r6",
                "r1",
                "r2",
                "r5",
                "r1",
                "r2",
                "r5",
                "r3",
                "r4",
                "r6",
            ],
            "row_index": [2, 3, 5, 0, 1, 4, 0, 1, 4, 2, 3, 5],
            "partition": [
                "train",
                "train",
                "train",
                "test",
                "test",
                "test",
                "train",
                "train",
                "train",
                "test",
                "test",
                "test",
            ],
            "validation_strategy": ["stratified"] * 12,
            "group_id": [None] * 12,
            "time": [None] * 12,
        }
    )

    results = evaluate_split_integrity(
        config,
        context,
        manifest,
    )

    frame = split_integrity_frame(results)

    assert not frame.empty
    assert frame["passed"].all()
    assert "no_train_test_overlap" in set(frame["check"])
    assert "one_test_assignment_per_repeat" in set(frame["check"])


def test_group_leakage_is_returned_as_failed_result(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    config = make_test_config(csv_path, strategy="stratified_group")

    context = SplitContext(
        X=pd.DataFrame({"x": [1, 2, 3, 4]}),
        y=pd.Series(["a", "a", "b", "b"]),
        row_ids=pd.Series(["r1", "r2", "r3", "r4"]),
        groups=pd.Series(["g1", "g1", "g2", "g3"]),
        times=None,
    )

    manifest = pd.DataFrame(
        {
            "repeat": [1, 1, 1, 1],
            "fold": [1, 1, 1, 1],
            "row_id": ["r1", "r2", "r3", "r4"],
            "row_index": [0, 1, 2, 3],
            "partition": ["train", "test", "train", "test"],
            "validation_strategy": ["stratified_group"] * 4,
            "group_id": ["g1", "g1", "g2", "g3"],
            "time": [None] * 4,
        }
    )

    results = evaluate_split_integrity(
        config,
        context,
        manifest,
    )

    leakage = [result for result in results if result.check == "no_group_leakage"]

    assert len(leakage) == 1
    assert leakage[0].passed is False
    assert leakage[0].repeat == 1
    assert leakage[0].fold == 1
    assert leakage[0].observed == "1"


def test_temporal_leakage_records_boundary_values(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    config = make_test_config(csv_path, strategy="time")

    context = SplitContext(
        X=pd.DataFrame({"x": [1, 2]}),
        y=pd.Series(["a", "b"]),
        row_ids=pd.Series(["r1", "r2"]),
        groups=None,
        times=pd.to_datetime(
            [
                "2025-01-02T00:00:00Z",
                "2025-01-01T00:00:00Z",
            ],
            utc=True,
        ).to_series(index=[0, 1]),
    )

    manifest = pd.DataFrame(
        {
            "repeat": [1, 1],
            "fold": [1, 1],
            "row_id": ["r1", "r2"],
            "row_index": [0, 1],
            "partition": ["train", "test"],
            "validation_strategy": ["time", "time"],
            "group_id": [None, None],
            "time": [
                "2025-01-02T00:00:00Z",
                "2025-01-01T00:00:00Z",
            ],
        }
    )

    results = evaluate_split_integrity(
        config,
        context,
        manifest,
    )

    temporal = [result for result in results if result.check == "strict_temporal_order"]

    assert len(temporal) == 1
    assert temporal[0].passed is False
    assert "maximum_train_time=" in temporal[0].observed
    assert "minimum_test_time=" in temporal[0].observed


def setup_sample_run(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "row_id": [f"r{i}" for i in range(1, 11)],
            "x": list(range(10)),
            "target": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    config_path = tmp_path / "config.yaml"
    config = make_test_config(csv_path, strategy="stratified")

    from gam_app.config import dump_config_dict
    from gam_app.io_utils import write_yaml_atomic

    write_yaml_atomic(config_path, dump_config_dict(config))

    workspace = tmp_path / "workspace"
    run_dir = create_run(config_path, workspace)
    return run_dir


def test_run_writes_split_integrity_csv_and_manifest(tmp_path: Path) -> None:
    run_directory = setup_sample_run(tmp_path)
    execute_run(run_directory)

    integrity_path = run_directory / "diagnostics" / "split_integrity.csv"
    assert integrity_path.exists()

    frame = pd.read_csv(integrity_path)
    assert list(frame.columns) == [
        "strategy",
        "scope",
        "repeat",
        "fold",
        "check",
        "passed",
        "observed",
        "expected",
        "details",
    ]
    assert not frame.empty
    assert frame["passed"].all()

    text = integrity_path.read_text(encoding="utf-8")
    assert ",True," in text
    assert ",False," not in text

    manifest_path = run_directory / "diagnostics" / "diagnostics_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "split_integrity" in payload
    info = payload["split_integrity"]
    assert info["artifact"] == "diagnostics/split_integrity.csv"
    assert info["passed"] is True
    assert info["result_count"] == len(frame)
    assert info["distinct_check_count"] == len(set(frame["check"]))
    assert info["failed_result_count"] == 0
    assert "evaluated_checks" in info
    assert "not_applicable_checks" in info


def test_existing_manifest_is_revalidated_on_resume(tmp_path: Path) -> None:
    run_directory = setup_sample_run(tmp_path)
    execute_run(run_directory)

    split_path = run_directory / "split_manifest.csv"
    assert split_path.exists()

    manifest = pd.read_csv(split_path)

    test_row = manifest.index[manifest["partition"] == "test"][0]
    manifest.loc[test_row, "partition"] = "train"
    manifest.to_csv(split_path, index=False)

    with pytest.raises(
        DataValidationError,
        match="Split-integrity validation failed",
    ):
        execute_run(run_directory)

    integrity_path = run_directory / "diagnostics" / "split_integrity.csv"
    assert integrity_path.exists()

    integrity = pd.read_csv(integrity_path)
    assert not integrity["passed"].all()
