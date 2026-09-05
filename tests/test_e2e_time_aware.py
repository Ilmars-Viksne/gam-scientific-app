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


def make_time_frame() -> pd.DataFrame:
    row_count = 48
    timestamps = pd.date_range(
        "2025-01-01",
        periods=row_count,
        freq="D",
        tz="UTC",
    )
    targets = [("A", "B", "C")[i % 3] for i in range(row_count)]

    return pd.DataFrame(
        {
            "row_id": [f"row-{i:03d}" for i in range(row_count)],
            "observed_at": timestamps.astype(str),
            "x1": [float(i) for i in range(row_count)],
            "x2": [float(i % 5) for i in range(row_count)],
            "target": targets,
        }
    )


@pytest.mark.e2e
def test_time_aware_run_satisfies_chronology_and_gap(tmp_path: Path) -> None:
    frame = make_time_frame()
    data_path = tmp_path / "data.csv"
    write_dataset(frame, data_path)

    config_payload = {
        "schema_version": "1.1",
        "experiment": {
            "name": "e2e_time_aware",
            "primary_metric": "log_loss",
        },
        "data": {
            "path": str(data_path),
            "target": "target",
            "row_id": "row_id",
            "group": None,
            "time": "observed_at",
        },
        "features": {
            "x1": {"role": "smooth"},
            "x2": {"role": "smooth"},
            "observed_at": {"role": "exclude"},
            "row_id": {"role": "exclude"},
        },
        "models": [
            {
                "id": "gam_main",
                "interactions": "none",
            }
        ],
        "validation": {
            "strategy": "time",
            "outer_splits": 3,
            "outer_repeats": 1,
            "inner_splits": 2,
            "random_state": 42,
            "gap": 2,
            "test_size": 6,
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
                "near_duplicate_threshold": 0.98,
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

    # 2. Read split manifest & assert single repeat
    split_manifest = pd.read_csv(run_directory / "split_manifest.csv")
    assert set(split_manifest["repeat"]) == {1}

    # 3. Assert test sizes
    test_sizes = (
        split_manifest.loc[split_manifest["partition"] == "test"]
        .groupby(["repeat", "fold"])
        .size()
    )
    assert (test_sizes == 6).all()

    # 4. Assert chronology and gap
    time_by_row = frame.assign(
        parsed_time=pd.to_datetime(frame["observed_at"], utc=True)
    ).set_index("row_id")["parsed_time"]

    row_position = {row_id: index for index, row_id in enumerate(frame["row_id"])}

    test_window_starts = []

    for _, fold_frame in split_manifest.groupby(["repeat", "fold"], sort=True):
        train_ids = fold_frame.loc[fold_frame["partition"] == "train", "row_id"]
        test_ids = fold_frame.loc[fold_frame["partition"] == "test", "row_id"]

        # Strict chronological ordering: train max < test min
        assert time_by_row.loc[train_ids].max() < time_by_row.loc[test_ids].min()

        # Gap assertion: row index difference >= gap (2) + 1
        last_train_pos = max(row_position[rid] for rid in train_ids)
        first_test_pos = min(row_position[rid] for rid in test_ids)
        assert first_test_pos - last_train_pos - 1 >= 2

        test_window_starts.append(time_by_row.loc[test_ids].min())

    # Forward progression
    assert test_window_starts == sorted(test_window_starts)

    # 5. Verify temporal split-integrity artifacts
    integrity = pd.read_csv(run_directory / "diagnostics/split_integrity.csv")
    assert integrity["passed"].all()

    check_names = set(integrity["check"])
    assert "strict_temporal_order" in check_names
    assert "non_overlapping_temporal_tests" in check_names
    assert "outer_repeats_equal_one" in check_names

    diag_manifest = load_json_object(
        run_directory / "diagnostics/diagnostics_manifest.json"
    )
    assert diag_manifest["split_integrity"]["passed"] is True

    # 6. Report assertions
    report_text = (run_directory / "reports/report.html").read_text(encoding="utf-8")
    assert "Validation design" in report_text
    assert "time" in report_text
    assert "observed_at" in report_text
    assert "PASS" in report_text
