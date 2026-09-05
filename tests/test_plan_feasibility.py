import json
from pathlib import Path

import pandas as pd
import pytest

from gam_app.cli import build_parser


@pytest.fixture
def plan_dataset(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "id": [f"id_{i}" for i in range(20)],
            "target": ["A"] * 10 + ["B"] * 10,
            "group_id": [f"g_{i // 2}" for i in range(20)],
            "time_col": pd.date_range("2020-01-01", periods=20, freq="D").astype(str),
            "x1": [0.1 * i for i in range(20)],
            "x2": [0.5, 1.5] * 10,
        }
    )
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)
    return data_path


@pytest.fixture
def plan_config_path(plan_dataset: Path, tmp_path: Path) -> Path:
    cfg_path = tmp_path / "config.yaml"
    parser = build_parser()
    args = parser.parse_args(
        [
            "configure",
            "--data",
            str(plan_dataset),
            "--target",
            "target",
            "--output",
            str(cfg_path),
            "--row-id",
            "id",
            "--group",
            "group_id",
            "--time",
            "time_col",
            "--outer-splits",
            "3",
            "--outer-repeats",
            "1",
            "--inner-splits",
            "2",
            "--non-interactive",
        ]
    )
    args.func(args)
    return cfg_path


def test_plan_feasible_stratified(plan_config_path: Path, capsys) -> None:
    capsys.readouterr()  # Clear configure output
    parser = build_parser()
    args = parser.parse_args(["plan", "--config", str(plan_config_path)])
    args.func(args)

    captured = capsys.readouterr().out
    assert "Validation design" in captured
    assert "Validation feasibility" in captured
    assert "Candidate fit estimation" in captured


def test_plan_feasible_json(plan_config_path: Path, capsys) -> None:
    capsys.readouterr()  # Clear configure output
    parser = build_parser()
    args = parser.parse_args(["plan", "--config", str(plan_config_path), "--json"])
    args.func(args)

    captured = capsys.readouterr().out
    data = json.loads(captured)
    assert data["feasible"] is True
    assert data["schema_name"] == "gam_plan"
    assert len(data["models"]) > 0


def test_plan_infeasible_too_few_class_observations(tmp_path: Path, capsys) -> None:
    df = pd.DataFrame(
        {
            "id": range(10),
            "target": ["A"] * 9 + ["B"] * 1,  # B has 1 obs, but outer_splits=3
            "x1": [0.1 * i for i in range(10)],
            "x2": [1, 2] * 5,
        }
    )
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    cfg_path = tmp_path / "config.yaml"
    parser = build_parser()
    args_cfg = parser.parse_args(
        [
            "configure",
            "--data",
            str(data_path),
            "--target",
            "target",
            "--output",
            str(cfg_path),
            "--outer-splits",
            "3",
            "--non-interactive",
        ]
    )
    args_cfg.func(args_cfg)
    capsys.readouterr()  # Clear configure output

    args_plan = parser.parse_args(["plan", "--config", str(cfg_path)])
    with pytest.raises(SystemExit) as exc_info:
        args_plan.func(args_plan)

    assert exc_info.value.code == 2
    captured = capsys.readouterr().out
    assert "The configured validation design is not feasible." in captured
    assert "outer_class_support" in captured


def test_plan_infeasible_json(tmp_path: Path, capsys) -> None:
    df = pd.DataFrame(
        {
            "id": range(10),
            "target": ["A"] * 9 + ["B"] * 1,
            "x1": [0.1 * i for i in range(10)],
            "x2": [1, 2] * 5,
        }
    )
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    cfg_path = tmp_path / "config.yaml"
    parser = build_parser()
    args_cfg = parser.parse_args(
        [
            "configure",
            "--data",
            str(data_path),
            "--target",
            "target",
            "--output",
            str(cfg_path),
            "--outer-splits",
            "3",
            "--non-interactive",
        ]
    )
    args_cfg.func(args_cfg)
    capsys.readouterr()  # Clear configure output

    args_plan = parser.parse_args(["plan", "--config", str(cfg_path), "--json"])
    with pytest.raises(SystemExit) as exc_info:
        args_plan.func(args_plan)

    assert exc_info.value.code == 2
    captured = capsys.readouterr().out
    data = json.loads(captured)
    assert data["feasible"] is False
    assert data["models"] == []
