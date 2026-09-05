import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gam_app.workflow import CreatedRun, create_run, create_run_result


def test_create_run_returns_created_directory(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    data_file = tmp_path / "data.csv"
    data_file.write_text("x1,target\n1,A\n2,B\n", encoding="utf-8")

    from gam_app.config import dump_config_dict, parse_config_payload
    from gam_app.io_utils import write_yaml_atomic

    payload = {
        "schema_version": "1.1",
        "experiment": {"name": "test-exp", "primary_metric": "log_loss"},
        "data": {"path": str(data_file), "target": "target"},
        "features": {"x1": {"role": "smooth"}},
        "models": [{"id": "gam_main", "interactions": "none"}],
    }
    config = parse_config_payload(payload, base_directory=tmp_path)
    write_yaml_atomic(config_file, dump_config_dict(config))

    workspace = tmp_path / "workspace"
    run_dir = create_run(config_file, workspace)

    assert isinstance(run_dir, Path)
    assert run_dir.exists()
    assert (run_dir / "run.json").exists()


def test_create_run_result_returns_rich_dataclass(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    data_file = tmp_path / "data.csv"
    data_file.write_text("x1,target\n1,A\n2,B\n", encoding="utf-8")

    from gam_app.config import dump_config_dict, parse_config_payload
    from gam_app.io_utils import write_yaml_atomic

    payload = {
        "schema_version": "1.1",
        "experiment": {
            "name": "test-exp",
            "primary_metric": "log_loss",
            "tags": ["tag1"],
            "metadata": {"meta1": "val1"},
        },
        "data": {"path": str(data_file), "target": "target"},
        "features": {"x1": {"role": "smooth"}},
        "models": [{"id": "gam_main", "interactions": "none"}],
    }
    config = parse_config_payload(payload, base_directory=tmp_path)
    write_yaml_atomic(config_file, dump_config_dict(config))

    workspace = tmp_path / "workspace"
    created = create_run_result(config_file, workspace)

    assert isinstance(created, CreatedRun)
    assert created.run_directory.exists()
    assert created.metadata_path == created.run_directory / "run.json"
    assert created.status_path == created.run_directory / "status.json"

    metadata = json.loads(created.metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_name"] == "gam_run_metadata"
    assert metadata["schema_version"] == "1.0"
    assert metadata["created_run_path"] == created.run_directory.resolve().as_posix()
    assert metadata["created_workspace_path"] == workspace.resolve().as_posix()
    assert metadata["tags"] == ["tag1"]
    assert metadata["metadata"] == {"meta1": "val1"}


def test_command_run_prints_path_before_execute_run(
    capsys: pytest.CaptureFixture, tmp_path: Path
) -> None:
    config_file = tmp_path / "config.yaml"
    data_file = tmp_path / "data.csv"
    data_file.write_text("x1,target\n1,A\n2,B\n", encoding="utf-8")

    from gam_app.cli import build_parser
    from gam_app.config import dump_config_dict, parse_config_payload
    from gam_app.io_utils import write_yaml_atomic

    payload = {
        "schema_version": "1.1",
        "experiment": {"name": "test-exp", "primary_metric": "log_loss"},
        "data": {"path": str(data_file), "target": "target"},
        "features": {"x1": {"role": "smooth"}},
        "models": [{"id": "gam_main", "interactions": "none"}],
    }
    config = parse_config_payload(payload, base_directory=tmp_path)
    write_yaml_atomic(config_file, dump_config_dict(config))

    workspace = tmp_path / "workspace"
    run_path_file = tmp_path / "latest_run.txt"

    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--config",
            str(config_file),
            "--workspace",
            str(workspace),
            "--run-path-file",
            str(run_path_file),
            "--json",
        ]
    )

    with patch("gam_app.cli.execute_run") as mock_execute:
        mock_execute.side_effect = RuntimeError("Simulated execution failure")
        with pytest.raises(RuntimeError, match="Simulated execution failure"):
            args.func(args)

    captured = capsys.readouterr()
    assert "gam_run_creation" in captured.out
    assert run_path_file.exists()
    file_content = run_path_file.read_text(encoding="utf-8")
    assert file_content.endswith("\n")
    assert "workspace/runs/run-" in file_content


def test_create_only_does_not_call_execute_run(
    capsys: pytest.CaptureFixture, tmp_path: Path
) -> None:
    config_file = tmp_path / "config.yaml"
    data_file = tmp_path / "data.csv"
    data_file.write_text("x1,target\n1,A\n2,B\n", encoding="utf-8")

    from gam_app.cli import build_parser
    from gam_app.config import dump_config_dict, parse_config_payload
    from gam_app.io_utils import write_yaml_atomic

    payload = {
        "schema_version": "1.1",
        "experiment": {"name": "test-exp", "primary_metric": "log_loss"},
        "data": {"path": str(data_file), "target": "target"},
        "features": {"x1": {"role": "smooth"}},
        "models": [{"id": "gam_main", "interactions": "none"}],
    }
    config = parse_config_payload(payload, base_directory=tmp_path)
    write_yaml_atomic(config_file, dump_config_dict(config))

    workspace = tmp_path / "workspace"

    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--config",
            str(config_file),
            "--workspace",
            str(workspace),
            "--create-only",
            "--json",
        ]
    )

    with patch("gam_app.cli.execute_run") as mock_execute:
        args.func(args)
        mock_execute.assert_not_called()

    captured = capsys.readouterr()
    payload_out = json.loads(captured.out)
    assert payload_out["state"] == "created"
    assert payload_out["execution_started"] is False


def test_failed_initialization_cleans_up_directory(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    data_file = tmp_path / "data.csv"
    data_file.write_text("x1,target\n1,A\n2,B\n", encoding="utf-8")

    from gam_app.config import dump_config_dict, parse_config_payload
    from gam_app.io_utils import write_yaml_atomic

    payload = {
        "schema_version": "1.1",
        "experiment": {"name": "test-exp", "primary_metric": "log_loss"},
        "data": {"path": str(data_file), "target": "target"},
        "features": {"x1": {"role": "smooth"}},
        "models": [{"id": "gam_main", "interactions": "none"}],
    }
    config = parse_config_payload(payload, base_directory=tmp_path)
    write_yaml_atomic(config_file, dump_config_dict(config))

    workspace = tmp_path / "workspace"

    with patch(
        "gam_app.workflow.write_json_atomic", side_effect=RuntimeError("IO Error")
    ):
        with pytest.raises(RuntimeError, match="IO Error"):
            create_run(config_file, workspace)

    runs_dir = workspace / "runs"
    if runs_dir.exists():
        assert len(list(runs_dir.iterdir())) == 0
