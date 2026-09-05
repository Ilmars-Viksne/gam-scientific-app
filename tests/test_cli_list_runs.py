import json
from pathlib import Path

import pytest

from gam_app.cli import build_parser
from gam_app.workflow import create_run_result


def test_cli_list_runs_text_and_json_output(
    capsys: pytest.CaptureFixture, tmp_path: Path
) -> None:
    data_file = tmp_path / "data.csv"
    data_file.write_text("x1,target\n1,A\n2,B\n", encoding="utf-8")

    config_file = tmp_path / "config.yaml"
    from gam_app.config import dump_config_dict, parse_config_payload
    from gam_app.io_utils import write_yaml_atomic

    payload = {
        "schema_version": "1.1",
        "experiment": {
            "name": "exp-test",
            "primary_metric": "log_loss",
            "tags": ["candidate"],
            "metadata": {"project": "study-a"},
        },
        "data": {"path": str(data_file), "target": "target"},
        "features": {"x1": {"role": "smooth"}},
        "models": [{"id": "gam_main", "interactions": "none"}],
    }
    config = parse_config_payload(payload, base_directory=tmp_path)
    write_yaml_atomic(config_file, dump_config_dict(config))

    workspace = tmp_path / "workspace"
    created = create_run_result(config_file, workspace)

    parser = build_parser()

    # Test text output mode
    args_text = parser.parse_args(
        [
            "list-runs",
            "--workspace",
            str(workspace),
            "--tag",
            "candidate",
        ]
    )
    args_text.func(args_text)

    captured = capsys.readouterr()
    assert created.run_id in captured.out
    assert "exp-test" in captured.out
    assert "candidate" in captured.out

    # Test JSON output mode
    args_json = parser.parse_args(
        [
            "list-runs",
            "--workspace",
            str(workspace),
            "--metadata",
            "project=study-a",
            "--json",
        ]
    )
    args_json.func(args_json)

    captured_json = capsys.readouterr()
    catalog_data = json.loads(captured_json.out)
    assert catalog_data["schema_name"] == "gam_run_catalog"
    assert catalog_data["matched_count"] == 1
    assert catalog_data["runs"][0]["run_id"] == created.run_id
    assert catalog_data["runs"][0]["tags"] == ["candidate"]
    assert catalog_data["runs"][0]["metadata"] == {"project": "study-a"}


def test_cli_list_runs_empty_result(
    capsys: pytest.CaptureFixture, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"

    parser = build_parser()

    # Empty text
    args_text = parser.parse_args(["list-runs", "--workspace", str(workspace)])
    args_text.func(args_text)
    captured = capsys.readouterr()
    assert "No runs matched the supplied filters." in captured.out

    # Empty JSON
    args_json = parser.parse_args(
        ["list-runs", "--workspace", str(workspace), "--json"]
    )
    args_json.func(args_json)
    captured_json = capsys.readouterr()
    payload = json.loads(captured_json.out)
    assert payload["matched_count"] == 0
    assert payload["runs"] == []
