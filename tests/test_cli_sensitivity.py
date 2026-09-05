from __future__ import annotations

import json
from pathlib import Path

from test_run_comparability import create_mock_run

from gam_app.cli import build_parser


def test_cli_create_and_show_sensitivity(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "workspace"
    ref_run = create_mock_run(workspace / "runs", "run-a", search_grid={"C": [0.1]})
    var_run = create_mock_run(workspace / "runs", "run-b", search_grid={"C": [1.0]})

    parser = build_parser()

    # create-sensitivity
    args_create = parser.parse_args(
        [
            "create-sensitivity",
            "--workspace",
            str(workspace),
            "--id",
            "sens-1",
            "--name",
            "Test sensitivity",
            "--reference-run",
            str(ref_run),
            "--variant-run",
            str(var_run),
            "--vary",
            "search.C",
            "--invariant",
            "dataset",
            "--json",
        ]
    )
    args_create.func(args_create)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["schema_name"] == "gam_sensitivity_manifest"
    assert payload["sensitivity_id"] == "sens-1"

    manifest_file = workspace / "sensitivity" / "sens-1" / "sensitivity_manifest.json"
    assert manifest_file.is_file()

    # show-sensitivity
    args_show = parser.parse_args(
        [
            "show-sensitivity",
            "--manifest",
            str(manifest_file),
            "--json",
        ]
    )
    args_show.func(args_show)

    captured_show = capsys.readouterr()
    payload_show = json.loads(captured_show.out)
    assert payload_show["sensitivity_id"] == "sens-1"
