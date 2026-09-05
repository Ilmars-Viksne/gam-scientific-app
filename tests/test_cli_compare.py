from __future__ import annotations

import json
from pathlib import Path

from test_run_comparability import create_mock_run

from gam_app.cli import build_parser


def test_compare_cli_success(tmp_path: Path, capsys) -> None:
    run1 = create_mock_run(tmp_path, "run-1")
    run2 = create_mock_run(tmp_path, "run-2")
    out_dir = tmp_path / "comparison_out"

    parser = build_parser()
    args = parser.parse_args(
        [
            "compare",
            "--left",
            str(run1),
            "--left-model",
            "gam_main",
            "--right",
            str(run2),
            "--right-model",
            "gam_pairwise",
            "--output",
            str(out_dir),
        ]
    )
    args.func(args)

    assert (out_dir / "comparison.csv").is_file()
    assert (out_dir / "summary.csv").is_file()
    assert (out_dir / "comparison_manifest.json").is_file()

    manifest = json.loads(
        (out_dir / "comparison_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_name"] == "gam_comparison_manifest"
    assert manifest["comparability"]["comparable"] is True


def test_compare_cli_check_only(tmp_path: Path) -> None:
    run1 = create_mock_run(tmp_path, "run-1")
    run2 = create_mock_run(tmp_path, "run-2")
    out_dir = tmp_path / "comparison_out"

    parser = build_parser()
    args = parser.parse_args(
        [
            "compare",
            "--left",
            str(run1),
            "--left-model",
            "gam_main",
            "--right",
            str(run2),
            "--right-model",
            "gam_pairwise",
            "--check-only",
        ]
    )
    try:
        args.func(args)
    except SystemExit as e:
        assert e.code == 0

    assert not out_dir.exists()


def test_compare_cli_json_mode(tmp_path: Path, capsys) -> None:
    run1 = create_mock_run(tmp_path, "run-1")
    run2 = create_mock_run(tmp_path, "run-2")
    out_dir = tmp_path / "comparison_out"

    parser = build_parser()
    args = parser.parse_args(
        [
            "compare",
            "--left",
            str(run1),
            "--left-model",
            "gam_main",
            "--right",
            str(run2),
            "--right-model",
            "gam_pairwise",
            "--output",
            str(out_dir),
            "--json",
        ]
    )
    args.func(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["schema_name"] == "gam_comparison"
    assert payload["comparable"] is True


def test_compare_cli_failure_non_comparable(tmp_path: Path, capsys) -> None:
    run1 = create_mock_run(tmp_path, "run-1", target="target_A")
    run2 = create_mock_run(tmp_path, "run-2", target="target_B")
    out_dir = tmp_path / "comparison_out"

    parser = build_parser()
    args = parser.parse_args(
        [
            "compare",
            "--left",
            str(run1),
            "--left-model",
            "gam_main",
            "--right",
            str(run2),
            "--right-model",
            "gam_main",
            "--output",
            str(out_dir),
        ]
    )

    try:
        args.func(args)
    except SystemExit as e:
        assert e.code == 2

    assert not out_dir.exists()
