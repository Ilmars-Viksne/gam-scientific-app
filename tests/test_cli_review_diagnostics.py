from __future__ import annotations

import json
from pathlib import Path

from test_diagnostic_review import create_mock_diagnostics
from test_run_comparability import create_mock_run

from gam_app.cli import build_parser


def test_cli_review_diagnostics_success(tmp_path: Path, capsys) -> None:
    run_dir = create_mock_run(tmp_path, "run-1")
    create_mock_diagnostics(run_dir)

    out_json = run_dir / "reviews" / "diagnostic_review.json"

    parser = build_parser()
    args = parser.parse_args(
        [
            "review-diagnostics",
            "--run",
            str(run_dir),
            "--output",
            str(out_json),
            "--json",
        ]
    )
    args.func(args)

    assert out_json.is_file()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["package_status"] == "valid"


def test_cli_review_diagnostics_strict_mode(tmp_path: Path) -> None:
    run_dir = create_mock_run(tmp_path, "run-1")
    create_mock_diagnostics(run_dir, warning_pair_count=1)

    parser = build_parser()
    args = parser.parse_args(
        [
            "review-diagnostics",
            "--run",
            str(run_dir),
            "--strict",
        ]
    )

    try:
        args.func(args)
    except SystemExit as e:
        assert e.code == 2


def test_cli_review_diagnostics_non_destructive(tmp_path: Path) -> None:
    run_dir = create_mock_run(tmp_path, "run-1")
    diag_dir = create_mock_diagnostics(run_dir, warning_pair_count=1)

    # Record hashes and mtimes of all diagnostic files
    file_info_before = {}
    for p in diag_dir.rglob("*"):
        if p.is_file():
            file_info_before[p] = (p.stat().st_mtime_ns, p.read_bytes())

    parser = build_parser()
    args = parser.parse_args(
        [
            "review-diagnostics",
            "--run",
            str(run_dir),
        ]
    )
    args.func(args)

    # Verify no diagnostic files were modified
    for p, (mtime, content) in file_info_before.items():
        assert p.is_file()
        assert p.stat().st_mtime_ns == mtime
        assert p.read_bytes() == content
