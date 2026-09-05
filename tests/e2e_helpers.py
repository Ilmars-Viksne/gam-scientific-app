from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from gam_app.workflow import create_run, execute_run


def write_dataset(
    frame: pd.DataFrame,
    path: Path,
) -> Path:
    frame.to_csv(path, index=False)
    return path


def write_config(
    payload: dict[str, Any],
    path: Path,
) -> Path:
    path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def create_and_execute(
    *,
    config_path: Path,
    workspace: Path,
) -> Path:
    run_directory = create_run(
        config_path,
        workspace,
    )
    execute_run(run_directory)
    return run_directory


def load_json_object(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def assert_run_completed(
    run_directory: Path,
) -> None:
    status = load_json_object(run_directory / "status.json")

    assert status["state"] == "completed"

    run_metadata = load_json_object(run_directory / "run.json")

    assert status["run_id"] == run_metadata["run_id"]
    assert run_directory.name == run_metadata["run_id"]
    assert not (run_directory / "run.lock").exists()
