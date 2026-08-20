from __future__ import annotations

import json
import platform
import sys
import uuid
from pathlib import Path
from time import perf_counter

import pandas as pd
import sklearn

from . import __version__
from .config import dump_config_dict, load_config
from .data import validate_training_data
from .evaluation import create_split_manifest, fit_final_model, run_model
from .io_utils import (
    format_duration,
    sha256_file,
    stable_hash,
    utc_now,
    write_json_atomic,
    write_yaml_atomic,
)
from .reporting import create_reports
from .run_store import FileRunStore


def create_run(config_path: Path, workspace: Path) -> Path:
    config = load_config(config_path)

    timestamp = utc_now().replace(":", "").replace("+00:00", "Z")
    identifier = uuid.uuid4().hex[:8]
    run_id = f"run-{timestamp}-{identifier}"

    run_directory = workspace / "runs" / run_id
    store = FileRunStore(run_directory)
    store.initialize()
    resolved = dump_config_dict(config)
    data_hash = sha256_file(config.data_path)
    config_hash = stable_hash(resolved)
    write_json_atomic(
        store.root / "run.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "created_at_utc": utc_now(),
            "data_hash": data_hash,
            "config_hash": config_hash,
            "application_version": __version__,
        },
    )
    write_yaml_atomic(store.root / "config.yaml", resolved)
    write_json_atomic(
        store.root / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    )
    write_json_atomic(store.status_path, {"state": "created", "run_id": run_id})
    return run_directory


def execute_run(run_directory: Path) -> None:
    run_started_at = perf_counter()
    store = FileRunStore(run_directory)
    store.acquire_lock()
    try:
        config = load_config(run_directory / "config.yaml")
        run = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
        X, y, row_ids = validate_training_data(config)

        write_json_atomic(
            run_directory / "data_manifest.json",
            {
                "path": str(config.data_path),
                "sha256": run["data_hash"],
                "rows": len(X),
                "predictors": list(X.columns),
                "target": config.target,
                "class_counts": y.value_counts().to_dict(),
            },
        )
        split_path = run_directory / "split_manifest.csv"
        if split_path.exists():
            splits = pd.read_csv(split_path)
        else:
            splits = create_split_manifest(config, X, y, row_ids)
            splits.to_csv(split_path, index=False)
        store.update_status(
            state="running", phase="evaluation", started_at_utc=utc_now()
        )
        store.event("run_started")
        for model in config.models:
            run_model(
                config,
                model,
                X,
                y,
                row_ids,
                splits,
                store,
                run["data_hash"],
                run["config_hash"],
            )
            if store.requested("PAUSE") or store.requested("CANCEL"):
                return
            print(
                f"[{model.id}] Fitting final model on {len(X)} observations",
                flush=True,
            )
            final_fit_started_at = perf_counter()
            fit_final_model(config, model, X, y, store)
            print(
                f"[{model.id}] Final model completed in "
                f"{format_duration(perf_counter() - final_fit_started_at)}",
                flush=True,
            )
        create_reports(config, store)
        store.update_status(
            state="completed", phase="reporting", completed_at_utc=utc_now()
        )
        store.event("run_completed")
        print(
            f"Run completed in {format_duration(perf_counter() - run_started_at)}",
            flush=True,
        )
    except Exception as error:
        store.update_status(state="failed", error=repr(error))
        store.event("run_failed", level="ERROR", error=repr(error))
        raise
    finally:
        store.release_lock()
