from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

from .exceptions import CheckpointError
from .io_utils import append_jsonl, read_json, utc_now, write_json_atomic


class FileRunStore:
    def __init__(self, run_directory: Path) -> None:
        self.root = run_directory
        self.status_path = self.root / "status.json"
        self.events_path = self.root / "events.jsonl"
        self.control = self.root / "control"
        self.checkpoints = self.root / "checkpoints"
        self.results = self.root / "results"
        self.models = self.root / "models"
        self.plots = self.root / "plots"
        self.reports = self.root / "reports"

    def initialize(self) -> None:
        for path in [
            self.root,
            self.control,
            self.checkpoints,
            self.results,
            self.models,
            self.plots,
            self.reports,
            self.root / "logs",
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def update_status(self, **values: Any) -> None:
        current = read_json(self.status_path) if self.status_path.exists() else {}
        current.update(values)
        current["updated_at_utc"] = utc_now()
        write_json_atomic(self.status_path, current)

    def event(self, event: str, level: str = "INFO", **values: Any) -> None:
        append_jsonl(self.events_path, {"level": level, "event": event, **values})

    def requested(self, name: str) -> bool:
        return (self.control / name).exists()

    def checkpoint_directory(self, model_id: str, repeat: int, fold: int) -> Path:
        return self.checkpoints / model_id / f"repeat-{repeat:02d}_fold-{fold:02d}"

    def checkpoint_complete(
        self, model_id: str, repeat: int, fold: int, data_hash: str, config_hash: str
    ) -> bool:
        directory = self.checkpoint_directory(model_id, repeat, fold)
        complete = directory / "COMPLETE"
        metadata = directory / "checkpoint.json"
        if not complete.exists() or not metadata.exists():
            return False
        payload = read_json(metadata)
        if payload["data_hash"] != data_hash or payload["config_hash"] != config_hash:
            raise CheckpointError("Checkpoint hashes do not match this run.")
        return True

    def acquire_lock(self) -> None:
        lock = self.root / "run.lock"
        if lock.exists():
            payload = read_json(lock)
            if payload.get("pid") != os.getpid():
                raise RuntimeError(f"Run is locked by PID {payload.get('pid')}.")
        write_json_atomic(
            lock,
            {"pid": os.getpid(), "host": socket.gethostname(), "created_at_utc": utc_now()},
        )

    def release_lock(self) -> None:
        lock = self.root / "run.lock"
        if lock.exists():
            lock.unlink()
