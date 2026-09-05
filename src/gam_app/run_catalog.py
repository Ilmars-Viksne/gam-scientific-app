from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any

from .config import load_config
from .io_utils import read_json

RUN_METADATA_SCHEMA_NAME = "gam_run_metadata"
RUN_METADATA_SCHEMA_VERSION = "1.0"
CATALOG_SCHEMA_NAME = "gam_run_catalog"
CATALOG_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    run_directory: Path
    created_at_utc: str | None
    updated_at_utc: str | None
    state: str
    phase: str | None
    experiment_name: str | None
    application_version: str | None
    configuration_schema_version: str | None
    dataset_basename: str | None
    data_hash: str | None
    config_hash: str | None
    target: str | None
    validation_strategy: str | None
    duplicate_group_policy: str | None
    model_ids: tuple[str, ...]
    tags: tuple[str, ...]
    metadata: Mapping[str, str]
    metadata_status: str  # "complete", "legacy", "incomplete", "invalid"
    sensitivity_ids: tuple[str, ...] = ()
    created_run_path: str | None = None
    created_workspace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "path": self.run_directory.resolve().as_posix(),
            "created_run_path": self.created_run_path,
            "created_workspace_path": self.created_workspace_path,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "state": self.state,
            "phase": self.phase,
            "experiment_name": self.experiment_name,
            "application_version": self.application_version,
            "configuration_schema_version": self.configuration_schema_version,
            "dataset_basename": self.dataset_basename,
            "data_hash": self.data_hash,
            "config_hash": self.config_hash,
            "target": self.target,
            "validation_strategy": self.validation_strategy,
            "duplicate_group_policy": self.duplicate_group_policy,
            "model_ids": list(self.model_ids),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "metadata_status": self.metadata_status,
            "sensitivity_ids": list(self.sensitivity_ids),
        }


@dataclass(frozen=True, slots=True)
class RunFilter:
    states: tuple[str, ...] = ()
    experiment_names: tuple[str, ...] = ()
    validation_strategies: tuple[str, ...] = ()
    duplicate_group_policies: tuple[str, ...] = ()
    model_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    sensitivity_ids: tuple[str, ...] = ()
    metadata_equals: tuple[tuple[str, str], ...] = ()
    created_after: datetime | None = None
    created_before: datetime | None = None
    data_hash: str | None = None
    config_hash: str | None = None


def parse_datetime_argument(
    value: str | datetime,
    *,
    is_end_of_day: bool = False,
) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError(
                f"Timestamp '{value.isoformat()}' has no timezone offset. "
                "Use 'Z' for UTC or provide an explicit offset such as '+03:00'."
            )
        return value.astimezone(UTC)

    value_str = value.strip()
    # Check date-only YYYY-MM-DD
    if len(value_str) == 10 and value_str.count("-") == 2:
        try:
            d = datetime.strptime(value_str, "%Y-%m-%d").date()
            if is_end_of_day:
                return datetime.combine(d, time.max, tzinfo=UTC)
            return datetime.combine(d, time.min, tzinfo=UTC)
        except ValueError as err:
            raise ValueError(f"Invalid date format '{value_str}': {err}") from err

    # Datetime string
    try:
        dt = datetime.fromisoformat(value_str.replace("Z", "+00:00"))
    except ValueError as err:
        raise ValueError(f"Invalid datetime format '{value_str}': {err}") from err

    if dt.tzinfo is None:
        raise ValueError(
            f"Timestamp '{value_str}' has no timezone offset. "
            "Use 'Z' for UTC or provide an explicit offset such as '+03:00'."
        )

    return dt.astimezone(UTC)


def _load_run_summary(run_dir: Path) -> RunSummary:
    run_json_path = run_dir / "run.json"
    status_json_path = run_dir / "status.json"

    if not run_json_path.is_file():
        raise FileNotFoundError(f"Missing run.json in {run_dir}")

    run_data = read_json(run_json_path)
    if not isinstance(run_data, dict):
        raise ValueError(f"run.json in {run_dir} is not a valid JSON object")

    status_data = {}
    if status_json_path.is_file():
        try:
            raw_status = read_json(status_json_path)
            if isinstance(raw_status, dict):
                status_data = raw_status
        except Exception:
            pass

    state = status_data.get("state", "unknown")
    phase = status_data.get("phase")
    updated_at_utc = status_data.get("updated_at_utc")

    # Discover local sensitivity memberships
    sens_ids: list[str] = []
    sens_dir = run_dir / "sensitivity"
    if sens_dir.is_dir():
        for p in sens_dir.glob("*.json"):
            try:
                data = read_json(p)
                sid = data.get("sensitivity_id")
                if sid and data.get("schema_name") == "gam_sensitivity_membership":
                    declared_mem_id = data.get("member_run_id")
                    actual_run_id = str(run_data.get("run_id", run_dir.name))
                    if declared_mem_id is None or declared_mem_id == actual_run_id:
                        if sid not in sens_ids:
                            sens_ids.append(sid)
            except Exception:
                pass

    sorted_sens_ids = tuple(sorted(set(sens_ids)))

    schema_name = run_data.get("schema_name")
    is_v10_metadata = schema_name == RUN_METADATA_SCHEMA_NAME

    if is_v10_metadata:
        exp_info = run_data.get("experiment") or {}
        ds_info = run_data.get("dataset") or {}
        val_info = run_data.get("validation") or {}
        models = run_data.get("models") or []
        tags = run_data.get("tags") or []
        meta = run_data.get("metadata") or run_data.get("labels") or {}

        return RunSummary(
            run_id=str(run_data.get("run_id", run_dir.name)),
            run_directory=run_dir,
            created_at_utc=run_data.get("created_at_utc"),
            updated_at_utc=updated_at_utc,
            state=state,
            phase=phase,
            experiment_name=exp_info.get("name"),
            application_version=run_data.get("application_version"),
            configuration_schema_version=run_data.get(
                "configuration_schema_version", run_data.get("schema_version")
            ),
            dataset_basename=ds_info.get("basename"),
            data_hash=run_data.get("data_hash"),
            config_hash=run_data.get("config_hash"),
            target=ds_info.get("target"),
            validation_strategy=val_info.get("strategy"),
            duplicate_group_policy=val_info.get("duplicate_group_policy"),
            model_ids=tuple(str(m) for m in models),
            tags=tuple(str(t) for t in tags),
            metadata={str(k): str(v) for k, v in meta.items()},
            metadata_status="complete" if status_json_path.is_file() else "incomplete",
            sensitivity_ids=sorted_sens_ids,
            created_run_path=run_data.get("created_run_path"),
            created_workspace_path=run_data.get("created_workspace_path"),
        )

    # Legacy run.json adapter
    config_yaml_path = run_dir / "config.yaml"
    exp_name = None
    target = None
    val_strategy = None
    dup_policy = None
    model_ids: list[str] = []
    tags_list: list[str] = []
    meta_dict: dict[str, str] = {}
    ds_basename = None

    if config_yaml_path.is_file():
        try:
            cfg = load_config(config_yaml_path)
            exp_name = cfg.name
            target = cfg.target
            val_strategy = cfg.validation.strategy
            dup_policy = cfg.validation.duplicate_group_policy
            model_ids = [m.id for m in cfg.models]
            tags_list = list(cfg.tags)
            meta_dict = dict(cfg.metadata)
            ds_basename = cfg.data_path.name
        except Exception:
            pass

    raw_meta = run_data.get("metadata", run_data.get("labels", {}))
    if isinstance(raw_meta, dict):
        for k, v in raw_meta.items():
            meta_dict[str(k)] = str(v)

    raw_tags = run_data.get("tags", [])
    if isinstance(raw_tags, list):
        for t in raw_tags:
            if str(t) not in tags_list:
                tags_list.append(str(t))

    return RunSummary(
        run_id=str(run_data.get("run_id", run_dir.name)),
        run_directory=run_dir,
        created_at_utc=run_data.get("created_at_utc"),
        updated_at_utc=updated_at_utc,
        state=state,
        phase=phase,
        experiment_name=exp_name,
        application_version=run_data.get("application_version"),
        configuration_schema_version=run_data.get("schema_version"),
        dataset_basename=ds_basename,
        data_hash=run_data.get("data_hash"),
        config_hash=run_data.get("config_hash"),
        target=target,
        validation_strategy=val_strategy,
        duplicate_group_policy=dup_policy,
        model_ids=tuple(model_ids),
        tags=tuple(tags_list),
        metadata=meta_dict,
        metadata_status="legacy",
        sensitivity_ids=sorted_sens_ids,
        created_run_path=run_data.get("created_run_path"),
        created_workspace_path=run_data.get("created_workspace_path"),
    )


@dataclass(frozen=True, slots=True)
class CatalogResult:
    workspace: Path
    runs: tuple[RunSummary, ...]
    matched_count: int
    invalid_run_count: int
    warnings: tuple[str, ...] = ()


def discover_runs(
    workspace: Path,
    filters: RunFilter | None = None,
    limit: int | None = None,
    include_invalid: bool = False,
) -> CatalogResult:
    runs_dir = workspace / "runs"
    discovered_runs: list[RunSummary] = []
    invalid_count = 0
    warnings: list[str] = []

    if runs_dir.is_dir():
        for item in runs_dir.iterdir():
            if not item.is_dir() or item.name.startswith("."):
                continue

            try:
                summary = _load_run_summary(item)
                discovered_runs.append(summary)
            except Exception as err:
                invalid_count += 1
                warnings.append(
                    f"Run directory {item.name!r} could not be loaded: {err}"
                )

    if filters is not None:
        filtered_runs = [r for r in discovered_runs if run_matches_filter(r, filters)]
    else:
        filtered_runs = discovered_runs

    # Deterministic sorting
    def sort_key(run: RunSummary) -> tuple[int, datetime, str]:
        if run.created_at_utc is not None:
            try:
                dt = parse_datetime_argument(run.created_at_utc)
                return (1, dt, run.run_id)
            except Exception:
                pass
        return (0, datetime.min.replace(tzinfo=UTC), run.run_id)

    sorted_runs = sorted(filtered_runs, key=sort_key, reverse=True)

    matched_count = len(sorted_runs)
    if limit is not None and limit >= 0:
        sorted_runs = sorted_runs[:limit]

    return CatalogResult(
        workspace=workspace,
        runs=tuple(sorted_runs),
        matched_count=matched_count,
        invalid_run_count=invalid_count,
        warnings=tuple(warnings),
    )


def run_matches_filter(run: RunSummary, filters: RunFilter) -> bool:
    if filters.states and run.state not in filters.states:
        return False

    if filters.experiment_names and run.experiment_name not in filters.experiment_names:
        return False

    if (
        filters.validation_strategies
        and run.validation_strategy not in filters.validation_strategies
    ):
        return False

    if (
        filters.duplicate_group_policies
        and run.duplicate_group_policy not in filters.duplicate_group_policies
    ):
        return False

    if filters.model_ids and not any(
        model_id in run.model_ids for model_id in filters.model_ids
    ):
        return False

    if filters.data_hash and run.data_hash != filters.data_hash:
        return False

    if filters.config_hash and run.config_hash != filters.config_hash:
        return False

    if filters.tags:
        run_tags_lower = {t.casefold() for t in run.tags}
        if any(req_tag.casefold() not in run_tags_lower for req_tag in filters.tags):
            return False

    if filters.sensitivity_ids:
        if not any(sid in run.sensitivity_ids for sid in filters.sensitivity_ids):
            return False

    if filters.metadata_equals:
        for key, expected_val in filters.metadata_equals:
            if run.metadata.get(key) != expected_val:
                return False

    if filters.created_after or filters.created_before:
        if run.created_at_utc is None:
            return False
        try:
            created_dt = parse_datetime_argument(run.created_at_utc)
        except Exception:
            return False

        if filters.created_after and created_dt < filters.created_after:
            return False
        if filters.created_before and created_dt > filters.created_before:
            return False

    return True
