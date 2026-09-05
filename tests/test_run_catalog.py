from datetime import UTC, datetime
from pathlib import Path

import pytest

from gam_app.run_catalog import (
    RunFilter,
    RunSummary,
    discover_runs,
    parse_datetime_argument,
    run_matches_filter,
)


def test_parse_datetime_argument_utc_conversion() -> None:
    dt = parse_datetime_argument("2026-09-05T12:00:00Z")
    assert dt == datetime(2026, 9, 5, 12, 0, 0, tzinfo=UTC)

    d_start = parse_datetime_argument("2026-09-05", is_end_of_day=False)
    assert d_start == datetime(2026, 9, 5, 0, 0, 0, tzinfo=UTC)

    d_end = parse_datetime_argument("2026-09-05", is_end_of_day=True)
    assert d_end == datetime(2026, 9, 5, 23, 59, 59, 999999, tzinfo=UTC)


def test_parse_datetime_argument_rejects_naive() -> None:
    with pytest.raises(ValueError, match="has no timezone offset"):
        parse_datetime_argument("2026-09-05T12:00:00")


def test_run_matches_filter_logic() -> None:
    summary = RunSummary(
        run_id="run-1",
        run_directory=Path("/tmp/run-1"),
        created_at_utc="2026-09-05T12:00:00Z",
        updated_at_utc=None,
        state="completed",
        phase="reporting",
        experiment_name="exp1",
        application_version="0.1.0",
        configuration_schema_version="1.1",
        dataset_basename="data.csv",
        data_hash="hash123",
        config_hash="cfg123",
        target="y",
        validation_strategy="stratified",
        duplicate_group_policy="report",
        model_ids=("gam_main", "gam_pairwise"),
        tags=("candidate", "grouped"),
        metadata={"project": "bridge-study", "cohort": "2026-q3"},
        metadata_status="complete",
    )

    # State filter
    assert run_matches_filter(summary, RunFilter(states=("completed",)))
    assert not run_matches_filter(summary, RunFilter(states=("failed",)))

    # Tag filter (AND logic, case-insensitive)
    assert run_matches_filter(summary, RunFilter(tags=("CANDIDATE", "GROUPED")))
    assert not run_matches_filter(summary, RunFilter(tags=("candidate", "other")))

    # Metadata filter
    assert run_matches_filter(
        summary, RunFilter(metadata_equals=(("project", "bridge-study"),))
    )
    assert not run_matches_filter(
        summary, RunFilter(metadata_equals=(("project", "other"),))
    )

    # Creation range filter
    after = parse_datetime_argument("2026-09-01", is_end_of_day=False)
    before = parse_datetime_argument("2026-09-10", is_end_of_day=True)
    assert run_matches_filter(
        summary, RunFilter(created_after=after, created_before=before)
    )

    after_too_late = parse_datetime_argument("2026-09-06", is_end_of_day=False)
    assert not run_matches_filter(summary, RunFilter(created_after=after_too_late))


def test_discover_runs_handles_legacy_and_invalid(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    runs_dir = workspace / "runs"
    runs_dir.mkdir(parents=True)

    # Valid v1.0 metadata run
    run1 = runs_dir / "run-1"
    run1.mkdir()
    (run1 / "run.json").write_text(
        '{"schema_name": "gam_run_metadata", "run_id": "run-1", "created_at_utc": "2026-09-05T10:00:00Z", "experiment": {"name": "exp1"}, "tags": ["tag1"], "metadata": {"k": "v"}}',
        encoding="utf-8",
    )
    (run1 / "status.json").write_text('{"state": "completed"}', encoding="utf-8")

    # Invalid run (corrupted json)
    run2 = runs_dir / "run-2"
    run2.mkdir()
    (run2 / "run.json").write_text("invalid json", encoding="utf-8")

    catalog = discover_runs(workspace)
    assert catalog.matched_count == 1
    assert catalog.invalid_run_count == 1
    assert catalog.runs[0].run_id == "run-1"
    assert catalog.runs[0].tags == ("tag1",)
    assert catalog.runs[0].metadata == {"k": "v"}
