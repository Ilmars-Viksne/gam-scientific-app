from __future__ import annotations

from pathlib import Path

import pandas as pd

from gam_app.config import ExperimentConfig, FeatureConfig, ModelConfig
from gam_app.diagnostic_schema import (
    DIAGNOSTICS_SCHEMA_NAME,
    DIAGNOSTICS_SCHEMA_VERSION,
    build_artifact_manifest_entry,
    update_diagnostics_manifest,
)
from gam_app.diagnostics import (
    StandaloneDiagnosticSettings,
    calculate_correlation_analysis,
    calculate_standalone_diagnostics,
    write_diagnostics,
    write_standalone_diagnostics,
)
from gam_app.io_utils import read_json


def make_sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target": ["A", "A", "B", "B", "A"],
        }
    )


def test_manifest_contains_schema_identity_and_version(tmp_path: Path) -> None:
    df = make_sample_dataframe()
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    settings = StandaloneDiagnosticSettings()
    diag = calculate_standalone_diagnostics(
        frame=df, target="target", settings=settings
    )

    out_dir = tmp_path / "diag"
    write_standalone_diagnostics(
        diagnostics=diag,
        output_directory=out_dir,
        settings=settings,
        data_path=csv_path,
        target="target",
    )

    manifest = read_json(out_dir / "diagnostics_manifest.json")
    assert manifest["schema_name"] == DIAGNOSTICS_SCHEMA_NAME
    assert manifest["schema_version"] == DIAGNOSTICS_SCHEMA_VERSION
    assert "generated_at_utc" in manifest
    assert manifest["generator"]["application"] == "gam-app"


def test_manifest_contains_dataset_provenance(tmp_path: Path) -> None:
    df = make_sample_dataframe()
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    settings = StandaloneDiagnosticSettings()
    diag = calculate_standalone_diagnostics(
        frame=df, target="target", settings=settings
    )

    out_dir = tmp_path / "diag"
    write_standalone_diagnostics(
        diagnostics=diag,
        output_directory=out_dir,
        settings=settings,
        data_path=csv_path,
        target="target",
        row_count=5,
        column_count=3,
        data_hash="abc123hash",
    )

    manifest = read_json(out_dir / "diagnostics_manifest.json")
    dataset_sec = manifest["dataset"]
    assert dataset_sec["sha256"] == "abc123hash"
    assert dataset_sec["row_count"] == 5
    assert dataset_sec["column_count"] == 3
    assert dataset_sec["target"] == "target"


def test_manifest_contains_context_kind(tmp_path: Path) -> None:
    df = make_sample_dataframe()
    settings = StandaloneDiagnosticSettings()
    diag = calculate_standalone_diagnostics(
        frame=df, target="target", settings=settings
    )

    out_dir = tmp_path / "diag"
    write_standalone_diagnostics(
        diagnostics=diag,
        output_directory=out_dir,
        settings=settings,
    )

    manifest = read_json(out_dir / "diagnostics_manifest.json")
    assert manifest["context"]["kind"] == "standalone"
    assert manifest["context"]["command"] == "profile"


def test_every_analysis_has_status(tmp_path: Path) -> None:
    df = make_sample_dataframe()
    settings = StandaloneDiagnosticSettings()
    diag = calculate_standalone_diagnostics(
        frame=df, target="target", settings=settings
    )

    out_dir = tmp_path / "diag"
    write_standalone_diagnostics(
        diagnostics=diag,
        output_directory=out_dir,
        settings=settings,
    )

    manifest = read_json(out_dir / "diagnostics_manifest.json")
    analyses = manifest["analyses"]

    expected_analyses = [
        "correlation",
        "derived_relations",
        "duplicate_groups",
        "data_dictionary",
        "split_integrity",
    ]
    for key in expected_analyses:
        assert key in analyses
        assert "status" in analyses[key]


def test_zero_findings_are_distinct_from_disabled(tmp_path: Path) -> None:
    df = pd.DataFrame({"x1": [1.0, 2.0], "target": ["A", "B"]})
    settings = StandaloneDiagnosticSettings()
    diag = calculate_standalone_diagnostics(
        frame=df, target="target", settings=settings
    )

    out_dir = tmp_path / "diag"
    write_standalone_diagnostics(
        diagnostics=diag,
        output_directory=out_dir,
        settings=settings,
    )

    manifest = read_json(out_dir / "diagnostics_manifest.json")
    corr_sec = manifest["analyses"]["correlation"]
    assert corr_sec["status"] == "completed"
    assert corr_sec["results"]["high_correlation_pair_count"] == 0


def test_manifest_lists_every_written_diagnostic_artifact(tmp_path: Path) -> None:
    df = make_sample_dataframe()
    settings = StandaloneDiagnosticSettings()
    diag = calculate_standalone_diagnostics(
        frame=df, target="target", settings=settings
    )

    out_dir = tmp_path / "diag"
    write_standalone_diagnostics(
        diagnostics=diag,
        output_directory=out_dir,
        settings=settings,
    )

    manifest = read_json(out_dir / "diagnostics_manifest.json")
    artifacts = manifest["artifacts"]
    artifact_ids = {a["id"] for a in artifacts}

    expected_ids = {
        "pearson_matrix",
        "spearman_matrix",
        "high_correlation_pairs",
        "predictor_dictionary",
        "suspected_derived_relations",
        "exact_duplicate_groups",
        "near_duplicate_groups",
        "conflicting_duplicate_targets",
    }
    assert artifact_ids == expected_ids

    for artifact in artifacts:
        assert artifact["status"] == "written"
        assert artifact["sha256"] is not None
        assert artifact["byte_count"] is not None


def test_manifest_artifact_row_counts_match_csv_files(tmp_path: Path) -> None:
    df = make_sample_dataframe()
    settings = StandaloneDiagnosticSettings(correlation_review_threshold=0.5)
    diag = calculate_standalone_diagnostics(
        frame=df, target="target", settings=settings
    )

    out_dir = tmp_path / "diag"
    write_standalone_diagnostics(
        diagnostics=diag,
        output_directory=out_dir,
        settings=settings,
    )

    manifest = read_json(out_dir / "diagnostics_manifest.json")
    artifact_map = {a["path"]: a for a in manifest["artifacts"]}

    high_pairs_csv = pd.read_csv(out_dir / "high_correlation_pairs.csv")
    assert artifact_map["high_correlation_pairs.csv"]["row_count"] == len(
        high_pairs_csv
    )

    dict_csv = pd.read_csv(out_dir / "numeric_predictor_dictionary.csv")
    assert artifact_map["numeric_predictor_dictionary.csv"]["row_count"] == len(
        dict_csv
    )


def test_split_integrity_update_preserves_all_existing_sections(tmp_path: Path) -> None:
    df = make_sample_dataframe()
    settings = StandaloneDiagnosticSettings()
    diag = calculate_standalone_diagnostics(
        frame=df, target="target", settings=settings
    )

    out_dir = tmp_path / "diag"
    write_standalone_diagnostics(
        diagnostics=diag,
        output_directory=out_dir,
        settings=settings,
    )

    manifest_path = out_dir / "diagnostics_manifest.json"
    before = read_json(manifest_path)

    update_diagnostics_manifest(
        manifest_path,
        split_integrity_updates={"status": "completed", "passed": True},
        validation_updates={"split_integrity_passed": True},
        artifact_entries=[
            build_artifact_manifest_entry(
                artifact_id="split_integrity",
                relative_path="split_integrity.csv",
                media_type="text/csv",
                schema_id="split_integrity/1.0",
                file_path=out_dir / "high_correlation_pairs.csv",
            )
        ],
    )

    after = read_json(manifest_path)
    assert after["schema_name"] == before["schema_name"]
    assert after["dataset"] == before["dataset"]
    assert after["analyses"]["correlation"] == before["analyses"]["correlation"]
    assert after["split_integrity"]["passed"] is True
    assert any(a["id"] == "split_integrity" for a in after["artifacts"])


def test_manifest_paths_are_relative_to_diagnostics_directory(tmp_path: Path) -> None:
    df = make_sample_dataframe()
    settings = StandaloneDiagnosticSettings()
    diag = calculate_standalone_diagnostics(
        frame=df, target="target", settings=settings
    )

    out_dir = tmp_path / "diag"
    write_standalone_diagnostics(
        diagnostics=diag,
        output_directory=out_dir,
        settings=settings,
    )

    manifest = read_json(out_dir / "diagnostics_manifest.json")
    for artifact in manifest["artifacts"]:
        p = artifact["path"]
        assert not Path(p).is_absolute()
        assert ".." not in p


def test_manifest_contains_no_nan_or_infinity_json_values(tmp_path: Path) -> None:
    df = make_sample_dataframe()
    config = ExperimentConfig(
        name="test_experiment",
        data_path=tmp_path / "data.csv",
        target="target",
        row_id=None,
        features={
            "x1": FeatureConfig(role="smooth"),
            "x2": FeatureConfig(role="smooth"),
        },
        models=(ModelConfig(id="m1"),),
    )
    corr = calculate_correlation_analysis(df, config)

    out_dir = tmp_path / "diag"
    write_diagnostics(
        artifacts=corr,
        output_directory=out_dir,
        context_kind="run",
        settings=config.profiling.correlation,
    )

    manifest_text = (out_dir / "diagnostics_manifest.json").read_text(encoding="utf-8")
    assert "NaN" not in manifest_text
    assert "Infinity" not in manifest_text
