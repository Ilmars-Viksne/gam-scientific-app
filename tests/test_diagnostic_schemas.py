from __future__ import annotations

from pathlib import Path

import pandas as pd

from gam_app.config import ExperimentConfig, FeatureConfig, ModelConfig
from gam_app.diagnostic_schema import (
    CONFLICTING_DUPLICATE_TARGET_COLUMNS,
    EXACT_DUPLICATE_GROUP_COLUMNS,
    HIGH_CORRELATION_PAIR_COLUMNS,
    NEAR_DUPLICATE_GROUP_COLUMNS,
    PREDICTOR_DICTIONARY_COLUMNS,
    SUSPECTED_DERIVED_RELATION_COLUMNS,
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
            "x2": [1.0, 2.0, 3.0, 4.0, 5.1],
            "cat": ["a", "b", "a", "b", "a"],
            "target": ["A", "A", "B", "B", "A"],
        }
    )


def make_sample_config(data_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="test_experiment",
        data_path=data_path,
        target="target",
        row_id=None,
        features={
            "x1": FeatureConfig(role="smooth"),
            "x2": FeatureConfig(role="smooth"),
            "cat": FeatureConfig(role="categorical"),
        },
        models=(ModelConfig(id="gam_main"),),
    )


def test_standalone_and_run_high_pair_columns_are_identical(tmp_path: Path) -> None:
    df = make_sample_dataframe()
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    settings = StandaloneDiagnosticSettings(correlation_review_threshold=0.5)
    standalone_diag = calculate_standalone_diagnostics(
        frame=df,
        target="target",
        settings=settings,
    )

    config = make_sample_config(csv_path)
    run_corr = calculate_correlation_analysis(df, config)

    assert (
        tuple(standalone_diag.high_correlation_pairs.columns)
        == HIGH_CORRELATION_PAIR_COLUMNS
    )
    assert (
        tuple(run_corr.high_correlation_pairs.columns) == HIGH_CORRELATION_PAIR_COLUMNS
    )


def test_standalone_and_run_dictionary_columns_are_identical(tmp_path: Path) -> None:
    df = make_sample_dataframe()
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    settings = StandaloneDiagnosticSettings()
    standalone_diag = calculate_standalone_diagnostics(
        frame=df,
        target="target",
        settings=settings,
    )

    config = make_sample_config(csv_path)
    run_corr = calculate_correlation_analysis(df, config)

    assert (
        tuple(standalone_diag.numeric_predictor_dictionary.columns)
        == PREDICTOR_DICTIONARY_COLUMNS
    )
    assert (
        tuple(run_corr.numeric_predictor_dictionary.columns)
        == PREDICTOR_DICTIONARY_COLUMNS
    )


def test_empty_and_populated_artifacts_have_identical_columns(tmp_path: Path) -> None:
    df_empty = pd.DataFrame({"target": ["A", "B"]})
    settings = StandaloneDiagnosticSettings()

    empty_diag = calculate_standalone_diagnostics(
        frame=df_empty,
        target="target",
        settings=settings,
    )

    assert (
        tuple(empty_diag.high_correlation_pairs.columns)
        == HIGH_CORRELATION_PAIR_COLUMNS
    )
    assert (
        tuple(empty_diag.numeric_predictor_dictionary.columns)
        == PREDICTOR_DICTIONARY_COLUMNS
    )
    assert (
        tuple(empty_diag.exact_duplicate_groups.columns)
        == EXACT_DUPLICATE_GROUP_COLUMNS
    )
    assert (
        tuple(empty_diag.near_duplicate_groups.columns) == NEAR_DUPLICATE_GROUP_COLUMNS
    )
    assert (
        tuple(empty_diag.conflicting_duplicate_targets.columns)
        == CONFLICTING_DUPLICATE_TARGET_COLUMNS
    )
    assert (
        tuple(empty_diag.suspected_derived_relations.columns)
        == SUSPECTED_DERIVED_RELATION_COLUMNS
    )


def test_standalone_and_run_manifest_top_level_keys_are_identical(
    tmp_path: Path,
) -> None:
    df = make_sample_dataframe()
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    settings = StandaloneDiagnosticSettings()
    standalone_diag = calculate_standalone_diagnostics(
        frame=df,
        target="target",
        settings=settings,
    )

    out_standalone = tmp_path / "diag_standalone"
    write_standalone_diagnostics(
        diagnostics=standalone_diag,
        output_directory=out_standalone,
        settings=settings,
        data_path=csv_path,
        target="target",
    )
    standalone_manifest = read_json(out_standalone / "diagnostics_manifest.json")

    config = make_sample_config(csv_path)
    run_corr = calculate_correlation_analysis(df, config)
    out_run = tmp_path / "diag_run"
    write_diagnostics(
        artifacts=run_corr,
        output_directory=out_run,
        context_kind="run",
        settings=config.profiling.correlation,
        data_path=csv_path,
        target="target",
    )
    run_manifest = read_json(out_run / "diagnostics_manifest.json")

    expected_keys = {
        "schema_name",
        "schema_version",
        "generated_at_utc",
        "generator",
        "context",
        "dataset",
        "analyses",
        "validation",
        "split_integrity",
        "artifacts",
    }
    assert set(standalone_manifest.keys()) == expected_keys
    assert set(run_manifest.keys()) == expected_keys


def test_manifest_uses_rounding_decimals_in_both_contexts(tmp_path: Path) -> None:
    df = make_sample_dataframe()
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    settings = StandaloneDiagnosticSettings(near_duplicate_decimals=6)
    standalone_diag = calculate_standalone_diagnostics(
        frame=df,
        target="target",
        settings=settings,
    )

    out_dir = tmp_path / "diag"
    write_standalone_diagnostics(
        diagnostics=standalone_diag,
        output_directory=out_dir,
        settings=settings,
        data_path=csv_path,
        target="target",
    )

    manifest = read_json(out_dir / "diagnostics_manifest.json")
    dup_params = manifest["analyses"]["duplicate_groups"]["parameters"]
    assert "rounding_decimals" in dup_params
    assert dup_params["rounding_decimals"] == 6
    assert "near_duplicate_decimals" not in dup_params


def test_disabled_analysis_still_writes_schema_valid_artifact(tmp_path: Path) -> None:
    df = make_sample_dataframe()
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    settings = StandaloneDiagnosticSettings()
    standalone_diag = calculate_standalone_diagnostics(
        frame=df,
        target="target",
        settings=settings,
    )

    out_dir = tmp_path / "diag"
    write_standalone_diagnostics(
        diagnostics=standalone_diag,
        output_directory=out_dir,
        settings=settings,
    )

    high_pairs_file = out_dir / "high_correlation_pairs.csv"
    assert high_pairs_file.exists()
    read_pairs = pd.read_csv(high_pairs_file)
    assert tuple(read_pairs.columns) == HIGH_CORRELATION_PAIR_COLUMNS


def test_no_numeric_predictors_produces_canonical_empty_outputs(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "cat1": ["a", "b", "c"],
            "cat2": ["x", "y", "z"],
            "target": ["A", "B", "A"],
        }
    )

    settings = StandaloneDiagnosticSettings()
    diag = calculate_standalone_diagnostics(
        frame=df,
        target="target",
        settings=settings,
    )

    assert diag.pearson.empty
    assert diag.spearman.empty
    assert tuple(diag.high_correlation_pairs.columns) == HIGH_CORRELATION_PAIR_COLUMNS
    assert len(diag.numeric_predictor_dictionary) == 2


def test_run_only_manifest_sections_are_not_applicable_for_standalone(
    tmp_path: Path,
) -> None:
    df = make_sample_dataframe()
    settings = StandaloneDiagnosticSettings()
    standalone_diag = calculate_standalone_diagnostics(
        frame=df,
        target="target",
        settings=settings,
    )

    out_dir = tmp_path / "diag"
    write_standalone_diagnostics(
        diagnostics=standalone_diag,
        output_directory=out_dir,
        settings=settings,
    )

    manifest = read_json(out_dir / "diagnostics_manifest.json")
    assert manifest["validation"]["status"] == "not_applicable"
    assert manifest["split_integrity"]["status"] == "not_applicable"
