from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from e2e_helpers import sha256_file

from gam_app.config import (
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    ValidationConfig,
)
from gam_app.diagnostic_schema import (
    CONFLICTING_DUPLICATE_TARGET_COLUMNS,
    EXACT_DUPLICATE_GROUP_COLUMNS,
    HIGH_CORRELATION_PAIR_COLUMNS,
    NEAR_DUPLICATE_GROUP_COLUMNS,
    PREDICTOR_DICTIONARY_COLUMNS,
    SPLIT_INTEGRITY_COLUMNS,
    SUSPECTED_DERIVED_RELATION_COLUMNS,
)
from gam_app.diagnostics import (
    StandaloneDiagnosticSettings,
    calculate_standalone_diagnostics,
    write_standalone_diagnostics,
)
from gam_app.workflow import create_run, execute_run


@dataclass(frozen=True, slots=True)
class CsvArtifactContract:
    artifact_id: str
    filename: str
    schema_id: str
    expected_columns: tuple[str, ...]


NEAR_DUPLICATE_EDGE_COLUMNS: tuple[str, ...] = (
    "left_row_id",
    "right_row_id",
    "matched_column_count",
    "compared_column_count",
    "match_fraction",
    "threshold",
)

EFFECTIVE_VALIDATION_GROUP_COLUMNS: tuple[str, ...] = (
    "row_id",
    "configured_group",
    "exact_duplicate_group_id",
    "near_duplicate_group_id",
    "effective_group_id",
    "group_sources",
)

CSV_ARTIFACT_CONTRACTS: tuple[CsvArtifactContract, ...] = (
    CsvArtifactContract(
        artifact_id="high_correlation_pairs",
        filename="high_correlation_pairs.csv",
        schema_id="high_correlation_pairs/1.0",
        expected_columns=HIGH_CORRELATION_PAIR_COLUMNS,
    ),
    CsvArtifactContract(
        artifact_id="predictor_dictionary",
        filename="numeric_predictor_dictionary.csv",
        schema_id="predictor_dictionary/1.0",
        expected_columns=PREDICTOR_DICTIONARY_COLUMNS,
    ),
    CsvArtifactContract(
        artifact_id="suspected_derived_relations",
        filename="suspected_derived_relations.csv",
        schema_id="suspected_derived_relations/1.0",
        expected_columns=SUSPECTED_DERIVED_RELATION_COLUMNS,
    ),
    CsvArtifactContract(
        artifact_id="exact_duplicate_groups",
        filename="exact_duplicate_groups.csv",
        schema_id="exact_duplicate_groups/1.0",
        expected_columns=EXACT_DUPLICATE_GROUP_COLUMNS,
    ),
    CsvArtifactContract(
        artifact_id="near_duplicate_groups",
        filename="near_duplicate_groups.csv",
        schema_id="near_duplicate_groups/1.0",
        expected_columns=NEAR_DUPLICATE_GROUP_COLUMNS,
    ),
    CsvArtifactContract(
        artifact_id="conflicting_duplicate_targets",
        filename="conflicting_duplicate_targets.csv",
        schema_id="conflicting_duplicate_targets/1.0",
        expected_columns=CONFLICTING_DUPLICATE_TARGET_COLUMNS,
    ),
    CsvArtifactContract(
        artifact_id="split_integrity",
        filename="split_integrity.csv",
        schema_id="split_integrity/1.0",
        expected_columns=SPLIT_INTEGRITY_COLUMNS,
    ),
    CsvArtifactContract(
        artifact_id="near_duplicate_edges",
        filename="near_duplicate_edges.csv",
        schema_id="near_duplicate_edges/1.0",
        expected_columns=NEAR_DUPLICATE_EDGE_COLUMNS,
    ),
    CsvArtifactContract(
        artifact_id="effective_validation_groups",
        filename="effective_validation_groups.csv",
        schema_id="effective_validation_groups/1.0",
        expected_columns=EFFECTIVE_VALIDATION_GROUP_COLUMNS,
    ),
)


def make_populated_dataset() -> pd.DataFrame:
    rows = []
    for i in range(12):
        x1 = float(i)
        rows.append(
            {
                "row_id": f"row-{i:02d}",
                "batch_id": f"batch-{i:02d}",
                "x1": x1,
                "x2": 2.0 * x1,
                "x3": float(i % 2),
                "target": "A" if i < 6 else "B",
            }
        )
    df = pd.DataFrame(rows)
    # Plant exact duplicate
    df.loc[1, ["x1", "x2", "x3"]] = df.loc[0, ["x1", "x2", "x3"]].to_numpy()
    # Plant target conflict
    df.loc[7, ["x1", "x2", "x3"]] = df.loc[6, ["x1", "x2", "x3"]].to_numpy()
    df.loc[7, "target"] = "B"
    df.loc[6, "target"] = "A"
    return df


def generate_run_artifacts(tmp_path: Path) -> Path:
    data_df = make_populated_dataset()
    data_path = tmp_path / "data.csv"
    data_df.to_csv(data_path, index=False)

    config = ExperimentConfig(
        name="test_schema_run",
        data_path=data_path,
        target="target",
        row_id="row_id",
        group_column="batch_id",
        features={
            "x1": FeatureConfig(role="smooth"),
            "x2": FeatureConfig(role="smooth"),
            "x3": FeatureConfig(role="smooth"),
            "batch_id": FeatureConfig(role="exclude"),
            "row_id": FeatureConfig(role="exclude"),
        },
        models=(ModelConfig(id="gam_main"),),
        validation=ValidationConfig(
            strategy="stratified_group",
            outer_splits=3,
            outer_repeats=1,
            inner_splits=2,
            duplicate_group_policy="group",
        ),
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml_dump_config(config),
        encoding="utf-8",
    )

    workspace = tmp_path / "workspace"
    run_dir = create_run(config_path, workspace)
    execute_run(run_dir)
    return run_dir


def yaml_dump_config(config: ExperimentConfig) -> str:
    import yaml

    from gam_app.config import dump_config_dict

    return yaml.safe_dump(dump_config_dict(config), sort_keys=False)


@pytest.mark.parametrize(
    "contract", CSV_ARTIFACT_CONTRACTS, ids=lambda c: c.artifact_id
)
def test_csv_artifact_exact_columns(
    tmp_path: Path, contract: CsvArtifactContract
) -> None:
    run_dir = generate_run_artifacts(tmp_path)
    file_path = run_dir / "diagnostics" / contract.filename
    if not file_path.exists():
        pytest.skip(
            f"Artifact {contract.filename} was not generated for this run configuration"
        )

    frame = pd.read_csv(file_path)
    assert tuple(frame.columns) == contract.expected_columns


def test_empty_and_populated_csv_artifact_columns(tmp_path: Path) -> None:
    empty_df = pd.DataFrame({"target": ["A", "B"]})
    settings = StandaloneDiagnosticSettings()

    empty_diag = calculate_standalone_diagnostics(
        frame=empty_df,
        target="target",
        settings=settings,
    )

    out_dir = tmp_path / "empty_diag"
    write_standalone_diagnostics(
        diagnostics=empty_diag,
        output_directory=out_dir,
        settings=settings,
    )

    for contract in CSV_ARTIFACT_CONTRACTS:
        file_path = out_dir / contract.filename
        if file_path.exists():
            read_df = pd.read_csv(file_path)
            assert tuple(read_df.columns) == contract.expected_columns


def test_controlled_vocabularies(tmp_path: Path) -> None:
    run_dir = generate_run_artifacts(tmp_path)
    diag_dir = run_dir / "diagnostics"

    high_pairs = pd.read_csv(diag_dir / "high_correlation_pairs.csv")
    if not high_pairs.empty:
        assert set(high_pairs["severity"].dropna()) <= {"review", "warning"}
        assert set(high_pairs["dominant_method"].dropna()) <= {
            "pearson",
            "spearman",
            "tie",
            "none",
        }

    dictionary = pd.read_csv(diag_dir / "numeric_predictor_dictionary.csv")
    if not dictionary.empty:
        assert set(dictionary["metadata_status"].dropna()) <= {
            "provided",
            "not_provided",
        }
        assert set(dictionary["derived_status"].dropna()) <= {
            "declared",
            "not_declared",
            "suspected",
            "not_evaluated",
        }


def test_identifier_uniqueness(tmp_path: Path) -> None:
    run_dir = generate_run_artifacts(tmp_path)
    diag_dir = run_dir / "diagnostics"

    high_pairs = pd.read_csv(diag_dir / "high_correlation_pairs.csv")
    if not high_pairs.empty:
        assert not high_pairs["rank"].duplicated().any()
        assert high_pairs["rank"].tolist() == list(range(1, len(high_pairs) + 1))

    exact_dups = pd.read_csv(diag_dir / "exact_duplicate_groups.csv")
    if not exact_dups.empty:
        assert not exact_dups[["duplicate_group_id", "row_id"]].duplicated().any()

    near_dups = pd.read_csv(diag_dir / "near_duplicate_groups.csv")
    if not near_dups.empty:
        assert not near_dups[["near_duplicate_group_id", "row_id"]].duplicated().any()


def test_semantic_column_relationships(tmp_path: Path) -> None:
    run_dir = generate_run_artifacts(tmp_path)
    diag_dir = run_dir / "diagnostics"

    high_pairs = pd.read_csv(diag_dir / "high_correlation_pairs.csv")
    if not high_pairs.empty:
        max_abs = high_pairs["maximum_absolute_correlation"].to_numpy()
        calc_max = (
            high_pairs[["absolute_pearson", "absolute_spearman"]].max(axis=1).to_numpy()
        )
        np.testing.assert_allclose(max_abs, calc_max, rtol=1e-5, equal_nan=True)

        tie_rows = high_pairs.loc[high_pairs["dominant_method"] == "tie"]
        if not tie_rows.empty:
            assert tie_rows["dominant_correlation"].isna().all()

        expected_frac = high_pairs["complete_pair_count"] / high_pairs["row_count"]
        np.testing.assert_allclose(
            high_pairs["complete_pair_fraction"],
            expected_frac,
            rtol=1e-5,
        )

    dictionary = pd.read_csv(diag_dir / "numeric_predictor_dictionary.csv")
    not_provided = dictionary.loc[dictionary["metadata_status"] == "not_provided"]
    if not not_provided.empty:
        assert (not_provided["derived_status"] == "not_evaluated").all()

    near_groups = pd.read_csv(diag_dir / "near_duplicate_groups.csv")
    if not near_groups.empty:
        unique_exact_sigs = near_groups.groupby("near_duplicate_group_id")[
            "exact_signature"
        ].nunique()
        assert (unique_exact_sigs >= 2).all()


def test_manifest_inventory_against_disk(tmp_path: Path) -> None:
    run_dir = generate_run_artifacts(tmp_path)
    diag_dir = run_dir / "diagnostics"
    manifest_path = diag_dir / "diagnostics_manifest.json"

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    artifacts = manifest.get("artifacts", [])
    assert len(artifacts) > 0

    for entry in artifacts:
        if entry["status"] == "written":
            rel_path = entry["path"]
            artifact_path = diag_dir / rel_path
            assert artifact_path.is_file(), f"File missing: {artifact_path}"
            assert artifact_path.stat().st_size == entry["byte_count"], (
                f"Size mismatch for {rel_path}"
            )
            assert sha256_file(artifact_path) == entry["sha256"], (
                f"SHA256 mismatch for {rel_path}"
            )

            if rel_path.endswith(".csv") and "matrix" not in entry["id"]:
                df = pd.read_csv(artifact_path)
                assert len(df) == entry["row_count"], (
                    f"Row count mismatch for {rel_path}"
                )


def test_json_strictness(tmp_path: Path) -> None:
    run_dir = generate_run_artifacts(tmp_path)
    manifest_path = run_dir / "diagnostics" / "diagnostics_manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")

    def raise_on_constant(val: str) -> Any:
        pytest.fail(f"Non-finite JSON constant detected: {val}")

    loaded = json.loads(manifest_text, parse_constant=raise_on_constant)
    assert isinstance(loaded, dict)


def test_standalone_and_run_schema_equivalence(tmp_path: Path) -> None:
    df = make_populated_dataset()
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    settings = StandaloneDiagnosticSettings()
    standalone_diag = calculate_standalone_diagnostics(
        frame=df,
        target="target",
        settings=settings,
    )

    standalone_out = tmp_path / "diag_standalone"
    write_standalone_diagnostics(
        diagnostics=standalone_diag,
        output_directory=standalone_out,
        settings=settings,
        data_path=csv_path,
        target="target",
    )

    run_dir = generate_run_artifacts(tmp_path)
    run_diag = run_dir / "diagnostics"

    s_dictionary = pd.read_csv(standalone_out / "numeric_predictor_dictionary.csv")
    r_dictionary = pd.read_csv(run_diag / "numeric_predictor_dictionary.csv")
    assert tuple(s_dictionary.columns) == tuple(r_dictionary.columns)

    s_high = pd.read_csv(standalone_out / "high_correlation_pairs.csv")
    r_high = pd.read_csv(run_diag / "high_correlation_pairs.csv")
    assert tuple(s_high.columns) == tuple(r_high.columns)

    with (standalone_out / "diagnostics_manifest.json").open(
        "r", encoding="utf-8"
    ) as f:
        s_manifest = json.load(f)
    with (run_diag / "diagnostics_manifest.json").open("r", encoding="utf-8") as f:
        r_manifest = json.load(f)

    assert s_manifest["schema_name"] == r_manifest["schema_name"]
    assert s_manifest["schema_version"] == r_manifest["schema_version"]
    assert set(s_manifest["analyses"].keys()) == set(r_manifest["analyses"].keys())
