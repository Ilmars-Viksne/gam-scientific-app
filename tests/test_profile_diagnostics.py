from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from gam_app.cli import command_profile
from gam_app.diagnostics import StandaloneDiagnosticSettings


def test_command_profile_writes_all_eleven_artifacts(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    output_directory = tmp_path / "profile_out"

    frame = pd.DataFrame(
        {
            "x": [1.0, 1.0, 3.0, 4.0, 5.0],
            "two_x": [2.0, 2.0, 6.0, 8.0, 10.0],
            "near_x": [1.000000000, 1.000000001, 3.000000000, 4.000000000, 5.000000000],
            "category": ["a", "a", "b", "b", "c"],
            "target": ["A", "A", "B", "B", "C"],
        }
    )
    frame.to_csv(data_path, index=False)

    command_profile(
        Namespace(
            data=data_path,
            target="target",
            output=output_directory,
            review_correlation=0.75,
            warn_correlation=0.90,
            near_duplicate_decimals=8,
        )
    )

    expected_files = {
        "profile.json",
        "columns.csv",
        "diagnostics_manifest.json",
        "correlation_pearson.csv",
        "correlation_spearman.csv",
        "high_correlation_pairs.csv",
        "numeric_predictor_dictionary.csv",
        "suspected_derived_relations.csv",
        "exact_duplicate_groups.csv",
        "near_duplicate_groups.csv",
        "conflicting_duplicate_targets.csv",
    }

    actual_files = {path.name for path in output_directory.iterdir() if path.is_file()}

    assert expected_files == actual_files

    high_pairs = pd.read_csv(output_directory / "high_correlation_pairs.csv")
    assert not high_pairs.empty
    matching = high_pairs.loc[
        (high_pairs["left"] == "two_x") & (high_pairs["right"] == "x")
    ]
    if len(matching) == 0:
        matching = high_pairs.loc[
            (high_pairs["left"] == "x") & (high_pairs["right"] == "two_x")
        ]
    assert len(matching) == 1
    assert matching.iloc[0]["severity"] == "warning"

    near_dups = pd.read_csv(output_directory / "near_duplicate_groups.csv")
    assert not near_dups.empty


def test_standalone_diagnostic_settings_validation() -> None:
    with pytest.raises(
        ValueError, match="correlation review threshold must be between 0 and 1"
    ):
        StandaloneDiagnosticSettings(correlation_review_threshold=1.5).validate()

    with pytest.raises(
        ValueError, match="correlation warning threshold must be between 0 and 1"
    ):
        StandaloneDiagnosticSettings(correlation_warning_threshold=-0.1).validate()

    with pytest.raises(ValueError, match="cannot be smaller than the review threshold"):
        StandaloneDiagnosticSettings(
            correlation_review_threshold=0.8,
            correlation_warning_threshold=0.7,
        ).validate()

    with pytest.raises(ValueError, match="near_duplicate_decimals cannot be negative"):
        StandaloneDiagnosticSettings(near_duplicate_decimals=-1).validate()
