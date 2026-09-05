from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gam_app.config import ExperimentConfig, FeatureConfig
from gam_app.diagnostics import calculate_correlation_analysis


def make_correlation_test_config(
    tmp_path: Path,
    frame: pd.DataFrame,
    features_dict: dict | None = None,
) -> ExperimentConfig:
    data_path = tmp_path / "data.csv"
    frame.to_csv(data_path, index=False)
    if features_dict is None:
        features_dict = {
            col: FeatureConfig(role="smooth" if col != "target" else "exclude")
            for col in frame.columns
            if col != "target"
        }
    cfg = ExperimentConfig(
        name="corr_test",
        data_path=data_path,
        target="target",
        row_id=None,
        features=features_dict,
    )
    return cfg


def test_high_correlation_pairs_detect_both_signs(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "positive": [2.0, 4.0, 6.0, 8.0, 10.0],
            "negative": [-1.0, -2.0, -3.0, -4.0, -5.0],
            "target": ["A", "A", "B", "B", "B"],
        }
    )

    config = make_correlation_test_config(tmp_path, frame)
    analysis = calculate_correlation_analysis(frame, config)

    pairs = {
        frozenset((row.left, row.right)): row
        for row in analysis.high_pairs.itertuples()
    }

    assert pairs[frozenset(("x", "positive"))].pearson == pytest.approx(1.0)
    assert pairs[frozenset(("x", "negative"))].pearson == pytest.approx(-1.0)


def test_high_pair_report_excludes_diagonal(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "positive": [2.0, 4.0, 6.0, 8.0, 10.0],
            "target": ["A", "A", "B", "B", "B"],
        }
    )
    config = make_correlation_test_config(tmp_path, frame)
    analysis = calculate_correlation_analysis(frame, config)

    assert not (analysis.high_pairs["left"] == analysis.high_pairs["right"]).any()


def test_high_pair_report_contains_unique_pairs(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "positive": [2.0, 4.0, 6.0, 8.0, 10.0],
            "target": ["A", "A", "B", "B", "B"],
        }
    )
    config = make_correlation_test_config(tmp_path, frame)
    analysis = calculate_correlation_analysis(frame, config)

    normalized = analysis.high_pairs.apply(
        lambda row: tuple(sorted((row["left"], row["right"]))),
        axis=1,
    )
    assert not normalized.duplicated().any()


def test_declared_derivation_is_reported(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "log_x": [0.0, 0.693147, 1.098612, 1.386294, 1.609438],
            "target": ["A", "A", "B", "B", "B"],
        }
    )
    features = {
        "x": FeatureConfig(role="smooth"),
        "log_x": FeatureConfig(
            role="smooth",
            derived="declared",
            derived_from=("x",),
            derivation="log(x)",
        ),
    }
    config = make_correlation_test_config(tmp_path, frame, features)
    analysis = calculate_correlation_analysis(frame, config)

    row = analysis.high_pairs.loc[
        (
            (analysis.high_pairs["left"] == "x")
            & (analysis.high_pairs["right"] == "log_x")
        )
        | (
            (analysis.high_pairs["left"] == "log_x")
            & (analysis.high_pairs["right"] == "x")
        )
    ].iloc[0]

    assert row["declared_derivation_relation"] == "yes"


def test_constant_predictor_does_not_create_high_pair(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "constant": [1.0, 1.0, 1.0, 1.0, 1.0],
            "target": ["A", "A", "B", "B", "B"],
        }
    )
    config = make_correlation_test_config(tmp_path, frame)
    analysis = calculate_correlation_analysis(frame, config)

    if not analysis.high_pairs.empty:
        assert "constant" not in set(analysis.high_pairs["left"])
        assert "constant" not in set(analysis.high_pairs["right"])


def test_high_pair_ordering_warning_before_review_and_rank_consecutive(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2_warning": [1.0, 2.0, 3.0, 4.0, 5.0],  # corr 1.0
            "x3_review": [1.0, 2.0, 3.0, 4.0, 4.5],  # corr ~0.8
            "target": ["A", "A", "B", "B", "B"],
        }
    )
    config = make_correlation_test_config(tmp_path, frame)
    analysis = calculate_correlation_analysis(frame, config)

    high_pairs = analysis.high_pairs
    assert not high_pairs.empty
    assert list(high_pairs["rank"]) == list(range(1, len(high_pairs) + 1))

    severities = list(high_pairs["severity"])
    if "warning" in severities and "review" in severities:
        first_review = severities.index("review")
        last_warning = len(severities) - 1 - severities[::-1].index("warning")
        assert last_warning < first_review


def test_dominant_method_tie_produces_none_dominant_correlation(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target": ["A", "A", "B", "B", "B"],
        }
    )
    config = make_correlation_test_config(tmp_path, frame)
    analysis = calculate_correlation_analysis(frame, config)

    high_pairs = analysis.high_pairs
    assert not high_pairs.empty
    row = high_pairs.iloc[0]
    assert row["dominant_method"] == "tie"
    assert pd.isna(row["dominant_correlation"])


def test_trigger_methods_uses_pipe_delimiter(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target": ["A", "A", "B", "B", "B"],
        }
    )
    config = make_correlation_test_config(tmp_path, frame)
    analysis = calculate_correlation_analysis(frame, config)

    row = analysis.high_pairs.iloc[0]
    assert row["trigger_methods"] == "pearson|spearman"
