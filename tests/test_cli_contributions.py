from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gam_app.cli import (
    build_parser,
    command_contributions,
)
from gam_app.config import (
    ExecutionConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    SearchConfig,
    ValidationConfig,
)
from gam_app.models import build_pipeline


def test_contributions_parser_registration() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "contributions",
            "--model",
            "model.joblib",
            "--input",
            "scenarios.csv",
            "--output",
            "contributions.csv",
        ]
    )

    assert args.func is command_contributions
    assert args.model == Path("model.joblib")
    assert args.input == Path("scenarios.csv")
    assert args.output == Path("contributions.csv")
    assert args.top == 10


def test_contributions_parser_accepts_top() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "contributions",
            "--model",
            "model.joblib",
            "--input",
            "scenarios.csv",
            "--output",
            "contributions.csv",
            "--top",
            "25",
        ]
    )

    assert args.top == 25


def test_command_contributions_execution(
    multiclass_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    cfg = ExperimentConfig(
        name="contributions-test",
        data_path=tmp_path / "unused.csv",
        target="target",
        row_id=None,
        features={
            "x1": FeatureConfig(role="smooth"),
            "x2": FeatureConfig(role="smooth"),
            "x3": FeatureConfig(role="categorical", categories=("high", "low")),
        },
        models=(ModelConfig(id="main"),),
        validation=ValidationConfig(
            outer_splits=2,
            outer_repeats=1,
            inner_splits=2,
            random_state=42,
        ),
        search=SearchConfig(
            n_knots=(3,),
            degree=(2,),
            C=(1.0,),
            interaction_scale=(1.0,),
        ),
        execution=ExecutionConfig(workers=1),
    )

    model = build_pipeline(
        cfg,
        cfg.models[0],
        n_knots=3,
        degree=2,
        C=1.0,
        interaction_scale=1.0,
    )

    predictor_columns = ["x1", "x2", "x3"]
    X = multiclass_frame[predictor_columns]
    y = multiclass_frame["target"]
    model.fit(X, y)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    scenarios = (
        multiclass_frame[predictor_columns]
        .iloc[[0, 10, 20]]
        .reset_index(drop=True)
        .copy()
    )
    input_path = tmp_path / "scenarios.csv"
    scenarios.to_csv(input_path, index=False)

    output_path = tmp_path / "results" / "contributions.csv"
    summary_path = (
        output_path.parent / f"{output_path.stem}_score_summary{output_path.suffix}"
    )

    args = Namespace(
        model=model_path,
        input=input_path,
        output=output_path,
        top=10,
    )

    command_contributions(args)

    assert output_path.is_file()
    assert summary_path.is_file()

    contributions = pd.read_csv(output_path)
    summary = pd.read_csv(summary_path)

    expected_columns = {
        "scenario_id",
        "observation_index",
        "class",
        "predicted_class",
        "class_probability",
        "class_score",
        "component",
        "component_type",
        "component_group",
        "transformed_value",
        "coefficient",
        "contribution",
        "absolute_contribution",
    }
    assert expected_columns.issubset(contributions.columns)

    np.testing.assert_allclose(
        summary["reconstructed_score"],
        summary["expected_score"],
        rtol=1e-12,
        atol=1e-12,
    )

    scenario_count = len(scenarios)
    class_count = len(model.classes_)
    intercepts = contributions.loc[contributions["component"] == "intercept"]
    expected_intercept_rows = scenario_count * class_count
    assert len(intercepts) == expected_intercept_rows
