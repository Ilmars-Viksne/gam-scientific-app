from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gam_app.cli import command_predict
from gam_app.config import (
    ExecutionConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    SearchConfig,
    ValidationConfig,
)
from gam_app.models import build_pipeline


def config(
    tmp_path: Path,
) -> ExperimentConfig:
    return ExperimentConfig(
        name="prediction-test",
        data_path=tmp_path / "unused.csv",
        target="target",
        row_id=None,
        features={
            "x1": FeatureConfig(
                role="smooth",
            ),
            "x2": FeatureConfig(
                role="smooth",
            ),
            "x3": FeatureConfig(
                role="categorical",
                categories=(
                    "high",
                    "low",
                ),
            ),
        },
        models=(
            ModelConfig(
                id="main",
            ),
        ),
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
        execution=ExecutionConfig(
            workers=1,
        ),
    )


def fit_and_save_model(
    multiclass_frame: pd.DataFrame,
    tmp_path: Path,
) -> tuple[Path, object]:
    cfg = config(tmp_path)

    model = build_pipeline(
        cfg,
        cfg.models[0],
        n_knots=3,
        degree=2,
        C=1.0,
        interaction_scale=1.0,
    )

    X = multiclass_frame[
        [
            "x1",
            "x2",
            "x3",
        ]
    ]

    y = multiclass_frame["target"]

    model.fit(X, y)

    model_path = tmp_path / "model.joblib"

    joblib.dump(
        model,
        model_path,
    )

    return model_path, model


def test_command_predict_writes_detailed_results(
    multiclass_frame: pd.DataFrame,
    tmp_path: Path,
    capsys,
) -> None:
    model_path, model = fit_and_save_model(
        multiclass_frame,
        tmp_path,
    )

    predictor_columns = [
        "x1",
        "x2",
        "x3",
    ]

    scenarios = (
        multiclass_frame[predictor_columns]
        .iloc[
            [
                0,
                20,
                40,
                60,
            ]
        ]
        .reset_index(drop=True)
        .copy()
    )

    scenarios["unused_note"] = [
        "first",
        "second",
        "third",
        "fourth",
    ]

    input_path = tmp_path / "scenarios.csv"

    output_path = tmp_path / "nested" / "predictions" / "results.csv"

    scenarios.to_csv(
        input_path,
        index=False,
    )

    args = Namespace(
        model=model_path,
        input=input_path,
        output=output_path,
    )

    command_predict(args)

    captured = capsys.readouterr()

    assert output_path.is_file()

    results = pd.read_csv(output_path)

    classifier = model.named_steps["classifier"]

    classes = [str(value) for value in classifier.classes_]

    required_predictors = {
        "x1",
        "x2",
        "x3",
    }

    assert required_predictors.issubset(results.columns)

    # Extra input metadata should be preserved in the output,
    # although it must not be passed to the fitted model.
    assert "unused_note" in results.columns

    pd.testing.assert_series_equal(
        results["unused_note"],
        scenarios["unused_note"],
        check_names=True,
    )

    expected_diagnostic_columns = {
        "scenario_id",
        "predicted_class",
        "maximum_probability",
        "second_highest_class",
        "second_highest_probability",
        "confidence_margin",
        "prediction_entropy",
        "normalized_entropy",
        "probability_sum",
    }

    assert expected_diagnostic_columns.issubset(results.columns)

    for class_name in classes:
        assert f"score_{class_name}" in results.columns

        assert f"probability_{class_name}" in results.columns

    assert len(results) == len(scenarios)

    np.testing.assert_array_equal(
        results["scenario_id"].to_numpy(),
        np.arange(
            1,
            len(results) + 1,
        ),
    )

    probability_columns = [f"probability_{class_name}" for class_name in classes]

    score_columns = [f"score_{class_name}" for class_name in classes]

    probabilities = results[probability_columns].to_numpy(dtype=np.float64)

    scores = results[score_columns].to_numpy(dtype=np.float64)

    assert np.isfinite(probabilities).all()

    assert np.isfinite(scores).all()

    numeric_diagnostic_columns = [
        "maximum_probability",
        "second_highest_probability",
        "confidence_margin",
        "prediction_entropy",
        "normalized_entropy",
        "probability_sum",
    ]

    assert np.isfinite(
        results[numeric_diagnostic_columns].to_numpy(dtype=np.float64)
    ).all()

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        results["probability_sum"].to_numpy(dtype=np.float64),
        1.0,
        rtol=0.0,
        atol=1e-12,
    )

    model_scenarios = scenarios.loc[
        :,
        predictor_columns,
    ]

    expected_probabilities = np.asarray(
        model.predict_proba(model_scenarios),
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        probabilities,
        expected_probabilities,
        rtol=1e-12,
        atol=1e-12,
    )

    expected_predictions = np.asarray(
        model.predict(model_scenarios),
        dtype=str,
    )

    np.testing.assert_array_equal(
        results["predicted_class"].astype(str).to_numpy(),
        expected_predictions,
    )

    expected_scores = np.asarray(
        model.decision_function(model_scenarios),
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        scores,
        expected_scores,
        rtol=1e-12,
        atol=1e-12,
    )

    probability_order = np.argsort(
        expected_probabilities,
        axis=1,
    )

    expected_top_indices = probability_order[:, -1]

    expected_second_indices = probability_order[:, -2]

    row_indices = np.arange(len(expected_probabilities))

    expected_maximum = expected_probabilities[
        row_indices,
        expected_top_indices,
    ]

    expected_second = expected_probabilities[
        row_indices,
        expected_second_indices,
    ]

    expected_second_classes = np.asarray(
        classes,
        dtype=object,
    )[expected_second_indices]

    np.testing.assert_allclose(
        results["maximum_probability"].to_numpy(dtype=np.float64),
        expected_maximum,
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_array_equal(
        results["second_highest_class"].astype(str).to_numpy(),
        expected_second_classes.astype(str),
    )

    np.testing.assert_allclose(
        results["second_highest_probability"].to_numpy(dtype=np.float64),
        expected_second,
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        results["confidence_margin"].to_numpy(dtype=np.float64),
        expected_maximum - expected_second,
        rtol=1e-12,
        atol=1e-12,
    )

    safe_probabilities = np.clip(
        expected_probabilities,
        np.finfo(np.float64).tiny,
        1.0,
    )

    expected_entropy = -np.sum(
        safe_probabilities * np.log(safe_probabilities),
        axis=1,
    )

    expected_normalized_entropy = expected_entropy / np.log(len(classes))

    np.testing.assert_allclose(
        results["prediction_entropy"].to_numpy(dtype=np.float64),
        expected_entropy,
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        results["normalized_entropy"].to_numpy(dtype=np.float64),
        expected_normalized_entropy,
        rtol=1e-12,
        atol=1e-12,
    )

    normalized_entropy = results["normalized_entropy"].to_numpy(dtype=np.float64)

    assert np.all(normalized_entropy >= -1e-12)

    assert np.all(normalized_entropy <= 1.0 + 1e-12)

    assert (
        "Extra input columns will be preserved in the output "
        "but excluded from model prediction: ['unused_note']" in captured.out
    )

    assert "Prediction results" in captured.out

    assert "Prediction summary" in captured.out

    assert f"Scenarios processed: {len(results)}" in captured.out

    assert f"Predictor count: {len(predictor_columns)}" in captured.out

    assert "Extra columns preserved: 1" in captured.out

    assert f"Target class count: {len(classes)}" in captured.out

    assert f"Target classes: {classes}" in captured.out

    assert "Maximum probability-sum error:" in captured.out

    assert "Predicted class counts" in captured.out

    assert str(model_path.resolve()) in captured.out


def test_command_predict_supports_binary_scores(
    binary_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    cfg = config(tmp_path)

    model = build_pipeline(
        cfg,
        cfg.models[0],
        n_knots=3,
        degree=2,
        C=1.0,
        interaction_scale=1.0,
    )

    X = binary_frame[
        [
            "x1",
            "x2",
            "x3",
        ]
    ]

    y = binary_frame["target"]

    model.fit(X, y)

    model_path = tmp_path / "binary_model.joblib"

    joblib.dump(
        model,
        model_path,
    )

    scenarios = (
        X.iloc[
            [
                0,
                10,
                30,
            ]
        ]
        .reset_index(drop=True)
        .copy()
    )

    input_path = tmp_path / "binary_scenarios.csv"

    output_path = tmp_path / "binary_results.csv"

    scenarios.to_csv(
        input_path,
        index=False,
    )

    command_predict(
        Namespace(
            model=model_path,
            input=input_path,
            output=output_path,
        )
    )

    results = pd.read_csv(output_path)

    classes = [str(value) for value in model.named_steps["classifier"].classes_]

    assert len(classes) == 2

    score_columns = [f"score_{class_name}" for class_name in classes]

    probability_columns = [f"probability_{class_name}" for class_name in classes]

    scores = results[score_columns].to_numpy(dtype=np.float64)

    probabilities = results[probability_columns].to_numpy(dtype=np.float64)

    decision_scores = model.decision_function(scenarios)

    np.testing.assert_allclose(
        scores[:, 0],
        0.0,
        rtol=0.0,
        atol=0.0,
    )

    np.testing.assert_allclose(
        scores[:, 1],
        decision_scores,
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        probabilities,
        model.predict_proba(scenarios),
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=1e-12,
    )
