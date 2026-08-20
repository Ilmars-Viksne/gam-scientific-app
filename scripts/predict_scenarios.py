from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: python predict_scenarios.py "
            "<model.joblib> <scenarios.csv> <results.csv>"
        )

    model_path = Path(sys.argv[1])
    scenario_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    if not scenario_path.is_file():
        raise FileNotFoundError(f"Scenario file does not exist: {scenario_path}")

    model = joblib.load(model_path)
    scenarios = pd.read_csv(scenario_path)

    if scenarios.empty:
        raise ValueError("The scenario file contains no observations.")

    transformer = model.named_steps["features"]
    classifier = model.named_steps["classifier"]

    required_columns = [str(value) for value in transformer.feature_names_in_]

    missing_columns = sorted(set(required_columns) - set(scenarios.columns))

    if missing_columns:
        raise ValueError(
            f"Scenario data is missing required predictors: {missing_columns}"
        )

    extra_columns = sorted(set(scenarios.columns) - set(required_columns))

    if extra_columns:
        print(
            f"Ignoring extra scenario columns: {extra_columns}",
            flush=True,
        )

    scenarios = scenarios.loc[
        :,
        required_columns,
    ].copy()

    probabilities = np.asarray(
        model.predict_proba(scenarios),
        dtype=np.float64,
    )

    predicted_classes = model.predict(scenarios)

    raw_scores = np.asarray(
        model.decision_function(scenarios),
        dtype=np.float64,
    )

    classes = [str(value) for value in classifier.classes_]

    if raw_scores.ndim == 1:
        score_matrix = np.column_stack(
            [
                np.zeros_like(raw_scores),
                raw_scores,
            ]
        )
    elif raw_scores.ndim == 2:
        score_matrix = raw_scores
    else:
        raise ValueError(
            "The classifier returned decision scores with an "
            f"unsupported shape: {raw_scores.shape}."
        )

    expected_shape = (
        len(scenarios),
        len(classes),
    )

    if probabilities.shape != expected_shape:
        raise ValueError(
            "The probability matrix has an unexpected shape. "
            f"Received {probabilities.shape}; "
            f"expected {expected_shape}."
        )

    if score_matrix.shape != expected_shape:
        raise ValueError(
            "The decision-score matrix has an unexpected shape. "
            f"Received {score_matrix.shape}; "
            f"expected {expected_shape}."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError("The predicted probabilities contain nonfinite values.")

    if not np.isfinite(score_matrix).all():
        raise ValueError("The decision scores contain nonfinite values.")

    results = scenarios.copy()

    results.insert(
        0,
        "scenario_id",
        np.arange(
            1,
            len(results) + 1,
        ),
    )

    for class_index, class_name in enumerate(classes):
        results[f"score_{class_name}"] = score_matrix[:, class_index]

    for class_index, class_name in enumerate(classes):
        results[f"probability_{class_name}"] = probabilities[:, class_index]

    results["predicted_class"] = predicted_classes

    results["maximum_probability"] = probabilities.max(axis=1)

    results["probability_sum"] = probabilities.sum(axis=1)

    maximum_probability_sum_error = float(
        np.max(np.abs(results["probability_sum"] - 1.0))
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print()
    print(f"Scenarios processed: {len(results)}")
    print(f"Target classes: {classes}")
    print(f"Maximum probability-sum error: {maximum_probability_sum_error:.17g}")
    print(f"Results written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
