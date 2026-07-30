from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .logistic import (
    decision_scores_to_probabilities,
    extract_class_score_parameters,
)


def inspect_model(
    path: Path,
    output: Path,
    reference_class: str | None = None,
) -> None:
    model = joblib.load(path)

    transformer = model.named_steps["features"]
    classifier = model.named_steps["classifier"]

    feature_names = transformer.get_feature_names_out()
    parameters = extract_class_score_parameters(classifier)

    classes = [str(value) for value in parameters.classes]

    if reference_class is None:
        reference_class = classes[0]

    if reference_class not in classes:
        raise ValueError(
            f"Reference class {reference_class!r} is unavailable. "
            f"Available classes: {classes}"
        )

    output.mkdir(parents=True, exist_ok=True)

    component_rows: list[dict[str, object]] = []
    equation_lines: list[str] = []

    for class_index, class_name in enumerate(classes):
        intercept = float(parameters.intercepts[class_index])

        component_rows.append(
            {
                "class": class_name,
                "component": "intercept",
                "coefficient": intercept,
            }
        )

        equation_lines.append(f"eta_{class_name}(x) = {intercept:.12g}")

        for feature_name, coefficient in zip(
            feature_names,
            parameters.coefficients[class_index],
            strict=True,
        ):
            coefficient = float(coefficient)

            component_rows.append(
                {
                    "class": class_name,
                    "component": str(feature_name),
                    "coefficient": coefficient,
                }
            )

            if coefficient == 0.0:
                continue

            sign = "+" if coefficient >= 0.0 else "-"

            equation_lines.append(
                f"    {sign} {abs(coefficient):.12g} * {feature_name}"
            )

        equation_lines.append("")

    pd.DataFrame(component_rows).to_csv(
        output / "components.csv",
        index=False,
    )

    (output / "equations.txt").write_text(
        "\n".join(equation_lines),
        encoding="utf-8",
    )

    reference_index = classes.index(reference_class)
    contrast_rows: list[dict[str, object]] = []

    for class_index, class_name in enumerate(classes):
        if class_index == reference_index:
            continue

        comparison = f"{class_name}_versus_{reference_class}"

        contrast_rows.append(
            {
                "comparison": comparison,
                "component": "intercept",
                "coefficient": float(
                    parameters.intercepts[class_index]
                    - parameters.intercepts[reference_index]
                ),
            }
        )

        differences = (
            parameters.coefficients[class_index]
            - parameters.coefficients[reference_index]
        )

        contrast_rows.extend(
            {
                "comparison": comparison,
                "component": str(feature_name),
                "coefficient": float(coefficient),
            }
            for feature_name, coefficient in zip(
                feature_names,
                differences,
                strict=True,
            )
        )

    pd.DataFrame(contrast_rows).to_csv(
        output / "reference_equations.csv",
        index=False,
    )


def verify_link(
    model_path: Path,
    data: pd.DataFrame,
    output: Path,
) -> float:
    model = joblib.load(model_path)

    decision_scores = model.decision_function(data)
    reconstructed = decision_scores_to_probabilities(decision_scores)
    actual = model.predict_proba(data)

    if reconstructed.shape != actual.shape:
        raise ValueError(
            "Reconstructed probabilities and predict_proba output "
            "have different shapes."
        )

    error = float(np.max(np.abs(reconstructed - actual)))

    classifier = model.named_steps["classifier"]
    classes = [str(value) for value in classifier.classes_]

    raw_scores = np.asarray(
        decision_scores,
        dtype=np.float64,
    )

    if raw_scores.ndim == 1:
        score_matrix = np.column_stack(
            [
                np.zeros_like(raw_scores),
                raw_scores,
            ]
        )
    else:
        score_matrix = raw_scores

    output.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        score_matrix,
        columns=[f"score_{class_name}" for class_name in classes],
    ).to_csv(
        output / "scores.csv",
        index=False,
    )

    pd.DataFrame(
        actual,
        columns=[f"probability_{class_name}" for class_name in classes],
    ).to_csv(
        output / "probabilities.csv",
        index=False,
    )

    pd.DataFrame(
        reconstructed,
        columns=[f"reconstructed_probability_{class_name}" for class_name in classes],
    ).to_csv(
        output / "reconstructed_probabilities.csv",
        index=False,
    )

    classification_type = "binary" if len(classes) == 2 else "multiclass"

    (output / "verification.txt").write_text(
        "\n".join(
            [
                f"classification_type={classification_type}",
                f"class_count={len(classes)}",
                f"classes={classes}",
                f"maximum_probability_error={error:.17g}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return error
