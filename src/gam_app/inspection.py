from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def inspect_model(path: Path, output: Path, reference_class: str = "O") -> None:
    model = joblib.load(path)
    transformer = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    names = transformer.get_feature_names_out()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    equation_lines = []
    for class_index, class_name in enumerate(classifier.classes_):
        equation_lines.append(f"eta_{class_name}(x) = {classifier.intercept_[class_index]:.12g}")
        for name, coefficient in zip(names, classifier.coef_[class_index], strict=True):
            rows.append({"class": class_name, "component": name, "coefficient": float(coefficient)})
            sign = "+" if coefficient >= 0 else "-"
            equation_lines.append(f"    {sign} {abs(coefficient):.12g} * {name}")
        equation_lines.append("")
    pd.DataFrame(rows).to_csv(output / "components.csv", index=False)
    (output / "equations.txt").write_text("\n".join(equation_lines), encoding="utf-8")
    classes = list(classifier.classes_)
    if reference_class not in classes:
        reference_class = classes[-1]
    reference = classes.index(reference_class)
    contrasts = []
    for index, class_name in enumerate(classes):
        if index == reference:
            continue
        contrasts.append({"comparison": f"{class_name}_versus_{reference_class}", "component": "intercept", "coefficient": classifier.intercept_[index] - classifier.intercept_[reference]})
        differences = classifier.coef_[index] - classifier.coef_[reference]
        contrasts.extend({"comparison": f"{class_name}_versus_{reference_class}", "component": name, "coefficient": value} for name, value in zip(names, differences, strict=True))
    pd.DataFrame(contrasts).to_csv(output / "reference_equations.csv", index=False)


def verify_link(model_path: Path, data: pd.DataFrame, output: Path) -> float:
    model = joblib.load(model_path)
    scores = model.decision_function(data)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    shifted = scores - scores.max(axis=1, keepdims=True)
    manual = np.exp(shifted)
    manual /= manual.sum(axis=1, keepdims=True)
    actual = model.predict_proba(data)
    error = float(np.max(np.abs(manual - actual)))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores, columns=[f"score_{c}" for c in model.named_steps["classifier"].classes_]).to_csv(output / "scores.csv", index=False)
    pd.DataFrame(actual, columns=[f"probability_{c}" for c in model.named_steps["classifier"].classes_]).to_csv(output / "probabilities.csv", index=False)
    (output / "verification.txt").write_text(f"maximum_softmax_error={error:.17g}\n", encoding="utf-8")
    return error
