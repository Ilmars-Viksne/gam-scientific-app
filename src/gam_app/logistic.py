from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

ClassificationType = Literal["binary", "multiclass"]


@dataclass(frozen=True, slots=True)
class ClassScoreParameters:
    """Uniform one-score-per-class representation."""

    classes: NDArray[np.object_]
    intercepts: NDArray[np.float64]
    coefficients: NDArray[np.float64]

    @property
    def class_count(self) -> int:
        return len(self.classes)

    @property
    def classification_type(self) -> ClassificationType:
        if self.class_count == 2:
            return "binary"
        return "multiclass"


def extract_class_score_parameters(
    classifier: LogisticRegression,
) -> ClassScoreParameters:
    """Return one score equation per target class."""

    classes = np.asarray(classifier.classes_, dtype=object)
    raw_intercepts = np.asarray(
        classifier.intercept_,
        dtype=np.float64,
    )
    raw_coefficients = np.asarray(
        classifier.coef_,
        dtype=np.float64,
    )

    if classes.ndim != 1 or len(classes) < 2:
        raise ValueError("The classifier must contain at least two classes.")

    if raw_coefficients.ndim != 2:
        raise ValueError("Classifier coefficients must be two-dimensional.")

    if len(classes) == 2:
        if raw_coefficients.shape[0] != 1:
            raise ValueError("A binary classifier must contain one coefficient row.")

        if raw_intercepts.shape != (1,):
            raise ValueError("A binary classifier must contain one intercept.")

        return ClassScoreParameters(
            classes=classes,
            intercepts=np.array(
                [0.0, raw_intercepts[0]],
                dtype=np.float64,
            ),
            coefficients=np.vstack(
                [
                    np.zeros_like(raw_coefficients[0]),
                    raw_coefficients[0],
                ]
            ),
        )

    if raw_coefficients.shape[0] != len(classes):
        raise ValueError("Coefficient rows do not match target classes.")

    if raw_intercepts.shape != (len(classes),):
        raise ValueError("Intercept count does not match target classes.")

    return ClassScoreParameters(
        classes=classes,
        intercepts=raw_intercepts,
        coefficients=raw_coefficients,
    )


def decision_scores_to_probabilities(
    scores: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Reconstruct binary or multiclass probabilities."""

    score_array = np.asarray(scores, dtype=np.float64)

    if score_array.ndim == 1:
        score_array = np.column_stack(
            [
                np.zeros_like(score_array),
                score_array,
            ]
        )
    elif score_array.ndim != 2:
        raise ValueError("Decision scores must be one- or two-dimensional.")

    shifted = score_array - score_array.max(
        axis=1,
        keepdims=True,
    )
    exponentials = np.exp(shifted)

    return exponentials / exponentials.sum(
        axis=1,
        keepdims=True,
    )
