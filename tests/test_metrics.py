import numpy as np
import pandas as pd
import pytest

from gam_app.evaluation import _fold_metrics


def test_specificity_binary() -> None:
    y_true = pd.Series(
        [
            "negative",
            "negative",
            "negative",
            "positive",
            "positive",
        ]
    )

    predictions = np.asarray(
        [
            "negative",
            "positive",
            "negative",
            "positive",
            "negative",
        ]
    )

    probabilities = np.asarray(
        [
            [0.9, 0.1],
            [0.4, 0.6],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.7, 0.3],
        ],
        dtype=np.float64,
    )

    classes = np.asarray(
        [
            "negative",
            "positive",
        ]
    )

    metrics = _fold_metrics(
        y_true,
        predictions,
        probabilities,
        classes,
    )

    # For the positive class:
    # TN = 2 and FP = 1.
    # Specificity = TN / (TN + FP) = 2 / 3.
    assert metrics["specificity_positive"] == pytest.approx(
        2.0 / 3.0,
    )

    # For the negative class:
    # TN = 1 and FP = 1.
    # Specificity = TN / (TN + FP) = 1 / 2.
    assert metrics["specificity_negative"] == pytest.approx(
        1.0 / 2.0,
    )

    assert metrics["macro_specificity"] == pytest.approx((2.0 / 3.0 + 1.0 / 2.0) / 2.0)
