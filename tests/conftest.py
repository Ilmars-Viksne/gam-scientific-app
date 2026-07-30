import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def multiclass_frame():
    rng = np.random.default_rng(7)
    n = 90

    x1 = rng.normal(size=n)
    x2 = rng.uniform(-1, 1, size=n)
    x3 = rng.choice(["low", "high"], size=n)

    score = x1 + 0.5 * x2

    y = np.where(
        score > 0.6,
        "C",
        np.where(score < -0.6, "A", "B"),
    )

    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "target": y,
        }
    )


@pytest.fixture
def binary_frame(multiclass_frame):
    frame = multiclass_frame.copy()

    frame["target"] = np.where(
        frame["target"] == "A",
        "M",
        "O",
    )

    return frame
