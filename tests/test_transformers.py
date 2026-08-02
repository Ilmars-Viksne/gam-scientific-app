from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from gam_app.transformers import GAMFeatureTransformer


def test_transformer_shape_and_names(
    multiclass_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    X = multiclass_frame[
        [
            "x1",
            "x2",
            "x3",
        ]
    ]

    transformer = GAMFeatureTransformer(
        smooth_features=(
            "x1",
            "x2",
        ),
        linear_features=(),
        categorical_features=("x3",),
        categorical_levels=(
            (
                "high",
                "low",
            ),
        ),
        interaction_pairs=(
            (
                "x1",
                "x2",
            ),
        ),
        n_knots=3,
        degree=2,
    )

    matrix = transformer.fit_transform(X)
    feature_names = transformer.get_feature_names_out()

    assert matrix.shape[0] == len(X)
    assert matrix.shape[1] == len(feature_names)
    assert np.isfinite(matrix).all()

    assert len(feature_names) == len(set(feature_names))

    assert any(name.startswith("main_spline__x1") for name in feature_names)
    assert any(name.startswith("main_spline__x2") for name in feature_names)
    assert any(name.startswith("main_categorical__x3") for name in feature_names)
    assert any(name.startswith("interaction__x1:x2") for name in feature_names)

    path = tmp_path / "transformer.joblib"
    joblib.dump(transformer, path)

    loaded = joblib.load(path)
    loaded_matrix = loaded.transform(X)

    np.testing.assert_allclose(
        matrix,
        loaded_matrix,
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_array_equal(
        feature_names,
        loaded.get_feature_names_out(),
    )


def test_invalid_interaction_rejected(
    multiclass_frame: pd.DataFrame,
) -> None:
    X = multiclass_frame[
        [
            "x1",
            "x2",
            "x3",
        ]
    ]

    transformer = GAMFeatureTransformer(
        smooth_features=("x1",),
        linear_features=("x2",),
        categorical_features=("x3",),
        categorical_levels=(
            (
                "high",
                "low",
            ),
        ),
        interaction_pairs=(
            (
                "x1",
                "x2",
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="configured as smooth",
    ):
        transformer.fit(X)


def test_configured_category_absent_from_fit_data_is_supported() -> None:
    training = pd.DataFrame(
        {
            "x1": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "x3": [
                "low",
                "high",
                "low",
                "high",
            ],
        }
    )

    validation = pd.DataFrame(
        {
            "x1": [5.0],
            "x3": ["medium"],
        }
    )

    transformer = GAMFeatureTransformer(
        smooth_features=("x1",),
        linear_features=(),
        categorical_features=("x3",),
        categorical_levels=(
            (
                "high",
                "low",
                "medium",
            ),
        ),
        interaction_pairs=(),
        n_knots=3,
        degree=2,
    )

    transformer.fit(training)
    matrix = transformer.transform(validation)

    assert matrix.shape[0] == 1
    assert matrix.shape[1] == len(transformer.get_feature_names_out())
    assert np.isfinite(matrix).all()

    assert transformer.categorical_encoder_ is not None

    np.testing.assert_array_equal(
        transformer.categorical_encoder_.categories_[0],
        np.asarray(
            [
                "high",
                "low",
                "medium",
            ],
            dtype=object,
        ),
    )


def test_undeclared_category_rejected() -> None:
    training = pd.DataFrame(
        {
            "x1": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "x3": [
                "low",
                "high",
                "low",
                "high",
            ],
        }
    )

    invalid = pd.DataFrame(
        {
            "x1": [5.0],
            "x3": ["unknown"],
        }
    )

    transformer = GAMFeatureTransformer(
        smooth_features=("x1",),
        linear_features=(),
        categorical_features=("x3",),
        categorical_levels=(
            (
                "high",
                "low",
                "medium",
            ),
        ),
        interaction_pairs=(),
        n_knots=3,
        degree=2,
    )

    transformer.fit(training)

    with pytest.raises(
        ValueError,
        match="configured category vocabulary",
    ):
        transformer.transform(invalid)
