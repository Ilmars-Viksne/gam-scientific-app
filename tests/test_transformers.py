import joblib
import numpy as np

from gam_app.transformers import GAMFeatureTransformer


def test_transformer_shape_and_names(multiclass_frame, tmp_path):
    X = multiclass_frame[["x1", "x2", "x3"]]
    transformer = GAMFeatureTransformer(
        smooth_features=("x1", "x2"),
        linear_features=(),
        categorical_features=("x3",),
        interaction_pairs=(("x1", "x2"),),
        n_knots=3,
        degree=2,
    ).fit(X)
    matrix = transformer.transform(X)
    assert matrix.shape[0] == len(X)
    assert matrix.shape[1] == len(transformer.get_feature_names_out())
    assert np.isfinite(matrix).all()
    path = tmp_path / "transformer.joblib"
    joblib.dump(transformer, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(matrix, loaded.transform(X))


def test_invalid_interaction_rejected(multiclass_frame):
    X = multiclass_frame[["x1", "x2", "x3"]]
    transformer = GAMFeatureTransformer(
        smooth_features=("x1",),
        linear_features=("x2",),
        categorical_features=("x3",),
        interaction_pairs=(("x1", "x2"),),
    )
    try:
        transformer.fit(X)
    except ValueError as error:
        assert "smooth features" in str(error)
    else:
        raise AssertionError("Invalid interaction was accepted.")
