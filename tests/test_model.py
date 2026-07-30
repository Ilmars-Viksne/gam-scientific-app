import numpy as np

from gam_app.config import (
    ExecutionConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    SearchConfig,
    ValidationConfig,
)
from gam_app.logistic import (
    decision_scores_to_probabilities,
    extract_class_score_parameters,
)
from gam_app.models import build_pipeline


def config(tmp_path) -> ExperimentConfig:
    return ExperimentConfig(
        name="test",
        data_path=tmp_path / "unused.csv",
        target="target",
        row_id=None,
        features={
            "x1": FeatureConfig("smooth"),
            "x2": FeatureConfig("smooth"),
            "x3": FeatureConfig(
                "categorical",
                categories=("high", "low"),
            ),
        },
        models=(ModelConfig("main"),),
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
        execution=ExecutionConfig(),
    )


def test_multiclass_probabilities_and_softmax(
    multiclass_frame,
    tmp_path,
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

    X = multiclass_frame[["x1", "x2", "x3"]]
    y = multiclass_frame["target"]

    model.fit(X, y)

    actual = model.predict_proba(X)
    reconstructed = decision_scores_to_probabilities(model.decision_function(X))

    assert actual.shape == (
        len(X),
        y.nunique(),
    )

    np.testing.assert_allclose(
        actual.sum(axis=1),
        1.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        reconstructed,
        actual,
        atol=1e-12,
    )


def test_multiclass_class_score_parameters(
    multiclass_frame,
    tmp_path,
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

    X = multiclass_frame[["x1", "x2", "x3"]]
    y = multiclass_frame["target"]

    model.fit(X, y)

    classifier = model.named_steps["classifier"]
    parameters = extract_class_score_parameters(classifier)

    expected_class_count = y.nunique()
    transformed_feature_count = model.named_steps["features"].transform(X).shape[1]

    assert parameters.class_count == expected_class_count
    assert parameters.classification_type == "multiclass"

    assert parameters.intercepts.shape == (expected_class_count,)

    assert parameters.coefficients.shape == (
        expected_class_count,
        transformed_feature_count,
    )

    np.testing.assert_array_equal(
        parameters.classes,
        classifier.classes_,
    )


def test_binary_probabilities_and_logistic_link(
    binary_frame,
    tmp_path,
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

    X = binary_frame[["x1", "x2", "x3"]]
    y = binary_frame["target"]

    model.fit(X, y)

    decision_scores = model.decision_function(X)
    actual = model.predict_proba(X)

    reconstructed = decision_scores_to_probabilities(decision_scores)

    assert decision_scores.ndim == 1
    assert actual.shape == (len(X), 2)
    assert reconstructed.shape == actual.shape

    np.testing.assert_allclose(
        actual.sum(axis=1),
        1.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        reconstructed,
        actual,
        atol=1e-12,
    )


def test_binary_class_score_parameters(
    binary_frame,
    tmp_path,
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

    X = binary_frame[["x1", "x2", "x3"]]
    y = binary_frame["target"]

    model.fit(X, y)

    classifier = model.named_steps["classifier"]
    parameters = extract_class_score_parameters(classifier)

    transformed_feature_count = model.named_steps["features"].transform(X).shape[1]

    assert parameters.class_count == 2
    assert parameters.classification_type == "binary"

    assert parameters.intercepts.shape == (2,)

    assert parameters.coefficients.shape == (
        2,
        transformed_feature_count,
    )

    np.testing.assert_array_equal(
        parameters.classes,
        classifier.classes_,
    )

    # The first class is the zero-score reference equation.
    assert parameters.intercepts[0] == 0.0

    np.testing.assert_allclose(
        parameters.coefficients[0],
        0.0,
        atol=0.0,
    )

    # The second class contains the fitted scikit-learn binary equation.
    np.testing.assert_allclose(
        parameters.intercepts[1],
        classifier.intercept_[0],
        atol=0.0,
    )

    np.testing.assert_allclose(
        parameters.coefficients[1],
        classifier.coef_[0],
        atol=0.0,
    )


def test_binary_score_difference_matches_decision_function(
    binary_frame,
    tmp_path,
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

    X = binary_frame[["x1", "x2", "x3"]]
    y = binary_frame["target"]

    model.fit(X, y)

    transformed = model.named_steps["features"].transform(X)
    classifier = model.named_steps["classifier"]
    parameters = extract_class_score_parameters(classifier)

    class_scores = transformed @ parameters.coefficients.T + parameters.intercepts

    score_difference = class_scores[:, 1] - class_scores[:, 0]

    np.testing.assert_allclose(
        score_difference,
        model.decision_function(X),
        atol=1e-12,
    )
