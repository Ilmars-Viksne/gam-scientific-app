import numpy as np

from gam_app.config import (
    ExecutionConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    SearchConfig,
    ValidationConfig,
)
from gam_app.models import build_pipeline


def config(tmp_path):
    return ExperimentConfig(
        name="test",
        data_path=tmp_path / "unused.csv",
        target="target",
        row_id=None,
        features={
            "x1": FeatureConfig("smooth"),
            "x2": FeatureConfig("smooth"),
            "x3": FeatureConfig("categorical"),
        },
        models=(ModelConfig("main"),),
        validation=ValidationConfig(2, 1, 2, 42),
        search=SearchConfig((3,), (2,), (1.0,), (1.0,)),
        execution=ExecutionConfig(),
    )


def test_probabilities_and_softmax(multiclass_frame, tmp_path):
    cfg = config(tmp_path)
    model = build_pipeline(cfg, cfg.models[0], n_knots=3, degree=2, C=1.0, interaction_scale=1.0)
    X = multiclass_frame[["x1", "x2", "x3"]]
    y = multiclass_frame.target
    model.fit(X, y)
    probabilities = model.predict_proba(X)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    scores = model.decision_function(X)
    shifted = scores - scores.max(axis=1, keepdims=True)
    manual = np.exp(shifted)
    manual /= manual.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(probabilities, manual, atol=1e-12)
