from __future__ import annotations

from itertools import combinations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import ExperimentConfig, ModelConfig
from .transformers import GAMFeatureTransformer


def feature_groups(config: ExperimentConfig) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    smooth = tuple(name for name, spec in config.features.items() if spec.role == "smooth")
    linear = tuple(name for name, spec in config.features.items() if spec.role == "linear")
    categorical = tuple(
        name for name, spec in config.features.items() if spec.role == "categorical"
    )
    return smooth, linear, categorical


def interaction_pairs(config: ExperimentConfig, model: ModelConfig) -> tuple[tuple[str, str], ...]:
    smooth, _, _ = feature_groups(config)
    if model.interactions == "none":
        return ()
    if model.interactions == "explicit":
        return model.pairs
    return tuple(combinations(smooth, 2))


def build_pipeline(
    config: ExperimentConfig,
    model: ModelConfig,
    *,
    n_knots: int,
    degree: int,
    C: float,
    interaction_scale: float,
) -> Pipeline:
    smooth, linear, categorical = feature_groups(config)
    policies = tuple((name, spec.missing) for name, spec in config.features.items())
    transformer = GAMFeatureTransformer(
        smooth_features=smooth,
        linear_features=linear,
        categorical_features=categorical,
        interaction_pairs=interaction_pairs(config, model),
        missing_policies=policies,
        n_knots=n_knots,
        degree=degree,
        interaction_scale=interaction_scale,
    )
    classifier = LogisticRegression(
        solver="lbfgs",
        C=C,
        max_iter=20_000,
        random_state=config.validation.random_state,
    )
    return Pipeline([("features", transformer), ("classifier", classifier)])
