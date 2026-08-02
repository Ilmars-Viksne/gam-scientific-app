from __future__ import annotations

from itertools import combinations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import ExperimentConfig, ModelConfig
from .transformers import GAMFeatureTransformer


def feature_groups(
    config: ExperimentConfig,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    smooth = tuple(
        name for name, spec in config.features.items() if spec.role == "smooth"
    )
    linear = tuple(
        name for name, spec in config.features.items() if spec.role == "linear"
    )
    categorical = tuple(
        name for name, spec in config.features.items() if spec.role == "categorical"
    )
    return smooth, linear, categorical


def interaction_pairs(
    config: ExperimentConfig, model: ModelConfig
) -> tuple[tuple[str, str], ...]:
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
    smooth_features = tuple(
        feature_name
        for feature_name, feature_config in config.features.items()
        if feature_config.role == "smooth"
    )

    linear_features = tuple(
        feature_name
        for feature_name, feature_config in config.features.items()
        if feature_config.role == "linear"
    )

    categorical_features = tuple(
        feature_name
        for feature_name, feature_config in config.features.items()
        if feature_config.role == "categorical"
    )

    categorical_levels = tuple(
        tuple(str(category) for category in config.features[feature_name].categories)
        for feature_name in categorical_features
    )

    missing_policies = tuple(
        (
            feature_name,
            feature_config.missing,
        )
        for feature_name, feature_config in config.features.items()
        if feature_config.role != "exclude"
    )

    pairs = interaction_pairs(
        config,
        model,
    )

    transformer = GAMFeatureTransformer(
        smooth_features=smooth_features,
        linear_features=linear_features,
        categorical_features=categorical_features,
        categorical_levels=categorical_levels,
        interaction_pairs=pairs,
        missing_policies=missing_policies,
        n_knots=n_knots,
        degree=degree,
        interaction_scale=interaction_scale,
    )

    classifier = LogisticRegression(
        C=C,
        max_iter=5000,
    )

    return Pipeline(
        steps=[
            (
                "features",
                transformer,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )
