from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from .exceptions import ConfigurationError

Role = Literal["smooth", "linear", "categorical", "exclude"]


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    role: Role
    missing: Literal["error", "median", "most_frequent"] = "error"
    categories: tuple[str, ...] = ()
    reference: str | None = None


@dataclass(frozen=True, slots=True)
class ModelConfig:
    id: str
    interactions: Literal["none", "all_eligible", "explicit"] = "none"
    pairs: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    outer_splits: int = 5
    outer_repeats: int = 3
    inner_splits: int = 5
    random_state: int = 42


@dataclass(frozen=True, slots=True)
class SearchConfig:
    n_knots: tuple[int, ...] = (3, 4, 5)
    degree: tuple[int, ...] = (2, 3)
    C: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    interaction_scale: tuple[float, ...] = (0.5, 1.0)


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    workers: int = 1
    checkpoint_unit: Literal["outer_fold"] = "outer_fold"
    stop_on_convergence_warning: bool = True


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    data_path: Path
    target: str
    row_id: str | None
    features: dict[str, FeatureConfig]
    models: tuple[ModelConfig, ...]
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    primary_metric: Literal["log_loss"] = "log_loss"
    schema_version: str = "1.0"

    def validate(self) -> None:
        if not self.name.strip():
            raise ConfigurationError("Experiment name cannot be empty.")
        if not self.target.strip():
            raise ConfigurationError("Target column cannot be empty.")
        if self.validation.outer_splits < 2 or self.validation.inner_splits < 2:
            raise ConfigurationError("Inner and outer splits must be at least 2.")
        if self.validation.outer_repeats < 1:
            raise ConfigurationError("outer_repeats must be at least 1.")
        active = {name for name, spec in self.features.items() if spec.role != "exclude"}
        if not active:
            raise ConfigurationError("At least one active predictor is required.")
        model_ids = [model.id for model in self.models]
        if len(model_ids) != len(set(model_ids)):
            raise ConfigurationError("Model IDs must be unique.")
        for model in self.models:
            if model.interactions == "explicit":
                for left, right in model.pairs:
                    if left not in active or right not in active or left == right:
                        raise ConfigurationError(f"Invalid interaction pair: {left}:{right}")
        if any(value < 2 for value in self.search.n_knots):
            raise ConfigurationError("Every n_knots value must be at least 2.")
        if any(value < 1 for value in self.search.degree):
            raise ConfigurationError("Every degree value must be at least 1.")
        if any(value <= 0 for value in self.search.C):
            raise ConfigurationError("Every C value must be positive.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_path"] = str(self.data_path)
        return payload


def _tuples(values: Any) -> tuple[Any, ...]:
    return tuple(values or ())


def load_config(path: Path) -> ExperimentConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent.resolve()
    data_path = Path(payload["data"]["path"])
    if not data_path.is_absolute():
        data_path = (base / data_path).resolve()
    features = {
        name: FeatureConfig(
            role=spec["role"],
            missing=spec.get("missing", "error"),
            categories=_tuples(spec.get("categories")),
            reference=spec.get("reference"),
        )
        for name, spec in payload["features"].items()
    }
    models = tuple(
        ModelConfig(
            id=item["id"],
            interactions=item.get("interactions", "none"),
            pairs=tuple(tuple(pair) for pair in item.get("pairs", ())),
        )
        for item in payload["models"]
    )
    validation = ValidationConfig(**payload.get("validation", {}))
    search_raw = payload.get("search", {})
    search = SearchConfig(
        n_knots=_tuples(search_raw.get("n_knots", SearchConfig.n_knots)),
        degree=_tuples(search_raw.get("degree", SearchConfig.degree)),
        C=_tuples(search_raw.get("C", SearchConfig.C)),
        interaction_scale=_tuples(
            search_raw.get("interaction_scale", SearchConfig.interaction_scale)
        ),
    )
    execution = ExecutionConfig(**payload.get("execution", {}))
    config = ExperimentConfig(
        schema_version=str(payload.get("schema_version", "1.0")),
        name=payload["experiment"]["name"],
        primary_metric=payload["experiment"].get("primary_metric", "log_loss"),
        data_path=data_path,
        target=payload["data"]["target"],
        row_id=payload["data"].get("row_id"),
        features=features,
        models=models,
        validation=validation,
        search=search,
        execution=execution,
    )
    config.validate()
    return config


def dump_config_dict(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "schema_version": config.schema_version,
        "experiment": {"name": config.name, "primary_metric": config.primary_metric},
        "data": {
            "path": str(config.data_path),
            "target": config.target,
            "row_id": config.row_id,
        },
        "features": {
            name: {
                "role": spec.role,
                "missing": spec.missing,
                **({"categories": list(spec.categories)} if spec.categories else {}),
                **({"reference": spec.reference} if spec.reference else {}),
            }
            for name, spec in config.features.items()
        },
        "models": [
            {
                "id": model.id,
                "interactions": model.interactions,
                **({"pairs": [list(pair) for pair in model.pairs]} if model.pairs else {}),
            }
            for model in config.models
        ],
        "validation": asdict(config.validation),
        "search": asdict(config.search),
        "execution": asdict(config.execution),
    }
