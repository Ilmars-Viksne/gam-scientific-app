from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from .exceptions import ConfigurationError

MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

Role = Literal["smooth", "linear", "categorical", "exclude"]
ValidationStrategy = Literal["stratified", "stratified_group", "time"]
DerivedKind = Literal["none", "declared", "suspected"]


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    role: Role
    missing: Literal["error", "median", "most_frequent"] = "error"
    categories: tuple[str, ...] = ()
    reference: str | None = None

    # Data dictionary / derived feature metadata
    derived: DerivedKind = "none"
    derived_from: tuple[str, ...] = ()
    derivation: str | None = None
    description: str | None = None
    unit: str | None = None


@dataclass(frozen=True, slots=True)
class ModelConfig:
    id: str
    interactions: Literal["none", "all_eligible", "explicit"] = "none"
    pairs: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class CorrelationConfig:
    enabled: bool = True
    pearson: bool = True
    spearman: bool = True
    review_threshold: float = 0.75
    warning_threshold: float = 0.90
    minimum_complete_pairs: int = 3


@dataclass(frozen=True, slots=True)
class DuplicateGroupConfig:
    enabled: bool = True
    rounding_decimals: int = 8
    near_duplicate_threshold: float = 0.98
    include_target_in_signature: bool = False
    maximum_pairwise_rows: int = 10_000


@dataclass(frozen=True, slots=True)
class ProfilingConfig:
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)
    duplicate_groups: DuplicateGroupConfig = field(default_factory=DuplicateGroupConfig)


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    strategy: ValidationStrategy = "stratified"
    outer_splits: int = 5
    outer_repeats: int = 3
    inner_splits: int = 5
    random_state: int = 42
    gap: int = 0
    test_size: int | None = None
    duplicate_group_policy: Literal["report", "group", "error"] = "report"


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
    group_column: str | None = None
    time_column: str | None = None
    features: dict[str, FeatureConfig] = field(default_factory=dict)
    models: tuple[ModelConfig, ...] = ()
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    primary_metric: Literal["log_loss"] = "log_loss"
    schema_version: str = "1.1"

    def validate(self) -> None:
        supported_versions = {"1.0", "1.1"}
        if self.schema_version not in supported_versions:
            raise ConfigurationError(
                f"Unsupported schema_version {self.schema_version!r}. "
                f"Supported versions: {sorted(supported_versions)}."
            )

        if not self.name.strip():
            raise ConfigurationError("Experiment name cannot be empty.")
        if not self.target.strip():
            raise ConfigurationError("Target column cannot be empty.")
        if self.validation.outer_splits < 2 or self.validation.inner_splits < 2:
            raise ConfigurationError("Inner and outer splits must be at least 2.")
        if self.validation.outer_repeats < 1:
            raise ConfigurationError("outer_repeats must be at least 1.")
        if self.execution.workers < 1:
            raise ConfigurationError("execution.workers must be at least 1.")

        active = {
            name for name, spec in self.features.items() if spec.role != "exclude"
        }
        if not active:
            raise ConfigurationError("At least one active predictor is required.")

        model_ids = [model.id for model in self.models]
        if len(model_ids) != len(set(model_ids)):
            raise ConfigurationError("Model IDs must be unique.")
        for model_id in model_ids:
            if not MODEL_ID_PATTERN.fullmatch(model_id):
                raise ConfigurationError(
                    "Model IDs may contain only letters, numbers, "
                    "underscores, periods, and hyphens, and must start "
                    f"with a letter or number: {model_id!r}."
                )

        smooth = {name for name, spec in self.features.items() if spec.role == "smooth"}

        for model in self.models:
            if model.interactions == "explicit":
                for left, right in model.pairs:
                    if left == right or left not in smooth or right not in smooth:
                        raise ConfigurationError(
                            "Explicit interactions require two distinct "
                            f"smooth predictors: {left}:{right}"
                        )
        if any(value < 2 for value in self.search.n_knots):
            raise ConfigurationError("Every n_knots value must be at least 2.")
        if any(value < 1 for value in self.search.degree):
            raise ConfigurationError("Every degree value must be at least 1.")
        if any(value <= 0 for value in self.search.C):
            raise ConfigurationError("Every C value must be positive.")

        # Profiling correlation validation
        if not 0.0 <= self.profiling.correlation.review_threshold <= 1.0:
            raise ConfigurationError(
                "profiling.correlation.review_threshold must be between 0 and 1."
            )
        if not 0.0 <= self.profiling.correlation.warning_threshold <= 1.0:
            raise ConfigurationError(
                "profiling.correlation.warning_threshold must be between 0 and 1."
            )
        if (
            self.profiling.correlation.warning_threshold
            < self.profiling.correlation.review_threshold
        ):
            raise ConfigurationError(
                "The correlation warning threshold cannot be smaller "
                "than the review threshold."
            )
        if self.profiling.correlation.minimum_complete_pairs < 2:
            raise ConfigurationError("minimum_complete_pairs must be at least 2.")

        # Duplicate groups validation
        if self.profiling.duplicate_groups.rounding_decimals < 0:
            raise ConfigurationError(
                "profiling.duplicate_groups.rounding_decimals cannot be negative."
            )

        threshold = self.profiling.duplicate_groups.near_duplicate_threshold
        if not math.isfinite(threshold) or not (0.0 < threshold <= 1.0):
            raise ConfigurationError(
                "profiling.duplicate_groups.near_duplicate_threshold "
                "must satisfy 0.0 < threshold <= 1.0."
            )

        if self.profiling.duplicate_groups.maximum_pairwise_rows < 2:
            raise ConfigurationError(
                "profiling.duplicate_groups.maximum_pairwise_rows must be at least 2."
            )

        if self.profiling.duplicate_groups.include_target_in_signature:
            raise ConfigurationError(
                "profiling.duplicate_groups.include_target_in_signature=true "
                "is not supported. Duplicate identities used for validation "
                "must be constructed from predictors only. Target values "
                "are used only to detect conflicting duplicate targets."
            )

        # Validation strategy and policy checks
        if self.validation.gap < 0:
            raise ConfigurationError("validation.gap cannot be negative.")

        if (
            self.validation.duplicate_group_policy in {"group", "error"}
            and not self.profiling.duplicate_groups.enabled
        ):
            raise ConfigurationError(
                "validation.duplicate_group_policy requires "
                "profiling.duplicate_groups.enabled=true."
            )

        if (
            self.validation.duplicate_group_policy == "group"
            and self.validation.strategy != "stratified_group"
        ):
            raise ConfigurationError(
                "validation.duplicate_group_policy='group' requires "
                "validation.strategy='stratified_group'."
            )

        if self.validation.strategy == "stratified_group":
            configured_grouping = self.group_column is not None
            duplicate_grouping = (
                self.validation.duplicate_group_policy == "group"
                and self.profiling.duplicate_groups.enabled
            )
            if not configured_grouping and not duplicate_grouping:
                raise ConfigurationError(
                    "validation.strategy='stratified_group' requires either "
                    "data.group or validation.duplicate_group_policy='group' "
                    "with profiling.duplicate_groups.enabled=true."
                )

        if self.validation.strategy == "time":
            if self.time_column is None:
                raise ConfigurationError(
                    "data.time is required when validation.strategy is 'time'."
                )
            if self.validation.outer_repeats != 1:
                raise ConfigurationError(
                    "Time-aware validation requires outer_repeats=1."
                )

        # Feature derivation validation
        for name, spec in self.features.items():
            unknown_sources = sorted(set(spec.derived_from) - set(self.features))
            if unknown_sources:
                raise ConfigurationError(
                    f"Feature {name!r} declares unknown derivation sources: "
                    f"{unknown_sources}."
                )

            if name in spec.derived_from:
                raise ConfigurationError(
                    f"Feature {name!r} cannot be derived from itself."
                )

            if spec.derived == "declared" and not spec.derived_from:
                raise ConfigurationError(
                    f"Derived feature {name!r} must declare derived_from."
                )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_path"] = str(self.data_path)
        return payload


def _tuples(values: Any) -> tuple[Any, ...]:
    return tuple(values or ())


def load_config(path: Path) -> ExperimentConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent.resolve()
    data_raw = payload["data"]
    data_path = Path(data_raw["path"])
    if not data_path.is_absolute():
        data_path = (base / data_path).resolve()

    features = {
        name: FeatureConfig(
            role=spec["role"],
            missing=spec.get("missing", "error"),
            categories=_tuples(spec.get("categories")),
            reference=spec.get("reference"),
            derived=spec.get("derived", "none"),
            derived_from=_tuples(spec.get("derived_from")),
            derivation=spec.get("derivation"),
            description=spec.get("description"),
            unit=spec.get("unit"),
        )
        for name, spec in payload["features"].items()
    }

    models = tuple(
        ModelConfig(
            id=item["id"],
            interactions=item.get("interactions", "none"),
            pairs=tuple(tuple(pair) for pair in item.get("pairs", ())),
        )
        for item in payload.get("models", [])
    )

    profiling_raw = payload.get("profiling", {})
    correlation = CorrelationConfig(**profiling_raw.get("correlation", {}))
    duplicate_groups = DuplicateGroupConfig(**profiling_raw.get("duplicate_groups", {}))
    profiling = ProfilingConfig(
        correlation=correlation,
        duplicate_groups=duplicate_groups,
    )

    validation = ValidationConfig(**payload.get("validation", {}))

    search_raw: dict[str, Any] = payload.get("search", {})
    search_defaults = SearchConfig()
    search = SearchConfig(
        n_knots=_tuples(search_raw.get("n_knots", search_defaults.n_knots)),
        degree=_tuples(search_raw.get("degree", search_defaults.degree)),
        C=_tuples(search_raw.get("C", search_defaults.C)),
        interaction_scale=_tuples(
            search_raw.get("interaction_scale", search_defaults.interaction_scale)
        ),
    )

    execution = ExecutionConfig(**payload.get("execution", {}))

    config = ExperimentConfig(
        schema_version=str(payload.get("schema_version", "1.0")),
        name=payload["experiment"]["name"],
        primary_metric=payload["experiment"].get("primary_metric", "log_loss"),
        data_path=data_path,
        target=data_raw["target"],
        row_id=data_raw.get("row_id"),
        group_column=data_raw.get("group"),
        time_column=data_raw.get("time"),
        features=features,
        models=models,
        profiling=profiling,
        validation=validation,
        search=search,
        execution=execution,
    )
    config.validate()
    return config


def dump_config_dict(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "schema_version": config.schema_version,
        "experiment": {
            "name": config.name,
            "primary_metric": config.primary_metric,
        },
        "data": {
            "path": str(config.data_path),
            "target": config.target,
            "row_id": config.row_id,
            "group": config.group_column,
            "time": config.time_column,
        },
        "features": {
            name: {
                "role": spec.role,
                "missing": spec.missing,
                **({"categories": list(spec.categories)} if spec.categories else {}),
                **({"reference": spec.reference} if spec.reference else {}),
                "derived": spec.derived,
                **(
                    {"derived_from": list(spec.derived_from)}
                    if spec.derived_from
                    else {}
                ),
                **({"derivation": spec.derivation} if spec.derivation else {}),
                **({"description": spec.description} if spec.description else {}),
                **({"unit": spec.unit} if spec.unit else {}),
            }
            for name, spec in config.features.items()
        },
        "models": [
            {
                "id": model.id,
                "interactions": model.interactions,
                **(
                    {"pairs": [list(pair) for pair in model.pairs]}
                    if model.pairs
                    else {}
                ),
            }
            for model in config.models
        ],
        "profiling": asdict(config.profiling),
        "validation": asdict(config.validation),
        "search": asdict(config.search),
        "execution": asdict(config.execution),
    }
