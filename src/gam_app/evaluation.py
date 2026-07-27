from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from .config import ExperimentConfig, ModelConfig
from .io_utils import utc_now, write_json_atomic
from .models import build_pipeline
from .run_store import FileRunStore


def parameter_candidates(config: ExperimentConfig, model: ModelConfig):
    scales = config.search.interaction_scale if model.interactions != "none" else (1.0,)
    return itertools.product(config.search.n_knots, config.search.degree, config.search.C, scales)


def inner_search(
    config: ExperimentConfig,
    model: ModelConfig,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    splitter = StratifiedKFold(
        n_splits=config.validation.inner_splits, shuffle=True, random_state=seed
    )
    rows = []
    best_loss = float("inf")
    best: dict[str, Any] | None = None
    for n_knots, degree, C, scale in parameter_candidates(config, model):
        losses = []
        for train, valid in splitter.split(X, y):
            pipeline = build_pipeline(
                config,
                model,
                n_knots=n_knots,
                degree=degree,
                C=C,
                interaction_scale=scale,
            )
            with warnings.catch_warnings():
                if config.execution.stop_on_convergence_warning:
                    warnings.filterwarnings("error", category=ConvergenceWarning)
                pipeline.fit(X.iloc[train], y.iloc[train])
            probabilities = pipeline.predict_proba(X.iloc[valid])
            classes = pipeline.named_steps["classifier"].classes_
            losses.append(log_loss(y.iloc[valid], probabilities, labels=classes))
        mean_loss = float(np.mean(losses))
        row = {
            "n_knots": n_knots,
            "degree": degree,
            "C": C,
            "interaction_scale": scale,
            "mean_log_loss": mean_loss,
            "std_log_loss": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
        }
        rows.append(row)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best = row.copy()
    assert best is not None
    return best, pd.DataFrame(rows)


def _fold_metrics(y_true, predictions, probabilities, classes) -> dict[str, float]:
    return {
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
    }


def create_split_manifest(config: ExperimentConfig, X, y, row_ids) -> pd.DataFrame:
    splitter = RepeatedStratifiedKFold(
        n_splits=config.validation.outer_splits,
        n_repeats=config.validation.outer_repeats,
        random_state=config.validation.random_state,
    )
    rows = []
    for iteration, (train, test) in enumerate(splitter.split(X, y), start=1):
        repeat = (iteration - 1) // config.validation.outer_splits + 1
        fold = (iteration - 1) % config.validation.outer_splits + 1
        rows.extend(
            {"repeat": repeat, "fold": fold, "row_id": str(row_ids.iloc[i]), "row_index": int(i), "partition": "train"}
            for i in train
        )
        rows.extend(
            {"repeat": repeat, "fold": fold, "row_id": str(row_ids.iloc[i]), "row_index": int(i), "partition": "test"}
            for i in test
        )
    return pd.DataFrame(rows)


def run_model(
    config: ExperimentConfig,
    model: ModelConfig,
    X: pd.DataFrame,
    y: pd.Series,
    row_ids: pd.Series,
    splits: pd.DataFrame,
    store: FileRunStore,
    data_hash: str,
    config_hash: str,
) -> None:
    fold_rows = []
    prediction_frames = []
    group_keys = splits[["repeat", "fold"]].drop_duplicates().itertuples(index=False, name=None)
    total = config.validation.outer_splits * config.validation.outer_repeats
    for iteration, (repeat, fold) in enumerate(group_keys, start=1):
        if store.requested("CANCEL"):
            store.update_status(state="cancelled")
            return
        if store.requested("PAUSE"):
            store.update_status(state="paused")
            return
        checkpoint = store.checkpoint_directory(model.id, repeat, fold)
        if store.checkpoint_complete(model.id, repeat, fold, data_hash, config_hash):
            fold_rows.append(json.loads((checkpoint / "metrics.json").read_text(encoding="utf-8")))
            prediction_frames.append(pd.read_parquet(checkpoint / "predictions.parquet"))
            continue
        subset = splits[(splits.repeat == repeat) & (splits.fold == fold)]
        train = subset.loc[subset.partition == "train", "row_index"].to_numpy()
        test = subset.loc[subset.partition == "test", "row_index"].to_numpy()
        store.update_status(
            state="running",
            phase="nested_cross_validation",
            model_id=model.id,
            repeat=int(repeat),
            fold=int(fold),
            completed_outer_folds=iteration - 1,
            total_outer_folds=total,
        )
        store.event("fold_started", model_id=model.id, repeat=repeat, fold=fold)
        best, trials = inner_search(
            config, model, X.iloc[train], y.iloc[train], config.validation.random_state + iteration
        )
        pipeline = build_pipeline(
            config,
            model,
            n_knots=int(best["n_knots"]),
            degree=int(best["degree"]),
            C=float(best["C"]),
            interaction_scale=float(best["interaction_scale"]),
        )
        pipeline.fit(X.iloc[train], y.iloc[train])
        probabilities = pipeline.predict_proba(X.iloc[test])
        predictions = pipeline.predict(X.iloc[test])
        classes = pipeline.named_steps["classifier"].classes_
        metrics = {
            "model_id": model.id,
            "repeat": int(repeat),
            "fold": int(fold),
            **_fold_metrics(y.iloc[test], predictions, probabilities, classes),
            **{f"best_{key}": value for key, value in best.items()},
        }
        prediction_frame = pd.DataFrame(
            {
                "model_id": model.id,
                "repeat": repeat,
                "fold": fold,
                "row_id": row_ids.iloc[test].to_numpy(),
                "observed_class": y.iloc[test].to_numpy(),
                "predicted_class": predictions,
            }
        )
        for index, class_name in enumerate(classes):
            prediction_frame[f"probability_{class_name}"] = probabilities[:, index]
        temporary = checkpoint.with_name(checkpoint.name + ".tmp")
        if temporary.exists():
            import shutil
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True, exist_ok=True)
        write_json_atomic(temporary / "checkpoint.json", {
            "model_id": model.id,
            "repeat": repeat,
            "fold": fold,
            "data_hash": data_hash,
            "config_hash": config_hash,
            "completed_at_utc": utc_now(),
        })
        write_json_atomic(temporary / "metrics.json", metrics)
        trials.to_parquet(temporary / "trials.parquet", index=False)
        prediction_frame.to_parquet(temporary / "predictions.parquet", index=False)
        joblib.dump(pipeline, temporary / "model.joblib")
        (temporary / "COMPLETE").touch()
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(checkpoint)
        fold_rows.append(metrics)
        prediction_frames.append(prediction_frame)
        store.event("fold_completed", model_id=model.id, repeat=repeat, fold=fold, log_loss=metrics["log_loss"])
    fold_frame = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    model_results = store.results / model.id
    model_results.mkdir(parents=True, exist_ok=True)
    fold_frame.to_csv(model_results / "fold_metrics.csv", index=False)
    predictions.to_parquet(model_results / "predictions.parquet", index=False)
    metric_columns = ["log_loss", "accuracy", "balanced_accuracy", "macro_f1"]
    fold_frame[metric_columns].agg(["mean", "std", "median", "min", "max"]).T.to_csv(
        model_results / "summary.csv"
    )


def fit_final_model(config, model, X, y, store):
    best, trials = inner_search(config, model, X, y, config.validation.random_state)
    pipeline = build_pipeline(
        config,
        model,
        n_knots=int(best["n_knots"]),
        degree=int(best["degree"]),
        C=float(best["C"]),
        interaction_scale=float(best["interaction_scale"]),
    )
    pipeline.fit(X, y)
    directory = store.models / model.id
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, directory / "model.joblib")
    write_json_atomic(directory / "best_parameters.json", best)
    trials.to_parquet(directory / "search_trials.parquet", index=False)
    transformer = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["classifier"]
    names = transformer.get_feature_names_out()
    rows = []
    for class_index, class_name in enumerate(classifier.classes_):
        rows.append({"class": class_name, "component": "intercept", "coefficient": classifier.intercept_[class_index]})
        rows.extend(
            {"class": class_name, "component": name, "coefficient": coefficient}
            for name, coefficient in zip(names, classifier.coef_[class_index], strict=True)
        )
    pd.DataFrame(rows).to_csv(directory / "components.csv", index=False)
