from __future__ import annotations

import itertools
import json
import shutil
import warnings
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
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
from .logistic import extract_class_score_parameters
from .models import build_pipeline
from .run_store import FileRunStore


@dataclass(frozen=True, slots=True)
class OuterFoldTask:
    model: ModelConfig
    repeat: int
    fold: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    seed: int


@dataclass(frozen=True, slots=True)
class OuterFoldResult:
    model_id: str
    repeat: int
    fold: int
    metrics: dict[str, Any]
    trials: pd.DataFrame
    predictions: pd.DataFrame
    fitted_model: Any


OuterFoldCallback = Callable[[OuterFoldResult], None]


def parameter_candidates(config: ExperimentConfig, model: ModelConfig):
    scales = config.search.interaction_scale if model.interactions != "none" else (1.0,)
    return itertools.product(
        config.search.n_knots, config.search.degree, config.search.C, scales
    )


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
    rows: list[dict[str, Any]] = []
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
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
    }


def build_outer_fold_tasks(
    config: ExperimentConfig,
    model: ModelConfig,
    splits: pd.DataFrame,
) -> list[OuterFoldTask]:
    tasks: list[OuterFoldTask] = []
    group_keys = (
        splits[["repeat", "fold"]].drop_duplicates().itertuples(index=False, name=None)
    )
    for iteration, (repeat, fold) in enumerate(group_keys, start=1):
        subset = splits[(splits.repeat == repeat) & (splits.fold == fold)]
        train = tuple(
            int(value)
            for value in subset.loc[subset.partition == "train", "row_index"].to_numpy()
        )
        test = tuple(
            int(value)
            for value in subset.loc[subset.partition == "test", "row_index"].to_numpy()
        )
        tasks.append(
            OuterFoldTask(
                model=model,
                repeat=int(repeat),
                fold=int(fold),
                train_indices=train,
                test_indices=test,
                seed=config.validation.random_state + iteration,
            )
        )
    return tasks


def create_split_manifest(config: ExperimentConfig, X, y, row_ids) -> pd.DataFrame:
    splitter = RepeatedStratifiedKFold(
        n_splits=config.validation.outer_splits,
        n_repeats=config.validation.outer_repeats,
        random_state=config.validation.random_state,
    )
    rows: list[dict[str, Any]] = []
    for iteration, (train, test) in enumerate(splitter.split(X, y), start=1):
        repeat = (iteration - 1) // config.validation.outer_splits + 1
        fold = (iteration - 1) % config.validation.outer_splits + 1
        rows.extend(
            {
                "repeat": repeat,
                "fold": fold,
                "row_id": str(row_ids.iloc[i]),
                "row_index": int(i),
                "partition": "train",
            }
            for i in train
        )
        rows.extend(
            {
                "repeat": repeat,
                "fold": fold,
                "row_id": str(row_ids.iloc[i]),
                "row_index": int(i),
                "partition": "test",
            }
            for i in test
        )
    return pd.DataFrame(rows)


def execute_outer_fold_task(
    task: OuterFoldTask,
    config: ExperimentConfig,
    X: pd.DataFrame,
    y: pd.Series,
    row_ids: pd.Series,
) -> OuterFoldResult:
    train = np.asarray(task.train_indices, dtype=int)
    test = np.asarray(task.test_indices, dtype=int)

    best, trials = inner_search(
        config,
        task.model,
        X.iloc[train],
        y.iloc[train],
        task.seed,
    )
    pipeline = build_pipeline(
        config,
        task.model,
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
        "model_id": task.model.id,
        "repeat": task.repeat,
        "fold": task.fold,
        **_fold_metrics(y.iloc[test], predictions, probabilities, classes),
        **{f"best_{key}": value for key, value in best.items()},
    }
    prediction_frame = pd.DataFrame(
        {
            "model_id": task.model.id,
            "repeat": task.repeat,
            "fold": task.fold,
            "row_id": row_ids.iloc[test].to_numpy(),
            "observed_class": y.iloc[test].to_numpy(),
            "predicted_class": predictions,
        }
    )
    for index, class_name in enumerate(classes):
        prediction_frame[f"probability_{class_name}"] = probabilities[:, index]

    return OuterFoldResult(
        model_id=task.model.id,
        repeat=task.repeat,
        fold=task.fold,
        metrics=metrics,
        trials=trials,
        predictions=prediction_frame,
        fitted_model=pipeline,
    )


def run_outer_folds_sequentially(
    tasks: Iterable[OuterFoldTask],
    config: ExperimentConfig,
    X: pd.DataFrame,
    y: pd.Series,
    row_ids: pd.Series,
    on_started: Callable[[OuterFoldTask], None],
    on_completed: OuterFoldCallback,
) -> None:
    for task in tasks:
        on_started(task)
        result = execute_outer_fold_task(task, config, X, y, row_ids)
        on_completed(result)


def run_outer_folds_in_parallel(
    tasks: Iterable[OuterFoldTask],
    config: ExperimentConfig,
    X: pd.DataFrame,
    y: pd.Series,
    row_ids: pd.Series,
    workers: int,
    on_started: Callable[[OuterFoldTask], None],
    on_completed: OuterFoldCallback,
    should_stop: Callable[[], bool],
) -> None:
    task_list = list(tasks)
    if not task_list:
        return

    maximum_workers = min(workers, len(task_list))
    future_tasks: dict[Future[OuterFoldResult], OuterFoldTask] = {}

    with ProcessPoolExecutor(max_workers=maximum_workers) as executor:
        for task in task_list:
            if should_stop():
                break
            on_started(task)
            future = executor.submit(
                execute_outer_fold_task, task, config, X, y, row_ids
            )
            future_tasks[future] = task

        pending = set(future_tasks)
        while pending:
            if should_stop():
                for future in pending:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                return

            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in done:
                task = future_tasks[future]
                try:
                    result = future.result()
                except BaseException as error:
                    for pending_future in pending:
                        pending_future.cancel()
                    raise RuntimeError(
                        "Outer-fold worker failed for "
                        f"model={task.model.id!r}, repeat={task.repeat}, "
                        f"fold={task.fold}."
                    ) from error
                on_completed(result)


def write_fold_checkpoint(
    result: OuterFoldResult,
    checkpoint: Path,
    data_hash: str,
    config_hash: str,
) -> None:
    temporary = checkpoint.with_name(checkpoint.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        temporary / "checkpoint.json",
        {
            "model_id": result.model_id,
            "repeat": result.repeat,
            "fold": result.fold,
            "data_hash": data_hash,
            "config_hash": config_hash,
            "completed_at_utc": utc_now(),
        },
    )
    write_json_atomic(temporary / "metrics.json", result.metrics)
    result.trials.to_parquet(temporary / "trials.parquet", index=False)
    result.predictions.to_parquet(temporary / "predictions.parquet", index=False)
    joblib.dump(result.fitted_model, temporary / "model.joblib")
    (temporary / "COMPLETE").touch()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    temporary.replace(checkpoint)


def rebuild_model_results(
    tasks: list[OuterFoldTask],
    store: FileRunStore,
    data_hash: str,
    config_hash: str,
) -> None:
    fold_rows = []
    prediction_frames = []
    for task in sorted(tasks, key=lambda item: (item.repeat, item.fold)):
        if not store.checkpoint_complete(
            task.model.id, task.repeat, task.fold, data_hash, config_hash
        ):
            raise RuntimeError(
                "Cannot build aggregate results because an outer fold is incomplete: "
                f"model={task.model.id!r}, repeat={task.repeat}, fold={task.fold}."
            )
        checkpoint = store.checkpoint_directory(task.model.id, task.repeat, task.fold)
        fold_rows.append(
            json.loads((checkpoint / "metrics.json").read_text(encoding="utf-8"))
        )
        prediction_frames.append(pd.read_parquet(checkpoint / "predictions.parquet"))

    fold_frame = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    model_results = store.results / tasks[0].model.id
    model_results.mkdir(parents=True, exist_ok=True)
    fold_frame.to_csv(model_results / "fold_metrics.csv", index=False)
    predictions.to_parquet(model_results / "predictions.parquet", index=False)
    metric_columns = ["log_loss", "accuracy", "balanced_accuracy", "macro_f1"]
    fold_frame[metric_columns].agg(["mean", "std", "median", "min", "max"]).T.to_csv(
        model_results / "summary.csv"
    )


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
    all_tasks = build_outer_fold_tasks(config, model, splits)
    total = config.validation.outer_splits * config.validation.outer_repeats
    pending_tasks = []
    for task in all_tasks:
        if store.requested("CANCEL"):
            store.update_status(state="cancelled")
            return
        if store.requested("PAUSE"):
            store.update_status(state="paused")
            return
        if store.checkpoint_complete(
            model.id, task.repeat, task.fold, data_hash, config_hash
        ):
            continue
        pending_tasks.append(task)

    completed_outer_folds = len(all_tasks) - len(pending_tasks)

    def should_stop() -> bool:
        if store.requested("CANCEL"):
            store.update_status(state="cancelled")
            return True
        if store.requested("PAUSE"):
            store.update_status(state="paused")
            return True
        return False

    def on_started(task: OuterFoldTask) -> None:
        store.update_status(
            state="running",
            phase="nested_cross_validation",
            model_id=model.id,
            repeat=task.repeat,
            fold=task.fold,
            completed_outer_folds=completed_outer_folds,
            total_outer_folds=total,
        )
        store.event(
            "fold_started", model_id=model.id, repeat=task.repeat, fold=task.fold
        )

    def on_completed(result: OuterFoldResult) -> None:
        nonlocal completed_outer_folds

        checkpoint = store.checkpoint_directory(
            result.model_id, result.repeat, result.fold
        )
        write_fold_checkpoint(result, checkpoint, data_hash, config_hash)
        completed_outer_folds += 1
        store.update_status(
            state="running",
            phase="nested_cross_validation",
            model_id=result.model_id,
            repeat=result.repeat,
            fold=result.fold,
            completed_outer_folds=completed_outer_folds,
            total_outer_folds=total,
        )
        store.event(
            "fold_completed",
            model_id=result.model_id,
            repeat=result.repeat,
            fold=result.fold,
            log_loss=result.metrics["log_loss"],
        )

    if config.execution.workers == 1:
        run_outer_folds_sequentially(
            pending_tasks,
            config,
            X,
            y,
            row_ids,
            on_started,
            on_completed,
        )
    else:
        run_outer_folds_in_parallel(
            pending_tasks,
            config,
            X,
            y,
            row_ids,
            config.execution.workers,
            on_started,
            on_completed,
            should_stop,
        )

    if should_stop():
        return

    rebuild_model_results(all_tasks, store, data_hash, config_hash)


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

    feature_names = transformer.get_feature_names_out()
    parameters = extract_class_score_parameters(classifier)

    if parameters.coefficients.shape[1] != len(feature_names):
        raise ValueError(
            "The fitted coefficient count does not match the transformed "
            "feature-name count."
        )

    rows: list[dict[str, object]] = []

    for class_index, class_name in enumerate(parameters.classes):
        rows.append(
            {
                "class": str(class_name),
                "component": "intercept",
                "coefficient": float(parameters.intercepts[class_index]),
            }
        )

        rows.extend(
            {
                "class": str(class_name),
                "component": str(feature_name),
                "coefficient": float(coefficient),
            }
            for feature_name, coefficient in zip(
                feature_names,
                parameters.coefficients[class_index],
                strict=True,
            )
        )

    pd.DataFrame(rows).to_csv(
        directory / "components.csv",
        index=False,
    )
