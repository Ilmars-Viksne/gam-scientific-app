from __future__ import annotations

import itertools
import json
import shutil
import time
import warnings
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from .config import ExperimentConfig, ModelConfig
from .io_utils import format_duration, utc_now, write_json_atomic
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
    class_metrics: pd.DataFrame
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


def _fold_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    """Calculate aggregate metrics for one outer test fold."""

    confusion_matrices = multilabel_confusion_matrix(
        y_true,
        predictions,
        labels=classes,
    )

    true_negatives = confusion_matrices[
        :,
        0,
        0,
    ].astype(np.float64)

    false_positives = confusion_matrices[
        :,
        0,
        1,
    ].astype(np.float64)

    specificity_by_class = _safe_ratio(
        true_negatives,
        true_negatives + false_positives,
    )

    class_support = np.asarray(
        [int((np.asarray(y_true) == class_name).sum()) for class_name in classes],
        dtype=np.float64,
    )

    macro_specificity = float(np.mean(specificity_by_class))

    if class_support.sum() == 0:
        weighted_specificity = 0.0
    else:
        weighted_specificity = float(
            np.average(
                specificity_by_class,
                weights=class_support,
            )
        )

    metrics: dict[str, float] = {
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=classes,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_specificity": macro_specificity,
        "weighted_specificity": weighted_specificity,
    }

    for class_index, class_name in enumerate(classes):
        safe_class_name = str(class_name)

        metrics[f"specificity_{safe_class_name}"] = float(
            specificity_by_class[class_index]
        )

    return metrics


def build_outer_fold_tasks(
    config: ExperimentConfig,
    model: ModelConfig,
    splits: pd.DataFrame,
    row_count: int,
) -> list[OuterFoldTask]:

    if row_count < 1:
        raise ValueError("row_count must be at least 1.")

    required_columns = {
        "repeat",
        "fold",
        "row_index",
        "partition",
    }

    missing_columns = sorted(required_columns - set(splits.columns))

    if missing_columns:
        raise ValueError(
            f"The split manifest is missing required columns: {missing_columns}."
        )

    if splits.empty:
        raise ValueError("The split manifest is empty.")

    valid_partitions = {
        "train",
        "test",
    }

    observed_partitions = set(splits["partition"].dropna().astype(str).unique())

    invalid_partitions = sorted(observed_partitions - valid_partitions)

    if invalid_partitions:
        raise ValueError(
            "The split manifest contains invalid partition values: "
            f"{invalid_partitions}."
        )

    if splits["partition"].isna().any():
        raise ValueError("The split manifest contains missing partition values.")

    if splits["row_index"].isna().any():
        raise ValueError("The split manifest contains missing row indices.")

    expected_indices = set(range(row_count))

    group_frame = (
        splits[
            [
                "repeat",
                "fold",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "repeat",
                "fold",
            ]
        )
    )

    expected_task_count = (
        config.validation.outer_splits * config.validation.outer_repeats
    )

    if len(group_frame) != expected_task_count:
        raise ValueError(
            "The split manifest has an unexpected number of "
            "outer folds. "
            f"Expected {expected_task_count}, "
            f"received {len(group_frame)}."
        )

    tasks: list[OuterFoldTask] = []

    for repeat_value, fold_value in group_frame.itertuples(
        index=False,
        name=None,
    ):
        repeat = int(repeat_value)
        fold = int(fold_value)

        if not (1 <= repeat <= config.validation.outer_repeats):
            raise ValueError(
                f"The split manifest contains an invalid repeat number: {repeat}."
            )

        if not (1 <= fold <= config.validation.outer_splits):
            raise ValueError(
                f"The split manifest contains an invalid fold number: {fold}."
            )

        subset = splits.loc[(splits["repeat"] == repeat) & (splits["fold"] == fold)]

        train_values = subset.loc[
            subset["partition"] == "train",
            "row_index",
        ]

        test_values = subset.loc[
            subset["partition"] == "test",
            "row_index",
        ]

        train_indices = tuple(int(value) for value in train_values.to_numpy())

        test_indices = tuple(int(value) for value in test_values.to_numpy())

        if not train_indices:
            raise ValueError(
                "The outer training partition is empty for "
                f"repeat={repeat}, fold={fold}."
            )

        if not test_indices:
            raise ValueError(
                f"The outer test partition is empty for repeat={repeat}, fold={fold}."
            )

        if len(train_indices) != len(set(train_indices)):
            raise ValueError(
                "The outer training partition contains duplicate "
                "row indices for "
                f"repeat={repeat}, fold={fold}."
            )

        if len(test_indices) != len(set(test_indices)):
            raise ValueError(
                "The outer test partition contains duplicate row "
                "indices for "
                f"repeat={repeat}, fold={fold}."
            )

        train_set = set(train_indices)
        test_set = set(test_indices)

        overlapping_indices = sorted(train_set & test_set)

        if overlapping_indices:
            raise ValueError(
                "Outer training and test partitions overlap for "
                f"repeat={repeat}, fold={fold}. "
                "Overlapping row indices include: "
                f"{overlapping_indices[:10]}."
            )

        observed_indices = train_set | test_set

        missing_indices = sorted(expected_indices - observed_indices)

        unexpected_indices = sorted(observed_indices - expected_indices)

        if missing_indices:
            raise ValueError(
                "The outer fold does not cover every dataset row "
                f"for repeat={repeat}, fold={fold}. "
                "Missing row indices include: "
                f"{missing_indices[:10]}."
            )

        if unexpected_indices:
            raise ValueError(
                "The outer fold contains row indices outside the "
                f"dataset for repeat={repeat}, fold={fold}. "
                "Unexpected row indices include: "
                f"{unexpected_indices[:10]}."
            )

        seed = (
            config.validation.random_state
            + (repeat - 1) * config.validation.outer_splits
            + fold
        )

        tasks.append(
            OuterFoldTask(
                model=model,
                repeat=repeat,
                fold=fold,
                train_indices=train_indices,
                test_indices=test_indices,
                seed=seed,
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
    with warnings.catch_warnings():
        if config.execution.stop_on_convergence_warning:
            warnings.filterwarnings(
                "error",
                category=ConvergenceWarning,
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

    class_metrics = calculate_class_metrics(
        y_true=y.iloc[test],
        predictions=predictions,
        classes=classes,
        model_id=task.model.id,
        repeat=task.repeat,
        fold=task.fold,
    )

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
        class_metrics=class_metrics,
        trials=trials,
        predictions=prediction_frame,
        fitted_model=pipeline,
    )


def _check_stop_requested(should_stop: Callable[[], Any]) -> str | None:
    val = should_stop()
    if not val:
        return None
    if isinstance(val, str):
        return val
    return "stop"


def run_outer_folds_sequentially(
    tasks: Iterable[OuterFoldTask],
    config: ExperimentConfig,
    X: pd.DataFrame,
    y: pd.Series,
    row_ids: pd.Series,
    on_started: Callable[[OuterFoldTask], None],
    on_completed: OuterFoldCallback,
    should_stop: Callable[[], Any] = lambda: None,
) -> None:
    for task in tasks:
        if _check_stop_requested(should_stop) is not None:
            break
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
    should_stop: Callable[[], Any] = lambda: None,
) -> None:
    task_iterator = iter(tasks)
    maximum_workers = workers

    future_tasks: dict[
        Future[OuterFoldResult],
        OuterFoldTask,
    ] = {}

    stopping = False

    def submit_next(
        executor: ProcessPoolExecutor,
    ) -> bool:
        nonlocal stopping
        if stopping or _check_stop_requested(should_stop) is not None:
            stopping = True
            return False

        try:
            task = next(task_iterator)
        except StopIteration:
            return False

        on_started(task)

        future = executor.submit(
            execute_outer_fold_task,
            task,
            config,
            X,
            y,
            row_ids,
        )

        future_tasks[future] = task
        return True

    with ProcessPoolExecutor(
        max_workers=maximum_workers,
    ) as executor:
        for _ in range(maximum_workers):
            if not submit_next(executor):
                break

        while future_tasks:
            if not stopping and _check_stop_requested(should_stop) is not None:
                stopping = True
                for future in list(future_tasks.keys()):
                    if future.cancel():
                        future_tasks.pop(future, None)

            if not future_tasks:
                break

            done, _ = wait(
                future_tasks,
                timeout=1.0,
                return_when=FIRST_COMPLETED,
            )

            for future in done:
                task = future_tasks.pop(future)

                if future.cancelled():
                    continue

                try:
                    result = future.result()
                except BaseException as error:
                    for pending_future in future_tasks:
                        pending_future.cancel()

                    raise RuntimeError(
                        "Outer-fold worker failed for "
                        f"model={task.model.id!r}, "
                        f"repeat={task.repeat}, "
                        f"fold={task.fold}."
                    ) from error

                on_completed(result)
                if not stopping:
                    submit_next(executor)


def _commit_checkpoint_directory(
    temporary: Path,
    checkpoint: Path,
    *,
    attempts: int = 5,
    initial_delay: float = 0.05,
) -> None:
    """Commit a completed temporary checkpoint directory.

    Directory renames can temporarily fail on Windows when another
    process, antivirus scanner, or file indexer briefly holds one of
    the newly written files. Retry with a short exponential backoff.

    If an atomic rename remains unavailable, fall back to shutil.move().
    The COMPLETE marker still protects checkpoint validity.
    """

    if attempts < 1:
        raise ValueError("attempts must be at least 1.")

    if not temporary.is_dir():
        raise FileNotFoundError(
            f"Temporary checkpoint directory does not exist: {temporary}"
        )

    if not (temporary / "COMPLETE").is_file():
        raise RuntimeError(
            "Temporary checkpoint cannot be committed because "
            f"its COMPLETE marker is missing: {temporary}"
        )

    checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if checkpoint.exists():
        shutil.rmtree(
            checkpoint,
        )

    delay = initial_delay
    last_error: OSError | None = None

    for attempt in range(
        1,
        attempts + 1,
    ):
        try:
            # The destination has already been removed, so rename()
            # expresses the intended operation more directly than
            # replace() for a directory.
            temporary.rename(checkpoint)
            return

        except PermissionError as error:
            last_error = error

        except OSError as error:
            # On Windows, a transient sharing violation may be exposed
            # as a general OSError rather than PermissionError.
            last_error = error

        if attempt < attempts:
            time.sleep(delay)
            delay *= 2.0

    # Atomic directory renaming was unavailable. shutil.move() first
    # attempts a rename and can fall back to copying and deleting.
    try:
        shutil.move(
            str(temporary),
            str(checkpoint),
        )
    except OSError as error:
        if last_error is not None:
            raise RuntimeError(
                "Could not commit the completed checkpoint "
                "directory after repeated Windows filesystem "
                "retries. "
                f"Temporary directory: {temporary}; "
                f"destination: {checkpoint}."
            ) from last_error

        raise RuntimeError(
            f"Could not commit the completed checkpoint directory: {checkpoint}."
        ) from error

    if not (checkpoint / "COMPLETE").is_file():
        raise RuntimeError(
            "The checkpoint directory was moved, but its COMPLETE "
            f"marker is missing: {checkpoint}"
        )


def write_fold_checkpoint(
    result: OuterFoldResult,
    checkpoint: Path,
    data_hash: str,
    config_hash: str,
) -> None:
    """Write and commit one complete outer-fold checkpoint."""

    temporary = checkpoint.with_name(f"{checkpoint.name}.tmp")

    if temporary.exists():
        shutil.rmtree(
            temporary,
            ignore_errors=True,
        )

    temporary.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:
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

        write_json_atomic(
            temporary / "metrics.json",
            result.metrics,
        )

        result.class_metrics.to_parquet(
            temporary / "class_metrics.parquet",
            index=False,
        )

        result.trials.to_parquet(
            temporary / "trials.parquet",
            index=False,
        )

        result.predictions.to_parquet(
            temporary / "predictions.parquet",
            index=False,
        )

        joblib.dump(
            result.fitted_model,
            temporary / "model.joblib",
        )

        # Write COMPLETE last. Its presence means every checkpoint
        # artifact has been written successfully.
        (temporary / "COMPLETE").write_text(
            "completed\n",
            encoding="utf-8",
        )

        _commit_checkpoint_directory(
            temporary=temporary,
            checkpoint=checkpoint,
        )

    except BaseException:
        # If the temporary directory still exists, the commit did not
        # complete. Remove it so a future retry starts cleanly.
        if temporary.exists():
            shutil.rmtree(
                temporary,
                ignore_errors=True,
            )

        raise


def rebuild_model_results(
    tasks: list[OuterFoldTask],
    store: FileRunStore,
    data_hash: str,
    config_hash: str,
) -> None:
    fold_rows = []
    prediction_frames = []
    class_metric_frames = []
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
        class_metric_frames.append(
            pd.read_parquet(checkpoint / "class_metrics.parquet")
        )

    fold_frame = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    class_metrics = pd.concat(class_metric_frames, ignore_index=True)

    model_results = store.results / tasks[0].model.id
    model_results.mkdir(parents=True, exist_ok=True)
    fold_frame.to_csv(model_results / "fold_metrics.csv", index=False)
    predictions.to_parquet(model_results / "predictions.parquet", index=False)
    class_metrics.to_csv(model_results / "class_metrics.csv", index=False)

    metric_columns = [
        "log_loss",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_specificity",
        "weighted_specificity",
    ]

    fold_frame[metric_columns].agg(["mean", "std", "median", "min", "max"]).T.to_csv(
        model_results / "summary.csv"
    )

    class_metrics_summary = class_metrics.groupby("class", sort=True)[
        ["sensitivity", "specificity", "precision", "f1"]
    ].agg(["mean", "std", "median", "min", "max"])

    class_metrics_summary[("support", "mean_fold_support")] = class_metrics.groupby(
        "class", sort=True
    )["support"].mean()
    class_metrics_summary[("support", "total_oof_support")] = class_metrics.groupby(
        "class", sort=True
    )["support"].sum()

    class_metrics_summary.to_csv(model_results / "class_metrics_summary.csv")


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

    all_tasks = build_outer_fold_tasks(
        config=config,
        model=model,
        splits=splits,
        row_count=len(X),
    )

    total = config.validation.outer_splits * config.validation.outer_repeats

    stop_event_recorded = False

    def should_stop() -> str | None:
        nonlocal stop_event_recorded
        if store.requested("CANCEL"):
            if not stop_event_recorded:
                store.event("cancel_requested")
                stop_event_recorded = True
            return "cancelled"
        if store.requested("PAUSE"):
            if not stop_event_recorded:
                store.event("pause_requested")
                stop_event_recorded = True
            return "paused"
        return None

    pending_tasks = []
    for task in all_tasks:
        stop_state = should_stop()
        if stop_state is not None:
            store.update_status(state=stop_state)
            store.event(f"run_{stop_state}")
            return
        if store.checkpoint_complete(
            model.id, task.repeat, task.fold, data_hash, config_hash
        ):
            continue
        pending_tasks.append(task)

    completed_outer_folds = len(all_tasks) - len(pending_tasks)

    model_started_at = perf_counter()

    print(
        f"[{model.id}] Starting outer-fold evaluation: "
        f"{len(pending_tasks)} pending, "
        f"{completed_outer_folds} restored, "
        f"{total} total, "
        f"{config.execution.workers} worker(s)",
        flush=True,
    )

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

        elapsed = perf_counter() - model_started_at
        log_loss_value = float(result.metrics["log_loss"])

        print(
            f"[{result.model_id}] "
            f"{completed_outer_folds:>{len(str(total))}}/"
            f"{total} completed"
            f" | repeat {result.repeat}, fold {result.fold}"
            f" | log_loss={log_loss_value:.4f}"
            f" | elapsed={format_duration(elapsed)}",
            flush=True,
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
            should_stop,
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

    stop_state = should_stop()
    if stop_state is not None:
        store.update_status(state=stop_state)
        store.event(f"run_{stop_state}")
        return

    rebuild_model_results(all_tasks, store, data_hash, config_hash)

    elapsed = perf_counter() - model_started_at

    print(
        f"[{model.id}] Outer-fold evaluation completed in {format_duration(elapsed)}",
        flush=True,
    )


def fit_final_model(
    config: ExperimentConfig,
    model: ModelConfig,
    X: pd.DataFrame,
    y: pd.Series,
    store: FileRunStore,
) -> None:
    """Tune and fit the final model using the complete dataset."""

    if len(X) != len(y):
        raise ValueError(
            "Predictor and target row counts do not match: "
            f"X has {len(X)} rows and y has {len(y)} rows."
        )

    if X.empty:
        raise ValueError("Cannot fit the final model on an empty dataset.")

    if y.isna().any():
        raise ValueError("The final-model target contains missing values.")

    expected_classes = np.asarray(
        sorted(str(value) for value in y.unique()),
        dtype=object,
    )

    if len(expected_classes) < 2:
        raise ValueError("Final classification requires at least two target classes.")

    best, trials = inner_search(
        config=config,
        model=model,
        X=X,
        y=y,
        seed=config.validation.random_state,
    )

    pipeline = build_pipeline(
        config,
        model,
        n_knots=int(best["n_knots"]),
        degree=int(best["degree"]),
        C=float(best["C"]),
        interaction_scale=float(best["interaction_scale"]),
    )

    with warnings.catch_warnings():
        if config.execution.stop_on_convergence_warning:
            warnings.filterwarnings(
                "error",
                category=ConvergenceWarning,
            )

        pipeline.fit(X, y)

    transformer = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["classifier"]

    actual_classes = np.asarray(
        [str(value) for value in classifier.classes_],
        dtype=object,
    )

    if not np.array_equal(
        actual_classes,
        expected_classes,
    ):
        raise RuntimeError(
            "The final classifier classes do not match the "
            "complete training target. "
            f"Expected {expected_classes.tolist()}, "
            f"received {actual_classes.tolist()}."
        )

    feature_names = np.asarray(
        transformer.get_feature_names_out(),
        dtype=object,
    )

    parameters = extract_class_score_parameters(classifier)

    parameter_classes = np.asarray(
        [str(value) for value in parameters.classes],
        dtype=object,
    )

    if not np.array_equal(
        parameter_classes,
        actual_classes,
    ):
        raise RuntimeError(
            "Extracted class-score parameters do not match the "
            "fitted classifier classes. "
            f"Classifier classes: {actual_classes.tolist()}; "
            f"parameter classes: {parameter_classes.tolist()}."
        )

    if parameters.intercepts.shape != (len(actual_classes),):
        raise RuntimeError(
            "The number of normalized intercepts does not match "
            "the number of target classes. "
            f"Received {parameters.intercepts.shape}; expected "
            f"({len(actual_classes)},)."
        )

    expected_coefficient_shape = (
        len(actual_classes),
        len(feature_names),
    )

    if parameters.coefficients.shape != (expected_coefficient_shape):
        raise RuntimeError(
            "The normalized coefficient matrix has an unexpected "
            "shape. "
            f"Received {parameters.coefficients.shape}; expected "
            f"{expected_coefficient_shape}."
        )

    rows: list[dict[str, object]] = []

    for class_index, class_name in enumerate(parameter_classes):
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

    components = pd.DataFrame(
        rows,
        columns=[
            "class",
            "component",
            "coefficient",
        ],
    )

    expected_component_rows = len(actual_classes) * (len(feature_names) + 1)

    if len(components) != expected_component_rows:
        raise RuntimeError(
            "The exported component-row count is incorrect. "
            f"Received {len(components)} rows; expected "
            f"{expected_component_rows}."
        )

    directory = store.models / model.id
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = directory / "model.joblib"
    temporary_model_path = directory / "model.joblib.tmp"

    if temporary_model_path.exists():
        temporary_model_path.unlink()

    joblib.dump(
        pipeline,
        temporary_model_path,
    )

    temporary_model_path.replace(model_path)

    write_json_atomic(
        directory / "best_parameters.json",
        {
            "n_knots": int(best["n_knots"]),
            "degree": int(best["degree"]),
            "C": float(best["C"]),
            "interaction_scale": float(best["interaction_scale"]),
            "mean_log_loss": float(best["mean_log_loss"]),
            "std_log_loss": float(best["std_log_loss"]),
        },
    )

    trials.to_parquet(
        directory / "search_trials.parquet",
        index=False,
    )

    components.to_csv(
        directory / "components.csv",
        index=False,
    )

    write_json_atomic(
        directory / "model_metadata.json",
        {
            "model_id": model.id,
            "fitted_at_utc": utc_now(),
            "training_rows": len(X),
            "predictor_count": X.shape[1],
            "transformed_feature_count": len(feature_names),
            "classification_type": (
                "binary" if len(actual_classes) == 2 else "multiclass"
            ),
            "class_count": len(actual_classes),
            "classes": actual_classes.tolist(),
        },
    )


def _safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(
        numerator,
        dtype=np.float64,
    )

    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator != 0,
    )

    return result


def calculate_class_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    classes: np.ndarray,
    *,
    model_id: str,
    repeat: int,
    fold: int,
) -> pd.DataFrame:
    confusion_matrices = multilabel_confusion_matrix(
        y_true,
        predictions,
        labels=classes,
    )

    true_negatives = confusion_matrices[:, 0, 0]
    false_positives = confusion_matrices[:, 0, 1]
    false_negatives = confusion_matrices[:, 1, 0]
    true_positives = confusion_matrices[:, 1, 1]

    specificity = _safe_ratio(
        true_negatives.astype(np.float64),
        (true_negatives + false_positives).astype(np.float64),
    )

    precision, sensitivity, f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=classes,
        zero_division=0,
    )

    return pd.DataFrame(
        {
            "model_id": model_id,
            "repeat": repeat,
            "fold": fold,
            "class": [str(value) for value in classes],
            "true_negative": true_negatives,
            "false_positive": false_positives,
            "false_negative": false_negatives,
            "true_positive": true_positives,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1": f1,
            "support": support,
        }
    )
