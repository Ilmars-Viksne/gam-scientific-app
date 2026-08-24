from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from gam_app.config import (
    ExecutionConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    SearchConfig,
    ValidationConfig,
)
from gam_app.evaluation import (
    OuterFoldResult,
    build_outer_fold_tasks,
    execute_outer_fold_task,
    run_outer_folds_in_parallel,
    run_outer_folds_sequentially,
)
from gam_app.exceptions import ConfigurationError
from gam_app.splitting import SplitContext, create_split_manifest


def config(
    tmp_path: Path,
    *,
    workers: int = 1,
) -> ExperimentConfig:
    return ExperimentConfig(
        name="test",
        data_path=tmp_path / "unused.csv",
        target="target",
        row_id=None,
        features={
            "x1": FeatureConfig(
                role="smooth",
            ),
            "x2": FeatureConfig(
                role="smooth",
            ),
            "x3": FeatureConfig(
                role="categorical",
                categories=(
                    "high",
                    "low",
                ),
            ),
        },
        models=(
            ModelConfig(
                id="main",
            ),
        ),
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
        execution=ExecutionConfig(
            workers=workers,
        ),
    )


def assert_metrics_equal(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    """Compare fold metrics, allowing tiny numeric differences."""

    assert actual.keys() == expected.keys()

    for key in actual:
        actual_value = actual[key]
        expected_value = expected[key]

        if isinstance(actual_value, float) or isinstance(
            expected_value,
            float,
        ):
            assert actual_value == pytest.approx(
                expected_value,
                rel=1e-12,
                abs=1e-12,
            )
        else:
            assert actual_value == expected_value


def assert_outer_fold_results_equal(
    actual: OuterFoldResult,
    expected: OuterFoldResult,
) -> None:
    """Compare two outer-fold results."""

    assert actual.model_id == expected.model_id
    assert actual.repeat == expected.repeat
    assert actual.fold == expected.fold

    assert_metrics_equal(
        actual.metrics,
        expected.metrics,
    )

    pd.testing.assert_frame_equal(
        actual.class_metrics,
        expected.class_metrics,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )

    pd.testing.assert_frame_equal(
        actual.trials,
        expected.trials,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )

    pd.testing.assert_frame_equal(
        actual.predictions,
        expected.predictions,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def prepare_outer_fold_tasks(
    multiclass_frame: pd.DataFrame,
    tmp_path: Path,
    *,
    workers: int = 1,
):
    """Create the test configuration, data, and outer-fold tasks."""

    cfg = config(
        tmp_path,
        workers=workers,
    )

    model = cfg.models[0]

    X = multiclass_frame[
        [
            "x1",
            "x2",
            "x3",
        ]
    ].copy()

    y = multiclass_frame["target"].copy()

    row_ids = pd.Series(
        range(len(X)),
        index=X.index,
        name="row_id",
    )

    context = SplitContext(
        X=X,
        y=y,
        row_ids=pd.Series(row_ids.astype(str)),
        groups=None,
        times=None,
    )

    splits = create_split_manifest(
        cfg,
        context,
    )

    tasks = build_outer_fold_tasks(
        config=cfg,
        model=model,
        splits=splits,
        row_count=len(X),
    )

    return (
        cfg,
        model,
        X,
        y,
        row_ids,
        splits,
        tasks,
    )


def test_outer_fold_tasks_are_complete_and_ordered(
    multiclass_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    (
        cfg,
        _,
        X,
        _,
        _,
        _,
        tasks,
    ) = prepare_outer_fold_tasks(
        multiclass_frame,
        tmp_path,
    )

    expected_task_count = cfg.validation.outer_splits * cfg.validation.outer_repeats

    assert len(tasks) == expected_task_count

    assert [
        (
            task.repeat,
            task.fold,
        )
        for task in tasks
    ] == [
        (
            1,
            1,
        ),
        (
            1,
            2,
        ),
    ]

    expected_indices = set(range(len(X)))

    for task in tasks:
        train_indices = set(task.train_indices)
        test_indices = set(task.test_indices)

        assert train_indices
        assert test_indices

        assert not (train_indices & test_indices)

        assert (train_indices | test_indices) == expected_indices


def test_outer_fold_task_is_deterministic(
    multiclass_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    (
        cfg,
        _,
        X,
        y,
        row_ids,
        _,
        tasks,
    ) = prepare_outer_fold_tasks(
        multiclass_frame,
        tmp_path,
    )

    task = tasks[0]

    first = execute_outer_fold_task(
        task,
        cfg,
        X,
        y,
        row_ids,
    )

    second = execute_outer_fold_task(
        task,
        cfg,
        X,
        y,
        row_ids,
    )

    assert_outer_fold_results_equal(
        first,
        second,
    )


def test_parallel_outer_fold_results_match_sequential(
    multiclass_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    (
        cfg,
        _,
        X,
        y,
        row_ids,
        _,
        tasks,
    ) = prepare_outer_fold_tasks(
        multiclass_frame,
        tmp_path,
        workers=2,
    )

    sequential_started: list[tuple[int, int]] = []

    sequential_results: list[OuterFoldResult] = []

    run_outer_folds_sequentially(
        tasks=tasks,
        config=cfg,
        X=X,
        y=y,
        row_ids=row_ids,
        groups=None,
        times=None,
        on_started=lambda task: sequential_started.append(
            (
                task.repeat,
                task.fold,
            )
        ),
        on_completed=sequential_results.append,
    )

    parallel_started: list[tuple[int, int]] = []

    parallel_results: list[OuterFoldResult] = []

    run_outer_folds_in_parallel(
        tasks=tasks,
        config=cfg,
        X=X,
        y=y,
        row_ids=row_ids,
        groups=None,
        times=None,
        workers=2,
        on_started=lambda task: parallel_started.append(
            (
                task.repeat,
                task.fold,
            )
        ),
        on_completed=parallel_results.append,
        should_stop=lambda: False,
    )

    sequential_by_fold = {
        (
            result.repeat,
            result.fold,
        ): result
        for result in sequential_results
    }

    parallel_by_fold = {
        (
            result.repeat,
            result.fold,
        ): result
        for result in parallel_results
    }

    assert set(parallel_started) == set(sequential_started)

    assert parallel_by_fold.keys() == (sequential_by_fold.keys())

    for key, sequential_result in sequential_by_fold.items():
        parallel_result = parallel_by_fold[key]

        assert_outer_fold_results_equal(
            parallel_result,
            sequential_result,
        )


def test_invalid_worker_count_is_rejected(
    tmp_path: Path,
) -> None:
    cfg = config(
        tmp_path,
        workers=0,
    )

    with pytest.raises(
        ConfigurationError,
        match=r"execution\.workers must be at least 1",
    ):
        cfg.validate()
