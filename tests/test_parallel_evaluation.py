import pandas as pd

from gam_app.config import (
    ExecutionConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    SearchConfig,
    ValidationConfig,
)
from gam_app.evaluation import (
    build_outer_fold_tasks,
    create_split_manifest,
    execute_outer_fold_task,
    run_outer_folds_in_parallel,
    run_outer_folds_sequentially,
)


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


def test_outer_fold_task_is_deterministic(multiclass_frame, tmp_path) -> None:
    cfg = config(tmp_path)
    model = cfg.models[0]
    X = multiclass_frame[["x1", "x2", "x3"]]
    y = multiclass_frame["target"]
    row_ids = pd.Series(range(len(X)), index=X.index)
    splits = create_split_manifest(cfg, X, y, row_ids)
    task = build_outer_fold_tasks(cfg, model, splits)[0]

    first = execute_outer_fold_task(task, cfg, X, y, row_ids)
    second = execute_outer_fold_task(task, cfg, X, y, row_ids)

    assert first.metrics == second.metrics
    pd.testing.assert_frame_equal(first.trials, second.trials)
    pd.testing.assert_frame_equal(first.predictions, second.predictions)


def test_parallel_outer_fold_results_match_sequential(
    multiclass_frame, tmp_path
) -> None:
    cfg = config(tmp_path)
    model = cfg.models[0]
    X = multiclass_frame[["x1", "x2", "x3"]]
    y = multiclass_frame["target"]
    row_ids = pd.Series(range(len(X)), index=X.index)
    splits = create_split_manifest(cfg, X, y, row_ids)
    tasks = build_outer_fold_tasks(cfg, model, splits)

    sequential_results = []
    run_outer_folds_sequentially(
        tasks,
        cfg,
        X,
        y,
        row_ids,
        lambda task: None,
        sequential_results.append,
    )

    parallel_results = []
    run_outer_folds_in_parallel(
        tasks,
        cfg,
        X,
        y,
        row_ids,
        2,
        lambda task: None,
        parallel_results.append,
        lambda: False,
    )

    sequential_by_fold = {
        (result.repeat, result.fold): result for result in sequential_results
    }
    parallel_by_fold = {
        (result.repeat, result.fold): result for result in parallel_results
    }

    assert parallel_by_fold.keys() == sequential_by_fold.keys()
    for key, sequential in sequential_by_fold.items():
        parallel = parallel_by_fold[key]
        assert parallel.metrics == sequential.metrics
        pd.testing.assert_frame_equal(parallel.trials, sequential.trials)
        pd.testing.assert_frame_equal(parallel.predictions, sequential.predictions)
