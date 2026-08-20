import json

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
    OuterFoldTask,
    rebuild_model_results,
    write_fold_checkpoint,
)
from gam_app.reporting import create_reports
from gam_app.run_store import FileRunStore


@pytest.fixture
def test_experiment_config(tmp_path) -> ExperimentConfig:
    return ExperimentConfig(
        name="Test Experiment",
        data_path=tmp_path / "data.csv",
        target="target",
        row_id="id",
        features={
            "x1": FeatureConfig("smooth"),
        },
        models=(ModelConfig("gam_main"),),
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


class MockPipeline:
    def fit(self, X, y):
        return self


def test_rebuild_model_results_aggregates_class_metrics(
    test_experiment_config, tmp_path
) -> None:
    store = FileRunStore(tmp_path / "run_dir")
    store.initialize()

    data_hash = "data_hash_123"
    config_hash = "config_hash_456"

    model = test_experiment_config.models[0]

    tasks = [
        OuterFoldTask(
            model=model,
            repeat=1,
            fold=1,
            train_indices=(0, 1),
            test_indices=(2, 3),
            seed=42,
        ),
        OuterFoldTask(
            model=model,
            repeat=1,
            fold=2,
            train_indices=(2, 3),
            seed=43,
            test_indices=(0, 1),
        ),
    ]

    pipeline = MockPipeline()

    for task in tasks:
        metrics = {
            "model_id": model.id,
            "repeat": task.repeat,
            "fold": task.fold,
            "log_loss": 0.5,
            "accuracy": 0.8,
            "balanced_accuracy": 0.8,
            "macro_f1": 0.8,
            "macro_specificity": 0.8,
            "weighted_specificity": 0.8,
        }
        class_metrics = pd.DataFrame(
            {
                "model_id": [model.id, model.id],
                "repeat": [task.repeat, task.repeat],
                "fold": [task.fold, task.fold],
                "class": ["A", "B"],
                "true_negative": [5, 6],
                "false_positive": [1, 0],
                "false_negative": [1, 1],
                "true_positive": [4, 5],
                "sensitivity": [0.8, 0.833],
                "specificity": [0.833, 1.0],
                "precision": [0.8, 1.0],
                "f1": [0.8, 0.9],
                "support": [5, 6],
            }
        )
        trials = pd.DataFrame(
            [{"n_knots": 3, "degree": 2, "C": 1.0, "mean_log_loss": 0.5}]
        )
        predictions = pd.DataFrame(
            {
                "model_id": [model.id, model.id],
                "repeat": [task.repeat, task.repeat],
                "fold": [task.fold, task.fold],
                "row_id": ["r1", "r2"],
                "observed_class": ["A", "B"],
                "predicted_class": ["A", "B"],
            }
        )

        result = OuterFoldResult(
            model_id=model.id,
            repeat=task.repeat,
            fold=task.fold,
            metrics=metrics,
            class_metrics=class_metrics,
            trials=trials,
            predictions=predictions,
            fitted_model=pipeline,
        )

        checkpoint = store.checkpoint_directory(model.id, task.repeat, task.fold)
        write_fold_checkpoint(result, checkpoint, data_hash, config_hash)

    rebuild_model_results(tasks, store, data_hash, config_hash)

    result_dir = store.results / model.id
    assert (result_dir / "class_metrics.csv").exists()
    assert (result_dir / "class_metrics_summary.csv").exists()

    cm_df = pd.read_csv(result_dir / "class_metrics.csv")
    assert len(cm_df) == 4  # 2 tasks * 2 classes
    expected_cols = {
        "model_id",
        "repeat",
        "fold",
        "class",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "support",
    }
    assert expected_cols.issubset(set(cm_df.columns))

    summary_df = pd.read_csv(
        result_dir / "class_metrics_summary.csv", header=[0, 1], index_col=0
    )
    assert ("sensitivity", "mean") in summary_df.columns
    assert ("support", "mean_fold_support") in summary_df.columns
    assert ("support", "total_oof_support") in summary_df.columns
    assert summary_df.loc["A", ("support", "total_oof_support")] == 10  # 5 + 5


def test_create_reports_generates_expected_outputs(
    test_experiment_config, tmp_path
) -> None:
    store = FileRunStore(tmp_path / "run_dir")
    store.initialize()

    # Create data_manifest.json
    data_manifest = {
        "path": "data.csv",
        "sha256": "abc",
        "rows": 10,
        "predictors": ["x1"],
        "target": "target",
        "class_counts": {"A": 5, "B": 5},
    }
    (store.root / "data_manifest.json").write_text(
        json.dumps(data_manifest), encoding="utf-8"
    )

    model_id = test_experiment_config.models[0].id
    result_dir = store.results / model_id
    result_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        {"mean": [0.5], "std": [0.01], "median": [0.5], "min": [0.49], "max": [0.51]},
        index=["log_loss"],
    )
    summary.to_csv(result_dir / "summary.csv")

    predictions = pd.DataFrame(
        {
            "model_id": [model_id] * 4,
            "repeat": [1, 1, 1, 1],
            "fold": [1, 1, 2, 2],
            "row_id": ["r1", "r2", "r3", "r4"],
            "observed_class": ["A", "A", "B", "B"],
            "predicted_class": ["A", "B", "B", "B"],
        }
    )
    predictions.to_parquet(result_dir / "predictions.parquet", index=False)

    class_metrics = pd.DataFrame(
        {
            "model_id": [model_id] * 2,
            "repeat": [1, 1],
            "fold": [1, 2],
            "class": ["A", "B"],
            "true_negative": [2, 2],
            "false_positive": [0, 1],
            "false_negative": [1, 0],
            "true_positive": [2, 2],
            "sensitivity": [0.66, 1.0],
            "specificity": [1.0, 0.66],
            "precision": [1.0, 0.66],
            "f1": [0.8, 0.8],
            "support": [2, 2],
        }
    )
    class_metrics.to_csv(result_dir / "class_metrics.csv", index=False)

    # Add model_metadata.json & best_parameters.json
    model_dir = store.models / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model_metadata.json").write_text(
        json.dumps({"classification_type": "binary", "class_count": 2}),
        encoding="utf-8",
    )
    (model_dir / "best_parameters.json").write_text(
        json.dumps(
            {
                "n_knots": 3,
                "degree": 2,
                "C": 1.0,
                "interaction_scale": 1.0,
                "mean_log_loss": 0.5,
                "std_log_loss": 0.0,
            }
        ),
        encoding="utf-8",
    )

    create_reports(test_experiment_config, store)

    report_path = store.reports / "report.html"
    assert report_path.exists()
    html_content = report_path.read_text(encoding="utf-8")

    # Check key sections
    assert "Dataset overview" in html_content
    assert "Class distribution" in html_content
    assert "Final-model metadata" in html_content
    assert "Final full-data inner-CV hyperparameter selection" in html_content
    assert "Per-class performance summary" in html_content
    assert "Confusion matrices" in html_content
    assert "Pooled row-normalized matrix" in html_content
    assert (
        "These confusion matrices pool held-out outer-fold predictions"
        in html_content
    )
    assert "Detailed per-class fold metrics" in html_content

    # Check plot image generation
    assert (store.plots / f"{model_id}_confusion_matrix.png").exists()
    assert (store.plots / f"{model_id}_confusion_matrix_normalized.png").exists()
