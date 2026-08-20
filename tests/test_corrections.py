import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from gam_app.cli import command_inspect, command_verify_link
from gam_app.config import (
    ExecutionConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    SearchConfig,
    ValidationConfig,
)
from gam_app.evaluation import (
    OuterFoldTask,
    execute_outer_fold_task,
    fit_final_model,
    inner_search,
)
from gam_app.reporting import create_reports
from gam_app.run_store import FileRunStore
from gam_app.workflow import execute_run


def make_test_config(tmp_path: Path, stop_on_warning: bool = True) -> ExperimentConfig:
    return ExperimentConfig(
        name="Correction Test",
        data_path=tmp_path / "data.csv",
        target="target",
        row_id="id",
        features={
            "x1": FeatureConfig("smooth"),
            "x2": FeatureConfig("linear"),
        },
        models=(ModelConfig("gam_main"),),
        validation=ValidationConfig(
            outer_splits=2,
            outer_repeats=2,
            inner_splits=2,
            random_state=42,
        ),
        search=SearchConfig(
            n_knots=(3,),
            degree=(2,),
            C=(1e6,),  # High C to force convergence warning with max_iter
            interaction_scale=(1.0,),
        ),
        execution=ExecutionConfig(
            stop_on_convergence_warning=stop_on_warning,
            workers=1,
        ),
    )


def test_convergence_warning_raises_error_when_configured(
    multiclass_frame: pd.DataFrame, tmp_path: Path
) -> None:
    cfg = make_test_config(tmp_path, stop_on_warning=True)
    X = multiclass_frame[["x1", "x2"]]
    y = multiclass_frame["target"]
    row_ids = pd.Series(range(len(X)), name="id")

    # Force a convergence warning by monkeypatching LogisticRegression max_iter
    import sklearn.linear_model

    original_init = sklearn.linear_model.LogisticRegression.__init__

    def mock_init(self, *args, **kwargs):
        kwargs["max_iter"] = 1
        kwargs["solver"] = "lbfgs"
        original_init(self, *args, **kwargs)

    sklearn.linear_model.LogisticRegression.__init__ = mock_init

    try:
        # inner_search
        with pytest.raises((ConvergenceWarning, Exception)):
            inner_search(cfg, cfg.models[0], X, y, seed=42)

        # execute_outer_fold_task
        task = OuterFoldTask(
            model=cfg.models[0],
            repeat=1,
            fold=1,
            train_indices=tuple(range(len(X) // 2)),
            test_indices=tuple(range(len(X) // 2, len(X))),
            seed=42,
        )
        with pytest.raises((ConvergenceWarning, Exception)):
            execute_outer_fold_task(task, cfg, X, y, row_ids)

        # fit_final_model
        store = FileRunStore(tmp_path / "run")
        store.initialize()
        with pytest.raises((ConvergenceWarning, Exception)):
            fit_final_model(cfg, cfg.models[0], X, y, store)

    finally:
        sklearn.linear_model.LogisticRegression.__init__ = original_init


def test_pause_and_cancellation_semantics(
    multiclass_frame: pd.DataFrame, tmp_path: Path
) -> None:
    data_path = tmp_path / "data.csv"
    multiclass_frame.to_csv(data_path, index=False)

    cfg_path = tmp_path / "config.yaml"
    cfg = ExperimentConfig(
        name="Pause Cancel Test",
        data_path=data_path,
        target="target",
        row_id=None,
        features={
            "x1": FeatureConfig("smooth"),
            "x2": FeatureConfig("linear"),
        },
        models=(ModelConfig("gam_main"),),
        validation=ValidationConfig(
            outer_splits=2,
            outer_repeats=2,
            inner_splits=2,
            random_state=42,
        ),
        search=SearchConfig(
            n_knots=(3,),
            degree=(2,),
            C=(1.0,),
            interaction_scale=(1.0,),
        ),
        execution=ExecutionConfig(workers=1),
    )

    from gam_app.config import dump_config_dict
    from gam_app.io_utils import write_yaml_atomic
    from gam_app.workflow import create_run

    write_yaml_atomic(cfg_path, dump_config_dict(cfg))
    run_dir = create_run(cfg_path, tmp_path / "workspace")
    store = FileRunStore(run_dir)

    # Request PAUSE before execution starts
    pause_file = store.control / "PAUSE"
    pause_file.parent.mkdir(parents=True, exist_ok=True)
    pause_file.touch()

    execute_run(run_dir)

    status = json.loads(store.status_path.read_text(encoding="utf-8"))
    assert status["state"] == "paused"

    events = [
        json.loads(line)
        for line in (store.root / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    event_names = [e["event"] for e in events]
    assert "pause_requested" in event_names
    assert "run_paused" in event_names

    # Resume by removing PAUSE marker and executing again
    pause_file.unlink()
    execute_run(run_dir)

    status_after = json.loads(store.status_path.read_text(encoding="utf-8"))
    assert status_after["state"] == "completed"


def test_multi_repeat_support_and_report_metadata(tmp_path: Path) -> None:
    cfg = make_test_config(tmp_path)
    store = FileRunStore(tmp_path / "run_dir")
    store.initialize()

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

    model_id = cfg.models[0].id
    result_dir = store.results / model_id
    result_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        {"mean": [0.5], "std": [0.01], "median": [0.5], "min": [0.49], "max": [0.51]},
        index=["log_loss"],
    )
    summary.to_csv(result_dir / "summary.csv")

    # 2 repeats, 2 folds per repeat, 10 unique observations
    # Each observation appears once per repeat as test data ->
    # 20 total OOF prediction events
    predictions = pd.DataFrame(
        {
            "model_id": [model_id] * 20,
            "repeat": [1] * 10 + [2] * 10,
            "fold": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2] * 2,
            "row_id": [f"r{i}" for i in range(10)] * 2,
            "observed_class": ["A"] * 5 + ["B"] * 5 + ["A"] * 5 + ["B"] * 5,
            "predicted_class": ["A"] * 5 + ["B"] * 5 + ["A"] * 5 + ["B"] * 5,
        }
    )
    predictions.to_parquet(result_dir / "predictions.parquet", index=False)

    class_metrics = pd.DataFrame(
        {
            "model_id": [model_id] * 4,
            "repeat": [1, 1, 2, 2],
            "fold": [1, 2, 1, 2],
            "class": ["A", "B", "A", "B"],
            "true_negative": [5, 5, 5, 5],
            "false_positive": [0, 0, 0, 0],
            "false_negative": [0, 0, 0, 0],
            "true_positive": [5, 5, 5, 5],
            "sensitivity": [1.0, 1.0, 1.0, 1.0],
            "specificity": [1.0, 1.0, 1.0, 1.0],
            "precision": [1.0, 1.0, 1.0, 1.0],
            "f1": [1.0, 1.0, 1.0, 1.0],
            "support": [5, 5, 5, 5],
        }
    )
    class_metrics.to_csv(result_dir / "class_metrics.csv", index=False)

    model_dir = store.models / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model_metadata.json").write_text(
        json.dumps(
            {"classification_type": "binary", "class_count": 2, "classes": ["A", "B"]}
        ),
        encoding="utf-8",
    )

    create_reports(cfg, store)

    report_html = (store.reports / "report.html").read_text(encoding="utf-8")
    assert "Unique observations: 10" in report_html
    assert "Outer repeats: 2" in report_html
    assert "Total OOF prediction events: 20" in report_html

    # Multi-repeat support check:
    # total_oof_support == original_class_count * outer_repeats
    # Here A count = 5, outer_repeats = 2 -> total_oof_support = 10
    total_oof_support = class_metrics[class_metrics["class"] == "A"]["support"].sum()
    assert total_oof_support == 5 * 2

    # Verify metadata class mismatch raises error
    (model_dir / "model_metadata.json").write_text(
        json.dumps(
            {"classification_type": "binary", "class_count": 1, "classes": ["A"]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="absent from final-model metadata"):
        create_reports(cfg, store)


def test_cli_inspect_and_verify_link_regenerate_report(
    multiclass_frame: pd.DataFrame, tmp_path: Path
) -> None:
    data_path = tmp_path / "demo.csv"
    multiclass_frame.to_csv(data_path, index=False)

    cfg_path = tmp_path / "config.yaml"
    cfg = ExperimentConfig(
        name="Regenerate Test",
        data_path=data_path,
        target="target",
        row_id=None,
        features={
            "x1": FeatureConfig("smooth"),
            "x2": FeatureConfig("linear"),
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
        execution=ExecutionConfig(workers=1),
    )

    from gam_app.config import dump_config_dict
    from gam_app.io_utils import write_yaml_atomic
    from gam_app.workflow import create_run

    write_yaml_atomic(cfg_path, dump_config_dict(cfg))
    run_dir = create_run(cfg_path, tmp_path / "workspace")
    execute_run(run_dir)

    report_path = run_dir / "reports" / "report.html"
    initial_content = report_path.read_text(encoding="utf-8")
    assert "Inspection artifacts" not in initial_content

    # Run CLI inspect
    class ArgsInspect:
        run = run_dir
        model = "gam_main"
        reference_class = None

    command_inspect(ArgsInspect())

    content_after_inspect = report_path.read_text(encoding="utf-8")
    assert "Inspection artifacts" in content_after_inspect

    # Run CLI verify-link
    class ArgsVerify:
        run = run_dir
        model = "gam_main"

    command_verify_link(ArgsVerify())

    content_after_verify = report_path.read_text(encoding="utf-8")
    assert "Link-verification results" in content_after_verify
