from pathlib import Path

import pandas as pd

from gam_app.config import (
    ExecutionConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    SearchConfig,
    ValidationConfig,
)
from gam_app.io_utils import format_duration
from gam_app.workflow import create_run, execute_run


def test_format_duration():
    assert format_duration(0) == "00:00:00"
    assert format_duration(65) == "00:01:05"
    assert format_duration(3665) == "01:01:05"


def test_run_output_messages(multiclass_frame: pd.DataFrame, tmp_path: Path, capsys):
    data_path = tmp_path / "data.csv"
    multiclass_frame.to_csv(data_path, index=False)

    cfg = ExperimentConfig(
        name="test_progress",
        data_path=data_path,
        target="target",
        row_id=None,
        features={
            "x1": FeatureConfig(role="smooth"),
            "x2": FeatureConfig(role="smooth"),
            "x3": FeatureConfig(role="categorical", categories=("high", "low")),
        },
        models=(ModelConfig(id="gam_main"),),
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

    config_path = tmp_path / "config.yaml"
    from gam_app.config import dump_config_dict
    from gam_app.io_utils import write_yaml_atomic

    write_yaml_atomic(config_path, dump_config_dict(cfg))

    workspace = tmp_path / "workspace"
    run_dir = create_run(config_path, workspace)

    capsys.readouterr()
    execute_run(run_dir)
    captured = capsys.readouterr().out

    assert (
        "[gam_main] Starting outer-fold evaluation: "
        "2 pending, 0 restored, 2 total, 1 worker(s)"
    ) in captured
    assert "[gam_main] 1/2 completed | repeat 1, fold 1 | log_loss=" in captured
    assert "[gam_main] 2/2 completed | repeat 1, fold 2 | log_loss=" in captured
    assert "[gam_main] Outer-fold evaluation completed in 00:00:" in captured
    assert "[gam_main] Fitting final model on" in captured
    assert "[gam_main] Final model completed in 00:00:" in captured
    assert "Run completed in 00:00:" in captured
