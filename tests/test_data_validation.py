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
from gam_app.data import validate_training_data
from gam_app.exceptions import DataValidationError


def make_config(
    tmp_path,
    *,
    outer_splits: int = 5,
    inner_splits: int = 5,
) -> ExperimentConfig:
    data_file = tmp_path / "data.csv"
    return ExperimentConfig(
        name="test_class_counts",
        data_path=data_file,
        target="target",
        row_id=None,
        features={
            "x1": FeatureConfig(role="smooth"),
        },
        models=(ModelConfig(id="main"),),
        validation=ValidationConfig(
            outer_splits=outer_splits,
            outer_repeats=1,
            inner_splits=inner_splits,
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


def test_validate_training_data_less_than_two_classes(tmp_path):
    cfg = make_config(tmp_path)
    df = pd.DataFrame({"target": ["A", "A", "A"], "x1": [1.0, 2.0, 3.0]})
    df.to_csv(cfg.data_path, index=False)

    with pytest.raises(
        DataValidationError,
        match="Classification requires at least two target classes.",
    ):
        validate_training_data(cfg)


def test_validate_training_data_smallest_class_less_than_outer_splits(tmp_path):
    cfg = make_config(tmp_path, outer_splits=5, inner_splits=2)
    # Target has 2 classes, but class 'B' has only 3 observations < 5 outer_splits
    df = pd.DataFrame(
        {
            "target": ["A"] * 20 + ["B"] * 3,
            "x1": range(23),
        }
    )
    df.to_csv(cfg.data_path, index=False)

    with pytest.raises(
        DataValidationError,
        match=(
            r"The least frequent target class contains 3 observations, "
            r"but outer_splits=5\."
        ),
    ):
        validate_training_data(cfg)


def test_validate_training_data_outer_ok_inner_fail(tmp_path):
    # smallest_class_count = 5, outer_splits = 5, inner_splits = 5
    # ceil(5/5) = 1 outer test count
    # outer train count = 5 - 1 = 4
    # 4 < 5 inner_splits -> fail
    cfg = make_config(tmp_path, outer_splits=5, inner_splits=5)
    df = pd.DataFrame(
        {
            "target": ["A"] * 20 + ["B"] * 5,
            "x1": range(25),
        }
    )
    df.to_csv(cfg.data_path, index=False)

    with pytest.raises(
        DataValidationError,
        match=(
            r"The least frequent target class may contain only 4 observations "
            r"in an outer training partition, but inner_splits=5\."
        ),
    ):
        validate_training_data(cfg)


def test_validate_training_data_valid_class_counts(tmp_path):
    # smallest_class_count = 10, outer_splits = 5, inner_splits = 2
    # ceil(10/5) = 2
    # outer train count = 10 - 2 = 8 >= 2 inner_splits -> pass
    cfg = make_config(tmp_path, outer_splits=5, inner_splits=2)
    df = pd.DataFrame(
        {
            "target": ["A"] * 20 + ["B"] * 10,
            "x1": range(30),
        }
    )
    df.to_csv(cfg.data_path, index=False)

    X, target, row_ids = validate_training_data(cfg)
    assert len(X) == 30
    assert len(target) == 30
