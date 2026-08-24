import numpy as np
import pandas as pd

from gam_app.config import ExperimentConfig, FeatureConfig, ValidationConfig
from gam_app.splitting import SplitContext, create_split_manifest


def test_every_row_is_test_once_per_repeat(multiclass_frame):
    X = multiclass_frame[["x1", "x2", "x3"]]
    y = multiclass_frame.target
    ids = pd.Series(multiclass_frame.index.astype(str))

    cfg = ExperimentConfig(
        name="test_splits",
        data_path="dummy.csv",
        target="target",
        row_id=None,
        features={
            "x1": FeatureConfig(role="smooth"),
            "x2": FeatureConfig(role="smooth"),
            "x3": FeatureConfig(role="categorical"),
        },
        validation=ValidationConfig(
            outer_splits=3, outer_repeats=2, inner_splits=2, random_state=42
        ),
    )

    context = SplitContext(X=X, y=y, row_ids=ids, groups=None, times=None)
    manifest = create_split_manifest(cfg, context)
    test = manifest[manifest.partition == "test"]
    counts = test.groupby(["repeat", "row_id"]).size()
    assert np.all(counts.to_numpy() == 1)
