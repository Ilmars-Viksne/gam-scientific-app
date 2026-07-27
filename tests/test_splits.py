import numpy as np

from gam_app.config import ValidationConfig
from gam_app.evaluation import create_split_manifest


def test_every_row_is_test_once_per_repeat(multiclass_frame):
    class Config:
        validation = ValidationConfig(3, 2, 2, 42)
    X = multiclass_frame[["x1", "x2", "x3"]]
    y = multiclass_frame.target
    ids = multiclass_frame.index.to_series().astype(str)
    manifest = create_split_manifest(Config(), X, y, ids)
    test = manifest[manifest.partition == "test"]
    counts = test.groupby(["repeat", "row_id"]).size()
    assert np.all(counts.to_numpy() == 1)
