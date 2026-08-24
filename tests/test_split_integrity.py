from __future__ import annotations

import pandas as pd
import pytest

from gam_app.splitting import (
    validate_forward_time_order,
    validate_no_group_leakage,
    validate_one_test_assignment_per_repeat,
)


def test_validate_no_group_leakage_raises_on_leak() -> None:
    manifest = pd.DataFrame(
        {
            "repeat": [1, 1, 1, 1],
            "fold": [1, 1, 1, 1],
            "partition": ["train", "test", "train", "test"],
            "group_id": ["g1", "g1", "g2", "g3"],
        }
    )
    with pytest.raises(ValueError, match="Group leakage detected"):
        validate_no_group_leakage(manifest)


def test_validate_forward_time_order_raises_on_leak() -> None:
    manifest = pd.DataFrame(
        {
            "repeat": [1, 1],
            "fold": [1, 1],
            "partition": ["train", "test"],
            "time": ["2025-01-02T00:00:00Z", "2025-01-01T00:00:00Z"],
        }
    )
    with pytest.raises(ValueError, match="Temporal leakage"):
        validate_forward_time_order(manifest)


def test_validate_one_test_assignment_per_repeat_raises_on_duplicate() -> None:
    manifest = pd.DataFrame(
        {
            "repeat": [1, 1],
            "row_id": ["r1", "r1"],
            "partition": ["test", "test"],
        }
    )
    with pytest.raises(ValueError, match="Every row must appear exactly once"):
        validate_one_test_assignment_per_repeat(manifest)
