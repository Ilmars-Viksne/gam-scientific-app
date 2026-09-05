from __future__ import annotations

import pandas as pd
import pytest

from gam_app.config import (
    ConfigurationError,
    DuplicateGroupConfig,
    ExperimentConfig,
    FeatureConfig,
    ProfilingConfig,
)
from gam_app.diagnostics import (
    analyze_duplicate_groups,
    canonicalize_predictors,
)
from gam_app.exceptions import DataValidationError


def test_numeric_values_equal_after_configured_rounding() -> None:
    X = pd.DataFrame({"x": [1.00000001, 1.00000002]})
    row_ids = pd.Series(["r1", "r2"])
    y = pd.Series(["A", "A"])

    cfg = DuplicateGroupConfig(rounding_decimals=6, near_duplicate_threshold=1.0)
    analysis = analyze_duplicate_groups(X, y, row_ids, cfg)
    assert not analysis.proper_near_duplicate_groups.empty


def test_numeric_values_differ_above_rounding_precision() -> None:
    X = pd.DataFrame({"x": [1.001, 1.002]})
    row_ids = pd.Series(["r1", "r2"])
    y = pd.Series(["A", "A"])

    cfg = DuplicateGroupConfig(rounding_decimals=6, near_duplicate_threshold=1.0)
    analysis = analyze_duplicate_groups(X, y, row_ids, cfg)
    assert analysis.proper_near_duplicate_groups.empty


def test_string_edge_whitespace_is_ignored() -> None:
    X = pd.DataFrame({"s": ["hello ", "hello"]})
    canonical = canonicalize_predictors(X)
    assert canonical["s"].iloc[0] == canonical["s"].iloc[1]


def test_string_case_is_not_ignored() -> None:
    X = pd.DataFrame({"s": ["Hello", "hello"]})
    canonical = canonicalize_predictors(X)
    assert canonical["s"].iloc[0] != canonical["s"].iloc[1]


def test_missing_equals_missing_but_not_observed_value() -> None:
    X = pd.DataFrame({"x": [None, None, 1.0]})
    canonical = canonicalize_predictors(X)
    assert canonical["x"].iloc[0] == canonical["x"].iloc[1]
    assert canonical["x"].iloc[0] != canonical["x"].iloc[2]


def test_negative_and_positive_zero_are_equivalent() -> None:
    X = pd.DataFrame({"x": [-0.0, 0.0]})
    canonical = canonicalize_predictors(X)
    assert canonical["x"].iloc[0] == canonical["x"].iloc[1]


def test_threshold_is_inclusive_at_boundary() -> None:
    # 2 predictors, 1 matches => match_fraction = 0.5
    X = pd.DataFrame({"x1": [1.0, 1.0], "x2": [10.0, 20.0]})
    row_ids = pd.Series(["r1", "r2"])
    y = pd.Series(["A", "A"])

    cfg = DuplicateGroupConfig(near_duplicate_threshold=0.5)
    analysis = analyze_duplicate_groups(X, y, row_ids, cfg)
    assert not analysis.proper_near_duplicate_groups.empty


def test_pair_below_threshold_is_not_near_duplicate() -> None:
    X = pd.DataFrame({"x1": [1.0, 1.0], "x2": [10.0, 20.0]})
    row_ids = pd.Series(["r1", "r2"])
    y = pd.Series(["A", "A"])

    cfg = DuplicateGroupConfig(near_duplicate_threshold=0.51)
    analysis = analyze_duplicate_groups(X, y, row_ids, cfg)
    assert analysis.proper_near_duplicate_groups.empty


def test_threshold_one_requires_all_columns_to_match() -> None:
    X = pd.DataFrame({"x1": [1.0, 1.0], "x2": [10.0, 20.0]})
    row_ids = pd.Series(["r1", "r2"])
    y = pd.Series(["A", "A"])

    cfg = DuplicateGroupConfig(near_duplicate_threshold=1.0)
    analysis = analyze_duplicate_groups(X, y, row_ids, cfg)
    assert analysis.proper_near_duplicate_groups.empty


def test_threshold_boundaries_validation() -> None:
    with pytest.raises(ConfigurationError):
        ExperimentConfig(
            name="test",
            data_path=None,  # type: ignore
            target="y",
            row_id=None,
            features={"x": FeatureConfig(role="smooth")},
            profiling=ProfilingConfig(
                duplicate_groups=DuplicateGroupConfig(near_duplicate_threshold=0.0)
            ),
        ).validate()


def test_include_target_in_signature_rejected() -> None:
    with pytest.raises(
        ConfigurationError, match="include_target_in_signature=true is not supported"
    ):
        ExperimentConfig(
            name="test",
            data_path=None,  # type: ignore
            target="y",
            row_id=None,
            features={"x": FeatureConfig(role="smooth")},
            profiling=ProfilingConfig(
                duplicate_groups=DuplicateGroupConfig(include_target_in_signature=True)
            ),
        ).validate()


def test_exact_only_group_is_not_repeated_as_proper_near_group() -> None:
    X = pd.DataFrame({"x": [1.0, 1.0, 1.0]})
    row_ids = pd.Series(["r1", "r2", "r3"])
    y = pd.Series(["A", "A", "A"])

    cfg = DuplicateGroupConfig(near_duplicate_threshold=0.98)
    analysis = analyze_duplicate_groups(X, y, row_ids, cfg)
    assert not analysis.exact_duplicate_groups.empty
    assert analysis.proper_near_duplicate_groups.empty


def test_near_group_contains_multiple_exact_signatures() -> None:
    X = pd.DataFrame({"x1": [1.0, 1.00000001], "x2": [2.0, 2.0]})
    row_ids = pd.Series(["r1", "r2"])
    y = pd.Series(["A", "A"])

    cfg = DuplicateGroupConfig(rounding_decimals=6, near_duplicate_threshold=1.0)
    analysis = analyze_duplicate_groups(X, y, row_ids, cfg)
    assert analysis.exact_duplicate_groups.empty
    assert not analysis.proper_near_duplicate_groups.empty


def test_near_duplicate_groups_use_connected_components() -> None:
    # A ~ B (1 mismatch out of 4), B ~ C (1 mismatch out of 4),
    # A !~ C (2 mismatches out of 4)
    X = pd.DataFrame(
        {
            "c1": [1, 1, 0],
            "c2": [1, 1, 1],
            "c3": [1, 1, 1],
            "c4": [0, 1, 1],
        }
    )
    row_ids = pd.Series(["rA", "rB", "rC"])
    y = pd.Series(["A", "A", "A"])

    cfg = DuplicateGroupConfig(near_duplicate_threshold=0.75)
    analysis = analyze_duplicate_groups(X, y, row_ids, cfg)

    near_groups = analysis.proper_near_duplicate_groups
    assert not near_groups.empty
    assert set(near_groups["row_id"]) == {"rA", "rB", "rC"}
    assert near_groups["near_duplicate_group_id"].nunique() == 1


def test_near_duplicate_group_ids_are_stable_after_row_reordering() -> None:
    X1 = pd.DataFrame({"x1": [1.0, 1.00001], "x2": ["a", "a"]})
    r1 = pd.Series(["r1", "r2"])
    y1 = pd.Series(["A", "A"])

    X2 = X1.iloc[::-1].reset_index(drop=True)
    r2 = r1.iloc[::-1].reset_index(drop=True)
    y2 = y1.iloc[::-1].reset_index(drop=True)

    cfg = DuplicateGroupConfig(rounding_decimals=3)
    a1 = analyze_duplicate_groups(X1, y1, r1, cfg)
    a2 = analyze_duplicate_groups(X2, y2, r2, cfg)

    id1 = a1.proper_near_duplicate_groups["near_duplicate_group_id"].iloc[0]
    id2 = a2.proper_near_duplicate_groups["near_duplicate_group_id"].iloc[0]
    assert id1 == id2


def test_target_does_not_change_duplicate_membership() -> None:
    X = pd.DataFrame({"x": [1.0, 1.0]})
    y1 = pd.Series(["A", "A"])
    y2 = pd.Series(["A", "B"])
    row_ids = pd.Series(["r1", "r2"])

    cfg = DuplicateGroupConfig()
    a1 = analyze_duplicate_groups(X, y1, row_ids, cfg)
    a2 = analyze_duplicate_groups(X, y2, row_ids, cfg)

    assert (
        a1.exact_duplicate_groups["duplicate_group_id"].tolist()
        == a2.exact_duplicate_groups["duplicate_group_id"].tolist()
    )


def test_maximum_pairwise_rows_exceeded_raises_error() -> None:
    X = pd.DataFrame({"x": range(10)})
    y = pd.Series(["A"] * 10)
    row_ids = pd.Series([f"r{i}" for i in range(10)])

    cfg = DuplicateGroupConfig(maximum_pairwise_rows=5)
    with pytest.raises(
        DataValidationError,
        match="Near-duplicate analysis requires an exact pairwise scan",
    ):
        analyze_duplicate_groups(X, y, row_ids, cfg)
