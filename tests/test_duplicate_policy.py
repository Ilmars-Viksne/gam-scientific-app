from __future__ import annotations

import pandas as pd
import pytest

from gam_app.config import DuplicateGroupConfig
from gam_app.diagnostics import analyze_duplicate_groups
from gam_app.exceptions import DataValidationError
from gam_app.workflow import apply_duplicate_group_policy


def test_report_policy_does_not_create_effective_groups() -> None:
    X = pd.DataFrame({"x": [1.0, 1.0]})
    y = pd.Series(["A", "A"])
    row_ids = pd.Series(["r1", "r2"])
    cfg = DuplicateGroupConfig()
    analysis = analyze_duplicate_groups(X, y, row_ids, cfg)

    result = apply_duplicate_group_policy(
        policy="report",
        configured_groups=None,
        duplicate_analysis=analysis,
        row_ids=row_ids,
        row_count=len(X),
    )

    assert result.effective_groups is None


def test_report_policy_preserves_configured_groups() -> None:
    X = pd.DataFrame({"x": [1.0, 1.0]})
    y = pd.Series(["A", "A"])
    row_ids = pd.Series(["r1", "r2"])
    cfg = DuplicateGroupConfig()
    analysis = analyze_duplicate_groups(X, y, row_ids, cfg)
    configured_groups = pd.Series(["g1", "g2"])

    result = apply_duplicate_group_policy(
        policy="report",
        configured_groups=configured_groups,
        duplicate_analysis=analysis,
        row_ids=row_ids,
        row_count=len(X),
    )

    assert result.effective_groups.equals(configured_groups)


def test_error_policy_rejects_exact_duplicates() -> None:
    X = pd.DataFrame({"x": [1.0, 1.0]})
    y = pd.Series(["A", "A"])
    row_ids = pd.Series(["r1", "r2"])
    cfg = DuplicateGroupConfig()
    analysis = analyze_duplicate_groups(X, y, row_ids, cfg)

    with pytest.raises(DataValidationError, match="Duplicate observations were found"):
        apply_duplicate_group_policy(
            policy="error",
            configured_groups=None,
            duplicate_analysis=analysis,
            row_ids=row_ids,
            row_count=len(X),
        )


def test_error_policy_rejects_near_duplicates() -> None:
    X = pd.DataFrame({"x1": [1.0, 1.00000001], "x2": [2.0, 2.0]})
    y = pd.Series(["A", "A"])
    row_ids = pd.Series(["r1", "r2"])
    cfg = DuplicateGroupConfig(rounding_decimals=6, near_duplicate_threshold=1.0)
    analysis = analyze_duplicate_groups(X, y, row_ids, cfg)

    with pytest.raises(DataValidationError, match="Duplicate observations were found"):
        apply_duplicate_group_policy(
            policy="error",
            configured_groups=None,
            duplicate_analysis=analysis,
            row_ids=row_ids,
            row_count=len(X),
        )


def test_error_policy_allows_duplicate_free_data() -> None:
    X = pd.DataFrame({"x": [1.0, 2.0]})
    y = pd.Series(["A", "A"])
    row_ids = pd.Series(["r1", "r2"])
    cfg = DuplicateGroupConfig()
    analysis = analyze_duplicate_groups(X, y, row_ids, cfg)

    result = apply_duplicate_group_policy(
        policy="error",
        configured_groups=None,
        duplicate_analysis=analysis,
        row_ids=row_ids,
        row_count=len(X),
    )
    assert result.effective_groups is None


def test_group_policy_groups_exact_duplicates() -> None:
    X = pd.DataFrame({"x": [1.0, 1.0, 2.0]})
    y = pd.Series(["A", "A", "B"])
    row_ids = pd.Series(["r1", "r2", "r3"])
    cfg = DuplicateGroupConfig()
    analysis = analyze_duplicate_groups(X, y, row_ids, cfg)

    result = apply_duplicate_group_policy(
        policy="group",
        configured_groups=None,
        duplicate_analysis=analysis,
        row_ids=row_ids,
        row_count=len(X),
    )

    groups = result.effective_groups
    assert groups.iloc[0] == groups.iloc[1]
    assert groups.iloc[0] != groups.iloc[2]


def test_group_policy_groups_near_duplicates() -> None:
    X = pd.DataFrame({"x1": [1.0, 1.00000001, 5.0], "x2": [2.0, 2.0, 10.0]})
    y = pd.Series(["A", "A", "B"])
    row_ids = pd.Series(["r1", "r2", "r3"])
    cfg = DuplicateGroupConfig(rounding_decimals=6, near_duplicate_threshold=1.0)
    analysis = analyze_duplicate_groups(X, y, row_ids, cfg)

    result = apply_duplicate_group_policy(
        policy="group",
        configured_groups=None,
        duplicate_analysis=analysis,
        row_ids=row_ids,
        row_count=len(X),
    )

    groups = result.effective_groups
    assert groups.iloc[0] == groups.iloc[1]
    assert groups.iloc[0] != groups.iloc[2]


def test_group_policy_works_without_configured_group_column() -> None:
    X = pd.DataFrame({"x": [1.0, 1.0]})
    y = pd.Series(["A", "A"])
    row_ids = pd.Series(["r1", "r2"])
    cfg = DuplicateGroupConfig()
    analysis = analyze_duplicate_groups(X, y, row_ids, cfg)

    result = apply_duplicate_group_policy(
        policy="group",
        configured_groups=None,
        duplicate_analysis=analysis,
        row_ids=row_ids,
        row_count=len(X),
    )

    assert result.effective_groups is not None


def test_transitive_merging_configured_near_and_exact() -> None:
    # rows 0 and 1 share configured group A
    # rows 1 and 2 are near duplicates (1 mismatch out of 4)
    # rows 2 and 3 are exact duplicates
    # row 4 is isolated
    X = pd.DataFrame(
        {
            "c1": [1, 1, 1, 1, 99],
            "c2": [2, 2, 2, 2, 99],
            "c3": [3, 3, 3, 3, 99],
            "c4": [4, 100, 2, 2, 99],
        }
    )
    y = pd.Series(["A"] * 5)
    row_ids = pd.Series(["r0", "r1", "r2", "r3", "r4"])
    cfg = DuplicateGroupConfig(near_duplicate_threshold=0.75)
    analysis = analyze_duplicate_groups(X, y, row_ids, cfg)

    configured_groups = pd.Series(["grpA", "grpA", "grpB", "grpC", "grpD"])

    result = apply_duplicate_group_policy(
        policy="group",
        configured_groups=configured_groups,
        duplicate_analysis=analysis,
        row_ids=row_ids,
        row_count=len(X),
    )

    eff = result.effective_groups
    # r0, r1, r2, r3 must receive the same effective group
    assert eff.iloc[0] == eff.iloc[1] == eff.iloc[2] == eff.iloc[3]
    assert eff.iloc[0] != eff.iloc[4]


def test_group_policy_duplicate_free_data_creates_singleton_groups() -> None:
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    y = pd.Series(["A", "B", "C"])
    row_ids = pd.Series(["r1", "r2", "r3"])
    cfg = DuplicateGroupConfig()
    analysis = analyze_duplicate_groups(X, y, row_ids, cfg)

    result = apply_duplicate_group_policy(
        policy="group",
        configured_groups=None,
        duplicate_analysis=analysis,
        row_ids=row_ids,
        row_count=len(X),
    )

    eff = result.effective_groups
    assert eff is not None
    assert eff.nunique() == 3
