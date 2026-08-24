from __future__ import annotations

import pandas as pd

from gam_app.diagnostics import (
    build_exact_predictor_signatures,
    build_near_duplicate_signatures,
    conflicting_duplicate_target_report,
)


def test_exact_duplicate_rows_share_signature() -> None:
    X = pd.DataFrame(
        {
            "x1": [1.0, 1.0, 2.0],
            "x2": ["a", "a", "b"],
        }
    )

    signatures = build_exact_predictor_signatures(X)

    assert signatures.iloc[0] == signatures.iloc[1]
    assert signatures.iloc[0] != signatures.iloc[2]


def test_different_targets_do_not_change_predictor_signature() -> None:
    X = pd.DataFrame(
        {
            "x1": [1.0, 1.0],
            "x2": ["a", "a"],
        }
    )

    y = pd.Series(["A", "B"])

    signatures = build_exact_predictor_signatures(X)

    assert signatures.iloc[0] == signatures.iloc[1]
    assert y.iloc[0] != y.iloc[1]


def test_conflicting_duplicate_targets_are_reported() -> None:
    X = pd.DataFrame(
        {
            "x1": [1.0, 1.0, 2.0],
        }
    )
    y = pd.Series(["A", "B", "A"])
    row_ids = pd.Series(["r1", "r2", "r3"])

    conflicts = conflicting_duplicate_target_report(
        X,
        y,
        row_ids,
    )

    assert set(conflicts["row_id"]) == {"r1", "r2"}


def test_near_duplicate_rounding_precision() -> None:
    X = pd.DataFrame(
        {
            "x": [1.00001, 1.00002, 2.0],
        }
    )

    rounded_3 = build_near_duplicate_signatures(
        X,
        decimals=3,
    )
    rounded_6 = build_near_duplicate_signatures(
        X,
        decimals=6,
    )

    assert rounded_3.iloc[0] == rounded_3.iloc[1]
    assert rounded_6.iloc[0] != rounded_6.iloc[1]
