from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from gam_app.config import DuplicateGroupConfig
from gam_app.diagnostics import (
    analyze_duplicate_groups,
    canonicalize_predictors,
)


# Hypothesis strategies
@st.composite
def numeric_dataframes(
    draw: st.DrawFn,
    min_rows: int = 2,
    max_rows: int = 10,
    min_cols: int = 1,
    max_cols: int = 6,
) -> pd.DataFrame:
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))

    cols = [f"x{i + 1}" for i in range(n_cols)]
    data = {}
    for c in cols:
        values = draw(
            st.lists(
                st.one_of(
                    st.floats(
                        allow_nan=False,
                        allow_infinity=False,
                        min_value=-1e5,
                        max_value=1e5,
                    ),
                    st.just(float("nan")),
                    st.just(0.0),
                    st.just(-0.0),
                ),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        data[c] = values
    return pd.DataFrame(data)


@st.composite
def mixed_dataframes(
    draw: st.DrawFn,
    min_rows: int = 2,
    max_rows: int = 8,
    min_cols: int = 1,
    max_cols: int = 4,
) -> pd.DataFrame:
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))

    cols = [f"feat_{i + 1}" for i in range(n_cols)]
    data = {}
    for c in cols:
        col_type = draw(st.sampled_from(["float", "string", "bool"]))
        if col_type == "float":
            vals = draw(
                st.lists(
                    st.one_of(
                        st.floats(
                            allow_nan=False,
                            allow_infinity=False,
                            min_value=-100,
                            max_value=100,
                        ),
                        st.just(float("nan")),
                    ),
                    min_size=n_rows,
                    max_size=n_rows,
                )
            )
        elif col_type == "string":
            vals = draw(
                st.lists(
                    st.text(
                        alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
                        max_size=10,
                    ),
                    min_size=n_rows,
                    max_size=n_rows,
                )
            )
        else:
            vals = draw(st.lists(st.booleans(), min_size=n_rows, max_size=n_rows))
        data[c] = vals

    return pd.DataFrame(data)


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(df=numeric_dataframes())
def test_match_fraction_is_symmetric_and_reflexive(df: pd.DataFrame) -> None:
    canonical = canonicalize_predictors(df, decimals=8)
    row_count = len(canonical)
    col_count = len(canonical.columns)
    matrix = canonical.to_numpy(dtype=str)

    # Reflexivity
    for i in range(row_count):
        row_i = matrix[i]
        matches = int(np.sum(row_i == row_i))
        score = matches / col_count if col_count > 0 else 1.0
        assert score == 1.0

    # Symmetry & Boundedness & Discrete Fraction
    for i in range(row_count):
        for j in range(i, row_count):
            row_i = matrix[i]
            row_j = matrix[j]
            matches_ij = int(np.sum(row_i == row_j))
            matches_ji = int(np.sum(row_j == row_i))
            assert matches_ij == matches_ji

            score_ij = matches_ij / col_count if col_count > 0 else 1.0
            score_ji = matches_ji / col_count if col_count > 0 else 1.0
            assert score_ij == score_ji

            assert 0.0 <= score_ij <= 1.0
            assert math.isfinite(score_ij)
            assert score_ij == pytest.approx(
                matches_ij / col_count if col_count > 0 else 1.0
            )


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    df=numeric_dataframes(),
    low_thresh=st.floats(min_value=0.1, max_value=0.5),
    high_thresh=st.floats(min_value=0.6, max_value=0.99),
)
def test_threshold_monotonicity(
    df: pd.DataFrame, low_thresh: float, high_thresh: float
) -> None:
    row_ids = pd.Series([f"r{i}" for i in range(len(df))])
    y = pd.Series(["A"] * len(df))

    cfg_low = DuplicateGroupConfig(
        enabled=True, near_duplicate_threshold=low_thresh, maximum_pairwise_rows=100
    )
    cfg_high = DuplicateGroupConfig(
        enabled=True, near_duplicate_threshold=high_thresh, maximum_pairwise_rows=100
    )

    res_low = analyze_duplicate_groups(df, y, row_ids, cfg_low)
    res_high = analyze_duplicate_groups(df, y, row_ids, cfg_high)

    assert len(res_low.near_edges) >= len(res_high.near_edges)


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(df=numeric_dataframes())
def test_exact_duplicates_are_never_proper_near_only(df: pd.DataFrame) -> None:
    assume(len(df) >= 2)
    # Duplicate row 0 to create exact duplicate
    df_dup = df.copy()
    df_dup.iloc[1] = df_dup.iloc[0]

    row_ids = pd.Series([f"r{i}" for i in range(len(df_dup))])
    y = pd.Series(["A"] * len(df_dup))

    cfg = DuplicateGroupConfig(
        enabled=True, near_duplicate_threshold=0.95, maximum_pairwise_rows=100
    )
    res = analyze_duplicate_groups(df_dup, y, row_ids, cfg)

    # Exactly duplicated rows must yield exact duplicate groups
    assert not res.exact_duplicate_groups.empty
    # And must not form proper-near-only groups if there are no other distinct exact signatures in the component
    if res.exact_signatures.nunique() == 1:
        assert res.proper_near_duplicate_groups.empty


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(df=mixed_dataframes())
def test_target_invariance(df: pd.DataFrame) -> None:
    row_ids = pd.Series([f"r{i}" for i in range(len(df))])
    y_one = pd.Series(["ClassA"] * len(df))
    y_two = pd.Series([f"Class_{i % 3}" for i in range(len(df))])

    cfg = DuplicateGroupConfig(
        enabled=True, near_duplicate_threshold=0.8, maximum_pairwise_rows=100
    )

    res_one = analyze_duplicate_groups(df, y_one, row_ids, cfg)
    res_two = analyze_duplicate_groups(df, y_two, row_ids, cfg)

    # Duplicate membership and edge connectivity must be identical
    pd.testing.assert_frame_equal(
        res_one.exact_duplicate_groups, res_two.exact_duplicate_groups
    )
    pd.testing.assert_frame_equal(
        res_one.proper_near_duplicate_groups, res_two.proper_near_duplicate_groups
    )
    pd.testing.assert_frame_equal(res_one.near_edges, res_two.near_edges)


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(df=numeric_dataframes())
def test_row_permutation_invariance(df: pd.DataFrame) -> None:
    assume(len(df) >= 3)
    row_ids = pd.Series([f"r{i:02d}" for i in range(len(df))])
    y = pd.Series(["A"] * len(df))

    perm = np.random.RandomState(42).permutation(len(df))
    df_perm = df.iloc[perm].reset_index(drop=True)
    row_ids_perm = row_ids.iloc[perm].reset_index(drop=True)
    y_perm = y.iloc[perm].reset_index(drop=True)

    cfg = DuplicateGroupConfig(
        enabled=True, near_duplicate_threshold=0.8, maximum_pairwise_rows=100
    )

    res_orig = analyze_duplicate_groups(df, y, row_ids, cfg)
    res_perm = analyze_duplicate_groups(df_perm, y_perm, row_ids_perm, cfg)

    # Check that sets of member row_ids in exact duplicate groups match
    def get_group_member_sets(
        groups_df: pd.DataFrame, col_id: str
    ) -> set[frozenset[str]]:
        if groups_df.empty:
            return set()
        return {
            frozenset(sub["row_id"].tolist()) for _, sub in groups_df.groupby(col_id)
        }

    orig_exact_sets = get_group_member_sets(
        res_orig.exact_duplicate_groups, "duplicate_group_id"
    )
    perm_exact_sets = get_group_member_sets(
        res_perm.exact_duplicate_groups, "duplicate_group_id"
    )
    assert orig_exact_sets == perm_exact_sets

    orig_near_sets = get_group_member_sets(
        res_orig.proper_near_duplicate_groups, "near_duplicate_group_id"
    )
    perm_near_sets = get_group_member_sets(
        res_perm.proper_near_duplicate_groups, "near_duplicate_group_id"
    )
    assert orig_near_sets == perm_near_sets


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    val=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e4, max_value=1e4),
    s=st.text(alphabet=st.characters(blacklist_categories=("Cs", "Cc")), max_size=15),
)
def test_canonicalization_properties(val: float, s: str) -> None:
    # Signed zero equivalence
    df_zero = pd.DataFrame({"x": [0.0, -0.0]})
    canon_zero = canonicalize_predictors(df_zero)
    assert canon_zero.iloc[0, 0] == canon_zero.iloc[1, 0]

    # String whitespace normalization
    df_str = pd.DataFrame({"s": [s, f"  {s}  "]})
    canon_str = canonicalize_predictors(df_str)
    assert canon_str.iloc[0, 0] == canon_str.iloc[1, 0]

    # String case swap preservation
    if s.swapcase() != s:
        df_case = pd.DataFrame({"s": [s, s.swapcase()]})
        canon_case = canonicalize_predictors(df_case)
        assert canon_case.iloc[0, 0] != canon_case.iloc[1, 0]
