from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gam_app.cli import (
    build_parser,
    command_grouped_contributions,
)


def _write_component_contributions(
    path: Path,
) -> pd.DataFrame:
    """Write a valid component-level contribution fixture."""

    rows = [
        # Scenario 1, class A:
        # 0.2 + 0.3 - 0.1 + 0.4 = 0.8
        {
            "scenario_id": "scenario_1",
            "observation_index": 0,
            "class": "A",
            "predicted_class": "A",
            "class_probability": 0.75,
            "class_score": 0.8,
            "component": "intercept",
            "component_type": "intercept",
            "component_group": "intercept",
            "transformed_value": 1.0,
            "coefficient": 0.2,
            "contribution": 0.2,
            "absolute_contribution": 0.2,
        },
        {
            "scenario_id": "scenario_1",
            "observation_index": 0,
            "class": "A",
            "predicted_class": "A",
            "class_probability": 0.75,
            "class_score": 0.8,
            "component": "main_spline__x1__basis_0",
            "component_type": "smooth",
            "component_group": "x1",
            "transformed_value": 0.6,
            "coefficient": 0.5,
            "contribution": 0.3,
            "absolute_contribution": 0.3,
        },
        {
            "scenario_id": "scenario_1",
            "observation_index": 0,
            "class": "A",
            "predicted_class": "A",
            "class_probability": 0.75,
            "class_score": 0.8,
            "component": "main_spline__x1__basis_1",
            "component_type": "smooth",
            "component_group": "x1",
            "transformed_value": 0.5,
            "coefficient": -0.2,
            "contribution": -0.1,
            "absolute_contribution": 0.1,
        },
        {
            "scenario_id": "scenario_1",
            "observation_index": 0,
            "class": "A",
            "predicted_class": "A",
            "class_probability": 0.75,
            "class_score": 0.8,
            "component": "main_linear__x2",
            "component_type": "linear",
            "component_group": "x2",
            "transformed_value": 0.8,
            "coefficient": 0.5,
            "contribution": 0.4,
            "absolute_contribution": 0.4,
        },
        # Scenario 1, class B:
        # -0.2 + 0.1 + 0.2 - 0.4 = -0.3
        {
            "scenario_id": "scenario_1",
            "observation_index": 0,
            "class": "B",
            "predicted_class": "A",
            "class_probability": 0.25,
            "class_score": -0.3,
            "component": "intercept",
            "component_type": "intercept",
            "component_group": "intercept",
            "transformed_value": 1.0,
            "coefficient": -0.2,
            "contribution": -0.2,
            "absolute_contribution": 0.2,
        },
        {
            "scenario_id": "scenario_1",
            "observation_index": 0,
            "class": "B",
            "predicted_class": "A",
            "class_probability": 0.25,
            "class_score": -0.3,
            "component": "main_spline__x1__basis_0",
            "component_type": "smooth",
            "component_group": "x1",
            "transformed_value": 0.5,
            "coefficient": 0.2,
            "contribution": 0.1,
            "absolute_contribution": 0.1,
        },
        {
            "scenario_id": "scenario_1",
            "observation_index": 0,
            "class": "B",
            "predicted_class": "A",
            "class_probability": 0.25,
            "class_score": -0.3,
            "component": "main_spline__x1__basis_1",
            "component_type": "smooth",
            "component_group": "x1",
            "transformed_value": 0.5,
            "coefficient": 0.4,
            "contribution": 0.2,
            "absolute_contribution": 0.2,
        },
        {
            "scenario_id": "scenario_1",
            "observation_index": 0,
            "class": "B",
            "predicted_class": "A",
            "class_probability": 0.25,
            "class_score": -0.3,
            "component": "main_linear__x2",
            "component_type": "linear",
            "component_group": "x2",
            "transformed_value": 0.8,
            "coefficient": -0.5,
            "contribution": -0.4,
            "absolute_contribution": 0.4,
        },
        # Scenario 2, class A:
        # 0.2 - 0.3 + 0.1 + 0.2 = 0.2
        {
            "scenario_id": "scenario_2",
            "observation_index": 1,
            "class": "A",
            "predicted_class": "B",
            "class_probability": 0.40,
            "class_score": 0.2,
            "component": "intercept",
            "component_type": "intercept",
            "component_group": "intercept",
            "transformed_value": 1.0,
            "coefficient": 0.2,
            "contribution": 0.2,
            "absolute_contribution": 0.2,
        },
        {
            "scenario_id": "scenario_2",
            "observation_index": 1,
            "class": "A",
            "predicted_class": "B",
            "class_probability": 0.40,
            "class_score": 0.2,
            "component": "main_spline__x1__basis_0",
            "component_type": "smooth",
            "component_group": "x1",
            "transformed_value": -0.6,
            "coefficient": 0.5,
            "contribution": -0.3,
            "absolute_contribution": 0.3,
        },
        {
            "scenario_id": "scenario_2",
            "observation_index": 1,
            "class": "A",
            "predicted_class": "B",
            "class_probability": 0.40,
            "class_score": 0.2,
            "component": "main_spline__x1__basis_1",
            "component_type": "smooth",
            "component_group": "x1",
            "transformed_value": -0.5,
            "coefficient": -0.2,
            "contribution": 0.1,
            "absolute_contribution": 0.1,
        },
        {
            "scenario_id": "scenario_2",
            "observation_index": 1,
            "class": "A",
            "predicted_class": "B",
            "class_probability": 0.40,
            "class_score": 0.2,
            "component": "main_linear__x2",
            "component_type": "linear",
            "component_group": "x2",
            "transformed_value": 0.4,
            "coefficient": 0.5,
            "contribution": 0.2,
            "absolute_contribution": 0.2,
        },
        # Scenario 2, class B:
        # -0.2 + 0.4 - 0.1 + 0.3 = 0.4
        {
            "scenario_id": "scenario_2",
            "observation_index": 1,
            "class": "B",
            "predicted_class": "B",
            "class_probability": 0.60,
            "class_score": 0.4,
            "component": "intercept",
            "component_type": "intercept",
            "component_group": "intercept",
            "transformed_value": 1.0,
            "coefficient": -0.2,
            "contribution": -0.2,
            "absolute_contribution": 0.2,
        },
        {
            "scenario_id": "scenario_2",
            "observation_index": 1,
            "class": "B",
            "predicted_class": "B",
            "class_probability": 0.60,
            "class_score": 0.4,
            "component": "main_spline__x1__basis_0",
            "component_type": "smooth",
            "component_group": "x1",
            "transformed_value": 0.8,
            "coefficient": 0.5,
            "contribution": 0.4,
            "absolute_contribution": 0.4,
        },
        {
            "scenario_id": "scenario_2",
            "observation_index": 1,
            "class": "B",
            "predicted_class": "B",
            "class_probability": 0.60,
            "class_score": 0.4,
            "component": "main_spline__x1__basis_1",
            "component_type": "smooth",
            "component_group": "x1",
            "transformed_value": 0.5,
            "coefficient": -0.2,
            "contribution": -0.1,
            "absolute_contribution": 0.1,
        },
        {
            "scenario_id": "scenario_2",
            "observation_index": 1,
            "class": "B",
            "predicted_class": "B",
            "class_probability": 0.60,
            "class_score": 0.4,
            "component": "main_linear__x2",
            "component_type": "linear",
            "component_group": "x2",
            "transformed_value": 0.6,
            "coefficient": 0.5,
            "contribution": 0.3,
            "absolute_contribution": 0.3,
        },
    ]

    frame = pd.DataFrame(rows)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        float_format="%.17g",
    )

    return frame


def test_grouped_contributions_parser_registration() -> None:
    """The parser should register the command and its defaults."""

    parser = build_parser()

    args = parser.parse_args(
        [
            "grouped-contributions",
            "--input",
            "component_contributions.csv",
            "--output",
            "grouped_contributions.csv",
        ]
    )

    assert args.func is command_grouped_contributions
    assert args.input == Path("component_contributions.csv")
    assert args.output == Path("grouped_contributions.csv")
    assert args.top == 10


def test_grouped_contributions_parser_accepts_top() -> None:
    """An explicit terminal-preview limit should be accepted."""

    parser = build_parser()

    args = parser.parse_args(
        [
            "grouped-contributions",
            "--input",
            "component_contributions.csv",
            "--output",
            "grouped_contributions.csv",
            "--top",
            "25",
        ]
    )

    assert args.top == 25


@pytest.mark.parametrize(
    "missing_option",
    [
        "--input",
        "--output",
    ],
)
def test_grouped_contributions_parser_requires_paths(
    missing_option: str,
) -> None:
    """Both input and output paths should be required."""

    parser = build_parser()

    arguments = [
        "grouped-contributions",
        "--input",
        "component_contributions.csv",
        "--output",
        "grouped_contributions.csv",
    ]

    option_index = arguments.index(missing_option)

    del arguments[option_index : option_index + 2]

    with pytest.raises(SystemExit) as error:
        parser.parse_args(arguments)

    assert error.value.code == 2


def test_grouped_contributions_aggregates_basis_terms(
    tmp_path: Path,
) -> None:
    """Spline basis components should aggregate by predictor."""

    input_path = tmp_path / "component_contributions.csv"

    output_path = tmp_path / "grouped_contributions.csv"

    source = _write_component_contributions(input_path)

    command_grouped_contributions(
        Namespace(
            input=input_path,
            output=output_path,
            top=10,
        )
    )

    grouped = pd.read_csv(output_path)

    assert output_path.is_file()

    assert len(source) == 16
    assert len(grouped) == 12

    scenario_1_class_a_x1 = grouped.loc[
        (grouped["scenario_id"] == "scenario_1")
        & (grouped["class"] == "A")
        & (grouped["component_type"] == "smooth")
        & (grouped["component_group"] == "x1"),
        "contribution",
    ]

    assert len(scenario_1_class_a_x1) == 1

    assert scenario_1_class_a_x1.iloc[0] == pytest.approx(0.2)

    scenario_1_class_b_x1 = grouped.loc[
        (grouped["scenario_id"] == "scenario_1")
        & (grouped["class"] == "B")
        & (grouped["component_type"] == "smooth")
        & (grouped["component_group"] == "x1"),
        "contribution",
    ]

    assert scenario_1_class_b_x1.iloc[0] == pytest.approx(0.3)

    scenario_2_class_a_x1 = grouped.loc[
        (grouped["scenario_id"] == "scenario_2")
        & (grouped["class"] == "A")
        & (grouped["component_type"] == "smooth")
        & (grouped["component_group"] == "x1"),
        "contribution",
    ]

    assert scenario_2_class_a_x1.iloc[0] == pytest.approx(-0.2)

    scenario_2_class_b_x1 = grouped.loc[
        (grouped["scenario_id"] == "scenario_2")
        & (grouped["class"] == "B")
        & (grouped["component_type"] == "smooth")
        & (grouped["component_group"] == "x1"),
        "contribution",
    ]

    assert scenario_2_class_b_x1.iloc[0] == pytest.approx(0.3)


def test_grouped_contributions_preserves_intercepts(
    tmp_path: Path,
) -> None:
    """There should be one intercept row per scenario and class."""

    input_path = tmp_path / "component_contributions.csv"

    output_path = tmp_path / "grouped_contributions.csv"

    _write_component_contributions(input_path)

    command_grouped_contributions(
        Namespace(
            input=input_path,
            output=output_path,
            top=10,
        )
    )

    grouped = pd.read_csv(output_path)

    intercepts = grouped.loc[grouped["component_type"] == "intercept"].reset_index(
        drop=True
    )

    assert len(intercepts) == 4

    assert set(intercepts["component_group"]) == {
        "intercept",
    }

    expected = {
        (
            "scenario_1",
            "A",
        ): 0.2,
        (
            "scenario_1",
            "B",
        ): -0.2,
        (
            "scenario_2",
            "A",
        ): 0.2,
        (
            "scenario_2",
            "B",
        ): -0.2,
    }

    actual = {
        (
            str(row.scenario_id),
            str(row.class_name),
        ): float(row.contribution)
        for row in intercepts.rename(
            columns={
                "class": "class_name",
            }
        ).itertuples(index=False)
    }

    assert actual == pytest.approx(expected)


def test_grouped_contributions_reconstructs_class_scores(
    tmp_path: Path,
) -> None:
    """Grouped contributions should sum to recorded class scores."""

    input_path = tmp_path / "component_contributions.csv"

    output_path = tmp_path / "grouped_contributions.csv"

    _write_component_contributions(input_path)

    command_grouped_contributions(
        Namespace(
            input=input_path,
            output=output_path,
            top=10,
        )
    )

    grouped = pd.read_csv(output_path)

    reconstructed = (
        grouped.groupby(
            [
                "scenario_id",
                "class",
            ],
            sort=False,
        )["contribution"]
        .sum()
        .rename("reconstructed_score")
        .reset_index()
    )

    expected = (
        grouped.loc[
            :,
            [
                "scenario_id",
                "class",
                "class_score",
            ],
        ]
        .drop_duplicates(
            subset=[
                "scenario_id",
                "class",
            ]
        )
        .reset_index(drop=True)
    )

    verification = expected.merge(
        reconstructed,
        on=[
            "scenario_id",
            "class",
        ],
        validate="one_to_one",
    )

    np.testing.assert_allclose(
        verification["reconstructed_score"].to_numpy(dtype=np.float64),
        verification["class_score"].to_numpy(dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_grouped_contributions_writes_score_summary(
    tmp_path: Path,
) -> None:
    """The command should write its score-integrity summary."""

    input_path = tmp_path / "component_contributions.csv"

    output_path = tmp_path / "grouped_contributions.csv"

    _write_component_contributions(input_path)

    command_grouped_contributions(
        Namespace(
            input=input_path,
            output=output_path,
            top=10,
        )
    )

    summary_path = output_path.with_name(f"{output_path.stem}_score_summary.csv")

    assert summary_path.is_file()

    summary = pd.read_csv(summary_path)

    expected_columns = {
        "scenario_id",
        "observation_index",
        "class",
        "class_score",
        "component_score",
        "grouped_score",
        "component_score_error",
        "grouped_score_error",
        "aggregation_error",
    }

    assert expected_columns.issubset(summary.columns)

    assert len(summary) == 4

    np.testing.assert_allclose(
        summary["component_score_error"].to_numpy(dtype=np.float64),
        0.0,
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        summary["grouped_score_error"].to_numpy(dtype=np.float64),
        0.0,
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        summary["aggregation_error"].to_numpy(dtype=np.float64),
        0.0,
        rtol=0.0,
        atol=1e-12,
    )


def test_grouped_contributions_adds_csv_extension(
    tmp_path: Path,
) -> None:
    """A missing output extension should become .csv."""

    input_path = tmp_path / "component_contributions.csv"

    output_without_suffix = tmp_path / "grouped_contributions"

    expected_output = output_without_suffix.with_suffix(".csv")

    expected_summary = expected_output.with_name(
        f"{expected_output.stem}_score_summary.csv"
    )

    _write_component_contributions(input_path)

    command_grouped_contributions(
        Namespace(
            input=input_path,
            output=output_without_suffix,
            top=10,
        )
    )

    assert expected_output.is_file()
    assert expected_summary.is_file()
    assert not output_without_suffix.exists()


def test_grouped_contributions_rejects_non_csv_output(
    tmp_path: Path,
) -> None:
    """A misleading non-CSV extension should be rejected."""

    input_path = tmp_path / "component_contributions.csv"

    _write_component_contributions(input_path)

    with pytest.raises(
        ValueError,
        match=("must use the '.csv' extension"),
    ):
        command_grouped_contributions(
            Namespace(
                input=input_path,
                output=(tmp_path / "grouped_contributions.xlsx"),
                top=10,
            )
        )


def test_grouped_contributions_rejects_missing_input(
    tmp_path: Path,
) -> None:
    """A nonexistent input file should produce a clear error."""

    missing_path = tmp_path / "missing.csv"

    with pytest.raises(
        FileNotFoundError,
        match=("input file does not exist"),
    ):
        command_grouped_contributions(
            Namespace(
                input=missing_path,
                output=(tmp_path / "grouped.csv"),
                top=10,
            )
        )


def test_grouped_contributions_rejects_empty_input(
    tmp_path: Path,
) -> None:
    """A CSV containing headers but no rows should be rejected."""

    input_path = tmp_path / "empty.csv"

    output_path = tmp_path / "grouped.csv"

    pd.DataFrame(
        columns=[
            "scenario_id",
            "observation_index",
            "class",
            "predicted_class",
            "class_probability",
            "class_score",
            "component_type",
            "component_group",
            "contribution",
        ]
    ).to_csv(
        input_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="contains no rows",
    ):
        command_grouped_contributions(
            Namespace(
                input=input_path,
                output=output_path,
                top=10,
            )
        )


def test_grouped_contributions_rejects_missing_columns(
    tmp_path: Path,
) -> None:
    """All required contribution columns should be validated."""

    input_path = tmp_path / "incomplete.csv"

    output_path = tmp_path / "grouped.csv"

    pd.DataFrame(
        {
            "scenario_id": [
                "scenario_1",
            ],
            "class": [
                "A",
            ],
            "contribution": [
                0.5,
            ],
        }
    ).to_csv(
        input_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        command_grouped_contributions(
            Namespace(
                input=input_path,
                output=output_path,
                top=10,
            )
        )


def test_grouped_contributions_rejects_invalid_top(
    tmp_path: Path,
) -> None:
    """The terminal row limit should be positive."""

    input_path = tmp_path / "component_contributions.csv"

    _write_component_contributions(input_path)

    with pytest.raises(
        ValueError,
        match="--top must be at least 1",
    ):
        command_grouped_contributions(
            Namespace(
                input=input_path,
                output=(tmp_path / "grouped.csv"),
                top=0,
            )
        )


def test_grouped_contributions_rejects_nonnumeric_values(
    tmp_path: Path,
) -> None:
    """Required numeric fields should not contain text."""

    input_path = tmp_path / "invalid_numeric.csv"

    output_path = tmp_path / "grouped.csv"

    frame = _write_component_contributions(input_path)

    frame["contribution"] = frame["contribution"].astype(object)

    frame.loc[
        0,
        "contribution",
    ] = "not-a-number"

    frame.to_csv(
        input_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match=("missing or nonnumeric values"),
    ):
        command_grouped_contributions(
            Namespace(
                input=input_path,
                output=output_path,
                top=10,
            )
        )


def test_grouped_contributions_rejects_nonfinite_values(
    tmp_path: Path,
) -> None:
    """Infinite component contributions should be rejected."""

    input_path = tmp_path / "nonfinite.csv"

    output_path = tmp_path / "grouped.csv"

    frame = _write_component_contributions(input_path)

    frame.loc[
        0,
        "contribution",
    ] = np.inf

    frame.to_csv(
        input_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match=("contributions contain nonfinite values"),
    ):
        command_grouped_contributions(
            Namespace(
                input=input_path,
                output=output_path,
                top=10,
            )
        )


def test_grouped_contributions_rejects_inconsistent_scores(
    tmp_path: Path,
) -> None:
    """Repeated rows for a scenario-class must share one score."""

    input_path = tmp_path / "inconsistent.csv"

    output_path = tmp_path / "grouped.csv"

    frame = _write_component_contributions(input_path)

    frame.loc[
        1,
        "class_score",
    ] = 999.0

    frame.to_csv(
        input_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match=("inconsistent repeated values"),
    ):
        command_grouped_contributions(
            Namespace(
                input=input_path,
                output=output_path,
                top=10,
            )
        )


def test_grouped_contributions_rejects_invalid_score_sum(
    tmp_path: Path,
) -> None:
    """Input contributions must reconstruct the class score."""

    input_path = tmp_path / "invalid_score_sum.csv"

    output_path = tmp_path / "grouped.csv"

    frame = _write_component_contributions(input_path)

    frame.loc[
        0,
        "contribution",
    ] = 0.25

    frame.loc[
        0,
        "absolute_contribution",
    ] = 0.25

    frame.to_csv(
        input_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match=("do not reconstruct the recorded class scores"),
    ):
        command_grouped_contributions(
            Namespace(
                input=input_path,
                output=output_path,
                top=10,
            )
        )


def test_top_limits_preview_but_not_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The --top option should not remove rows from the CSV."""

    input_path = tmp_path / "component_contributions.csv"

    output_path = tmp_path / "grouped_contributions.csv"

    _write_component_contributions(input_path)

    command_grouped_contributions(
        Namespace(
            input=input_path,
            output=output_path,
            top=1,
        )
    )

    captured = capsys.readouterr()

    grouped = pd.read_csv(output_path)

    assert len(grouped) == 12

    assert "Largest grouped class-score contributions" in captured.out

    assert "Grouped-contribution summary" in captured.out

    assert str(output_path.resolve()) in captured.out
