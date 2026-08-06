from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "X_Minimum",
    "X_Maximum",
    "Y_Minimum",
    "Y_Maximum",
    "Pixels_Areas",
    "X_Perimeter",
    "Y_Perimeter",
    "Sum_of_Luminosity",
    "Minimum_of_Luminosity",
    "Maximum_of_Luminosity",
    "Length_of_Conveyer",
    "TypeOfSteel_A300",
    "TypeOfSteel_A400",
    "Steel_Plate_Thickness",
    "Edges_Index",
    "Empty_Index",
    "Square_Index",
    "Outside_X_Index",
    "Edges_X_Index",
    "Edges_Y_Index",
    "Outside_Global_Index",
    "LogOfAreas",
    "Log_X_Index",
    "Log_Y_Index",
    "Orientation_Index",
    "Luminosity_Index",
    "SigmoidOfAreas",
]

TARGET_COLUMNS = [
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
]

ALL_COLUMNS = FEATURE_COLUMNS + TARGET_COLUMNS


def load_raw_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=ALL_COLUMNS,
    )

    if frame.shape[1] != len(ALL_COLUMNS):
        raise ValueError(
            "Unexpected raw column count: "
            f"expected {len(ALL_COLUMNS)}, "
            f"received {frame.shape[1]}."
        )

    return frame


def validate_targets(frame: pd.DataFrame) -> None:
    target_matrix = frame[TARGET_COLUMNS]

    invalid_values = ~target_matrix.isin([0, 1])

    if invalid_values.any().any():
        raise ValueError(
            "Target indicator columns contain values other than zero and one."
        )

    target_totals = target_matrix.sum(axis=1)

    invalid_rows = frame.index[target_totals != 1]

    if len(invalid_rows) > 0:
        preview = invalid_rows[:10].tolist()

        raise ValueError(
            "Every row must have exactly one active fault label. "
            f"Invalid row indices include: {preview}."
        )


def prepare_data(frame: pd.DataFrame) -> pd.DataFrame:
    validate_targets(frame)

    target_matrix = frame[TARGET_COLUMNS]

    prepared = frame[FEATURE_COLUMNS].copy()

    prepared["Y"] = target_matrix.idxmax(axis=1).astype("string")

    return prepared


def validate_prepared_data(
    frame: pd.DataFrame,
) -> None:
    if frame.empty:
        raise ValueError("The prepared dataset is empty.")

    if frame.columns.has_duplicates:
        raise ValueError("The prepared dataset has duplicate column names.")

    if frame.isna().any().any():
        missing = frame.isna().sum()
        missing = missing[missing > 0].to_dict()

        raise ValueError(f"Prepared data contains missing values: {missing}.")

    expected_classes = set(TARGET_COLUMNS)
    observed_classes = set(frame["Y"].astype(str).unique())

    if observed_classes != expected_classes:
        raise ValueError(
            "Prepared target classes do not match the expected "
            "fault types. "
            f"Expected {sorted(expected_classes)}, "
            f"received {sorted(observed_classes)}."
        )

    numeric_columns = frame.columns.drop("Y")

    numeric_values = frame[numeric_columns].to_numpy(dtype=np.float64)

    if not np.isfinite(numeric_values).all():
        raise ValueError("Prepared predictors contain nonfinite values.")


def main() -> None:
    project_directory = Path(__file__).resolve().parents[1]

    source_path = (
        project_directory / "data" / "raw" / "steel_plates_faults" / "Faults.NNA"
    )

    output_path = project_directory / "data" / "steel_plates_faults.csv"

    frame = load_raw_data(source_path)
    prepared = prepare_data(frame)
    validate_prepared_data(prepared)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared.to_csv(
        output_path,
        index=False,
    )

    print(f"Source rows: {len(frame)}")
    print(f"Prepared rows: {len(prepared)}")
    print(f"Predictors: {len(FEATURE_COLUMNS)}")
    print()
    print("Target counts:")
    print(prepared["Y"].value_counts().sort_index().to_string())
    print()
    print(f"Written to: {output_path}")


if __name__ == "__main__":
    main()
