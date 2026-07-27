from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .exceptions import DataValidationError
from .io_utils import sha256_file


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise DataValidationError(f"Unsupported data format: {suffix}")


def infer_role(series: pd.Series) -> tuple[str, str]:
    unique = series.nunique(dropna=True)
    ratio = unique / max(len(series), 1)
    if unique <= 1:
        return "exclude", "constant or empty column"
    if pd.api.types.is_bool_dtype(series):
        return "categorical", "boolean data"
    if not pd.api.types.is_numeric_dtype(series):
        if ratio > 0.98:
            return "exclude", "possible text identifier because almost every value is unique"
        return "categorical", "non-numeric data"
    if unique <= 10:
        return "categorical", f"low-cardinality numeric data ({unique} levels)"
    if pd.api.types.is_integer_dtype(series) and ratio > 0.98:
        return "exclude", "possible integer identifier because almost every value is unique"
    return "smooth", f"numeric data with {unique} unique values"


def profile_data(path: Path, target: str) -> dict[str, Any]:
    frame = load_table(path)
    if target not in frame.columns:
        raise DataValidationError(f"Target column {target!r} is absent.")
    columns: dict[str, Any] = {}
    for name in frame.columns:
        series = frame[name]
        role, reason = ("target", "selected target") if name == target else infer_role(series)
        info: dict[str, Any] = {
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
            "recommended_role": role,
            "reason": reason,
        }
        if pd.api.types.is_numeric_dtype(series) and series.notna().any():
            clean = pd.to_numeric(series, errors="coerce").dropna()
            info["minimum"] = float(clean.min())
            info["median"] = float(clean.median())
            info["maximum"] = float(clean.max())
        else:
            info["top_values"] = {
                str(key): int(value) for key, value in series.value_counts(dropna=False).head(10).items()
            }
        columns[name] = info
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "target": target,
        "target_counts": {
            str(key): int(value) for key, value in frame[target].value_counts(dropna=False).items()
        },
        "duplicate_rows": int(frame.duplicated().sum()),
        "column_profiles": columns,
    }


def validate_training_data(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    frame = load_table(config.data_path)
    required = [config.target, *config.features]
    missing_columns = sorted(set(required) - set(frame.columns))
    if missing_columns:
        raise DataValidationError(f"Missing configured columns: {missing_columns}")
    if frame.columns.duplicated().any():
        raise DataValidationError("Duplicate column names are not supported.")
    target = frame[config.target]
    if target.isna().any():
        raise DataValidationError("Missing target values are not supported.")
    if target.nunique() < 2:
        raise DataValidationError("Classification requires at least two target classes.")
    active = [name for name, spec in config.features.items() if spec.role != "exclude"]
    X = frame.loc[:, active].copy()
    for name in active:
        spec = config.features[name]
        if spec.missing == "error" and X[name].isna().any():
            raise DataValidationError(f"Feature {name!r} contains missing values.")
        if spec.role in {"smooth", "linear"}:
            X[name] = pd.to_numeric(X[name], errors="coerce")
            if X[name].isna().any() and spec.missing == "error":
                raise DataValidationError(f"Feature {name!r} is not fully numeric.")
        elif spec.role == "categorical":
            X[name] = X[name].astype("string")
    if config.row_id:
        row_ids = frame[config.row_id].astype(str)
        if row_ids.duplicated().any():
            raise DataValidationError("Configured row_id must be unique.")
    else:
        row_ids = pd.Series(np.arange(1, len(frame) + 1).astype(str), name="row_id")
    return X, target.astype(str), row_ids


def save_profile(profile: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    rows = []
    for name, info in profile["column_profiles"].items():
        rows.append({"column": name, **{k: v for k, v in info.items() if not isinstance(v, dict)}})
    pd.DataFrame(rows).to_csv(directory / "columns.csv", index=False)
