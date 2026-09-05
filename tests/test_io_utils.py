from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from gam_app.io_utils import write_csv_atomic


def test_write_csv_atomic_creates_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "output.csv"
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    write_csv_atomic(df, csv_path, index=False)

    assert csv_path.exists()
    loaded = pd.read_csv(csv_path)
    assert loaded.to_dict(orient="records") == [
        {"a": 1, "b": "x"},
        {"a": 2, "b": "y"},
    ]


def test_write_csv_atomic_replaces_existing(tmp_path: Path) -> None:
    csv_path = tmp_path / "output.csv"
    csv_path.write_text("old,data\n1,2\n", encoding="utf-8")

    new_df = pd.DataFrame({"c": [3], "d": ["new"]})
    write_csv_atomic(new_df, csv_path, index=False)

    loaded = pd.read_csv(csv_path)
    assert loaded.to_dict(orient="records") == [{"c": 3, "d": "new"}]


def test_write_csv_atomic_cleans_up_and_preserves_on_failure(tmp_path: Path) -> None:
    csv_path = tmp_path / "output.csv"
    csv_path.write_text("original,data\n1,2\n", encoding="utf-8")

    df = pd.DataFrame({"a": [1]})

    with patch.object(
        pd.DataFrame, "to_csv", side_effect=RuntimeError("disk write failed")
    ):
        with pytest.raises(RuntimeError, match="disk write failed"):
            write_csv_atomic(df, csv_path)

    # Destination file should retain original content
    assert csv_path.read_text(encoding="utf-8") == "original,data\n1,2\n"

    # No leftover .tmp files
    tmp_files = list(tmp_path.glob(".*.tmp"))
    assert tmp_files == []


def test_write_csv_atomic_supports_unicode_and_spaces(tmp_path: Path) -> None:
    dir_with_spaces = tmp_path / "folder with spaces"
    csv_path = dir_with_spaces / "unicode data.csv"

    df = pd.DataFrame({"greeting": ["Hello 🌍", "Bonjour 🥖"]})
    write_csv_atomic(df, csv_path, index=False, encoding="utf-8")

    assert csv_path.exists()
    loaded = pd.read_csv(csv_path, encoding="utf-8")
    assert loaded["greeting"].tolist() == ["Hello 🌍", "Bonjour 🥖"]
