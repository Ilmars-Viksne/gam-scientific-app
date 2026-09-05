from __future__ import annotations

import pandas as pd
import pytest
import yaml

from gam_app.config import ExperimentConfig, FeatureConfig, load_config
from gam_app.diagnostics import (
    StandaloneDiagnosticSettings,
    calculate_correlation_analysis,
    calculate_standalone_diagnostics,
)
from gam_app.exceptions import ConfigurationError


def test_data_dictionary_metadata_loaded_and_dumped(tmp_path) -> None:
    data_path = tmp_path / "data.csv"
    data_path.write_text("x,src_x,target\n1,1,A\n2,2,B\n", encoding="utf-8")

    cfg_yaml = {
        "schema_version": "1.1",
        "experiment": {"name": "dict_test", "primary_metric": "log_loss"},
        "data": {"path": str(data_path), "target": "target"},
        "features": {
            "x": {
                "role": "smooth",
                "derived": "declared",
                "derived_from": ["src_x"],
                "derivation": "src_x + 1",
                "description": "Offset feature",
                "unit": "meters",
            },
            "src_x": {
                "role": "smooth",
                "derived": "none",
            },
        },
        "models": [{"id": "main", "interactions": "none"}],
    }

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(cfg_yaml), encoding="utf-8")

    config = load_config(cfg_file)
    spec = config.features["x"]

    assert spec.derived == "declared"
    assert spec.derived_from == ("src_x",)
    assert spec.derivation == "src_x + 1"
    assert spec.description == "Offset feature"
    assert spec.unit == "meters"


def test_config_derived_none_maps_to_diagnostic_not_declared(tmp_path) -> None:
    df = pd.DataFrame({"x1": [1.0, 2.0], "target": ["A", "B"]})
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    config = ExperimentConfig(
        name="test_dict",
        data_path=data_path,
        target="target",
        row_id=None,
        features={
            "x1": FeatureConfig(role="smooth", derived="none"),
        },
    )

    analysis = calculate_correlation_analysis(df, config)
    dict_df = analysis.numeric_predictor_dictionary
    assert not dict_df.empty
    row = dict_df.loc[dict_df["predictor"] == "x1"].iloc[0]
    assert row["metadata_status"] == "provided"
    assert row["derived_status"] == "not_declared"


def test_standalone_dictionary_uses_not_provided_and_not_evaluated(tmp_path) -> None:
    df = pd.DataFrame({"x1": [1.0, 2.0], "target": ["A", "B"]})
    settings = StandaloneDiagnosticSettings()

    diag = calculate_standalone_diagnostics(
        frame=df,
        target="target",
        settings=settings,
    )

    dict_df = diag.numeric_predictor_dictionary
    assert not dict_df.empty
    row = dict_df.iloc[0]
    assert row["metadata_status"] == "not_provided"
    assert row["derived_status"] == "not_evaluated"


def test_invalid_derived_feature_configurations_are_rejected(tmp_path) -> None:
    # 1. Self derivation
    with pytest.raises(ConfigurationError, match="cannot be derived from itself"):
        ExperimentConfig(
            name="test",
            data_path=tmp_path / "data.csv",
            target="target",
            row_id=None,
            features={
                "x": FeatureConfig(
                    role="smooth", derived="declared", derived_from=("x",)
                ),
            },
        ).validate()

    # 2. Declared without derived_from
    with pytest.raises(ConfigurationError, match="must declare derived_from"):
        ExperimentConfig(
            name="test",
            data_path=tmp_path / "data.csv",
            target="target",
            row_id=None,
            features={
                "x": FeatureConfig(role="smooth", derived="declared", derived_from=()),
            },
        ).validate()

    # 3. Derived none with derived_from
    with pytest.raises(
        ConfigurationError, match="has derived='none' but declares derived_from"
    ):
        ExperimentConfig(
            name="test",
            data_path=tmp_path / "data.csv",
            target="target",
            row_id=None,
            features={
                "src": FeatureConfig(role="smooth"),
                "x": FeatureConfig(
                    role="smooth", derived="none", derived_from=("src",)
                ),
            },
        ).validate()
