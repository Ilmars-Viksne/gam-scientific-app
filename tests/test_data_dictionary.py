from __future__ import annotations

import yaml

from gam_app.config import load_config


def test_data_dictionary_metadata_loaded_and_dumped(tmp_path) -> None:
    data_path = tmp_path / "data.csv"
    data_path.write_text("x,target\n1,A\n2,B\n", encoding="utf-8")

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
