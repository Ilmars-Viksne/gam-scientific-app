from pathlib import Path

import pytest
import yaml

from gam_app.cli import build_parser
from gam_app.config import load_config
from gam_app.config_migration import assess_config_schema, migrate_config_payload


@pytest.fixture
def schema_1_0_payload() -> dict:
    return {
        "experiment": {"name": "legacy_exp", "primary_metric": "log_loss"},
        "data": {
            "path": "../data/sample.csv",
            "target": "target",
            "row_id": "id",
        },
        "features": {
            "id": {"role": "exclude"},
            "x1": {"role": "smooth"},
        },
        "models": [{"id": "gam_main", "interactions": "none"}],
        "validation": {
            "outer_splits": 5,
            "outer_repeats": 3,
            "inner_splits": 5,
        },
    }


def test_assess_config_schema_legacy(schema_1_0_payload: dict) -> None:
    assessment = assess_config_schema(schema_1_0_payload)
    assert assessment.detected_version == "1.0"
    assert assessment.supported is True
    assert assessment.migration_recommended is True
    assert len(assessment.guidance) > 0


def test_assess_config_schema_current() -> None:
    payload = {"schema_version": "1.1"}
    assessment = assess_config_schema(payload)
    assert assessment.detected_version == "1.1"
    assert assessment.supported is True
    assert assessment.migration_recommended is False


def test_migrate_config_payload_adds_defaults(schema_1_0_payload: dict) -> None:
    result = migrate_config_payload(schema_1_0_payload)
    assert result.source_version == "1.0"
    assert result.target_version == "1.1"
    assert result.payload["schema_version"] == "1.1"
    assert result.payload["validation"]["strategy"] == "stratified"
    assert result.payload["validation"]["gap"] == 0
    assert result.payload["validation"]["test_size"] is None
    assert result.payload["validation"]["duplicate_group_policy"] == "report"


def test_migrate_config_payload_rebases_relative_path(
    tmp_path: Path, schema_1_0_payload: dict
) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    result = migrate_config_payload(
        schema_1_0_payload,
        input_directory=in_dir,
        output_directory=out_dir,
    )
    # in_dir / "../data/sample.csv" -> tmp_path / "data/sample.csv"
    # relative to out_dir -> "../data/sample.csv"
    assert result.payload["data"]["path"] == "../data/sample.csv"


def test_migrate_config_cli_command(tmp_path: Path, schema_1_0_payload: dict) -> None:
    data_file = tmp_path / "sample.csv"
    data_file.write_text("id,target,x1\n1,0,10\n2,1,20\n3,0,30\n4,1,40\n5,0,50\n")

    in_cfg = tmp_path / "legacy.yaml"
    out_cfg = tmp_path / "current.yaml"

    payload = dict(schema_1_0_payload)
    payload["data"]["path"] = str(data_file)
    in_cfg.write_text(yaml.dump(payload))

    parser = build_parser()
    args = parser.parse_args(
        [
            "migrate-config",
            "--input",
            str(in_cfg),
            "--output",
            str(out_cfg),
        ]
    )
    args.func(args)

    assert out_cfg.exists()
    config = load_config(out_cfg)
    assert config.schema_version == "1.1"
    assert config.validation.strategy == "stratified"


def test_migrate_config_cli_prevents_overwrite_without_flag(
    tmp_path: Path, schema_1_0_payload: dict
) -> None:
    in_cfg = tmp_path / "cfg.yaml"
    out_cfg = tmp_path / "cfg.yaml"
    in_cfg.write_text(yaml.dump(schema_1_0_payload))

    parser = build_parser()
    args = parser.parse_args(
        [
            "migrate-config",
            "--input",
            str(in_cfg),
            "--output",
            str(out_cfg),
        ]
    )
    with pytest.raises(
        FileExistsError, match="Input and output paths refer to the same file"
    ):
        args.func(args)
