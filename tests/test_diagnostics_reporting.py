from pathlib import Path

import pandas as pd
import pytest

from gam_app.config import (
    ExecutionConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    SearchConfig,
    ValidationConfig,
)
from gam_app.reporting import (
    _build_predictor_diagnostics_view,
    _build_validation_design_view,
    _prepare_high_pair_preview,
    _render_duplicate_conflict_warning,
    _render_high_correlation_preview,
    _render_validation_design,
)


@pytest.fixture
def base_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="Test Experiment",
        data_path=tmp_path / "data.csv",
        target="target",
        row_id="id",
        features={"x1": FeatureConfig("smooth")},
        models=(ModelConfig("gam_main"),),
        validation=ValidationConfig(
            strategy="stratified",
            outer_splits=5,
            outer_repeats=3,
            inner_splits=5,
            random_state=42,
            gap=0,
            test_size=None,
            duplicate_group_policy="report",
        ),
        search=SearchConfig(),
        execution=ExecutionConfig(),
    )


# --- GAM-109 Tests: Validation design ---


def test_validation_design_reports_stratified_configuration(
    base_config: ExperimentConfig,
) -> None:
    view = _build_validation_design_view(config=base_config, diagnostics_manifest=None)
    html = _render_validation_design(view)

    assert "<h2>Validation design</h2>" in html
    assert "stratified cross-validation" in html
    assert "<dt>Strategy</dt>" in html
    assert "<dd>stratified</dd>" in html
    assert "<dt>Configured group column</dt>" in html
    assert "<dd>Not configured</dd>" in html
    assert "<dt>Configured time column</dt>" in html
    assert "<dd>Not configured</dd>" in html
    assert "<dt>Temporal gap</dt>" in html
    assert "<dd>Not applicable</dd>" in html
    assert "<dt>Temporal test size</dt>" in html
    assert "<dd>Not applicable</dd>" in html


def test_validation_design_reports_group_column(
    base_config: ExperimentConfig,
) -> None:
    manifest = {
        "validation": {
            "strategy": "stratified_group",
            "configured_group_column": "hospital_id",
            "duplicate_group_policy": "group",
            "duplicate_grouping_applied": True,
            "outer_splits": 5,
            "outer_repeats": 1,
            "inner_splits": 3,
            "random_state": 123,
            "gap": 0,
            "test_size": None,
        }
    }
    view = _build_validation_design_view(
        config=base_config, diagnostics_manifest=manifest
    )
    html = _render_validation_design(view)

    assert "<dd>stratified_group</dd>" in html
    assert "<dd>hospital_id</dd>" in html
    assert "Effective groups include duplicate-derived constraints" in html
    assert "<dd>Yes</dd>" in html


def test_validation_design_reports_duplicate_only_grouped_strategy(
    base_config: ExperimentConfig,
) -> None:
    manifest = {
        "validation": {
            "strategy": "stratified_group",
            "configured_group_column": None,
            "duplicate_group_policy": "group",
            "duplicate_grouping_applied": True,
            "outer_splits": 5,
            "outer_repeats": 1,
            "inner_splits": 3,
            "random_state": 42,
            "gap": 0,
            "test_size": None,
        }
    }
    view = _build_validation_design_view(
        config=base_config, diagnostics_manifest=manifest
    )
    html = _render_validation_design(view)

    assert "<dd>stratified_group</dd>" in html
    assert "<dd>Not configured</dd>" in html
    assert "<dd>Yes</dd>" in html


def test_validation_design_reports_time_configuration(
    base_config: ExperimentConfig,
) -> None:
    manifest = {
        "validation": {
            "strategy": "time",
            "configured_time_column": "timestamp",
            "gap": 10,
            "test_size": None,
            "outer_splits": 3,
            "outer_repeats": 1,
            "inner_splits": 3,
            "random_state": 42,
        }
    }
    view = _build_validation_design_view(
        config=base_config, diagnostics_manifest=manifest
    )
    html = _render_validation_design(view)

    assert "forward temporal splits" in html
    assert "<dd>timestamp</dd>" in html
    assert "<dt>Temporal gap</dt>" in html
    assert "<dd>10</dd>" in html
    assert "<dt>Temporal test size</dt>" in html
    assert "<dd>Not configured</dd>" in html


def test_validation_design_distinguishes_missing_integrity_status(
    base_config: ExperimentConfig,
) -> None:
    view = _build_validation_design_view(config=base_config, diagnostics_manifest=None)
    html = _render_validation_design(view)

    assert "Not available" in html
    assert "PASS" not in html
    assert "FAIL" not in html


def test_validation_design_reports_pass_and_failed_integrity(
    base_config: ExperimentConfig,
) -> None:
    pass_manifest = {
        "split_integrity": {
            "passed": True,
            "result_count": 38,
            "failed_result_count": 0,
            "artifact": "split_integrity.csv",
        }
    }
    view_pass = _build_validation_design_view(
        config=base_config, diagnostics_manifest=pass_manifest
    )
    html_pass = _render_validation_design(view_pass)

    assert "PASS" in html_pass
    assert "Result rows: 38, Failed: 0" in html_pass
    assert "Open integrity checks" in html_pass
    assert "alert-danger" not in html_pass

    fail_manifest = {
        "split_integrity": {
            "passed": False,
            "result_count": 38,
            "failed_result_count": 2,
            "artifact": "split_integrity.csv",
        }
    }
    view_fail = _build_validation_design_view(
        config=base_config, diagnostics_manifest=fail_manifest
    )
    html_fail = _render_validation_design(view_fail)

    assert "FAIL" in html_fail
    assert "Result rows: 38, Failed: 2" in html_fail
    assert "Split integrity failed." in html_fail


def test_validation_design_escapes_html(
    base_config: ExperimentConfig,
) -> None:
    manifest = {
        "validation": {
            "strategy": "<script>alert('xss')</script>",
            "configured_group_column": "<group_col>",
            "configured_time_column": "time & space",
            "duplicate_group_policy": "group",
            "duplicate_grouping_applied": False,
            "outer_splits": 5,
            "outer_repeats": 1,
            "inner_splits": 3,
            "random_state": 42,
            "gap": 0,
            "test_size": None,
        }
    }
    view = _build_validation_design_view(
        config=base_config, diagnostics_manifest=manifest
    )
    html = _render_validation_design(view)

    assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in html
    assert "&lt;group_col&gt;" in html
    assert "time &amp; space" in html


# --- GAM-110 Tests: Predictor-diagnostics summary ---


def test_predictor_summary_renders_all_required_counts() -> None:
    manifest = {
        "status": "completed",
        "duplicate_group_policy": {
            "configured": "report",
            "enforcement_applied": False,
        },
        "analyses": {
            "correlation": {
                "status": "completed",
                "results": {"review_pair_count": 5, "warning_pair_count": 2},
            },
            "data_dictionary": {
                "status": "completed",
                "results": {"derived_declared_count": 3},
            },
            "derived_relations": {
                "status": "completed",
                "results": {"suspected_relation_count": 1},
            },
            "duplicate_groups": {
                "status": "completed",
                "results": {
                    "exact_group_count": 4,
                    "proper_near_group_count": 2,
                    "conflicting_target_group_count": 1,
                },
            },
        },
    }
    view = _build_predictor_diagnostics_view(manifest)
    assert view.review_pair_count == 5
    assert view.warning_pair_count == 2
    assert view.declared_derived_feature_count == 3
    assert view.suspected_derived_relation_count == 1
    assert view.exact_duplicate_group_count == 4
    assert view.proper_near_duplicate_group_count == 2
    assert view.conflicting_duplicate_target_group_count == 1

    # Check sum
    assert view.review_pair_count + view.warning_pair_count == 7


def test_disabled_and_failed_analysis_are_not_rendered_as_zero_findings() -> None:
    manifest = {
        "status": "completed",
        "duplicate_group_policy": {
            "configured": "report",
            "enforcement_applied": False,
        },
        "analyses": {
            "correlation": {
                "status": "disabled",
                "results": {"review_pair_count": 0, "warning_pair_count": 0},
            },
            "data_dictionary": {
                "status": "failed",
                "results": {},
            },
            "derived_relations": {
                "status": "not_applicable",
                "results": {},
            },
            "duplicate_groups": {
                "status": "completed",
                "results": {
                    "exact_group_count": 0,
                    "proper_near_group_count": 0,
                    "conflicting_target_group_count": 0,
                },
            },
        },
    }
    view = _build_predictor_diagnostics_view(manifest)
    from gam_app.reporting import _render_predictor_summary

    html = _render_predictor_summary(view)

    assert "Not evaluated" in html  # correlation disabled
    assert "Unavailable" in html  # data_dictionary failed
    assert "Not applicable" in html  # derived_relations not_applicable
    assert ">0<" in html  # duplicate_groups completed with 0 count


def test_predictor_summary_artifact_links(tmp_path: Path) -> None:
    diag_dir = tmp_path / "diagnostics"
    diag_dir.mkdir()
    (diag_dir / "correlation_pearson.csv").write_text("a,b\n1,1")
    (diag_dir / "numeric_predictor_dictionary.csv").write_text("col\nx")

    manifest = {
        "schema_name": "gam_diagnostics_manifest",
        "schema_version": "1.0",
        "artifacts": [
            {
                "id": "pearson_matrix",
                "path": "correlation_pearson.csv",
                "status": "written",
            },
            {
                "id": "predictor_dictionary",
                "path": "numeric_predictor_dictionary.csv",
                "status": "written",
            },
            {
                "id": "spearman_matrix",
                "path": "correlation_spearman.csv",
                "status": "not_written",
            },
        ],
    }

    from gam_app.reporting import _render_diagnostic_artifact_links

    html = _render_diagnostic_artifact_links(manifest, diag_dir)

    assert "Pearson correlation matrix" in html
    assert "Predictor dictionary" in html
    assert "Numeric predictor dictionary" not in html
    assert "Spearman correlation matrix" not in html


# --- GAM-111 Tests: High-correlation preview ---


def test_high_correlation_preview_is_limited_to_ten_rows() -> None:
    rows = []
    for i in range(25):
        rows.append(
            {
                "rank": i + 1,
                "left": f"var_{i}",
                "right": f"var_{i + 1}",
                "pearson": 0.8,
                "spearman": 0.8,
                "maximum_absolute_correlation": 0.8,
                "dominant_method": "pearson",
                "complete_pair_count": 100,
                "complete_pair_fraction": 1.0,
                "severity": "review",
                "declared_derivation_relation": "no",
                "recommended_action": "Do something wide that shouldn't appear",
            }
        )
    df = pd.DataFrame(rows)

    preview_df = _prepare_high_pair_preview(df)
    assert len(preview_df) == 10
    assert "Recommended action" not in preview_df.columns

    html = _render_high_correlation_preview(frame=df, analysis_status="completed")
    assert "Showing 10 of 25 high-correlation pairs." in html
    assert "high_correlation_pairs.csv" in html


def test_high_correlation_preview_preserves_priority_order() -> None:
    rows = [
        {
            "rank": 2,
            "left": "b",
            "right": "c",
            "maximum_absolute_correlation": 0.85,
            "severity": "review",
        },
        {
            "rank": 1,
            "left": "a",
            "right": "b",
            "maximum_absolute_correlation": 0.95,
            "severity": "warning",
        },
    ]
    df = pd.DataFrame(rows)
    preview_df = _prepare_high_pair_preview(df)

    assert preview_df.iloc[0]["Predictor 1"] == "a"
    assert preview_df.iloc[1]["Predictor 1"] == "b"


def test_high_correlation_preview_formats_coverage_and_floats() -> None:
    df = pd.DataFrame(
        [
            {
                "left": "x",
                "right": "y",
                "pearson": 0.123456,
                "spearman": 0.654321,
                "maximum_absolute_correlation": 0.654321,
                "complete_pair_fraction": 0.9542,
                "severity": "review",
            }
        ]
    )
    preview_df = _prepare_high_pair_preview(df)

    assert preview_df.iloc[0]["Pearson"] == "0.1235"
    assert preview_df.iloc[0]["Spearman"] == "0.6543"
    assert preview_df.iloc[0]["Coverage"] == "95.4%"


def test_high_correlation_preview_handles_fewer_than_ten_rows() -> None:
    df = pd.DataFrame(
        [
            {
                "left": "x",
                "right": "y",
                "pearson": 0.8,
                "spearman": 0.8,
                "maximum_absolute_correlation": 0.8,
                "severity": "review",
            }
        ]
    )
    html = _render_high_correlation_preview(frame=df, analysis_status="completed")
    assert "Showing all 1 high-correlation pair." in html


# --- GAM-112 Tests: Duplicate conflict warning ---


def test_conflicting_target_warning_is_prominent() -> None:
    html_single = _render_duplicate_conflict_warning(
        conflict_count=1, artifact_available=True, analysis_status="completed"
    )
    assert "Conflicting duplicate targets detected." in html_single
    assert "1 group of predictor-identical records" in html_single
    assert "cannot distinguish records" in html_single
    assert "conflicting_duplicate_targets.csv" in html_single

    html_plural = _render_duplicate_conflict_warning(
        conflict_count=3, artifact_available=True, analysis_status="completed"
    )
    assert "3 groups of predictor-identical records" in html_plural


def test_no_conflict_warning_when_count_is_zero() -> None:
    html = _render_duplicate_conflict_warning(
        conflict_count=0, artifact_available=True, analysis_status="completed"
    )
    assert html == ""


def test_conflict_warning_when_disabled_or_unavailable() -> None:
    html = _render_duplicate_conflict_warning(
        conflict_count=None, artifact_available=False, analysis_status="disabled"
    )
    assert "Duplicate-target conflict status was not evaluated." in html
