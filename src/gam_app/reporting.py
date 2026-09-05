from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .config import ExperimentConfig
from .run_store import FileRunStore

HIGH_CORRELATION_PREVIEW_LIMIT: Final[int] = 10

HIGH_PAIR_DISPLAY_LABELS: Final[dict[str, str]] = {
    "rank": "Rank",
    "left": "Predictor 1",
    "right": "Predictor 2",
    "pearson": "Pearson",
    "spearman": "Spearman",
    "maximum_absolute_correlation": "Maximum |correlation|",
    "dominant_method": "Dominant method",
    "complete_pair_count": "Complete pairs",
    "complete_pair_fraction": "Coverage",
    "severity": "Severity",
    "declared_derivation_relation": "Declared derivation",
}

ANALYSIS_STATUS_DISPLAY: Final[dict[str, str | None]] = {
    "completed": None,
    "disabled": "Not evaluated",
    "not_applicable": "Not applicable",
    "deferred": "Deferred",
    "failed": "Unavailable",
}

DIAGNOSTIC_ARTIFACT_LINKS: Final[dict[str, tuple[str, str]]] = {
    "pearson_matrix": (
        "correlation_pearson.csv",
        "Pearson correlation matrix",
    ),
    "spearman_matrix": (
        "correlation_spearman.csv",
        "Spearman correlation matrix",
    ),
    "high_correlation_pairs": (
        "high_correlation_pairs.csv",
        "High-correlation pair report",
    ),
    "predictor_dictionary": (
        "numeric_predictor_dictionary.csv",
        "Predictor dictionary",
    ),
    "exact_duplicate_groups": (
        "exact_duplicate_groups.csv",
        "Exact duplicate groups",
    ),
    "near_duplicate_groups": (
        "near_duplicate_groups.csv",
        "Proper near-duplicate groups",
    ),
    "near_duplicate_edges": (
        "near_duplicate_edges.csv",
        "Near-duplicate edge evidence",
    ),
    "effective_validation_groups": (
        "effective_validation_groups.csv",
        "Effective validation groups",
    ),
    "conflicting_duplicate_targets": (
        "conflicting_duplicate_targets.csv",
        "Conflicting duplicate targets",
    ),
    "suspected_derived_relations": (
        "suspected_derived_relations.csv",
        "Suspected derived relations",
    ),
}

REPORT_STYLE: Final[str] = """
body {
    font-family: Arial, sans-serif;
    max-width: 1100px;
    margin: 2rem auto;
    padding: 0 1rem;
    color: #202124;
    line-height: 1.45;
}

section {
    margin: 2rem 0;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 1rem 0;
}

th,
td {
    border: 1px solid #d0d0d0;
    padding: 0.45rem;
    text-align: left;
    vertical-align: top;
}

th {
    background: #f3f5f7;
}

.summary-grid {
    display: grid;
    grid-template-columns: minmax(13rem, 1fr) 2fr;
    gap: 0.35rem 1rem;
}

.summary-grid dt {
    font-weight: 700;
}

.summary-grid dd {
    margin: 0;
}

.summary-cards {
    display: grid;
    grid-template-columns: repeat(
        auto-fit,
        minmax(10rem, 1fr)
    );
    gap: 0.75rem;
    margin: 1rem 0;
}

.summary-card {
    border: 1px solid #d0d0d0;
    border-radius: 0.3rem;
    padding: 0.75rem;
    background: #fafafa;
}

.summary-value {
    font-size: 1.5rem;
    font-weight: 700;
}

.summary-label {
    margin-top: 0.25rem;
}

.summary-warning {
    border-color: #a66a00;
    background: #fff8e6;
}

.summary-danger {
    border-color: #a4262c;
    background: #fdecec;
}

.alert {
    border: 1px solid;
    border-left-width: 0.4rem;
    border-radius: 0.25rem;
    padding: 0.85rem 1rem;
    margin: 1rem 0;
}

.alert-danger {
    border-color: #a4262c;
    background: #fdecec;
}

.alert-warning {
    border-color: #a66a00;
    background: #fff8e6;
}

.alert-neutral {
    border-color: #6b7280;
    background: #f3f4f6;
}

.status-badge {
    display: inline-block;
    border-radius: 999px;
    padding: 0.15rem 0.55rem;
    font-size: 0.9rem;
    font-weight: 700;
}

.status-pass {
    background: #e7f5e9;
    color: #126b2e;
}

.status-fail {
    background: #fdecec;
    color: #8c1d23;
}

.status-warning {
    background: #fff3cd;
    color: #6f4e00;
}

.status-neutral {
    background: #eceff1;
    color: #4b5563;
}

.table-wrapper {
    overflow-x: auto;
}

img {
    max-width: 700px;
}
"""


@dataclass(frozen=True, slots=True)
class ValidationDesignView:
    strategy: str
    configured_group_column: str | None
    configured_time_column: str | None
    duplicate_group_policy: str
    duplicate_grouping_applied: bool
    outer_splits: int
    outer_repeats: int
    inner_splits: int
    random_state: int
    gap: int
    test_size: int | None
    split_integrity_status: str
    split_integrity_result_count: int | None
    split_integrity_failed_count: int | None
    split_integrity_artifact: str | None


@dataclass(frozen=True, slots=True)
class PredictorDiagnosticsView:
    diagnostics_status: str
    correlation_status: str
    data_dictionary_status: str
    derived_relations_status: str
    duplicate_groups_status: str
    review_pair_count: int
    warning_pair_count: int
    declared_derived_feature_count: int
    suspected_derived_relation_count: int
    exact_duplicate_group_count: int
    proper_near_duplicate_group_count: int
    conflicting_duplicate_target_group_count: int
    duplicate_group_policy: str
    duplicate_grouping_applied: bool


def _load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")

    return payload


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    return pd.read_csv(path)


def _escape(value: object) -> str:
    return html.escape(str(value))


def _display_optional(
    value: object | None,
    *,
    missing: str = "Not configured",
) -> str:
    if value is None or value == "":
        return html.escape(missing)

    return html.escape(str(value))


def _status_badge(status: str) -> str:
    normalized = status.lower()

    css_class = {
        "pass": "status-pass",
        "passed": "status-pass",
        "completed": "status-pass",
        "warning": "status-warning",
        "fail": "status-fail",
        "failed": "status-fail",
        "not_applicable": "status-neutral",
        "not_available": "status-neutral",
        "unknown": "status-neutral",
    }.get(normalized, "status-neutral")

    return f'<span class="status-badge {css_class}">{html.escape(status)}</span>'


def _diagnostic_link(*, path: str, label: str) -> str:
    safe_path = html.escape(path, quote=True)
    safe_label = html.escape(label)

    return f'<a href="../diagnostics/{safe_path}">{safe_label}</a>'


def _build_validation_design_view(
    *,
    config: ExperimentConfig,
    diagnostics_manifest: dict[str, Any] | None,
) -> ValidationDesignView:
    validation_manifest = (
        diagnostics_manifest.get("validation", {}) if diagnostics_manifest else {}
    )

    integrity_manifest = (
        diagnostics_manifest.get("split_integrity", {}) if diagnostics_manifest else {}
    )

    passed = integrity_manifest.get("passed")

    if passed is True:
        integrity_status = "PASS"
    elif passed is False:
        integrity_status = "FAIL"
    else:
        integrity_status = "Not available"

    return ValidationDesignView(
        strategy=str(
            validation_manifest.get(
                "strategy",
                config.validation.strategy,
            )
        ),
        configured_group_column=validation_manifest.get(
            "configured_group_column",
            config.group_column,
        ),
        configured_time_column=validation_manifest.get(
            "configured_time_column",
            config.time_column,
        ),
        duplicate_group_policy=str(
            validation_manifest.get(
                "duplicate_group_policy",
                config.validation.duplicate_group_policy,
            )
        ),
        duplicate_grouping_applied=bool(
            validation_manifest.get(
                "duplicate_grouping_applied",
                False,
            )
        ),
        outer_splits=int(
            validation_manifest.get(
                "outer_splits",
                config.validation.outer_splits,
            )
        ),
        outer_repeats=int(
            validation_manifest.get(
                "outer_repeats",
                config.validation.outer_repeats,
            )
        ),
        inner_splits=int(
            validation_manifest.get(
                "inner_splits",
                config.validation.inner_splits,
            )
        ),
        random_state=int(
            validation_manifest.get(
                "random_state",
                config.validation.random_state,
            )
        ),
        gap=int(
            validation_manifest.get(
                "gap",
                config.validation.gap,
            )
        ),
        test_size=validation_manifest.get(
            "test_size",
            config.validation.test_size,
        ),
        split_integrity_status=integrity_status,
        split_integrity_result_count=integrity_manifest.get("result_count"),
        split_integrity_failed_count=integrity_manifest.get("failed_result_count"),
        split_integrity_artifact=integrity_manifest.get("artifact"),
    )


def _render_validation_design(
    view: ValidationDesignView,
) -> str:
    integrity = _status_badge(view.split_integrity_status)

    integrity_link = ""
    if view.split_integrity_artifact:
        integrity_link = " " + _diagnostic_link(
            path="split_integrity.csv",
            label="Open integrity checks",
        )

    strategy_desc = ""
    if view.strategy == "stratified":
        strategy_desc = (
            "<p><em>Rows were partitioned using stratified cross-validation. "
            "No scientific group or chronological ordering constraint was "
            "applied.</em></p>"
        )
    elif view.strategy == "stratified_group":
        desc = (
            "<p><em>Rows were partitioned using stratified group cross-validation. "
            "Every effective group was kept entirely within one partition."
        )
        if view.duplicate_grouping_applied:
            desc += (
                " Effective groups include duplicate-derived constraints "
                "and any configured scientific groups."
            )
        desc += "</em></p>"
        strategy_desc = desc
    elif view.strategy == "time":
        strategy_desc = (
            "<p><em>Rows were evaluated using forward temporal splits. "
            "Training observations precede test observations.</em></p>"
        )

    if view.strategy == "time":
        temporal_gap_display = _escape(view.gap)
        temporal_test_size_display = _display_optional(
            view.test_size, missing="Not configured"
        )
    else:
        temporal_gap_display = _escape("Not applicable")
        temporal_test_size_display = _escape("Not applicable")

    integrity_details = ""
    if (
        view.split_integrity_result_count is not None
        and view.split_integrity_failed_count is not None
    ):
        integrity_details = (
            f" (Result rows: {view.split_integrity_result_count}, "
            f"Failed: {view.split_integrity_failed_count})"
        )

    integrity_warning = ""
    if view.split_integrity_status == "FAIL":
        integrity_warning = (
            '<div class="alert alert-danger" role="alert">'
            "<strong>Split integrity failed.</strong> "
            "One or more validation-design checks did not pass. "
            "Review the integrity report before interpreting model results."
            "</div>"
        )

    return "".join(
        [
            '<section id="validation-design">',
            "<h2>Validation design</h2>",
            strategy_desc,
            '<dl class="summary-grid">',
            "<dt>Strategy</dt>",
            f"<dd>{_escape(view.strategy)}</dd>",
            "<dt>Configured group column</dt>",
            f"<dd>{_display_optional(view.configured_group_column)}</dd>",
            "<dt>Configured time column</dt>",
            f"<dd>{_display_optional(view.configured_time_column)}</dd>",
            "<dt>Duplicate-group policy</dt>",
            f"<dd>{_escape(view.duplicate_group_policy)}</dd>",
            "<dt>Duplicate grouping applied</dt>",
            ("<dd>Yes</dd>" if view.duplicate_grouping_applied else "<dd>No</dd>"),
            "<dt>Outer splits</dt>",
            f"<dd>{view.outer_splits}</dd>",
            "<dt>Outer repeats</dt>",
            f"<dd>{view.outer_repeats}</dd>",
            "<dt>Inner splits</dt>",
            f"<dd>{view.inner_splits}</dd>",
            "<dt>Random state</dt>",
            f"<dd>{view.random_state}</dd>",
            "<dt>Temporal gap</dt>",
            f"<dd>{temporal_gap_display}</dd>",
            "<dt>Temporal test size</dt>",
            f"<dd>{temporal_test_size_display}</dd>",
            "<dt>Split integrity</dt>",
            f"<dd>{integrity}{integrity_details}{integrity_link}</dd>",
            "</dl>",
            integrity_warning,
            "</section>",
        ]
    )


def _build_predictor_diagnostics_view(
    manifest: dict[str, Any] | None,
    diagnostics_directory: Path | None = None,
) -> PredictorDiagnosticsView:
    analyses = manifest.get("analyses", {}) if manifest else {}

    correlation = analyses.get("correlation", {})
    correlation_results = correlation.get("results", {})

    dictionary = analyses.get("data_dictionary", {})
    dictionary_results = dictionary.get("results", {})

    derivations = analyses.get("derived_relations", {})
    derivation_results = derivations.get("results", {})

    duplicates = analyses.get("duplicate_groups", {})
    duplicate_results = duplicates.get("results", {})

    policy = manifest.get("duplicate_group_policy", {}) if manifest else {}

    diag_status = str(
        manifest.get("status", "not_available") if manifest else "not_available"
    )

    corr_status = str(correlation.get("status", "not_available"))
    dict_status = str(dictionary.get("status", "not_available"))
    deriv_status = str(derivations.get("status", "not_available"))
    dup_status = str(duplicates.get("status", "not_available"))

    # Manifest counts
    review_pairs = correlation_results.get("review_pair_count")
    warning_pairs = correlation_results.get("warning_pair_count")
    declared_derived = dictionary_results.get("derived_declared_count")
    suspected_relations = derivation_results.get("suspected_relation_count")
    exact_groups = duplicate_results.get("exact_group_count")
    proper_near_groups = duplicate_results.get("proper_near_group_count")
    conflicting_groups = duplicate_results.get("conflicting_target_group_count")

    # CSV fallback if manifest missing or field absent
    if diagnostics_directory and diagnostics_directory.exists():
        if review_pairs is None or warning_pairs is None:
            high_pairs_csv = diagnostics_directory / "high_correlation_pairs.csv"
            if high_pairs_csv.exists():
                df = pd.read_csv(high_pairs_csv)
                if not df.empty and "severity" in df.columns:
                    if review_pairs is None:
                        review_pairs = int((df["severity"] == "review").sum())
                    if warning_pairs is None:
                        warning_pairs = int((df["severity"] == "warning").sum())

        if exact_groups is None:
            exact_csv = diagnostics_directory / "exact_duplicate_groups.csv"
            if exact_csv.exists():
                df = pd.read_csv(exact_csv)
                if not df.empty and "duplicate_group_id" in df.columns:
                    exact_groups = int(df["duplicate_group_id"].nunique())
                else:
                    exact_groups = 0

        if proper_near_groups is None:
            near_csv = diagnostics_directory / "near_duplicate_groups.csv"
            if near_csv.exists():
                df = pd.read_csv(near_csv)
                if not df.empty and "near_duplicate_group_id" in df.columns:
                    proper_near_groups = int(df["near_duplicate_group_id"].nunique())
                else:
                    proper_near_groups = 0

        if conflicting_groups is None:
            conf_csv = diagnostics_directory / "conflicting_duplicate_targets.csv"
            if conf_csv.exists():
                df = pd.read_csv(conf_csv)
                if not df.empty:
                    if "signature" in df.columns:
                        conflicting_groups = int(df["signature"].nunique())
                    elif "duplicate_group_id" in df.columns:
                        conflicting_groups = int(df["duplicate_group_id"].nunique())
                else:
                    conflicting_groups = 0

    return PredictorDiagnosticsView(
        diagnostics_status=diag_status,
        correlation_status=corr_status,
        data_dictionary_status=dict_status,
        derived_relations_status=deriv_status,
        duplicate_groups_status=dup_status,
        review_pair_count=int(review_pairs or 0),
        warning_pair_count=int(warning_pairs or 0),
        declared_derived_feature_count=int(declared_derived or 0),
        suspected_derived_relation_count=int(suspected_relations or 0),
        exact_duplicate_group_count=int(exact_groups or 0),
        proper_near_duplicate_group_count=int(proper_near_groups or 0),
        conflicting_duplicate_target_group_count=int(conflicting_groups or 0),
        duplicate_group_policy=str(policy.get("configured", "unknown")),
        duplicate_grouping_applied=bool(policy.get("enforcement_applied", False)),
    )


def _summary_card(
    *,
    label: str,
    value: int | str,
    css_class: str = "",
) -> str:
    return (
        f'<div class="summary-card {css_class}">'
        f'<div class="summary-value">{_escape(value)}</div>'
        f'<div class="summary-label">{_escape(label)}</div>'
        "</div>"
    )


def _card_value(status: str, count: int) -> str:
    display_override = ANALYSIS_STATUS_DISPLAY.get(status)
    if display_override is not None:
        return display_override
    if status == "completed":
        return str(count)
    return "Not available"


def _render_predictor_summary(
    view: PredictorDiagnosticsView,
) -> str:
    corr_review_val = _card_value(view.correlation_status, view.review_pair_count)
    corr_warn_val = _card_value(view.correlation_status, view.warning_pair_count)
    decl_val = _card_value(
        view.data_dictionary_status, view.declared_derived_feature_count
    )
    susp_val = _card_value(
        view.derived_relations_status, view.suspected_derived_relation_count
    )
    exact_val = _card_value(
        view.duplicate_groups_status, view.exact_duplicate_group_count
    )
    near_val = _card_value(
        view.duplicate_groups_status, view.proper_near_duplicate_group_count
    )
    conf_val = _card_value(
        view.duplicate_groups_status,
        view.conflicting_duplicate_target_group_count,
    )

    warning_class = (
        "summary-warning"
        if view.warning_pair_count > 0 and view.correlation_status == "completed"
        else ""
    )
    conflict_class = (
        "summary-danger"
        if view.conflicting_duplicate_target_group_count > 0
        and view.duplicate_groups_status == "completed"
        else ""
    )

    cards = [
        _summary_card(
            label="Review-level correlation pairs",
            value=corr_review_val,
        ),
        _summary_card(
            label="Warning-level correlation pairs",
            value=corr_warn_val,
            css_class=warning_class,
        ),
        _summary_card(
            label="Declared derived predictors",
            value=decl_val,
        ),
        _summary_card(
            label="Suspected derived relations",
            value=susp_val,
        ),
        _summary_card(
            label="Exact duplicate groups",
            value=exact_val,
        ),
        _summary_card(
            label="Proper near-duplicate groups",
            value=near_val,
        ),
        _summary_card(
            label="Conflicting-target duplicate groups",
            value=conf_val,
            css_class=conflict_class,
        ),
    ]

    return '<div class="summary-cards">' + "".join(cards) + "</div>"


def _render_duplicate_conflict_warning(
    *,
    conflict_count: int | None,
    artifact_available: bool,
    analysis_status: str = "completed",
) -> str:
    if analysis_status == "disabled" or conflict_count is None:
        return (
            '<div class="alert alert-neutral" role="alert">'
            "Duplicate-target conflict status was not evaluated."
            "</div>"
        )

    if conflict_count == 0:
        return ""

    group_word = "group" if conflict_count == 1 else "groups"

    link = ""
    if artifact_available:
        link = (
            " "
            + _diagnostic_link(
                path="conflicting_duplicate_targets.csv",
                label="Review conflicting records",
            )
            + "."
        )

    return "".join(
        [
            '<div class="alert alert-danger" role="alert">',
            "<strong>Conflicting duplicate targets detected.</strong> ",
            f"{conflict_count} {group_word} of predictor-identical records ",
            "contain different target labels. A deterministic classifier ",
            "using only the available predictors cannot distinguish records ",
            "within these groups. Review the conflicting records and their ",
            "data provenance before interpreting model performance.",
            link,
            "</div>",
        ]
    )


def _ordered_high_correlation_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    required = {
        "severity",
        "maximum_absolute_correlation",
        "left",
        "right",
    }

    if not required.issubset(frame.columns):
        return frame.copy()

    result = frame.copy()

    severity_order = {
        "warning": 0,
        "review": 1,
    }

    result["_severity_order"] = result["severity"].map(severity_order).fillna(99)

    if "complete_pair_count" not in result:
        result["complete_pair_count"] = 0

    return (
        result.sort_values(
            [
                "_severity_order",
                "maximum_absolute_correlation",
                "complete_pair_count",
                "left",
                "right",
            ],
            ascending=[
                True,
                False,
                False,
                True,
                True,
            ],
            kind="stable",
        )
        .drop(columns="_severity_order")
        .reset_index(drop=True)
    )


def _prepare_high_pair_preview(
    frame: pd.DataFrame,
    *,
    limit: int = HIGH_CORRELATION_PREVIEW_LIMIT,
) -> pd.DataFrame:
    ordered = _ordered_high_correlation_pairs(frame)

    columns = [
        column
        for column in (
            "rank",
            "left",
            "right",
            "pearson",
            "spearman",
            "maximum_absolute_correlation",
            "dominant_method",
            "complete_pair_count",
            "complete_pair_fraction",
            "severity",
            "declared_derivation_relation",
        )
        if column in ordered.columns
    ]

    preview = ordered.loc[:, columns].head(limit).copy()

    if "complete_pair_fraction" in preview:
        preview["complete_pair_fraction"] = preview["complete_pair_fraction"].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.1%}"
        )

    for column in (
        "pearson",
        "spearman",
        "maximum_absolute_correlation",
    ):
        if column in preview:
            preview[column] = preview[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )

    return preview.rename(columns=HIGH_PAIR_DISPLAY_LABELS)


def _render_high_correlation_preview(
    *,
    frame: pd.DataFrame | None,
    analysis_status: str,
) -> str:
    sections = ["<h3>High predictor correlations</h3>"]

    if analysis_status == "disabled":
        sections.append("<p>Correlation analysis was disabled for this run.</p>")
        return "".join(sections)

    if frame is None:
        sections.append("<p>The correlation-pair artifact is unavailable.</p>")
        return "".join(sections)

    if frame.empty:
        sections.append(
            "<p>No predictor pairs exceeded the configured review threshold.</p>"
        )
        return "".join(sections)

    preview = _prepare_high_pair_preview(frame)

    table_html = preview.to_html(
        index=False,
        escape=True,
        classes=["diagnostic-table", "correlation-preview"],
        border=0,
    )

    sections.append(f'<div class="table-wrapper">{table_html}</div>')

    total_pairs = len(frame)
    shown = min(total_pairs, HIGH_CORRELATION_PREVIEW_LIMIT)

    if total_pairs <= HIGH_CORRELATION_PREVIEW_LIMIT:
        pair_word = "pair" if total_pairs == 1 else "pairs"
        sections.append(
            f"<p>Showing all {total_pairs} high-correlation {pair_word}.</p>"
        )
    else:
        sections.append(
            f"<p>Showing {shown} of {total_pairs} high-correlation pairs. "
            + _diagnostic_link(
                path="high_correlation_pairs.csv",
                label="Open the complete pair report",
            )
            + ".</p>"
        )

    return "".join(sections)


def _render_diagnostics_interpretation() -> str:
    return (
        "<p><em>High correlation identifies predictor redundancy "
        "or shared structure. It is not an automatic predictor "
        "deletion rule. Penalized prediction may remain stable while "
        "individual term attribution is not unique. Status 'not_evaluated' "
        "or 'not_provided' means metadata was unavailable, which does not "
        "constitute proof that a feature is non-derived.</em></p>"
    )


def _render_diagnostic_artifact_links(
    diagnostics_manifest: dict[str, Any] | None,
    diagnostics_directory: Path,
) -> str:
    artifacts_manifest = (
        diagnostics_manifest.get("artifacts", []) if diagnostics_manifest else []
    )
    artifact_map = {a.get("id"): a for a in artifacts_manifest if isinstance(a, dict)}

    is_current_schema = (
        diagnostics_manifest is not None
        and diagnostics_manifest.get("schema_name") == "gam_diagnostics_manifest"
        and diagnostics_manifest.get("schema_version") == "1.0"
    )

    links: list[str] = []

    for art_id, (filename, label) in DIAGNOSTIC_ARTIFACT_LINKS.items():
        file_path = diagnostics_directory / filename
        entry = artifact_map.get(art_id)

        available = False
        if entry is not None:
            available = entry.get("status") == "written" and file_path.exists()
        elif not is_current_schema:
            available = file_path.exists()

        if available:
            links.append(f"<li>{_diagnostic_link(path=filename, label=label)}</li>")

    if not links:
        return ""

    return f"<h3>Diagnostic artifacts</h3><ul>{''.join(links)}</ul>"


def _render_predictor_diagnostics(
    *,
    config: ExperimentConfig,
    store: FileRunStore,
    diagnostics_manifest: dict[str, Any] | None,
) -> str:
    diagnostics_directory = store.root / "diagnostics"
    if not diagnostics_directory.exists():
        return ""

    view = _build_predictor_diagnostics_view(
        diagnostics_manifest,
        diagnostics_directory=diagnostics_directory,
    )

    high_pairs = _load_csv(diagnostics_directory / "high_correlation_pairs.csv")
    conflict_artifact = diagnostics_directory / "conflicting_duplicate_targets.csv"

    parts = [
        '<section id="predictor-diagnostics">',
        "<h2>Predictor diagnostics</h2>",
        _render_predictor_summary(view),
        _render_duplicate_conflict_warning(
            conflict_count=view.conflicting_duplicate_target_group_count,
            artifact_available=conflict_artifact.exists(),
            analysis_status=view.duplicate_groups_status,
        ),
        _render_high_correlation_preview(
            frame=high_pairs,
            analysis_status=view.correlation_status,
        ),
        _render_diagnostics_interpretation(),
        _render_diagnostic_artifact_links(
            diagnostics_manifest,
            diagnostics_directory,
        ),
        "</section>",
    ]

    return "".join(parts)


def _render_report_heading(config: ExperimentConfig) -> str:
    return (
        f"<h1>{_escape(config.name)}</h1>"
        "<p>Penalized multinomial logistic additive B-spline experiment.</p>"
    )


def _render_dataset_overview(store: FileRunStore) -> str:
    data_manifest_path = store.root / "data_manifest.json"
    if not data_manifest_path.exists():
        return ""

    data_manifest = _load_json_object(data_manifest_path)
    if not data_manifest:
        return ""

    rows = data_manifest.get("rows", 0)
    predictors = data_manifest.get("predictors", [])
    target = data_manifest.get("target", "")
    class_counts = data_manifest.get("class_counts", {})

    dist_rows = []
    for cls_name, count in sorted(class_counts.items(), key=lambda x: str(x[0])):
        pct = (count / rows * 100.0) if rows > 0 else 0.0
        dist_rows.append(
            {
                "class": str(cls_name),
                "count": count,
                "percentage": f"{pct:.2f}%",
            }
        )
    dist_df = pd.DataFrame(dist_rows)

    parts = [
        '<section id="dataset-overview">',
        "<h2>Dataset overview</h2>",
        f"<p><strong>Number of observations:</strong> {rows}</p>",
        f"<p><strong>Number of active predictors:</strong> {len(predictors)}</p>",
        f"<p><strong>Target:</strong> {_escape(target)}</p>",
        "<h3>Class distribution</h3>",
        dist_df.to_html(index=False),
        "</section>",
    ]
    return "".join(parts)


def _render_model_results(config: ExperimentConfig, store: FileRunStore) -> str:
    parts: list[str] = []
    for model in config.models:
        result_dir = store.results / model.id
        if not (result_dir / "summary.csv").exists():
            continue

        summary = pd.read_csv(result_dir / "summary.csv", index_col=0)
        predictions = pd.read_parquet(result_dir / "predictions.parquet")

        model_parts = [f"<h2>{_escape(model.id)}</h2>"]

        metadata_path = store.models / model.id / "model_metadata.json"
        if metadata_path.exists():
            metadata = _load_json_object(metadata_path) or {}
            meta_df = pd.DataFrame(
                [{"property": k, "value": str(v)} for k, v in metadata.items()]
            )
            model_parts.append("<h3>Final-model metadata</h3>")
            model_parts.append(meta_df.to_html(index=False))

        params_path = store.models / model.id / "best_parameters.json"
        if params_path.exists():
            params = _load_json_object(params_path) or {}
            params_df = pd.DataFrame([params])
            model_parts.append(
                "<h3>Final full-data inner-CV hyperparameter selection</h3>"
            )
            model_parts.append(
                params_df.to_html(
                    index=False,
                    float_format=lambda value: f"{value:.6f}",
                )
            )

        model_parts.append("<h3>Nested CV metric summary</h3>")
        model_parts.append(summary.to_html(float_format=lambda value: f"{value:.6f}"))

        class_metrics_path = result_dir / "class_metrics.csv"
        if class_metrics_path.exists():
            class_metrics = pd.read_csv(class_metrics_path)
            class_summary = class_metrics.groupby("class", as_index=False).agg(
                sensitivity_mean=("sensitivity", "mean"),
                specificity_mean=("specificity", "mean"),
                precision_mean=("precision", "mean"),
                f1_mean=("f1", "mean"),
                mean_fold_support=("support", "mean"),
                total_oof_support=("support", "sum"),
            )
            class_summary["total_oof_support"] = class_summary[
                "total_oof_support"
            ].astype(int)
            model_parts.append("<h3>Per-class performance summary</h3>")
            model_parts.append(
                class_summary.to_html(
                    index=False,
                    float_format=lambda value: f"{value:.6f}",
                )
            )

        labels: list[str] = []
        if metadata_path.exists():
            metadata = _load_json_object(metadata_path) or {}
            labels = [str(value) for value in metadata.get("classes", [])]

        prediction_labels = set(predictions["observed_class"].astype(str)) | set(
            predictions["predicted_class"].astype(str)
        )

        if labels:
            missing_labels = prediction_labels - set(labels)
            if missing_labels:
                raise ValueError(
                    "Out-of-fold predictions contain classes that "
                    "are absent from final-model metadata: "
                    f"{sorted(missing_labels)}."
                )
        else:
            labels = sorted(prediction_labels)

        raw_matrix = confusion_matrix(
            predictions.observed_class, predictions.predicted_class, labels=labels
        )
        display = ConfusionMatrixDisplay(raw_matrix, display_labels=labels)
        display.plot()
        raw_path = store.plots / f"{model.id}_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(raw_path, dpi=160)
        plt.close()

        norm_matrix = confusion_matrix(
            predictions.observed_class,
            predictions.predicted_class,
            labels=labels,
            normalize="true",
        )
        norm_display = ConfusionMatrixDisplay(norm_matrix, display_labels=labels)
        norm_display.plot(values_format=".3f")
        norm_path = store.plots / f"{model.id}_confusion_matrix_normalized.png"
        plt.tight_layout()
        plt.savefig(norm_path, dpi=160)
        plt.close()

        model_parts.append("<h3>Confusion matrices</h3>")
        model_parts.append("<p><strong>Pooled raw count matrix:</strong></p>")
        model_parts.append(
            f'<img src="../plots/{raw_path.name}" alt="Confusion matrix">'
        )
        model_parts.append("<p><strong>Pooled row-normalized matrix:</strong></p>")
        model_parts.append(
            f'<img src="../plots/{norm_path.name}" alt="Normalized confusion matrix">'
        )

        model_parts.append(
            "<p><em>Note: These confusion matrices pool held-out outer-fold "
            "predictions across all outer repeats. Each observation contributes "
            "one out-of-fold prediction per repeat.</em></p>"
        )

        total_predictions = len(predictions)
        unique_obs = (
            predictions["row_id"].nunique() if "row_id" in predictions.columns else ""
        )
        outer_repeats = (
            predictions["repeat"].nunique() if "repeat" in predictions.columns else ""
        )

        count_info = []
        if unique_obs != "":
            count_info.append(f"Unique observations: {unique_obs}")
        if outer_repeats != "":
            count_info.append(f"Outer repeats: {outer_repeats}")
        count_info.append(f"Total OOF prediction events: {total_predictions}")

        model_parts.append(f"<p>{' | '.join(count_info)}</p>")

        model_parts.append("<h3>Detailed result files</h3>")
        mid = model.id
        links = [
            f'<li><a href="../results/{mid}/fold_metrics.csv">'
            "Detailed fold metrics</a></li>",
            f'<li><a href="../results/{mid}/class_metrics.csv">'
            "Detailed per-class fold metrics</a></li>",
            f'<li><a href="../results/{mid}/class_metrics_summary.csv">'
            "Per-class summary statistics</a></li>",
            f'<li><a href="../results/{mid}/predictions.parquet">'
            "Out-of-fold predictions</a></li>",
        ]
        model_parts.append(f"<ul>{''.join(links)}</ul>")

        inspection_dir = result_dir / "inspection"
        if inspection_dir.exists():
            model_parts.append("<h3>Inspection artifacts</h3>")
            insp_links = []
            if (inspection_dir / "equations.txt").exists():
                insp_links.append(
                    f'<li><a href="../results/{mid}/inspection/equations.txt">'
                    "View transformed-space equations</a></li>"
                )
            if (inspection_dir / "reference_equations.csv").exists():
                insp_links.append(
                    f'<li><a href="../results/{mid}/inspection/'
                    'reference_equations.csv">View reference-class contrasts</a></li>'
                )
            if (inspection_dir / "components.csv").exists():
                insp_links.append(
                    f'<li><a href="../results/{mid}/inspection/components.csv">'
                    "View coefficient components</a></li>"
                )
            if insp_links:
                model_parts.append(f"<ul>{''.join(insp_links)}</ul>")

        link_verification_dir = result_dir / "link_verification"
        if link_verification_dir.exists():
            model_parts.append("<h3>Link-verification results</h3>")
            ver_txt = link_verification_dir / "verification.txt"
            if ver_txt.exists():
                ver_content = ver_txt.read_text(encoding="utf-8")
                model_parts.append(f"<pre>{_escape(ver_content)}</pre>")
            ver_links = []
            if (link_verification_dir / "scores.csv").exists():
                ver_links.append(
                    f'<li><a href="../results/{mid}/link_verification/scores.csv">'
                    "Scores</a></li>"
                )
            if (link_verification_dir / "probabilities.csv").exists():
                ver_links.append(
                    f'<li><a href="../results/{mid}/link_verification/'
                    'probabilities.csv">Probabilities</a></li>'
                )
            if (link_verification_dir / "reconstructed_probabilities.csv").exists():
                ver_links.append(
                    f'<li><a href="../results/{mid}/link_verification/'
                    'reconstructed_probabilities.csv">'
                    "Reconstructed probabilities</a></li>"
                )
            if ver_links:
                model_parts.append(f"<ul>{''.join(ver_links)}</ul>")

        parts.append(
            f'<section id="model-{_escape(model.id)}">'
            + "".join(model_parts)
            + "</section>"
        )

    return "".join(parts)


def _build_html_document(*, title: str, sections: list[str]) -> str:
    html_parts = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{_escape(title)}</title>",
        f"<style>{REPORT_STYLE}</style>",
        "</head>",
        "<body>",
        "\n".join(sections),
        "</body>",
        "</html>",
    ]
    return "".join(html_parts)


def create_reports(config: ExperimentConfig, store: FileRunStore) -> None:
    diagnostics_manifest_path = store.root / "diagnostics" / "diagnostics_manifest.json"
    diagnostics_manifest = _load_json_object(diagnostics_manifest_path)

    validation_view = _build_validation_design_view(
        config=config,
        diagnostics_manifest=diagnostics_manifest,
    )

    sections = [
        _render_report_heading(config),
        _render_dataset_overview(store),
        _render_validation_design(validation_view),
        _render_predictor_diagnostics(
            config=config,
            store=store,
            diagnostics_manifest=diagnostics_manifest,
        ),
        _render_model_results(config, store),
    ]

    report = _build_html_document(
        title="GAM report",
        sections=[section for section in sections if section],
    )

    temp_path = store.reports / "report.html.tmp"
    store.reports.mkdir(parents=True, exist_ok=True)
    temp_path.write_text(report, encoding="utf-8")
    temp_path.replace(store.reports / "report.html")
