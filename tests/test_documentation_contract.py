import re
from pathlib import Path
from typing import Any, cast

import pytest

from gam_app.cli import build_parser

DOCUMENTED_COMMANDS = (
    "demo",
    "profile",
    "configure",
    "plan",
    "migrate-config",
    "run",
    "list-runs",
    "status",
    "pause",
    "resume",
    "cancel",
    "inspect",
    "verify-link",
    "compare",
    "predict",
    "transform",
    "contributions",
    "grouped-contributions",
    "review-diagnostics",
    "create-sensitivity",
    "show-sensitivity",
)


def test_documentation_guides_exist() -> None:
    assert Path("docs/powershell-workflow.md").is_file()
    assert Path("docs/scientific-interpretation.md").is_file()


def test_readme_links_documentation_guides() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "docs/powershell-workflow.md" in readme
    assert "docs/scientific-interpretation.md" in readme


@pytest.mark.parametrize(
    "stale_claim",
    [
        "grouped or temporal CV",
        "The included test suite completed successfully",
        "4 passed",
    ],
)
def test_readme_omits_known_stale_claims(stale_claim: str) -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert stale_claim not in readme


@pytest.mark.parametrize("command", DOCUMENTED_COMMANDS)
def test_all_public_commands_are_documented(command: str) -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    workflow = Path("docs/powershell-workflow.md").read_text(encoding="utf-8")
    combined = readme + "\n" + workflow
    assert command in combined


@pytest.mark.parametrize(
    "strategy",
    [
        "stratified",
        "stratified_group",
        "time",
    ],
)
def test_readme_documents_supported_validation_strategies(strategy: str) -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert strategy in readme


@pytest.mark.parametrize(
    "unsupported_capability",
    [
        "causal-effect estimation",
        "regression",
        "functional-ANOVA centering",
        "distributed",
    ],
)
def test_readme_documents_release_exclusions(
    unsupported_capability: str,
) -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert unsupported_capability in readme


def test_interpretation_guide_rejects_causal_and_significance_claims() -> None:
    guide = Path("docs/scientific-interpretation.md").read_text(encoding="utf-8")
    required_phrases = (
        "does not establish causality",
        "significance",
    )
    for phrase in required_phrases:
        assert phrase.casefold() in guide.casefold()


def test_interpretation_guide_does_not_prescribe_duplicate_cleaning() -> None:
    guide = Path("docs/scientific-interpretation.md").read_text(encoding="utf-8")
    assert "does not alter, delete, or clean source data" in guide.casefold()


def test_interpretation_guide_covers_all_validation_strategies() -> None:
    guide = Path("docs/scientific-interpretation.md").read_text(encoding="utf-8")
    for strategy in ("stratified", "stratified_group", "time"):
        assert strategy in guide


@pytest.mark.parametrize(
    "concept",
    [
        "Pearson",
        "Spearman",
        "declared",
        "suspected",
        "Exact Duplicates",
        "Proper Near-Duplicates",
        "Conflicting Duplicate Targets",
        "report",
        "error",
        "group",
    ],
)
def test_interpretation_guide_covers_diagnostic_concepts(concept: str) -> None:
    guide = Path("docs/scientific-interpretation.md").read_text(encoding="utf-8")
    assert concept.casefold() in guide.casefold()


def test_interpretation_guide_documents_comparison_direction() -> None:
    guide = Path("docs/scientific-interpretation.md").read_text(encoding="utf-8")
    assert "right" in guide
    assert "left" in guide
    assert "lower log loss is better" in guide.casefold()


@pytest.mark.parametrize(
    "required_fragment",
    [
        "Set-StrictMode",
        "$ErrorActionPreference",
        "$LASTEXITCODE",
        "Join-Path",
        "ConvertFrom-Json",
        "--run-path-file",
        "Test-Path",
    ],
)
def test_powershell_workflow_contains_audit_controls(
    required_fragment: str,
) -> None:
    workflow = Path("docs/powershell-workflow.md").read_text(encoding="utf-8")
    assert required_fragment in workflow


def test_workflow_uses_verified_transformation_and_contribution_commands() -> None:
    workflow = Path("docs/powershell-workflow.md").read_text(encoding="utf-8")
    assert "gam-app transform" in workflow
    assert "gam-app contributions" in workflow
    assert "gam-app grouped-contributions" in workflow
    assert "--model $ModelPath" in workflow
    assert "--input $PredictionInput" in workflow
    assert "--output $ContributionsOutput" in workflow


@pytest.mark.parametrize(
    "forbidden_fragment",
    [
        "C:\\Users\\",
        "/home/",
        "/Users/",
    ],
)
def test_documentation_contains_no_developer_specific_paths(
    forbidden_fragment: str,
) -> None:
    documentation = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "docs/powershell-workflow.md",
            "docs/scientific-interpretation.md",
        )
    )
    assert forbidden_fragment not in documentation


def test_readme_does_not_publish_a_fixed_test_count() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert not re.search(
        r"\b\d+\s+passed\b",
        readme,
        flags=re.IGNORECASE,
    )


def test_cli_options_in_workflow_exist_in_parser() -> None:
    parser = build_parser()
    subparsers_actions = [
        action for action in parser._actions if action.dest == "command"
    ]
    assert len(subparsers_actions) == 1
    choices = cast(dict[str, Any], subparsers_actions[0].choices)
    assert choices is not None

    configure_options = {
        opt for action in choices["configure"]._actions for opt in action.option_strings
    }
    assert "--validation-strategy" in configure_options
    assert "--duplicate-group-policy" in configure_options
    assert "--outer-splits" in configure_options
    assert "--outer-repeats" in configure_options
    assert "--inner-splits" in configure_options

    run_options = {
        opt for action in choices["run"]._actions for opt in action.option_strings
    }
    assert "--run-path-file" in run_options
    assert "--create-only" in run_options
    assert "--json" in run_options
