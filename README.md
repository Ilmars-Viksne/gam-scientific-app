# GAM Scientific App

A standalone, file-based, configuration-driven command-line application for
reproducible binary and multiclass classification with penalized generalized
additive models on tabular data.

The application supports ordinary stratified, group-aware, and time-aware
nested validation; predictor and duplicate diagnostics; resumable run
execution; machine-readable artifacts; HTML reporting; model inspection;
batch prediction; run comparison; and run discovery without requiring a
database or external service.

## Installation

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the application in editable development mode:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify installation:

```powershell
poetry run gam-app --help
```

## Quick start

Create demonstration data:

```powershell
poetry run gam-app demo `
  --output examples/demo.csv
```

Create a configuration:

```powershell
poetry run gam-app configure `
  --data examples/demo.csv `
  --target Y `
  --output configs/demo.yaml `
  --preset quick `
  --non-interactive
```

Check validation feasibility:

```powershell
poetry run gam-app plan `
  --config configs/demo.yaml
```

Run the experiment and persist its path:

```powershell
poetry run gam-app run `
  --config configs/demo.yaml `
  --workspace workspace `
  --run-path-file workspace/latest-run.txt
```

For the audited Windows PowerShell workflow, including exit-code handling, JSON output, run-path recovery, profiling, grouped validation, time-aware validation, inspection, contributions, and prediction, see `docs/powershell-workflow.md`.

## Capabilities

### Data and configuration

- CSV, TSV, and Parquet input
- categorical classification targets
- numeric and categorical predictors
- dataset profiling and feature-role recommendations
- interactive and noninteractive configuration generation
- versioned YAML configuration
- configuration migration from supported legacy schemas
- experiment tags and searchable key-value metadata

### Modelling

- penalized logistic additive models
- smooth B-spline terms
- standardized linear terms
- one-hot categorical terms
- excluded source columns
- main-effects models
- explicit or all-eligible smooth-by-smooth tensor-product interactions
- hyperparameter search inside nested validation

### Validation

- repeated nested stratified cross-validation
- group-aware stratified validation
- forward time-aware validation
- deterministic persisted outer split manifests
- configured scientific groups
- duplicate-derived effective validation groups
- validation-feasibility checks before execution
- persisted split-integrity diagnostics

### Predictor diagnostics

- Pearson and Spearman correlation matrices
- ranked high-correlation pair reports
- predictor data dictionary
- declared and suspected derived-variable metadata
- exact duplicate groups
- proper near-duplicate groups
- conflicting duplicate-target detection
- duplicate policies: report, error, and group
- schema-versioned diagnostic manifests
- artifact row counts, byte counts, and SHA-256 hashes

### Execution and reproducibility

- unique self-contained run directories
- dataset and configuration hashes
- environment and application-version metadata
- fold-level checkpoints
- pause, resume, and cancel controls
- immediate run-path return or persistence
- append-only event history
- atomic status updates
- workspace run discovery and metadata filtering

### Results and interpretation

- out-of-fold probabilities
- fold and aggregate classification metrics
- confusion matrices
- final fitted models
- transformed-space component exports
- exact transformed-space score equations
- reference-class equation exports
- link-function verification
- observation-level contribution exports
- grouped contribution summaries
- comparison of paired fold results
- batch prediction with input-schema validation
- HTML reports with validation and diagnostic summaries

## Validation strategies

Use the strategy that matches how the trained model will encounter future observations.

- `stratified`: use when observations can reasonably be treated as independent and class proportions should be preserved.
- `stratified_group`: use when related observations must remain together, such as observations from the same specimen, site, batch, participant, or duplicate-derived component.
- `time`: use when prediction is forward-looking and training observations must precede test observations.

Run `gam-app plan` before execution. It checks whether the configured dataset can support the requested outer and inner validation design.

For scientific background and guidance on selecting validation strategies, see `docs/scientific-interpretation.md`.

## Configuration

The generated YAML file is a complete, versioned experiment contract (current schema version 1.1). Users can modify configuration through `gam-app configure` or by editing the file directly; no source code changes are required.

Feature roles:
- `smooth`: univariate B-spline effect
- `linear`: standardized linear effect
- `categorical`: one-hot categorical effect
- `exclude`: omit column from active predictors

Interaction modes:
- `none`: main effects only
- `all_eligible`: all pairs among smooth predictors
- `explicit`: explicitly configured smooth-smooth predictor pairs

Reserved data-role columns:
- Columns configured under `data` (`target`, `row_id`, `group`, `time`) serve dedicated data roles and cannot be active model predictors. They must be set to `role: exclude` in `features` or omitted.

Schema migration:
- Upgrade legacy schema 1.0 configurations using `poetry run gam-app migrate-config --input configs/legacy.yaml --output configs/current.yaml`.

## Runs and outputs

Each experiment run creates a unique, self-contained directory in the workspace:

```text
run-id/
├── run.json
├── config.yaml
├── data_manifest.json
├── environment.json
├── split_manifest.csv
├── status.json
├── events.jsonl
├── run.lock
├── checkpoints/
├── diagnostics/
├── results/
├── models/
├── plots/
├── reports/
└── logs/
```

- `run.json`: overall run metadata including experiment tags and key-value metadata.
- `status.json`: atomically updated progress state (`created`, `running`, `paused`, `completed`, `failed`, `cancelled`).
- `events.jsonl`: append-only event log.
- `split_manifest.csv`: deterministic outer cross-validation fold assignments shared by all models in the run.
- `diagnostics/`: schema-versioned diagnostic artifacts (`diagnostics_manifest.json`, correlation matrices, duplicate groups, predictor dictionary).
- `reports/report.html`: standalone HTML report summarizing validation design, diagnostics, nested CV metrics, and confusion matrices.

For operational command details and artifact inspection steps, see `docs/powershell-workflow.md`.

## Diagnostics

The application runs dataset and duplicate diagnostics during profiling and experiment execution:

- Predictor correlations: Pearson and Spearman matrices with ranked high-correlation pair reporting.
- Derived variables: detection of declared and suspected functional relationships (affine, log, square, square root).
- Duplicate analysis: identification of exact duplicate groups and proper near-duplicate groups using transitive closure.
- Conflicting targets: detection of predictor-identical records with conflicting target labels.
- Duplicate policies (`report`, `error`, `group`): enforce error stops or merge duplicate constraints into effective validation groups.

For in-depth scientific interpretation of diagnostic findings, see `docs/scientific-interpretation.md`.

## Documentation

- `docs/powershell-workflow.md`: Audited operational PowerShell workflow with step-by-step commands, exit-code checking, JSON parsing, run recovery, and command lifecycle examples.
- `docs/scientific-interpretation.md`: Comprehensive scientific guide covering model mathematical formulation, predictor roles, regularization, nested validation, metrics, diagnostics, comparison sign conventions, review checklist, and reporting template.

## Development

```powershell
poetry run ruff format --check .
poetry run ruff check .
poetry run mypy src
poetry run pytest --cov=gam_app --cov-report=term-missing
```

The test suite includes focused unit tests, artifact-contract tests, property-based duplicate-analysis tests, and complete stratified, group-aware, and time-aware workflows.

Run the current suite to obtain the authoritative collected-test and coverage counts:

```powershell
poetry run pytest --cov=gam_app --cov-report=term-missing
```

## Security

Only load `.joblib` models created by a trusted installation. Joblib is pickle-based. The application records package versions and model metadata, but a serialized Python object is not a safe interchange format for untrusted files.

# Current release boundaries

## Included

This release supports:

- local standalone execution
- command-line operation
- file-based persistence without a database
- CSV, TSV, and Parquet input
- categorical binary and multiclass classification targets
- numeric and categorical predictors
- smooth, linear, categorical, and excluded feature roles
- main-effects additive models
- explicit and all-eligible smooth-by-smooth tensor-product interactions
- ordinary stratified nested validation
- stratified group-aware nested validation
- forward time-aware nested validation
- configured and duplicate-derived validation groups
- exact and proper near-duplicate diagnostics
- predictor-correlation diagnostics
- persisted validation and diagnostic manifests
- pause, resume, cancel, and fold-level checkpoint recovery
- final model fitting and batch prediction
- transformed-space inspection and contribution exports
- paired run or model comparison
- HTML and machine-readable outputs
- local run discovery and metadata filtering

## Not included

This release does not include:

- regression, survival, count, or continuous-target modelling
- causal-effect estimation
- classical unpenalized coefficient significance tests
- confidence intervals or p-values for smooth terms
- automated feature selection based on causal or scientific relevance
- stability-aware forward interaction selection
- categorical-by-smooth interaction terms
- functional-ANOVA centering of interaction surfaces
- automated calibration curves or recalibration
- local explanation dashboards
- a desktop graphical application
- a hosted web service
- distributed or cluster execution
- a database-backed run registry
- remote artifact storage
- automated correction of conflicting target labels
- automatic deletion or relabelling of duplicate observations
- an executable desktop installer
