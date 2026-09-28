# GAM Scientific App

GAM Scientific App is a local, configuration-driven command-line application for binary and multiclass classification of tabular data with penalized generalized additive models (GAMs). It combines explicit feature-role configuration, nested cross-validation, predictor and duplicate diagnostics, persisted split-integrity checks, resumable file-based execution, model inspection, and batch prediction.

The application is intended for reproducible predictive studies where users need to audit how a model was prepared, tuned, validated, and exported. It supports ordinary stratified, group-aware stratified, and forward time-aware validation. It does **not** establish causality, physical mechanisms, or classical coefficient significance.

## Contents

- [Model and mathematics](#model-and-mathematics)
- [Install](#install)
- [End-user workflow](#end-user-workflow)
- [Configuration reference](#configuration-reference)
- [Command reference](#command-reference)
- [Artifacts and outputs](#artifacts-and-outputs)
- [Architecture](#architecture)
- [Development and tests](#development-and-tests)
- [Troubleshooting and limitations](#troubleshooting-and-limitations)
- [Further documentation](#further-documentation)

## Model and mathematics

### Additive classification model

Let $n$ be the number of observations, $K \ge 2$ the number of target classes, and $\mathbf{x}_i$ the configured predictors for row $i$. The feature transformer maps predictors to a design vector $\boldsymbol{\phi}(\mathbf{x}_i) \in \mathbb{R}^{d}$. For each class $k$, the class score is

$$
\eta_{ik} = \beta_{0k} + \boldsymbol{\phi}(\mathbf{x}_i)^\top \boldsymbol{\beta}_k.
$$

The feature vector concatenates these terms:

- **Smooth**: a univariate B-spline basis $\mathbf{B}_j(x_{ij})$ fitted with scikit-learn's `SplineTransformer`, configured by `n_knots` and `degree`, with `include_bias=False`.
- **Linear**: a standardized value $\widetilde{x}_{ij}=(x_{ij}-\mu_j)/s_j$, with the mean and scale learned from the current training partition.
- **Categorical**: one-hot indicators for the configured category vocabulary, with the first configured level dropped as the reference level.
- **Excluded**: not included in the design matrix. Target, row ID, group, and time columns are data roles and cannot be active predictors.

For smooth predictor set $\mathcal{S}$, linear set $\mathcal{L}$, categorical set $\mathcal{C}$, and configured interaction pairs $\mathcal{I}$, the score can be written as

$$
\eta_k(\mathbf{x}) = \beta_{0k}
+ \sum_{j\in\mathcal{S}} \mathbf{B}_j(x_j)^\top\boldsymbol{\theta}_{jk}
+ \sum_{j\in\mathcal{L}} \widetilde{x}_j\beta_{jk}
+ \sum_{j\in\mathcal{C}} \boldsymbol{D}_j(x_j)^\top\boldsymbol{\gamma}_{jk}
+ \sum_{(r,s)\in\mathcal{I}} \left[\mathbf{B}_r(x_r)\otimes\mathbf{B}_s(x_s)\right]^\top\boldsymbol{\delta}_{rsk},
$$

where $\mathbf{D}_j$ is the dropped-first one-hot vector and $\otimes$ is the row-wise tensor product. In the implementation, if $\mathbf{B}_r\in\mathbb{R}^{n\times q_r}$ and $\mathbf{B}_s\in\mathbb{R}^{n\times q_s}$, their interaction contributes $q_rq_s$ columns. Every interaction column is multiplied by the configured `interaction_scale` before fitting. Interactions are available only between distinct smooth predictors and are fitted alongside both main effects; they are not functional-ANOVA-centered.

For $K$ classes, probabilities are obtained with the softmax link:

$$
p_{ik}=P(Y_i=k\mid\mathbf{x}_i)=\frac{\exp(\eta_{ik})}{\sum_{\ell=1}^{K}\exp(\eta_{i\ell})}.
$$

For a binary classifier, the exported score representation uses scores $(0,z_i)$, giving $P(Y_i=1)=\sigma(z_i)=1/(1+\exp(-z_i))$. Adding the same quantity to all class scores does not change softmax probabilities, so class-contrast equations are more meaningful than absolute class scores.

### Fitting and model selection

The transformed matrix has shape $n\times d$. It is passed to scikit-learn `LogisticRegression` (`max_iter=5000`); the configured `C` is its inverse regularization-strength parameter. For observations with one-hot labels $y_{ik}$, the fitted classifier minimizes multinomial cross-entropy with L2 regularization. In the usual averaged-loss notation this is

$$
\mathcal{L}(\boldsymbol{\beta}) = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K}y_{ik}\log p_{ik}
+ \frac{\lambda}{2}\sum_{k=1}^{K}\|\boldsymbol{\beta}_k\|_2^2,
$$

where $\lambda$ increases as `C` decreases; the exact internal loss scaling follows scikit-learn's solver implementation. The primary selection metric is log loss:

$$
\operatorname{LogLoss}=-\frac{1}{N}\sum_{i=1}^{N}\log p_{i,y_i}.
$$

Within each outer training partition, the application evaluates the Cartesian product of `n_knots`, `degree`, `C`, and, for models with interactions, `interaction_scale`. Each candidate is scored by mean inner-fold log loss; the lowest-loss candidate is selected. Preprocessing is part of the fitted pipeline, so spline knots, imputers, and linear scaling are learned from the corresponding training fold rather than from its validation or test rows.

The outer folds estimate out-of-fold performance after tuning has taken place using outer-training data only. The final deployable model is then refitted to all observations using the run's selected hyperparameters. This separation reduces the direct optimism caused by evaluating a model on the same folds used to select its hyperparameters. It does not guarantee transportability, calibration, or independence of the source data.

The selected methods serve different practical needs: spline bases provide nonlinear univariate effects while retaining additive, inspectable components; standardized linear and one-hot terms support numeric and categorical predictors in the same model; tensor products add explicit smooth pairwise variation; and nested validation separates selection from outer evaluation. L2 regularization controls coefficient magnitude and helps manage flexible bases, but it is not feature selection or a significance test.

### Validation designs and metrics

| Strategy | Split behavior | Use when |
| --- | --- | --- |
| `stratified` | Repeated stratified K-fold; preserves class proportions approximately in folds. | Rows can be treated as independent and identically distributed for the target deployment. |
| `stratified_group` | `StratifiedGroupKFold`; rows with the same effective group stay together. | Related observations, specimens, batches, sites, participants, or duplicates must not cross folds. |
| `time` | `TimeSeriesSplit`; sorts by timestamp and row ID, then uses forward training/test windows. Training timestamps must be strictly earlier than test timestamps. | The intended prediction is forward in time. |

For each outer repeat, every row is assigned to test once for stratified strategies. Time validation requires `outer_repeats: 1`; `gap` and `test_size` apply only to this strategy. Feasibility and split integrity are checked before model execution. `plan` is the recommended preflight command.

Fold-level metrics include log loss, accuracy, balanced accuracy, macro F1, macro and weighted specificity, and class-specific specificity. Class-level exports also report sensitivity, specificity, precision, F1, and support. Accuracy and F1 are higher-is-better; log loss is lower-is-better. Fold dispersion is descriptive, not a confidence interval.

### Interpretation boundaries

- Score contributions add in logit-score space, not probability or percentage space.
- Raw spline coefficients depend on the basis, knots, degree, scaling, reference categories, and regularization. Use component and contrast exports rather than treating one coefficient as a marginal probability effect.
- A fitted smooth or interaction describes predictive association conditional on the included terms; it does not establish a mechanism or causal effect.
- Prediction outside the training predictor support is extrapolation and is not scientifically validated by this application.

## Install

### Requirements

- Python 3.11 or newer.
- A local CPU-capable Python environment. The package does not depend on CUDA, GPU libraries, a database, or a hosted service.
- Disk space for source data, run artifacts, checkpoints, reports, and optional Parquet output. Memory and run time depend on row count, validation folds/repeats, search-grid size, and interaction count. Pairwise interactions can substantially increase the design width.

Runtime dependencies are declared in [`pyproject.toml`](pyproject.toml): joblib, matplotlib, NumPy, pandas, PyArrow, PyYAML, and scikit-learn. There are no required environment variables. On Windows, use PowerShell 5.1 or PowerShell 7+ for the commands below.

### Virtual environment and installation

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
gam-app --help
```

To install runtime dependencies only, use `python -m pip install -e .`. If PowerShell blocks activation under the current policy, activate the environment using your organization's approved policy or invoke its interpreter directly as `\.venv\Scripts\python.exe`.

## End-user workflow

### 1. Prepare and profile data

Input is a rectangular table with one header row and unique column names. Supported formats are CSV (`.csv`), tab-separated text (`.tsv`, `.txt`), and Parquet (`.parquet`, `.pq`). The target must be present, nonmissing, and contain at least two classes. Binary and multiclass targets are supported; regression is not.

Numeric predictors may be assigned `smooth` or `linear`; categorical predictors use a finite, explicit category vocabulary. Active predictor missing values default to an error. Numeric roles optionally support `median` imputation; categorical roles support `most_frequent`. Target values and group/time values cannot be missing. Configured row IDs must be unique. A time column is parsed as UTC timestamps.

Profile a dataset and write summaries plus standalone diagnostic artifacts:

```powershell
gam-app profile `
  --data data/steel_plates_faults.csv `
  --target FaultClass `
  --output profile/steel
```

`profile` writes `profile.json`, `columns.csv`, correlation and duplicate analysis CSVs, a predictor dictionary, and `diagnostics_manifest.json`. Check the profile's target counts, missingness, inferred roles, high correlations, suspected derived relationships, and duplicate/conflicting-target findings before choosing the validation design. The feature-role recommendations are heuristics, not automatic scientific decisions.

For a self-contained example, create deterministic demo data:

```powershell
gam-app demo --output examples/demo.csv --rows 300 --seed 42
```

The demo has numeric predictors `X1`, `X2`, `X4`, categorical `X3`, and four-class target `Y`.

### 2. Generate and check an experiment configuration

Generate a noninteractive quick configuration:

```powershell
gam-app configure `
  --data examples/demo.csv `
  --target Y `
  --output configs/demo.yaml `
  --name demo `
  --preset quick `
  --non-interactive

gam-app plan --config configs/demo.yaml
```

The configuration is YAML and current schema version is `1.1`. Relative data paths are resolved relative to the configuration file. `configure` creates `gam_main` (main effects) and `gam_pairwise` (all smooth-by-smooth pairs) models by default. Use the generated file as the full experiment contract; it can also be edited directly and checked with `plan`.

### 3. Run and capture the run path

```powershell
gam-app run `
  --config configs/demo.yaml `
  --workspace workspace `
  --run-path-file workspace/latest-run.txt

$RunPath = (Get-Content workspace/latest-run.txt -Raw).Trim()
gam-app status --run $RunPath
```

Runs are created under `workspace/runs/` and contain their resolved configuration, provenance hashes, status/event history, split manifest, diagnostics, fold checkpoints, fitted models, metrics, and report. Use `--create-only --json` to initialize a run without executing it. Use `status --follow`, `pause`, `resume`, and `cancel` to monitor or control a live run. Resume reuses complete fold checkpoints whose data and configuration hashes match.

### 4. Inspect results and deploy predictions

Open `reports/report.html`, review each model's results and diagnostics, then inspect a fitted model:

```powershell
gam-app inspect --run $RunPath --model gam_main
gam-app verify-link --run $RunPath --model gam_main
```

For batch prediction, provide a CSV with the trained predictor columns and compatible values. The target column is not required:

```powershell
gam-app predict `
  --model "$RunPath/models/gam_main/model.joblib" `
  --input predictions/new-observations.csv `
  --output predictions/scored-observations.csv
```

`transform` exports the fitted GAM design matrix; `contributions` exports additive score contributions for every class; `grouped-contributions` aggregates that CSV by predictor or interaction family. These exports aid model inspection and are not probability decompositions.

### Edge-case workflow examples

**Group-aware validation with configured and duplicate-derived grouping:**

Create a deterministic demo table with a synthetic batch label (for exercising the split mechanics only):

```powershell
gam-app demo --output examples/demo.csv --rows 300 --seed 42
$rows = Import-Csv examples/demo.csv
for ($index = 0; $index -lt $rows.Count; $index++) {
  $batch = "batch_{0:D2}" -f [int][math]::Floor($index / 10)
  Add-Member -InputObject $rows[$index] -NotePropertyName Batch -NotePropertyValue $batch
}
$rows | Export-Csv examples/demo-grouped.csv -NoTypeInformation
```

```powershell
gam-app configure `
  --data examples/demo-grouped.csv `
  --target Y `
  --group Batch `
  --validation-strategy stratified_group `
  --duplicate-group-policy group `
  --preset quick `
  --non-interactive `
  --output configs/steel-grouped.yaml

gam-app plan --config configs/steel-grouped.yaml --json
```

Effective groups merge configured groups, predictor-identical signatures, and proper near-duplicate links into connected components. Grouping prevents related rows from crossing folds; it does not modify or remove source rows. Group-level class scarcity can make a requested split infeasible, so review the plan before running.

**Forward time validation with a gap and fixed test window:**

Add unique, increasing timestamps to the demo data:

```powershell
gam-app demo --output examples/demo.csv --rows 300 --seed 42
$rows = Import-Csv examples/demo.csv
$origin = [datetime]::new(2026, 1, 1, 0, 0, 0, [datetimekind]::Utc)
for ($index = 0; $index -lt $rows.Count; $index++) {
  $timestamp = $origin.AddDays($index).ToString("o")
  Add-Member -InputObject $rows[$index] -NotePropertyName ObservedAt -NotePropertyValue $timestamp
}
$rows | Export-Csv examples/demo-time.csv -NoTypeInformation
```

```powershell
gam-app configure `
  --data examples/demo-time.csv `
  --target Y `
  --time ObservedAt `
  --validation-strategy time `
  --outer-splits 3 `
  --outer-repeats 1 `
  --inner-splits 2 `
  --gap 2 `
  --test-size 20 `
  --preset quick `
  --non-interactive `
  --output configs/steel-time.yaml

gam-app plan --config configs/steel-time.yaml
```

Time-aware splitting sorts timestamp ties deterministically by row ID, but rejects a fold whose latest training timestamp is equal to or later than its earliest test timestamp. Tied timestamps at a boundary may therefore require a larger gap or a different test-window setup.

**Explicit interactions:** edit a model entry in YAML to list only desired pairs; both names must be distinct features with role `smooth`:

```yaml
models:
  - id: gam_selected_pairs
    interactions: explicit
    pairs:
      - [X1, X2]
```

For a sensitivity analysis, run reference and variant configurations, then link their completed run paths with `create-sensitivity`. Comparisons require comparable data, target, validation design, and outer test assignments; use `compare --check-only` to check eligibility before writing a comparison.

## Configuration reference

The canonical examples are produced by `gam-app configure`; the full field definitions and validation rules are implemented in `src/gam_app/config.py`.

| Section / key | Meaning and supported values |
| --- | --- |
| `schema_version` | Current schema is `1.1`; supported legacy `1.0` configurations can be upgraded with `migrate-config`. |
| `experiment` | `name`, fixed `primary_metric: log_loss`, optional `tags` and searchable string `metadata`. |
| `data` | `path`, categorical `target`, optional unique `row_id`, optional `group`, optional `time`. These four roles must refer to distinct columns. |
| `features.<column>.role` | `smooth`, `linear`, `categorical`, or `exclude`. At least one predictor must be active. Reserved data-role columns must be excluded. |
| `features.<column>.missing` | `error` (default); numeric features additionally allow `median`; categorical features allow `most_frequent`. |
| `features.<column>.categories` | Required explicit ordered vocabulary for categorical predictors. Its first level is the one-hot reference level. Unconfigured/unknown values fail validation. |
| `features.<column>` metadata | Optional `reference`, `derived` (`none`, `declared`, `suspected`), `derived_from`, `derivation`, `description`, and `unit`. Declared derived features must list their source features. |
| `models[]` | Unique `id`; `interactions` is `none`, `all_eligible`, or `explicit`; `pairs` is used for explicit pairs of distinct smooth predictors. |
| `profiling.correlation` | `enabled`, Pearson/Spearman toggles, `review_threshold` and `warning_threshold` in $(0,1]$ with warning at least review, and `minimum_complete_pairs` of at least 2. |
| `profiling.duplicate_groups` | `enabled`, numeric `rounding_decimals` (nonnegative), `near_duplicate_threshold` in $(0,1]$, `maximum_pairwise_rows` of at least 2, and `include_target_in_signature` (must remain false). |
| `validation` | `strategy`, `outer_splits` (at least 2), `outer_repeats` (at least 1), `inner_splits` (at least 2), `random_state`, time-only `gap` (nonnegative) and `test_size` (positive if set), and duplicate policy `report`, `error`, or `group`. |
| `search` | Positive `C` values, `n_knots` values at least 2, `degree` values at least 1, and `interaction_scale`. Lists form a Cartesian product. |
| `execution` | `workers` at least 1, `checkpoint_unit: outer_fold`, and `stop_on_convergence_warning` (default true). |

For `stratified_group`, provide a group column or use duplicate policy `group` with duplicate diagnostics enabled. Duplicate policy `group` is not supported with `time`; use `report` or `error`. Policy `error` stops before split creation when duplicate groups are found. Policy `report` reports them without changing the split groups.

The `configure` search presets are:

| Preset | Outer folds × repeats | Inner folds | `n_knots` | `degree` | `C` |
| --- | --- | --- | --- | --- | --- |
| `quick` | 3 × 1 | 3 | `[3]` | `[2]` | `[0.1, 1.0, 10.0]` |
| `standard` | 5 × 3 | 5 | `[3, 4, 5]` | `[2, 3]` | `[0.01, 0.1, 1.0, 10.0]` |
| `thorough` | 5 × 5 | 5 | `[3, 4, 5, 6]` | `[2, 3]` | `[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]` |

`configure` sets `outer_repeats` to 1 by default for `time`, irrespective of preset; otherwise it uses the selected preset. Explicit split-count options override preset counts.

## Command reference

Run `gam-app <command> --help` for the live parser help. These are the public commands registered by the CLI:

| Command | Main arguments / purpose |
| --- | --- |
| `demo` | `--output PATH [--rows N] [--seed N]`; generate the four-class example dataset. |
| `profile` | `--data PATH --target COLUMN --output DIR`; add correlation and duplicate diagnostic thresholds with `--review-correlation`, `--warn-correlation`, `--near-duplicate-decimals`, `--near-duplicate-threshold`, and `--maximum-pairwise-rows`. |
| `configure` | `--data PATH --target COLUMN --output PATH`; accepts `--name`, `--row-id`, `--group`, `--time`, `--validation-strategy`, `--gap`, `--test-size`, `--preset`, explicit split counts, `--random-state`, duplicate/correlation controls, repeated `--tag` and `--metadata KEY=VALUE`, and `--non-interactive`. |
| `migrate-config` | `--input PATH --output PATH [--overwrite]`; migrate supported legacy YAML schema 1.0. |
| `plan` | `--config PATH [--json]`; summarize design and check validation feasibility. |
| `run` | `--config PATH [--workspace DIR] [--json] [--create-only] [--run-path-file PATH]`; create a run and execute unless create-only is set. |
| `status` | `--run DIR [--follow]`; inspect or follow run status. |
| `pause`, `resume`, `cancel` | `--run DIR`; control or restart a run. |
| `inspect` | `--run DIR --model ID [--reference-class LABEL]`; export/inspect model equations and components. |
| `verify-link` | `--run DIR --model ID`; verify exported scores against classifier probabilities. |
| `compare` | `--left DIR --left-model ID --right DIR --right-model ID`; optional `--output DIR`, `--check-only`, `--json`, `--overwrite`, `--sensitivity ID`. |
| `predict` | `--model MODEL_JOBLIB --input CSV --output CSV`; batch prediction with fitted preprocessing. |
| `transform` | `--model MODEL_JOBLIB --input CSV --output CSV`; export the transformed design matrix. |
| `contributions` | Same model/input/output paths; optional `--top N` (default 10; CSV retains all components). |
| `grouped-contributions` | `--input CONTRIBUTIONS_CSV --output CSV`; optional `--top N` and `--reference-class LABEL`. |
| `list-runs` | Optional `--workspace DIR`, repeated filters such as `--state`, `--experiment`, `--sensitivity`, `--strategy`, `--duplicate-policy`, `--model`, `--tag`, `--metadata`; also date/hash/limit filters, `--json`, `--include-invalid`. |
| `create-sensitivity` | Required `--id`, `--name`, `--reference-run`, and one or more `--variant-run`; optional repeated `--vary`, `--invariant`, `--output`, `--overwrite`, `--json`, `--workspace`. |
| `show-sensitivity` | `--manifest PATH [--json]`; display a sensitivity manifest. |
| `review-diagnostics` | `--run DIR`; optional `--json`, `--output PATH`, `--overwrite`, `--strict`, and `--[no-]verify-artifacts`. |

Planning emits exit code 2 when feasibility checks fail. Other command errors are printed to stderr and exit with code 1. For Windows PowerShell scripts that need robust exit-code handling, JSON parsing, and run-path recovery, follow [`docs/powershell-workflow.md`](docs/powershell-workflow.md).

## Artifacts and outputs

Runs are self-contained directories under `<workspace>/runs/<run-id>/`. Common files and directories are:

```text
run-id/
├── run.json                    # run identity, data/config hashes, experiment, validation, tags
├── config.yaml                 # resolved run configuration
├── environment.json            # Python/platform and selected dependency versions
├── status.json                 # atomic lifecycle state and timestamps
├── events.jsonl                # append-only execution event history
├── run.lock                    # active execution lock, when running
├── split_manifest.csv          # row-level train/test assignments by repeat and fold
├── control/                    # pause/cancel control markers
├── checkpoints/<model>/...     # fold metrics, predictions, class metrics, completion metadata
├── diagnostics/                # diagnostic package and split-integrity results
├── results/<model>/            # fold metrics, OOF predictions, class metrics and summaries
├── models/<model>/             # model.joblib, parameters, search trials, components, metadata
├── plots/                      # generated plots when applicable
├── reports/report.html         # standalone report
└── logs/                       # run logs
```

Model results include `fold_metrics.csv`, `predictions.parquet`, `class_metrics.csv`, `summary.csv`, and `class_metrics_summary.csv`. Model directories include `model.joblib`, `best_parameters.json`, `search_trials.parquet`, `components.csv`, and `model_metadata.json`. Diagnostics include `diagnostics_manifest.json`, `split_integrity.csv`, Pearson/Spearman correlation matrices, high-correlation pairs, numeric predictor dictionary, suspected derived relations, exact/near duplicate groups, and conflicting duplicate targets. Empty diagnostic artifacts remain schema-versioned artifacts rather than being mistaken for disabled or failed analyses.

`run.json` records SHA-256 data and configuration hashes; the diagnostic manifest records artifact metadata including row/byte counts and hashes. `status.json` lifecycle states include `created`, `running`, `paused`, `completed`, `failed`, and `cancelled`. The report summarizes validation, diagnostics, metrics, and confusion matrices. The run directory is the primary unit for archiving or transferring experiment evidence.

Comparison metric differences use $\Delta=\text{right}-\text{left}$. Positive accuracy, balanced accuracy, and macro F1 differences favor the right model; positive log-loss differences favor the left model because lower log loss is better. Comparisons require matching scientific conditions and outer test assignments, not merely similar metric names.

## Architecture

```text
src/gam_app/
├── cli.py                 # argparse command registration and command handlers
├── config.py              # immutable configuration dataclasses, schema parsing/validation
├── config_migration.py    # supported YAML schema migration
├── data.py                # tabular I/O, profiling, and training-data validation
├── transformers.py        # GAM basis, imputation, scaling, category encoding, interactions
├── models.py              # scikit-learn Pipeline construction
├── logistic.py            # class-score representation and softmax reconstruction
├── splitting.py           # outer/inner splits, group merging, manifest and integrity checks
├── planning.py            # pre-run validation feasibility checks
├── diagnostics.py         # correlations, derived-variable and duplicate analyses/artifacts
├── diagnostic_schema.py   # diagnostic schemas and artifact manifest
├── diagnostic_review.py   # independent persisted-diagnostic validation and summary
├── evaluation.py          # inner search, outer-fold execution, metrics and final fits
├── workflow.py            # run creation/orchestration, provenance, diagnostics and lifecycle
├── run_store.py           # file-backed status, events, checkpoints and control markers
├── reporting.py           # standalone HTML report generation
├── inspection.py          # equations, transformed components and link verification
├── comparison.py          # paired-run comparability and metric comparisons
├── sensitivity.py         # sensitivity-study manifest creation and validation
├── run_catalog.py         # workspace run discovery and filters
├── io_utils.py            # atomic writes, hashes, JSON/CSV helpers and formatting
└── exceptions.py          # domain-specific errors

tests/                     # unit, CLI, artifact-schema, property, integration and e2e tests
docs/                      # PowerShell operations and scientific interpretation
examples/                  # example experiment configuration
scripts/                   # dataset preparation and scenario prediction helpers
configs/ data/ profile/ predictions/ comparisons/ workspace/
                           # example inputs and generated experiment artifacts
```

The console entry point is `gam-app = gam_app.cli:main`. The project uses a pipeline boundary: configuration/data validation produce explicit inputs, splitting persists outer assignments, each model/fold gets an isolated pipeline fit, and the run store persists resumable outputs. Dataclass configuration objects are frozen and slotted. Public-facing source uses type annotations; domain validation raises explicit configuration/data errors, while CLI handling reports exceptions and returns a nonzero exit.

To add a feature role or model behavior, start with its owning boundary rather than placing it in the CLI: define/validate configuration in `config.py`, implement transformation in `transformers.py`, wire it through `models.py`, and ensure fitting/exports in `evaluation.py` and `inspection.py` stay consistent. Add focused transformer/config tests and an end-to-end test when the persisted workflow contract changes. To add a CLI workflow, register arguments in `build_parser()` and add handler logic in `cli.py`; reuse existing domain modules and add parser/behavior tests. A new loss or selection metric must be propagated consistently through inner selection, outer evaluation, summaries, comparisons, and reports; update the declared configuration contract and tests rather than changing only the estimator call.

## Development and tests

Install the development extras in the virtual environment using `python -m pip install -e ".[dev]"`. The project configures Ruff (88-character line length, Python 3.11 target; lint rules `E`, `F`, `I`, `B`, `UP`, with `E501` ignored), mypy (`check_untyped_defs`, unused-config warnings), pytest, pytest-cov, and Hypothesis. No pre-commit configuration or hook is defined in this repository.

Run focused or full checks from the repository root:

```powershell
python -m pytest tests/test_transformers.py
python -m pytest -m e2e
python -m pytest -m property
python -m pytest --cov=gam_app --cov-report=term-missing
python -m ruff format --check .
python -m ruff check .
python -m mypy src
```

Pytest's configured test directory is `tests/`; the `e2e` and `property` markers identify complete workflow and property-based tests. There is no separate performance benchmark suite or configured coverage threshold. The `pyproject.toml` mypy target is Python 3.14 even though the package's minimum supported runtime is Python 3.11; run mypy with a compatible installed checker and treat the package runtime requirement separately.

## Troubleshooting and limitations

- **Configuration or missing columns:** run `plan` and check the exact column names. Relative data paths resolve from the YAML file's directory.
- **Too few class observations:** stratified designs need enough observations per class for both outer and inner folds. Reduce fold counts only when scientifically defensible; grouped designs additionally require feasible class/group composition.
- **Group strategy rejected:** configure `data.group`, or use duplicate policy `group` with duplicate diagnostics enabled. `group` duplicate policy is incompatible with `time` and ordinary `stratified` validation.
- **Time split infeasible:** confirm timestamps parse, sort into strictly forward windows, configure `outer_repeats: 1`, and consider boundary ties, `gap`, and `test_size` in the plan output.
- **Missing or unknown predictor values:** choose a supported per-feature missing policy and keep categorical `categories` synchronized with training and prediction data. Unknown categories intentionally fail rather than silently map to a new level.
- **Convergence warning:** by default `stop_on_convergence_warning` is true. Inspect scaling, class/sample support, spline/search complexity, and interaction count before changing this strict behavior.
- **Unexpected resume rejection:** checkpoints are tied to configuration and data hashes; do not edit a run's copied config/data and expect its existing checkpoints to be reusable.
- **Large run cost:** candidate count is the product of the search-list lengths, multiplied by inner folds, outer folds/repeats, and models. Pairwise tensor terms grow as $q_rq_s$ per smooth pair. Start with `quick`, use only scientifically justified interactions, and scale workers/resources based on measured local runs.
- **Near-duplicate diagnostics too expensive:** `maximum_pairwise_rows` limits exact pairwise near-duplicate analysis. Reduce input size or raise that limit only after considering its pairwise cost.
- **PowerShell command discovery or exit codes:** activate the project environment, verify `gam-app --help`, and inspect `$LASTEXITCODE` after each CLI command in scripts.
- **Serialized model trust:** `model.joblib` is pickle-based. Only load models from a trusted source; a serialized Python object is not a safe untrusted interchange format.

Not included are regression or other continuous-target models, causal-effect estimation, classical coefficient significance tests or smooth-term p-values, confidence intervals, automated causal feature selection, stability-aware forward interaction selection, categorical-by-smooth interactions, functional-ANOVA centering, automated probability recalibration, a GUI/web service, distributed execution, database-backed run tracking, remote artifact storage, or automatic duplicate/label correction. Duplicate grouping constrains validation partitions only; it does not alter, delete, or clean source data.

## Further documentation

- [`docs/powershell-workflow.md`](docs/powershell-workflow.md): audited Windows workflow, exit-code handling, JSON parsing, run recovery, diagnostics review, inspection, prediction, comparisons, and validation variants.
- [`docs/scientific-interpretation.md`](docs/scientific-interpretation.md): extended mathematical formulation, validation guidance, metric interpretation, diagnostics, comparison conventions, scientific checklist, and reporting template.
- [`examples/quick-demo.yaml`](examples/quick-demo.yaml): compact example configuration (schema 1.0, supported and migratable).
- [`pyproject.toml`](pyproject.toml): package requirements, console script, optional development dependencies, and tool configuration.