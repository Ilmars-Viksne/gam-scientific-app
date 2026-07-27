# GAM Scientific App

A standalone, file-based, configuration-driven CLI for reproducible multiclass
Generalized Additive Model experiments on tabular datasets.

## Capabilities

- CSV, TSV, and Parquet input
- automatic profiling and role recommendations
- interactive or non-interactive configuration generation
- main-effect and tensor-product pairwise GAM-style models
- repeated nested stratified cross-validation
- deterministic split manifests shared by all models
- fold-level checkpoints, pause, cancel, and resume
- out-of-fold probabilities and class metrics
- final fitted models, exact transformed-space coefficients, and equations
- main effects, interaction surfaces, confusion matrices, and HTML reports
- comparison of runs with paired fold differences
- batch prediction with input-schema validation
- no database or external service

---

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Create demonstration data:

```powershell
gam-app demo --output examples/demo.csv
```

Profile it:

```powershell
gam-app profile --data examples/demo.csv --target Y --output profile
```

Create a configuration without editing Python:

```powershell
gam-app configure --data examples/demo.csv --target Y --output configs/demo.yaml
```

For unattended configuration, accept inferred roles and the standard preset:

```powershell
gam-app configure --data examples/demo.csv --target Y --output configs/demo.yaml --non-interactive
```

Validate and estimate the work:

```powershell
gam-app plan --config configs/demo.yaml
```

Run:

```powershell
gam-app run --config configs/demo.yaml --workspace workspace
```

Monitor or resume:

```powershell
gam-app status --run workspace/runs/<run-id>
gam-app status --run workspace/runs/<run-id> --follow
gam-app pause --run workspace/runs/<run-id>
gam-app resume --run workspace/runs/<run-id>
gam-app cancel --run workspace/runs/<run-id>
```

Inspect and predict:

```powershell
gam-app inspect --run workspace/runs/<run-id> --model gam_main
gam-app predict --model workspace/runs/<run-id>/models/gam_main/model.joblib --input new.csv --output predictions.csv
```

Compare two completed runs or model results:

```powershell
gam-app compare --left workspace/runs/<run-a> --left-model gam_main --right workspace/runs/<run-b> --right-model gam_pairwise --output comparison
```

## Configuration

The generated YAML is a complete experiment contract. Users can change it through
`gam-app configure`; no source-code edits are required. A resolved immutable copy
is stored with each run.

Feature roles:

- `smooth`: univariate B-spline effect
- `linear`: standardized linear effect
- `categorical`: one-hot categorical effect
- `exclude`: omit from modelling

Interaction modes:

- `none`
- `all_eligible`
- `explicit`

The application currently targets multiclass classification. Binary targets also
work through scikit-learn logistic regression, but multiclass is the validated
primary use case.

## Run directory

Each run is self-contained:

```text
run-id/
├── run.json
├── config.yaml
├── data_manifest.json
├── split_manifest.csv
├── status.json
├── events.jsonl
├── control/
├── checkpoints/
├── results/
├── models/
├── plots/
├── reports/
└── logs/
```

`status.json` is atomically replaced. `events.jsonl` is append-only. A fold
checkpoint is reusable only when its `COMPLETE` marker exists and its dataset and
configuration hashes match the run.

## Development

```powershell
ruff format --check .
ruff check .
mypy src
pytest --cov=gam_app --cov-report=term-missing
```

The unit tests use small synthetic datasets. The demonstration dataset is used by
end-to-end smoke and regression tests, not by every unit test.

## Scientific interpretation

This implementation creates penalized multinomial logistic additive B-spline
models. The inverse link is softmax. Coefficients are predictive regularized
components, not causal estimates or classical unregularized significance tests.
Tensor-product interaction blocks are included only with their constituent main
effects. Raw interaction blocks should be interpreted cautiously unless further
functional-ANOVA centering is implemented.

## Security

Only load `.joblib` models created by a trusted installation. Joblib is
pickle-based. The application records package versions and model metadata, but a
serialized Python object is not a safe interchange format for untrusted files.

---

## Included project structure

```text
gam-scientific-app/
├── pyproject.toml
├── README.md
├── .gitignore
├── examples/
│   └── quick-demo.yaml
├── src/
│   └── gam_app/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── exceptions.py
│       ├── inspection.py
│       ├── io_utils.py
│       ├── models.py
│       ├── reporting.py
│       ├── run_store.py
│       ├── transformers.py
│       └── workflow.py
└── tests/
    ├── conftest.py
    ├── test_model.py
    ├── test_splits.py
    └── test_transformers.py
```

---

# 1. Installation

Extract the ZIP file and open the resulting directory in VS Code.

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
gam-app --help
```

---

# 2. Complete user workflow

## Step 1 — Register or select a dataset

The application supports:

- CSV;
- TSV;
- Parquet.

A dataset should have:

- one categorical classification target;
- at least two target classes;
- one or more predictor columns;
- no missing target values.

The predictor names are arbitrary. Nothing is hard-coded to `X1`, `X2`, or `Y`.

---

## Step 2 — Profile the dataset

```powershell
gam-app profile `
  --data data/my_dataset.csv `
  --target outcome `
  --output profile
```

This generates:

```text
profile/
├── profile.json
└── columns.csv
```

The profile contains:

- row and column counts;
- target class distribution;
- data types;
- missing-value counts;
- unique-value counts;
- numerical ranges;
- common categorical values;
- duplicate-row count;
- recommended feature roles;
- explanations for recommendations;
- dataset SHA-256 hash.

---

## Step 3 — Create the model configuration

### Interactive configuration

```powershell
gam-app configure `
  --data data/my_dataset.csv `
  --target outcome `
  --output configs/my_experiment.yaml
```

For each predictor, the application shows:

- inferred type;
- number of unique values;
- recommended role;
- reason for the recommendation.

The user chooses among:

```text
smooth
linear
categorical
exclude
```

No Python editing is required.

### Unattended configuration

To accept automatic recommendations:

```powershell
gam-app configure `
  --data data/my_dataset.csv `
  --target outcome `
  --output configs/my_experiment.yaml `
  --preset standard `
  --non-interactive
```

Available search presets:

```text
quick
standard
thorough
```

---

# 3. Configuration-driven modelling

The generated YAML file is the complete experiment contract.

Example:

```yaml
schema_version: "1.0"

experiment:
  name: my_experiment
  primary_metric: log_loss

data:
  path: C:/data/my_dataset.csv
  target: outcome
  row_id: null

features:
  temperature:
    role: smooth
    missing: error

  pressure:
    role: smooth
    missing: median

  operating_mode:
    role: categorical
    missing: most_frequent
    categories:
      - low
      - standard
      - high

  calibration:
    role: linear
    missing: error

models:
  - id: gam_main
    interactions: none

  - id: gam_pairwise
    interactions: all_eligible

validation:
  outer_splits: 5
  outer_repeats: 3
  inner_splits: 5
  random_state: 42

search:
  n_knots:
    - 3
    - 4
    - 5

  degree:
    - 2
    - 3

  C:
    - 0.01
    - 0.1
    - 1.0
    - 10.0

  interaction_scale:
    - 0.5
    - 1.0

execution:
  workers: 1
  checkpoint_unit: outer_fold
  stop_on_convergence_warning: true
```

Users can generate this configuration through the CLI wizard. Advanced users may also edit or version the YAML without changing application code.

---

# 4. Supported predictor roles

## Smooth

```yaml
role: smooth
```

The predictor receives a univariate B-spline effect:

$$
f_{jk}(x_j) = \sum_m \theta_{jkm} B_{jm}(x_j).
$$

## Linear

```yaml
role: linear
```

The predictor is standardized inside every training fold and enters as:

$$
\beta_{jk}
\frac{x_j-\mu_j}{\sigma_j}.
$$

## Categorical

```yaml
role: categorical
```

The predictor is one-hot encoded with a reference level.

## Excluded

```yaml
role: exclude
```

The predictor remains in the source dataset but is not passed to the model.

---

# 5. Missing-value policies

Each variable can use one of these policies:

```yaml
missing: error
```

Reject missing values.

```yaml
missing: median
```

Use training-fold median imputation.

```yaml
missing: most_frequent
```

Use training-fold most-frequent-value imputation.

All fitted preprocessing occurs inside the model pipeline and is learned from training data only.

---

# 6. Model types

## Main-effects GAM

```yaml
- id: gam_main
  interactions: none
```

The class score has the general form:

$$
\eta_k(\mathbf{x})
=
\beta_{0k}
+\sum_j f_{jk}(x_j)
+\sum_l \beta_{lk}\widetilde{x}_l
+\sum_c \gamma_{ck}(x_c).
$$

## All-eligible pairwise GAM

```yaml
- id: gam_pairwise
  interactions: all_eligible
```

Every pair among smooth predictors is included:

$$
\eta_k(\mathbf{x})
=
\beta_{0k}
+\sum_j f_{jk}(x_j)
+\sum_{(r,s)} f_{rsk}(x_r,x_s).
$$

## Explicit pairs

The application model layer also supports explicitly configured pairs:

```yaml
- id: gam_selected_pairs
  interactions: explicit
  pairs:
    - [temperature, pressure]
    - [load, humidity]
```

The system validates that both members are distinct configured predictors.

---

# 7. Link function

Both model types use:

- multinomial response;
- multinomial-logit link;
- softmax inverse link;
- L2-penalized multinomial log loss.

Class probabilities are:

$$
P(Y=k\mid\mathbf{x})
=
\frac{\exp(\eta_k)}
{\sum_{\ell}\exp(\eta_\ell)}.
$$

---

# 8. Preview the execution plan

Before starting an expensive calculation:

```powershell
gam-app plan --config configs/my_experiment.yaml
```

Example output:

```text
       model  candidates  estimated_fits
    gam_main          60            7500
gam_pairwise         120           15000
```

This reports:

- models;
- hyperparameter candidates;
- approximate number of model fits.

The user can then choose a smaller preset or adjust the generated configuration through the configuration workflow.

---

# 9. Run the experiment

```powershell
gam-app run `
  --config configs/my_experiment.yaml `
  --workspace workspace
```

The application creates a unique run directory:

```text
workspace/runs/run-<timestamp>-<identifier>/
```

Each run is immutable with respect to its dataset and configuration hashes.

---

# 10. Run directory structure

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
├── control/
├── checkpoints/
│   ├── gam_main/
│   └── gam_pairwise/
├── results/
│   ├── gam_main/
│   └── gam_pairwise/
├── models/
│   ├── gam_main/
│   └── gam_pairwise/
├── plots/
├── reports/
└── logs/
```

The run stores:

- exact resolved configuration;
- dataset hash;
- configuration hash;
- Python/platform/package versions;
- persisted CV split assignments;
- progress state;
- event timeline;
- fold checkpoints;
- out-of-fold predictions;
- final models;
- reports and plots.

No database is required.

---

# 11. Monitoring

Show the current state:

```powershell
gam-app status `
  --run workspace/runs/<run-id>
```

Follow progress continuously:

```powershell
gam-app status `
  --run workspace/runs/<run-id> `
  --follow
```

`status.json` contains fields such as:

```json
{
  "state": "running",
  "phase": "nested_cross_validation",
  "model_id": "gam_pairwise",
  "repeat": 2,
  "fold": 4,
  "completed_outer_folds": 8,
  "total_outer_folds": 15,
  "updated_at_utc": "..."
}
```

`events.jsonl` provides an append-only event history.

---

# 12. Pause and resume

Request a safe pause:

```powershell
gam-app pause `
  --run workspace/runs/<run-id>
```

The worker:

1. finishes the current safe fold unit;
2. writes the checkpoint;
3. transitions the run to `paused`;
4. exits cleanly.

Resume:

```powershell
gam-app resume `
  --run workspace/runs/<run-id>
```

Completed checkpoints are validated and skipped.

---

# 13. Cancel

```powershell
gam-app cancel `
  --run workspace/runs/<run-id>
```

The application stops after the current safe calculation unit and marks the run as cancelled.

A cancelled run remains auditable, and completed checkpoints are retained.

---

# 14. Checkpoint design

Each completed outer fold contains:

```text
checkpoints/
└── gam_main/
    └── repeat-01_fold-01/
        ├── checkpoint.json
        ├── metrics.json
        ├── trials.parquet
        ├── predictions.parquet
        ├── model.joblib
        └── COMPLETE
```

A checkpoint is reusable only if:

- `COMPLETE` exists;
- its dataset hash matches;
- its configuration hash matches.

Incomplete temporary checkpoints are not treated as completed work.

JSON state updates use atomic file replacement to avoid partially written status files.

---

# 15. Reproducible outer splits

The application generates:

```text
split_manifest.csv
```

Example columns:

```text
repeat
fold
row_id
row_index
partition
```

Both main-effects and pairwise models consume the same persisted split manifest.

This protects comparisons from accidental differences caused by:

- row reordering;
- changed random-state handling;
- filtering in only one workflow;
- independently generated folds.

---

# 16. Outputs

For every model:

```text
results/<model-id>/
├── fold_metrics.csv
├── predictions.parquet
├── summary.csv
└── inspection/
```

Final model artifacts:

```text
models/<model-id>/
├── model.joblib
├── best_parameters.json
├── search_trials.parquet
└── components.csv
```

The report is generated at:

```text
reports/report.html
```

It includes:

- experiment identity;
- model summaries;
- nested-CV metrics;
- confusion matrices;
- links to detailed result tables.

---

# 17. Inspect fitted equations

```powershell
gam-app inspect `
  --run workspace/runs/<run-id> `
  --model gam_main
```

For the pairwise model:

```powershell
gam-app inspect `
  --run workspace/runs/<run-id> `
  --model gam_pairwise
```

Choose a reference class:

```powershell
gam-app inspect `
  --run workspace/runs/<run-id> `
  --model gam_main `
  --reference-class B
```

This generates:

```text
results/<model-id>/inspection/
├── components.csv
├── equations.txt
└── reference_equations.csv
```

`equations.txt` contains the exact transformed-space score equations.

---

# 18. Verify the link function

```powershell
gam-app verify-link `
  --run workspace/runs/<run-id> `
  --model gam_main
```

The application compares:

```python
model.predict_proba(X)
```

with:

```python
softmax(model.decision_function(X))
```

Outputs:

```text
results/<model-id>/link_verification/
├── scores.csv
├── probabilities.csv
└── verification.txt
```

The maximum error should be near floating-point precision.

---

# 19. Compare experiments or models

```powershell
gam-app compare `
  --left workspace/runs/<left-run> `
  --left-model gam_main `
  --right workspace/runs/<right-run> `
  --right-model gam_pairwise `
  --output comparisons/main-vs-pairwise
```

The application merges fold metrics by:

```text
repeat
fold
```

and calculates:

```text
log_loss_difference
accuracy_difference
balanced_accuracy_difference
macro_f1_difference
```

The direction is:

$$
\Delta=\text{right}-\text{left}.
$$

Outputs:

```text
comparisons/main-vs-pairwise/
├── comparison.csv
└── summary.csv
```

---

# 20. Batch prediction

```powershell
gam-app predict `
  --model workspace/runs/<run-id>/models/gam_main/model.joblib `
  --input new_observations.csv `
  --output predictions.csv
```

Output columns include:

```text
probability_<class-1>
probability_<class-2>
...
predicted_class
```

The fitted transformer selects the configured predictors by name, so input column order does not need to match the training file.

Only trusted application-generated `.joblib` files should be loaded.

---

# 21. Demonstration workflow

Generate synthetic demonstration data:

```powershell
gam-app demo `
  --output examples/demo.csv `
  --rows 300 `
  --seed 42
```

Create a quick configuration:

```powershell
gam-app configure `
  --data examples/demo.csv `
  --target Y `
  --output examples/generated-demo.yaml `
  --preset quick `
  --non-interactive
```

Preview:

```powershell
gam-app plan `
  --config examples/generated-demo.yaml
```

Run:

```powershell
gam-app run `
  --config examples/generated-demo.yaml `
  --workspace workspace
```

---

# 22. Automated tests

Run:

```powershell
pytest
```

Included tests cover:

- transformation dimensions;
- stable feature names;
- finite transformed values;
- pairwise tensor-product features;
- invalid interaction rejection;
- serialization round trip;
- probabilities summing to one;
- softmax reconstruction;
- reproducible outer split behaviour;
- every observation appearing once per repeat as test data.

The included test suite completed successfully:

```text
4 passed
```

---

# 23. Quality checks

```powershell
ruff format --check .
ruff check .
mypy src
pytest --cov=gam_app --cov-report=term-missing
```

The source package also passed:

```powershell
python -m compileall src tests
```

---

# 24. Current release boundaries

This reference release is intentionally scoped to:

- standalone local execution;
- CLI operation;
- file-based persistence;
- CSV, TSV, and Parquet;
- categorical classification targets;
- numeric and categorical predictors;
- smooth, linear, categorical, and excluded feature roles;
- repeated nested stratified CV;
- main-effects GAMs;
- explicit or all-eligible smooth–smooth interactions;
- batch prediction;
- reproducible HTML and machine-readable outputs.

Not yet included:

- desktop graphical wizard;
- grouped or temporal CV;
- stability-aware forward interaction selection;
- categorical–smooth interactions;
- functional-ANOVA interaction centering;
- calibration plots and local explanation dashboards;
- distributed execution;
- executable desktop installer.

The architecture keeps these additions separate from the modelling core, so they can be implemented without returning to dataset-specific scripts.

---
