# Audited PowerShell workflow

This guide provides a tested, copy-pasteable Windows PowerShell operational workflow for running, monitoring, inspecting, and predicting with `gam-app`.

## Prerequisites

- Windows PowerShell 5.1 or PowerShell 7+
- Python 3.11+ supported by the project
- Poetry installed
- Commands executed from the repository root directory

All commands use `poetry run gam-app ...` to guarantee execution inside the managed environment.

## 1. Environment setup and verification

```powershell
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

poetry run gam-app --help

if ($LASTEXITCODE -ne 0) {
    throw "gam-app installation verification failed."
}
```

## 2. Define project working paths safely

```powershell
$ProjectRoot = (Get-Location).Path
$DataDirectory = Join-Path $ProjectRoot "data"
$ConfigDirectory = Join-Path $ProjectRoot "configs"
$ProfileDirectory = Join-Path $ProjectRoot "profile"
$Workspace = Join-Path $ProjectRoot "workspace"
$PredictionDirectory = Join-Path $ProjectRoot "predictions"
$ComparisonDirectory = Join-Path $ProjectRoot "comparisons"

$null = New-Item `
  -ItemType Directory `
  -Force `
  -Path $DataDirectory,
        $ConfigDirectory,
        $ProfileDirectory,
        $Workspace,
        $PredictionDirectory,
        $ComparisonDirectory
```

## 3. Generate demonstration data

```powershell
$DataPath = Join-Path $DataDirectory "demo.csv"

poetry run gam-app demo `
  --output $DataPath `
  --rows 300 `
  --seed 42

if ($LASTEXITCODE -ne 0) {
    throw "Demonstration data creation failed."
}
```

## 4. Profile dataset with diagnostics

```powershell
$ProfilePath = Join-Path $ProfileDirectory "demo"

poetry run gam-app profile `
  --data $DataPath `
  --target Y `
  --output $ProfilePath

if ($LASTEXITCODE -ne 0) {
    throw "Dataset profiling failed."
}

Get-ChildItem -Recurse $ProfilePath
```

Review profile outputs:
- `profile.json`: row/column counts, missing values, target distributions, dataset SHA-256
- `columns.csv`: column summary table and recommended feature roles
- `diagnostics_manifest.json`: diagnostic execution inventory
- `correlation_pearson.csv` & `correlation_spearman.csv`: correlation matrices
- `high_correlation_pairs.csv`: ranked correlation findings
- `exact_duplicate_groups.csv` & `near_duplicate_groups.csv`: predictor duplicate analysis
- `conflicting_duplicate_targets.csv`: duplicate records with target label conflicts

## 5. Generate experiment configuration

```powershell
$ConfigPath = Join-Path $ConfigDirectory "demo.yaml"

poetry run gam-app configure `
  --data $DataPath `
  --target Y `
  --output $ConfigPath `
  --name demo-audited `
  --validation-strategy stratified `
  --outer-splits 3 `
  --outer-repeats 1 `
  --inner-splits 2 `
  --duplicate-group-policy report `
  --review-correlation 0.75 `
  --warn-correlation 0.90 `
  --near-duplicate-decimals 8 `
  --near-duplicate-threshold 0.98 `
  --maximum-pairwise-rows 10000 `
  --tag tutorial `
  --metadata workflow=audited-powershell `
  --preset quick `
  --non-interactive

if ($LASTEXITCODE -ne 0) {
    throw "Configuration generation failed."
}
```

## 6. Validate feasibility and plan

### Text mode planning

```powershell
poetry run gam-app plan `
  --config $ConfigPath

if ($LASTEXITCODE -eq 2) {
    throw "The configured validation design is not feasible."
}

if ($LASTEXITCODE -ne 0) {
    throw "Planning failed unexpectedly."
}
```

### JSON mode planning

```powershell
$PlanJson = poetry run gam-app plan `
  --config $ConfigPath `
  --json

if ($LASTEXITCODE -ne 0) {
    throw "Machine-readable planning failed."
}

$Plan = $PlanJson | ConvertFrom-Json

if (-not $Plan.feasible) {
    $Plan.checks |
      Where-Object { $_.level -eq "fail" } |
      Format-Table check, observed, required, details

    throw "Validation feasibility checks failed."
}
```

## 7. Run experiment and capture created path

Use `--run-path-file` to reliably capture the created run directory path:

```powershell
$RunPathFile = Join-Path $Workspace "latest-run.txt"

poetry run gam-app run `
  --config $ConfigPath `
  --workspace $Workspace `
  --run-path-file $RunPathFile

if ($LASTEXITCODE -ne 0) {
    if (Test-Path $RunPathFile) {
        $FailedRunPath = (
          Get-Content $RunPathFile -Raw
        ).Trim()

        Write-Warning (
          "Execution failed after creating run: " +
          $FailedRunPath
        )
    }

    throw "Experiment execution failed."
}

$RunPath = (
  Get-Content $RunPathFile -Raw
).Trim()

if (-not (Test-Path $RunPath)) {
    throw "The persisted run path does not exist: $RunPath"
}
```

### Alternative: Create run without immediate execution

```powershell
$CreationJson = poetry run gam-app run `
  --config $ConfigPath `
  --workspace $Workspace `
  --create-only `
  --json

if ($LASTEXITCODE -ne 0) {
    throw "Run creation failed."
}

$CreatedRun = $CreationJson | ConvertFrom-Json
$CreatedRunPath = $CreatedRun.run_path
```

## 8. Inspect run metadata and state

```powershell
$RunMetadata = Get-Content `
  (Join-Path $RunPath "run.json") `
  -Raw |
  ConvertFrom-Json

$RunStatus = Get-Content `
  (Join-Path $RunPath "status.json") `
  -Raw |
  ConvertFrom-Json

if ($RunMetadata.run_id -ne (Split-Path $RunPath -Leaf)) {
    throw "Run ID does not match its directory name."
}

if ($RunStatus.state -ne "completed") {
    throw "Run did not complete successfully."
}

$RunMetadata
$RunStatus
```

## 9. Open HTML report

```powershell
$ReportPath = Join-Path $RunPath "reports\report.html"

if (-not (Test-Path $ReportPath)) {
    throw "HTML report was not created."
}

Start-Process $ReportPath
```

## 10. Inspect and review diagnostic package

Review the diagnostic package for a run without modifying artifacts:

```powershell
$ReviewOutput = Join-Path $RunPath "reviews\diagnostic_review.json"

poetry run gam-app review-diagnostics `
  --run $RunPath `
  --output $ReviewOutput

if ($LASTEXITCODE -ne 0) {
    throw "Diagnostic review failed."
}

$DiagnosticsPath = Join-Path $RunPath "diagnostics"

$HighCorrFile = Join-Path $DiagnosticsPath "high_correlation_pairs.csv"
if (Test-Path $HighCorrFile) {
    Import-Csv $HighCorrFile |
      Select-Object -First 10 |
      Format-Table
}

$ConflictFile = Join-Path $DiagnosticsPath "conflicting_duplicate_targets.csv"
if (Test-Path $ConflictFile) {
    Import-Csv $ConflictFile |
      Format-Table
}
```

## 10b. Manage sensitivity studies

Link runs in a planned sensitivity analysis:

```powershell
$SensitivityOutput = Join-Path $Workspace "sensitivity\demo-study\sensitivity_manifest.json"

poetry run gam-app create-sensitivity `
  --workspace $Workspace `
  --id demo-study `
  --name "Demo Sensitivity Study" `
  --reference-run $RunPath `
  --variant-run $RunPath `
  --vary search.C `
  --invariant dataset `
  --invariant target `
  --output $SensitivityOutput

if ($LASTEXITCODE -ne 0) {
    throw "Sensitivity study creation failed."
}

poetry run gam-app show-sensitivity `
  --manifest $SensitivityOutput
```

## 11. Inspect model equations and verify link function

```powershell
poetry run gam-app inspect `
  --run $RunPath `
  --model gam_main

if ($LASTEXITCODE -ne 0) {
    throw "Model inspection failed."
}

Get-ChildItem (Join-Path $RunPath "results\gam_main\inspection")

poetry run gam-app verify-link `
  --run $RunPath `
  --model gam_main

if ($LASTEXITCODE -ne 0) {
    throw "Link-function verification failed."
}
```

## 12. Transform features and export score contributions

The sequence for understanding fitted model representations is:
1. `transform`: exports the fitted design matrix.
2. `contributions`: exports observation-level additive score contributions for each class.
3. `grouped-contributions`: aggregates observation-level contributions by predictor/interaction group.

Prepare sample input without target variable:

```powershell
$PredictionInput = Join-Path $PredictionDirectory "demo-input.csv"

Import-Csv $DataPath |
  Select-Object -First 10 |
  Select-Object -Property * -ExcludeProperty Y |
  Export-Csv -Path $PredictionInput -NoTypeInformation
```

### Export transformed feature matrix

```powershell
$ModelPath = Join-Path $RunPath "models\gam_main\model.joblib"
$TransformedOutput = Join-Path $PredictionDirectory "demo-transformed.csv"

poetry run gam-app transform `
  --model $ModelPath `
  --input $PredictionInput `
  --output $TransformedOutput

if ($LASTEXITCODE -ne 0) {
    throw "Predictor transformation failed."
}

Import-Csv $TransformedOutput |
  Select-Object -First 5 |
  Format-Table
```

### Export observation-level contributions

```powershell
$ContributionsOutput = Join-Path $PredictionDirectory "demo-contributions.csv"

poetry run gam-app contributions `
  --model $ModelPath `
  --input $PredictionInput `
  --output $ContributionsOutput

if ($LASTEXITCODE -ne 0) {
    throw "Contribution export failed."
}
```

### Export grouped contribution summary

`grouped-contributions` consumes the CSV generated by `contributions`:

```powershell
$GroupedContributionsOutput = Join-Path $PredictionDirectory "demo-grouped-contributions.csv"

poetry run gam-app grouped-contributions `
  --input $ContributionsOutput `
  --output $GroupedContributionsOutput

if ($LASTEXITCODE -ne 0) {
    throw "Grouped contribution export failed."
}

Import-Csv $GroupedContributionsOutput |
  Format-Table
```

## 13. Batch prediction

```powershell
$PredictionOutput = Join-Path $PredictionDirectory "demo-predictions.csv"

poetry run gam-app predict `
  --model $ModelPath `
  --input $PredictionInput `
  --output $PredictionOutput

if ($LASTEXITCODE -ne 0) {
    throw "Batch prediction failed."
}

Import-Csv $PredictionOutput |
  Format-Table
```

## 14. Compare paired models or runs

```powershell
$ComparisonOutput = Join-Path $ComparisonDirectory "main-vs-self"

poetry run gam-app compare `
  --left $RunPath `
  --left-model gam_main `
  --right $RunPath `
  --right-model gam_main `
  --output $ComparisonOutput

if ($LASTEXITCODE -ne 0) {
    throw "Model comparison failed."
}

Import-Csv (Join-Path $ComparisonOutput "summary.csv") |
  Format-Table
```

## 15. Discover and filter workspace runs

```powershell
poetry run gam-app list-runs `
  --workspace $Workspace `
  --state completed `
  --tag tutorial `
  --metadata workflow=audited-powershell
```

JSON output parsing:

```powershell
$CatalogJson = poetry run gam-app list-runs `
  --workspace $Workspace `
  --state completed `
  --tag tutorial `
  --json

if ($LASTEXITCODE -ne 0) {
    throw "Run listing failed."
}

$Catalog = $CatalogJson | ConvertFrom-Json

$Catalog.runs |
  Select-Object run_id, state, experiment_name, validation_strategy, path |
  Format-Table
```

## 16. Validation strategy variants

### Group-aware validation variant

When related observations share a specimen, site, or batch, use `stratified_group`:

```powershell
$GroupedConfigPath = Join-Path $ConfigDirectory "grouped-demo.yaml"

poetry run gam-app configure `
  --data $DataPath `
  --target Y `
  --group group_col `
  --output $GroupedConfigPath `
  --validation-strategy stratified_group `
  --duplicate-group-policy group `
  --outer-splits 3 `
  --outer-repeats 1 `
  --inner-splits 2 `
  --preset quick `
  --non-interactive
```

Scientific groups specified under `--group` and duplicate-derived component constraints are merged into effective validation groups to prevent fold leakage.

### Time-aware validation variant

When evaluating forward-looking prediction, use `time`:

```powershell
$TimeConfigPath = Join-Path $ConfigDirectory "time-demo.yaml"

poetry run gam-app configure `
  --data $DataPath `
  --target Y `
  --time time_col `
  --output $TimeConfigPath `
  --validation-strategy time `
  --outer-splits 3 `
  --outer-repeats 1 `
  --inner-splits 2 `
  --gap 2 `
  --test-size 20 `
  --preset quick `
  --non-interactive
```

Always run `poetry run gam-app plan --config $TimeConfigPath` before executing time-aware runs to verify temporal split feasibility.

## 17. Optional cleanup

```powershell
# Optional cleanup of profile and prediction output files
Remove-Item `
  -Recurse `
  -Force `
  $ProfilePath, $PredictionDirectory
```

Note: Do not delete `$Workspace` if you need to retain audit evidence or run artifacts.
