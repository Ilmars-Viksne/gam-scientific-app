#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string] $RunManifest,

    [string] $ModelId = "gam_main",
    [string] $ScenarioInput,

    [ValidateRange(1, 1000000)]
    [int] $ScenarioRowCount = 20,

    [string] $ReferenceClass,
    [string] $ComparisonLeftModel = "gam_main",
    [string] $ComparisonRightModel = "gam_pairwise",
    [switch] $CompareModels,
    [switch] $OpenReport
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-ExitCode {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)] [string] $Operation,
        [Parameter(Mandatory)] [int] $ExitCode
    )

    if ($ExitCode -ne 0) {
        throw "$Operation failed with exit code $ExitCode."
    }
}

function Get-Sha256 {
    [CmdletBinding()]
    param([Parameter(Mandatory)] [string] $Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Cannot calculate SHA-256 because the file does not exist: $Path"
    }

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Read-Json {
    [CmdletBinding()]
    param([Parameter(Mandatory)] [string] $Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "JSON file does not exist: $Path"
    }

    try {
        return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "Invalid JSON file '$Path': $($_.Exception.Message)"
    }
}

function Write-JsonAtomic {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)] [object] $Value,
        [Parameter(Mandatory)] [string] $Path
    )

    $Parent = Split-Path -Path $Path -Parent
    if (-not [string]::IsNullOrWhiteSpace($Parent)) {
        New-Item -ItemType Directory -Path $Parent -Force | Out-Null
    }

    $TemporaryPath = "$Path.tmp-$([Guid]::NewGuid().ToString('N'))"
    try {
        $Value |
            ConvertTo-Json -Depth 30 |
            Set-Content -LiteralPath $TemporaryPath -Encoding UTF8

        Move-Item -LiteralPath $TemporaryPath -Destination $Path -Force
    }
    finally {
        Remove-Item -LiteralPath $TemporaryPath -Force -ErrorAction SilentlyContinue
    }
}

function Test-Blank {
    [CmdletBinding()]
    param([AllowNull()] [object] $Value)

    return [string]::IsNullOrWhiteSpace([string] $Value)
}

function Resolve-ManifestPathValue {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)] [string] $Value,
        [Parameter(Mandatory)] [string] $BaseDirectory
    )

    if ([IO.Path]::IsPathRooted($Value)) {
        return $Value
    }

    return Join-Path $BaseDirectory $Value
}

function Get-ModelIds {
    [CmdletBinding()]
    param([AllowNull()] [object] $Models)

    $Ids = [System.Collections.Generic.List[string]]::new()

    foreach ($Model in @($Models)) {
        if ($null -eq $Model) {
            continue
        }

        if ($Model -is [string]) {
            if (-not [string]::IsNullOrWhiteSpace($Model)) {
                $Ids.Add($Model)
            }
            continue
        }

        $IdProperty = $Model.PSObject.Properties["id"]
        if ($null -ne $IdProperty -and -not (Test-Blank $IdProperty.Value)) {
            $Ids.Add([string] $IdProperty.Value)
            continue
        }

        $ModelIdProperty = $Model.PSObject.Properties["model_id"]
        if ($null -ne $ModelIdProperty -and -not (Test-Blank $ModelIdProperty.Value)) {
            $Ids.Add([string] $ModelIdProperty.Value)
            continue
        }

        throw "The run manifest contains a model entry without id or model_id."
    }

    return @($Ids | Select-Object -Unique)
}

function Assert-OutputFile {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)] [string] $Operation,
        [Parameter(Mandatory)] [string] $Path
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Operation did not create the expected output file: $Path"
    }
}

# -----------------------------------------------------------------------------
# Load and validate the Stage 2 run handoff.
# -----------------------------------------------------------------------------
$RunManifest = (Resolve-Path -LiteralPath $RunManifest -ErrorAction Stop).Path
$RunManifestDirectory = Split-Path -Path $RunManifest -Parent
$RunInfo = Read-Json -Path $RunManifest

if ($RunInfo.schema_name -ne "gam_run_handoff") {
    throw "Unsupported run handoff schema '$($RunInfo.schema_name)'."
}

if ($RunInfo.status -ne "completed") {
    throw "Run handoff is incomplete. Recorded status: '$($RunInfo.status)'."
}

if (Test-Blank $RunInfo.run_path) {
    throw "Run handoff does not contain run_path."
}

$RunPathCandidate = Resolve-ManifestPathValue `
    -Value ([string] $RunInfo.run_path) `
    -BaseDirectory $RunManifestDirectory
$RunPath = (Resolve-Path -LiteralPath $RunPathCandidate -ErrorAction Stop).Path

if (-not (Test-Path -LiteralPath $RunPath -PathType Container)) {
    throw "Run directory is missing: $RunPath"
}

$StatusPath = Join-Path $RunPath "status.json"
$Status = Read-Json -Path $StatusPath
if ($Status.state -ne "completed") {
    throw "The actual run state is '$($Status.state)', not 'completed'."
}

$Stage1ManifestValue = if (-not (Test-Blank $RunInfo.stage1_manifest)) {
    [string] $RunInfo.stage1_manifest
}
elseif (-not (Test-Blank $RunInfo.preprocessing_manifest)) {
    [string] $RunInfo.preprocessing_manifest
}
else {
    throw "Run handoff contains neither stage1_manifest nor preprocessing_manifest."
}

$Stage1ManifestCandidate = Resolve-ManifestPathValue `
    -Value $Stage1ManifestValue `
    -BaseDirectory $RunManifestDirectory
$Stage1Manifest = (Resolve-Path -LiteralPath $Stage1ManifestCandidate -ErrorAction Stop).Path
$Stage1 = Read-Json -Path $Stage1Manifest

if (-not (Test-Blank $RunInfo.stage1_manifest_sha256)) {
    $ActualStage1Hash = Get-Sha256 -Path $Stage1Manifest
    if ($ActualStage1Hash -ne ([string] $RunInfo.stage1_manifest_sha256).ToLowerInvariant()) {
        throw "Stage 1 manifest hash differs from the Stage 2 handoff."
    }
}

if (Test-Blank $Stage1.project_root) {
    throw "Stage 1 handoff does not contain project_root."
}

$ProjectRoot = (Resolve-Path -LiteralPath ([string] $Stage1.project_root) -ErrorAction Stop).Path
Set-Location -LiteralPath $ProjectRoot

$PyprojectPath = Join-Path $ProjectRoot "pyproject.toml"
$PackagePath = Join-Path $ProjectRoot "src\gam_app\__init__.py"
if (-not (Test-Path -LiteralPath $PyprojectPath -PathType Leaf)) {
    throw "pyproject.toml was not found: $PyprojectPath"
}
if (-not (Test-Path -LiteralPath $PackagePath -PathType Leaf)) {
    throw "Python package was not found: $PackagePath"
}

& poetry run gam-app --help | Out-Null
Assert-ExitCode -Operation "gam-app --help" -ExitCode $LASTEXITCODE

# Prefer model IDs from the Stage 2 handoff, but fall back to run.json.
$AvailableModelIds = Get-ModelIds -Models $RunInfo.models
if ($AvailableModelIds.Count -eq 0) {
    $RunMetadata = Read-Json -Path (Join-Path $RunPath "run.json")
    $AvailableModelIds = Get-ModelIds -Models $RunMetadata.models
}

if ($ModelId -notin $AvailableModelIds) {
    throw "Model '$ModelId' is absent. Available models: $($AvailableModelIds -join ', ')."
}

$ModelPath = Join-Path (Join-Path (Join-Path $RunPath "models") $ModelId) "model.joblib"
if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
    throw "Model file is missing: $ModelPath"
}

# -----------------------------------------------------------------------------
# Refresh model inspection and verify the fitted link.
# -----------------------------------------------------------------------------
& poetry run gam-app inspect --run $RunPath --model $ModelId
Assert-ExitCode -Operation "inspect" -ExitCode $LASTEXITCODE

& poetry run gam-app verify-link --run $RunPath --model $ModelId
Assert-ExitCode -Operation "verify-link" -ExitCode $LASTEXITCODE

# -----------------------------------------------------------------------------
# Create a unique scenario package and obtain scenario input.
# -----------------------------------------------------------------------------
$SessionId = "{0}-{1}" -f `
    ([DateTimeOffset]::UtcNow.ToString("yyyyMMdd-HHmmssfff")), `
    ([Guid]::NewGuid().ToString("N").Substring(0, 8))

$OutputDirectory = Join-Path `
    (Join-Path $ProjectRoot "predictions") `
    "scenario-$SessionId"
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

$ScenarioSource = if (Test-Blank $ScenarioInput) {
    "generated_from_prepared_data"
}
else {
    "user_supplied"
}

if (Test-Blank $ScenarioInput) {
    if (Test-Blank $Stage1.prepared_data_path) {
        throw "Stage 1 handoff does not contain prepared_data_path."
    }

    $PreparedDataCandidate = Resolve-ManifestPathValue `
        -Value ([string] $Stage1.prepared_data_path) `
        -BaseDirectory (Split-Path -Path $Stage1Manifest -Parent)
    $PreparedDataPath = (Resolve-Path -LiteralPath $PreparedDataCandidate -ErrorAction Stop).Path

    if (-not (Test-Blank $Stage1.prepared_data_sha256)) {
        $PreparedDataHash = Get-Sha256 -Path $PreparedDataPath
        if ($PreparedDataHash -ne ([string] $Stage1.prepared_data_sha256).ToLowerInvariant()) {
            throw "Prepared data hash differs from the Stage 1 handoff."
        }
    }

    $Target = [string] $Stage1.target
    if (Test-Blank $Target) {
        throw "Stage 1 handoff does not declare the target column."
    }

    $ScenarioInput = Join-Path $OutputDirectory "scenarios.csv"
    $ScenarioRows = @(
        Import-Csv -LiteralPath $PreparedDataPath |
            Select-Object -First $ScenarioRowCount
    )

    if ($ScenarioRows.Count -eq 0) {
        throw "Prepared data contains no rows for scenario generation."
    }

    $ScenarioRows |
        Select-Object -Property * -ExcludeProperty $Target |
        Export-Csv -LiteralPath $ScenarioInput -NoTypeInformation -Encoding UTF8
}
else {
    $ScenarioInput = (Resolve-Path -LiteralPath $ScenarioInput -ErrorAction Stop).Path
}

if (-not (Test-Path -LiteralPath $ScenarioInput -PathType Leaf)) {
    throw "Scenario input is missing: $ScenarioInput"
}

$ScenarioRowsForValidation = @(Import-Csv -LiteralPath $ScenarioInput)
if ($ScenarioRowsForValidation.Count -eq 0) {
    throw "Scenario input contains no observations: $ScenarioInput"
}

$ScenarioColumns = @($ScenarioRowsForValidation[0].PSObject.Properties.Name)
if ($ScenarioColumns.Count -ne (@($ScenarioColumns | Select-Object -Unique)).Count) {
    throw "Scenario input contains duplicate column names."
}

# -----------------------------------------------------------------------------
# Produce predictions, transformed components, and contribution artifacts.
# -----------------------------------------------------------------------------
$Predictions = Join-Path $OutputDirectory "predictions.csv"
$Transformed = Join-Path $OutputDirectory "transformed.csv"
$Contributions = Join-Path $OutputDirectory "contributions.csv"
$Grouped = Join-Path $OutputDirectory "grouped-contributions.csv"

& poetry run gam-app predict `
    --model $ModelPath `
    --input $ScenarioInput `
    --output $Predictions
Assert-ExitCode -Operation "predict" -ExitCode $LASTEXITCODE
Assert-OutputFile -Operation "predict" -Path $Predictions

& poetry run gam-app transform `
    --model $ModelPath `
    --input $ScenarioInput `
    --output $Transformed
Assert-ExitCode -Operation "transform" -ExitCode $LASTEXITCODE
Assert-OutputFile -Operation "transform" -Path $Transformed

& poetry run gam-app contributions `
    --model $ModelPath `
    --input $ScenarioInput `
    --output $Contributions `
    --top 10
Assert-ExitCode -Operation "contributions" -ExitCode $LASTEXITCODE
Assert-OutputFile -Operation "contributions" -Path $Contributions

$GroupedArguments = @(
    "run", "gam-app", "grouped-contributions",
    "--input", $Contributions,
    "--output", $Grouped,
    "--top", 10
)

if (-not (Test-Blank $ReferenceClass)) {
    $GroupedArguments += @("--reference-class", $ReferenceClass)
}

& poetry @GroupedArguments
Assert-ExitCode -Operation "grouped-contributions" -ExitCode $LASTEXITCODE
Assert-OutputFile -Operation "grouped-contributions" -Path $Grouped

# -----------------------------------------------------------------------------
# Optionally compare both generated models from the same completed run.
# -----------------------------------------------------------------------------
$ComparisonPath = $null
$ComparisonManifestPath = $null

if ($CompareModels) {
    if (($ComparisonLeftModel -notin $AvailableModelIds) -or ($ComparisonRightModel -notin $AvailableModelIds)) {
        throw "Comparison models are absent. Available models: $($AvailableModelIds -join ', ')."
    }

    $ComparisonPath = Join-Path `
        (Join-Path $ProjectRoot "comparisons") `
        "$ComparisonLeftModel-vs-$ComparisonRightModel-$SessionId"

    & poetry run gam-app compare `
        --left $RunPath `
        --left-model $ComparisonLeftModel `
        --right $RunPath `
        --right-model $ComparisonRightModel `
        --check-only
    Assert-ExitCode -Operation "compare --check-only" -ExitCode $LASTEXITCODE

    & poetry run gam-app compare `
        --left $RunPath `
        --left-model $ComparisonLeftModel `
        --right $RunPath `
        --right-model $ComparisonRightModel `
        --output $ComparisonPath
    Assert-ExitCode -Operation "compare" -ExitCode $LASTEXITCODE

    if (-not (Test-Path -LiteralPath $ComparisonPath -PathType Container)) {
        throw "compare did not create the expected output directory: $ComparisonPath"
    }

    $ComparisonManifestCandidate = Join-Path $ComparisonPath "comparison_manifest.json"
    if (Test-Path -LiteralPath $ComparisonManifestCandidate -PathType Leaf) {
        $ComparisonManifestPath = $ComparisonManifestCandidate
    }
}

# -----------------------------------------------------------------------------
# Write an auditable scenario-analysis manifest.
# -----------------------------------------------------------------------------
$ScenarioManifestPath = Join-Path $OutputDirectory "scenario_manifest.json"
$ScenarioManifest = [ordered]@{
    schema_name = "gam_scenario_analysis"
    schema_version = "1.1"
    created_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
    run_manifest = $RunManifest
    run_manifest_sha256 = Get-Sha256 -Path $RunManifest
    run_id = $RunInfo.run_id
    run_path = $RunPath
    model_id = $ModelId
    model_path = $ModelPath
    model_sha256 = Get-Sha256 -Path $ModelPath
    scenario_source = $ScenarioSource
    scenario_input = $ScenarioInput
    scenario_input_sha256 = Get-Sha256 -Path $ScenarioInput
    scenario_row_count = $ScenarioRowsForValidation.Count
    reference_class = if (Test-Blank $ReferenceClass) { $null } else { $ReferenceClass }
    artifacts = [ordered]@{
        predictions = [ordered]@{
            path = $Predictions
            sha256 = Get-Sha256 -Path $Predictions
        }
        transformed_features = [ordered]@{
            path = $Transformed
            sha256 = Get-Sha256 -Path $Transformed
        }
        contributions = [ordered]@{
            path = $Contributions
            sha256 = Get-Sha256 -Path $Contributions
        }
        grouped_contributions = [ordered]@{
            path = $Grouped
            sha256 = Get-Sha256 -Path $Grouped
        }
        comparison = if ($null -eq $ComparisonPath) {
            $null
        }
        else {
            [ordered]@{
                path = $ComparisonPath
                manifest_path = $ComparisonManifestPath
                manifest_sha256 = if ($null -eq $ComparisonManifestPath) {
                    $null
                }
                else {
                    Get-Sha256 -Path $ComparisonManifestPath
                }
            }
        }
    }
}

Write-JsonAtomic -Value $ScenarioManifest -Path $ScenarioManifestPath

$ReportPath = if (-not (Test-Blank $RunInfo.report_path)) {
    [string] $RunInfo.report_path
}
else {
    Join-Path (Join-Path $RunPath "reports") "report.html"
}

if ($OpenReport) {
    if (Test-Path -LiteralPath $ReportPath -PathType Leaf) {
        Start-Process -FilePath $ReportPath
    }
    else {
        Write-Warning "The report could not be opened because it does not exist: $ReportPath"
    }
}

Write-Host ""
Write-Host "Stage 3 completed"
Write-Host "Scenario package: $OutputDirectory"
Write-Host "Predictions: $Predictions"
Write-Host "Grouped contributions: $Grouped"
Write-Host "Manifest: $ScenarioManifestPath"
Write-Output $ScenarioManifestPath
