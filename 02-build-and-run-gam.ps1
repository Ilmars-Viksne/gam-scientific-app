#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string] $PreprocessingManifest,

    [ValidateSet("stratified", "stratified_group", "time")]
    [string] $Strategy = "stratified",

    [string] $GroupColumn,
    [string] $TimeColumn,

    [ValidateRange(0, 1024)]
    [int] $Workers = 0,

    [ValidateRange(2, 50)]
    [int] $OuterSplits = 3,

    [ValidateRange(1, 50)]
    [int] $OuterRepeats = 1,

    [ValidateRange(2, 50)]
    [int] $InnerSplits = 2,

    [ValidateRange(0, 1000000)]
    [int] $Gap = 0,

    [ValidateRange(1, 1000000)]
    [int] $TestSize = 100,

    [ValidateSet("report", "error", "group")]
    [string] $DuplicatePolicy = "report",

    [ValidateSet("quick", "standard", "thorough")]
    [string] $Preset = "quick",

    [switch] $MainEffectsOnly,
    [switch] $PauseForConfigurationReview
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

function ConvertFrom-NativeJson {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $Lines,
        [Parameter(Mandatory)] [string] $Operation
    )

    $Text = $Lines -join [Environment]::NewLine
    if ([string]::IsNullOrWhiteSpace($Text)) {
        throw "$Operation returned empty JSON."
    }

    try {
        return $Text | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "$Operation returned invalid JSON: $($_.Exception.Message)`n$Text"
    }
}

function Test-Blank {
    [CmdletBinding()]
    param([AllowNull()] [object] $Value)

    return [string]::IsNullOrWhiteSpace([string] $Value)
}

function Resolve-HandoffPathValue {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)] [string] $Value,
        [Parameter(Mandatory)] [string] $ManifestDirectory
    )

    if ([IO.Path]::IsPathRooted($Value)) {
        return $Value
    }

    return Join-Path $ManifestDirectory $Value
}

function Update-GamConfig {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)] [string] $Path,
        [Parameter(Mandatory)] [int] $WorkerCount,
        [Parameter(Mandatory)] [bool] $KeepMainOnly
    )

    $TemporaryScript = Join-Path `
        ([IO.Path]::GetTempPath()) `
        ("gam-stage2-{0}.py" -f [Guid]::NewGuid().ToString("N"))

    $Code = @'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
workers = int(sys.argv[2])
main_only = sys.argv[3].lower() == "true"

payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
if not isinstance(payload, dict):
    raise ValueError("Configuration root must be a YAML mapping.")

payload.setdefault("execution", {})["workers"] = workers

if main_only:
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError("Configuration does not contain a models list.")
    selected = [model for model in models if model.get("id") == "gam_main"]
    if len(selected) != 1:
        raise ValueError("Expected exactly one model with id 'gam_main'.")
    payload["models"] = selected

path.write_text(
    yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
'@

    try {
        Set-Content -LiteralPath $TemporaryScript -Value $Code -Encoding UTF8
        & poetry run python $TemporaryScript $Path $WorkerCount $KeepMainOnly.ToString()
        Assert-ExitCode -Operation "YAML update" -ExitCode $LASTEXITCODE
    }
    finally {
        Remove-Item -LiteralPath $TemporaryScript -Force -ErrorAction SilentlyContinue
    }
}

# -----------------------------------------------------------------------------
# Load and validate the Stage 1 handoff.
# -----------------------------------------------------------------------------
$PreprocessingManifest = (
    Resolve-Path -LiteralPath $PreprocessingManifest -ErrorAction Stop
).Path
$ManifestDirectory = Split-Path -Path $PreprocessingManifest -Parent
$Pre = Read-Json -Path $PreprocessingManifest

$SupportedHandoffSchemas = @(
    "gam_preparation_profile_handoff",
    "gam_preprocessing_handoff"
)

if ($Pre.schema_name -notin $SupportedHandoffSchemas) {
    throw "Unsupported Stage 1 handoff schema '$($Pre.schema_name)'."
}

if ($Pre.schema_name -eq "gam_preparation_profile_handoff") {
    if ($Pre.stage -ne "data_preparation_and_profile") {
        throw "Unsupported Stage 1 handoff stage '$($Pre.stage)'."
    }
    if ($Pre.preprocessing_fitted -eq $true -or $Pre.model_fitted -eq $true) {
        throw "Stage 1 handoff unexpectedly reports fitted preprocessing or a fitted model."
    }
}

if (Test-Blank $Pre.project_root) {
    throw "Stage 1 handoff does not contain project_root."
}

$ProjectRoot = (Resolve-Path -LiteralPath ([string] $Pre.project_root) -ErrorAction Stop).Path
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

if (Test-Blank $Pre.prepared_data_path) {
    throw "Stage 1 handoff does not contain prepared_data_path."
}

$DataPathCandidate = Resolve-HandoffPathValue `
    -Value ([string] $Pre.prepared_data_path) `
    -ManifestDirectory $ManifestDirectory
$DataPath = (Resolve-Path -LiteralPath $DataPathCandidate -ErrorAction Stop).Path

if (-not (Test-Path -LiteralPath $DataPath -PathType Leaf)) {
    throw "Prepared data is missing: $DataPath"
}

if (Test-Blank $Pre.prepared_data_sha256) {
    throw "Stage 1 handoff does not contain prepared_data_sha256."
}

$ActualPreparedHash = Get-Sha256 -Path $DataPath
if ($ActualPreparedHash -ne ([string] $Pre.prepared_data_sha256).ToLowerInvariant()) {
    throw "Prepared data hash differs from the Stage 1 handoff."
}

$Target = [string] $Pre.target
$RowId = [string] $Pre.row_id
if ((Test-Blank $Target) -or (Test-Blank $RowId)) {
    throw "Stage 1 handoff must declare target and row_id."
}

$FirstRow = Import-Csv -LiteralPath $DataPath | Select-Object -First 1
if ($null -eq $FirstRow) {
    throw "Prepared data contains no records: $DataPath"
}

$Columns = @($FirstRow.PSObject.Properties.Name)
if ($Target -notin $Columns -or $RowId -notin $Columns) {
    throw "Prepared data lacks target '$Target' or row ID '$RowId'."
}

# -----------------------------------------------------------------------------
# Validate strategy-specific columns and options.
# -----------------------------------------------------------------------------
if ($Strategy -eq "stratified_group") {
    if (Test-Blank $GroupColumn) {
        $GroupColumn = [string] $Pre.group
    }

    if ((Test-Blank $GroupColumn) -or ($GroupColumn -notin $Columns)) {
        throw "A genuine existing group column is required for stratified_group."
    }

    if ($DuplicatePolicy -eq "report") {
        $DuplicatePolicy = "group"
    }
}
elseif ($DuplicatePolicy -eq "group") {
    throw "DuplicatePolicy 'group' requires Strategy 'stratified_group'."
}

if ($Strategy -eq "time") {
    if (Test-Blank $TimeColumn) {
        $TimeColumn = [string] $Pre.time
    }

    if ((Test-Blank $TimeColumn) -or ($TimeColumn -notin $Columns)) {
        throw "A genuine existing time column is required for time validation."
    }

    $OuterRepeats = 1
}
elseif ($Gap -ne 0) {
    throw "Gap applies only to Strategy 'time'."
}

# -----------------------------------------------------------------------------
# Establish deterministic paths and a bounded worker count.
# -----------------------------------------------------------------------------
$LogicalProcessors = [Environment]::ProcessorCount
$WorkerCount = if ($Workers -eq 0) {
    [Math]::Min(4, [Math]::Max(1, $LogicalProcessors - 1))
}
else {
    [Math]::Max(1, [Math]::Min($Workers, $LogicalProcessors))
}

$SessionId = "{0}-{1}" -f `
    ([DateTimeOffset]::UtcNow.ToString("yyyyMMdd-HHmmssfff")), `
    ([Guid]::NewGuid().ToString("N").Substring(0, 8))

$ConfigDirectory = Join-Path $ProjectRoot "configs"
$Workspace = Join-Path $ProjectRoot "workspace"
$HandoffDirectory = Join-Path $ManifestDirectory "runs"

foreach ($Directory in @($ConfigDirectory, $Workspace, $HandoffDirectory)) {
    New-Item -ItemType Directory -Path $Directory -Force | Out-Null
}

$ConfigPath = Join-Path $ConfigDirectory "steel-$Strategy-$SessionId.yaml"
$Name = "steel-$Strategy-$SessionId"
$ManifestHash = Get-Sha256 -Path $PreprocessingManifest

# Keep metadata compact because configuration metadata values have a bounded size.
$ConfigureArguments = @(
    "run", "gam-app", "configure",
    "--data", $DataPath,
    "--target", $Target,
    "--output", $ConfigPath,
    "--name", $Name,
    "--row-id", $RowId,
    "--validation-strategy", $Strategy,
    "--outer-splits", $OuterSplits,
    "--outer-repeats", $OuterRepeats,
    "--inner-splits", $InnerSplits,
    "--random-state", 42,
    "--duplicate-group-policy", $DuplicatePolicy,
    "--review-correlation", 0.75,
    "--warn-correlation", 0.90,
    "--minimum-complete-pairs", 3,
    "--near-duplicate-decimals", 8,
    "--near-duplicate-threshold", 0.98,
    "--maximum-pairwise-rows", 10000,
    "--correlation-diagnostics",
    "--duplicate-groups",
    "--tag", "steel-plates",
    "--tag", $Strategy,
    "--metadata", "stage=2",
    "--metadata", "stage1_sha256=$ManifestHash",
    "--preset", $Preset,
    "--non-interactive"
)

if ($Strategy -eq "stratified_group") {
    $ConfigureArguments += @("--group", $GroupColumn)
}

if ($Strategy -eq "time") {
    $ConfigureArguments += @(
        "--time", $TimeColumn,
        "--gap", $Gap,
        "--test-size", $TestSize
    )
}

& poetry @ConfigureArguments
Assert-ExitCode -Operation "configure" -ExitCode $LASTEXITCODE

if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw "configure did not create the expected YAML file: $ConfigPath"
}

Update-GamConfig `
    -Path $ConfigPath `
    -WorkerCount $WorkerCount `
    -KeepMainOnly $MainEffectsOnly.IsPresent

if ($PauseForConfigurationReview) {
    Write-Host "Review derived metadata and feature roles in: $ConfigPath"
    Read-Host "Press Enter after saving the configuration" | Out-Null
}

# -----------------------------------------------------------------------------
# Validate feasibility before creating a run.
# -----------------------------------------------------------------------------
$PlanLines = @(& poetry run gam-app plan --config $ConfigPath --json)
$PlanExitCode = $LASTEXITCODE
$Plan = ConvertFrom-NativeJson -Lines $PlanLines -Operation "plan"

if ($PlanExitCode -eq 2 -or $Plan.feasible -ne $true) {
    @($Plan.checks) |
        Where-Object { $_.level -eq "fail" } |
        Format-Table -AutoSize
    throw "Validation design is infeasible."
}
Assert-ExitCode -Operation "plan" -ExitCode $PlanExitCode

# -----------------------------------------------------------------------------
# Run with one numerical-library thread per worker process.
# -----------------------------------------------------------------------------
$RunPathFile = Join-Path $Workspace "steel-$Strategy-$SessionId.txt"
$ThreadVariableNames = @(
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS"
)
$PreviousThreadValues = @{}

foreach ($NameOfVariable in $ThreadVariableNames) {
    $Item = Get-Item "Env:$NameOfVariable" -ErrorAction SilentlyContinue
    $PreviousThreadValues[$NameOfVariable] = if ($null -eq $Item) {
        $null
    }
    else {
        $Item.Value
    }
    Set-Item "Env:$NameOfVariable" "1"
}

$RunExitCode = $null
try {
    & poetry run gam-app run `
        --config $ConfigPath `
        --workspace $Workspace `
        --run-path-file $RunPathFile
    $RunExitCode = $LASTEXITCODE
}
finally {
    foreach ($NameOfVariable in $ThreadVariableNames) {
        if ($null -eq $PreviousThreadValues[$NameOfVariable]) {
            Remove-Item "Env:$NameOfVariable" -ErrorAction SilentlyContinue
        }
        else {
            Set-Item "Env:$NameOfVariable" $PreviousThreadValues[$NameOfVariable]
        }
    }
}

$RunPath = if (Test-Path -LiteralPath $RunPathFile -PathType Leaf) {
    (Get-Content -LiteralPath $RunPathFile -Raw).Trim()
}
else {
    $null
}

if ($RunExitCode -ne 0) {
    if (-not (Test-Blank $RunPath)) {
        Write-Warning "Failed run: $RunPath"
    }
    throw "run failed with exit code $RunExitCode."
}

if (Test-Blank $RunPath) {
    throw "Run path file is empty or missing: $RunPathFile"
}

$RunPath = (Resolve-Path -LiteralPath $RunPath -ErrorAction Stop).Path
if (-not (Test-Path -LiteralPath $RunPath -PathType Container)) {
    throw "Valid run path was not persisted."
}

# -----------------------------------------------------------------------------
# Verify completion and create the diagnostic review.
# -----------------------------------------------------------------------------
$StatusPath = Join-Path $RunPath "status.json"
$RunJsonPath = Join-Path $RunPath "run.json"
$Status = Read-Json -Path $StatusPath

if ($Status.state -ne "completed") {
    throw "Run state is '$($Status.state)'."
}

$ReviewDirectory = Join-Path $RunPath "reviews"
New-Item -ItemType Directory -Path $ReviewDirectory -Force | Out-Null
$ReviewPath = Join-Path $ReviewDirectory "diagnostic_review.json"

$ReviewLines = @(
    & poetry run gam-app review-diagnostics `
        --run $RunPath `
        --output $ReviewPath `
        --json
)
$ReviewExitCode = $LASTEXITCODE

$DiagnosticReviewCreated = $false
$DiagnosticReviewError = $null
$DiagnosticReviewCommandOutput = $ReviewLines -join [Environment]::NewLine

if ($ReviewExitCode -eq 0) {
    try {
        $null = ConvertFrom-NativeJson `
            -Lines $ReviewLines `
            -Operation "review-diagnostics"

        if (-not (Test-Path -LiteralPath $ReviewPath -PathType Leaf)) {
            throw "The command succeeded but did not create: $ReviewPath"
        }

        $DiagnosticReviewCreated = $true
    }
    catch {
        $DiagnosticReviewError = $_.Exception.Message
        Write-Warning "Diagnostic review validation failed: $DiagnosticReviewError"
    }
}
else {
    $DiagnosticReviewError = @"
review-diagnostics failed with exit code $ReviewExitCode.
Command output:
$DiagnosticReviewCommandOutput
"@

    Write-Warning $DiagnosticReviewError
    Write-Warning "The GAM run completed successfully. Stage 2 will continue without a diagnostic review artifact."
}

$RunMetadata = Read-Json -Path $RunJsonPath
$ModelIds = @($RunMetadata.models)
$ReportPath = Join-Path (Join-Path $RunPath "reports") "report.html"

if (-not (Test-Path -LiteralPath $ReportPath -PathType Leaf)) {
    Write-Warning "Expected HTML report was not found: $ReportPath"
}

# -----------------------------------------------------------------------------
# Write the Stage 2 handoff with hashes for auditable downstream use.
# -----------------------------------------------------------------------------
$HandoffPath = Join-Path $HandoffDirectory "run_${Strategy}_$SessionId.json"
$Handoff = [ordered]@{
    schema_name = "gam_run_handoff"
    schema_version = "1.1"
    created_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
    stage1_manifest = $PreprocessingManifest
    stage1_manifest_sha256 = $ManifestHash
    strategy = $Strategy
    config_path = $ConfigPath
    config_sha256 = Get-Sha256 -Path $ConfigPath
    run_path = $RunPath
    run_id = $RunMetadata.run_id
    status = $Status.state
    models = $ModelIds
    diagnostic_review = [ordered]@{
        created = $DiagnosticReviewCreated
        path = if ($DiagnosticReviewCreated) {
            $ReviewPath
        }
        else {
            $null
        }
        sha256 = if ($DiagnosticReviewCreated) {
            Get-Sha256 -Path $ReviewPath
        }
        else {
            $null
        }
        exit_code = $ReviewExitCode
        error = $DiagnosticReviewError
    }
    report_path = $ReportPath
    workers = $WorkerCount
}

Write-JsonAtomic -Value $Handoff -Path $HandoffPath

Write-Host ""
Write-Host "Stage 2 completed"
Write-Host "Configuration: $ConfigPath"
Write-Host "Run: $RunPath"
Write-Host "Diagnostic review: $ReviewPath"
Write-Host "Handoff: $HandoffPath"
Write-Output $HandoffPath

