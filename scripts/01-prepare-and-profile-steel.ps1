#Requires -Version 5.1
[CmdletBinding()]
param(
    [string] $ProjectRoot = (Get-Location).Path,
    [string] $InputFile = "steel_plates_faults.csv",
    [string] $TargetColumn = "FaultClass",
    [string] $RowIdColumn = "row_id",
    [string] $GroupColumn,
    [string] $TimeColumn,
    [double] $ReviewCorrelation = 0.75,
    [double] $WarnCorrelation = 0.90,
    [int] $NearDuplicateDecimals = 8,
    [double] $NearDuplicateThreshold = 0.98,
    [int] $MaximumPairwiseRows = 10000,
    [switch] $SkipInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-ExitCode {
    param([string] $Operation, [int] $ExitCode)
    if ($ExitCode -ne 0) { throw "$Operation failed with exit code $ExitCode." }
}

function Get-Sha256 {
    param([string] $Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-JsonAtomic {
    param([object] $Value, [string] $Path)
    $Parent = Split-Path -Path $Path -Parent
    if (-not [string]::IsNullOrWhiteSpace($Parent)) {
        New-Item -ItemType Directory -Path $Parent -Force | Out-Null
    }
    $TemporaryPath = "$Path.tmp-$([Guid]::NewGuid().ToString('N'))"
    try {
        $Value | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $TemporaryPath -Encoding UTF8
        Move-Item -LiteralPath $TemporaryPath -Destination $Path -Force
    }
    finally {
        Remove-Item -LiteralPath $TemporaryPath -Force -ErrorAction SilentlyContinue
    }
}

function ConvertTo-BinaryIndicator {
    param([string] $Value, [string] $Column, [int] $LineNumber)
    $Parsed = 0
    if (-not [int]::TryParse($Value, [ref] $Parsed)) {
        throw "CSV line $LineNumber, column '$Column' contains noninteger value '$Value'."
    }
    if ($Parsed -notin @(0, 1)) {
        throw "CSV line $LineNumber, column '$Column' must contain 0 or 1."
    }
    return $Parsed
}

$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot -ErrorAction Stop).Path
Set-Location -LiteralPath $ProjectRoot
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "pyproject.toml") -PathType Leaf)) {
    throw "pyproject.toml was not found. Run from the repository root or pass -ProjectRoot."
}

if (-not $SkipInstall) {
    & poetry install --extras dev
    Assert-ExitCode -Operation "poetry install" -ExitCode $LASTEXITCODE
}
& poetry run gam-app --help | Out-Null
Assert-ExitCode -Operation "gam-app --help" -ExitCode $LASTEXITCODE

$SessionId = Get-Date -Format "yyyyMMdd-HHmmss"
$DataDirectory = Join-Path $ProjectRoot "data"
$PreparedDirectory = Join-Path $DataDirectory "prepared"
$ProfileDirectory = Join-Path (Join-Path $ProjectRoot "profile") "steel-$SessionId"
$HandoffDirectory = Join-Path (Join-Path $ProjectRoot "workflow") "steel-$SessionId"
foreach ($Directory in @($DataDirectory, $PreparedDirectory, $ProfileDirectory, $HandoffDirectory)) {
    New-Item -ItemType Directory -Path $Directory -Force | Out-Null
}

$SourcePath = if ([IO.Path]::IsPathRooted($InputFile)) { $InputFile } else { Join-Path $DataDirectory $InputFile }
if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
    $CsvFallback = Join-Path $DataDirectory "steel_plates_faults.csv"
    if (Test-Path -LiteralPath $CsvFallback -PathType Leaf) { $SourcePath = $CsvFallback }
    else { throw "Dataset not found: $SourcePath" }
}
$SourcePath = (Resolve-Path -LiteralPath $SourcePath).Path

$Header = Get-Content -LiteralPath $SourcePath -TotalCount 1
if ([string]::IsNullOrWhiteSpace($Header)) { throw "The source dataset is empty." }
[char] $Delimiter = if ($Header.Contains(";") -and -not $Header.Contains(",")) { ";" } elseif ($Header.Contains(",")) { "," } elseif ($Header.Contains("`t")) { "`t" } else { throw "Could not detect comma, semicolon, or tab delimiter." }
$RawRows = @(Import-Csv -LiteralPath $SourcePath -Delimiter $Delimiter)
if ($RawRows.Count -eq 0) { throw "The dataset contains no records." }
$Columns = @($RawRows[0].PSObject.Properties.Name)
if ($Columns.Count -ne (@($Columns | Select-Object -Unique)).Count) { throw "The dataset contains duplicate column names." }

$FaultColumns = @("Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults")
$HasOneHotTarget = (@($FaultColumns | Where-Object { $_ -notin $Columns }).Count -eq 0)
$PreparedRows = @()
if ($HasOneHotTarget) {
    for ($Index = 0; $Index -lt $RawRows.Count; $Index++) {
        $Row = $RawRows[$Index]
        $Line = $Index + 2
        $Active = @()
        foreach ($Fault in $FaultColumns) {
            $Indicator = ConvertTo-BinaryIndicator -Value ([string] $Row.PSObject.Properties[$Fault].Value) -Column $Fault -LineNumber $Line
            if ($Indicator -eq 1) { $Active += $Fault }
        }
        if ($Active.Count -ne 1) { throw "CSV line $Line has $($Active.Count) active fault indicators; exactly one is required." }
        $Output = [ordered]@{}
        $Output[$RowIdColumn] = "steel-{0:D6}" -f ($Index + 1)
        foreach ($Property in $Row.PSObject.Properties) {
            if ($Property.Name -notin $FaultColumns -and $Property.Name -ne $RowIdColumn) { $Output[$Property.Name] = $Property.Value }
        }
        $Output[$TargetColumn] = $Active[0]
        $PreparedRows += [pscustomobject] $Output
    }
}
elseif ($TargetColumn -in $Columns) {
    for ($Index = 0; $Index -lt $RawRows.Count; $Index++) {
        $Row = $RawRows[$Index]
        $Output = [ordered]@{}
        if ($RowIdColumn -in $Columns) { $Output[$RowIdColumn] = $Row.PSObject.Properties[$RowIdColumn].Value }
        else { $Output[$RowIdColumn] = "steel-{0:D6}" -f ($Index + 1) }
        foreach ($Property in $Row.PSObject.Properties) {
            if ($Property.Name -ne $RowIdColumn) { $Output[$Property.Name] = $Property.Value }
        }
        $PreparedRows += [pscustomobject] $Output
    }
}
else { throw "Neither the expected one-hot fault columns nor categorical target '$TargetColumn' were found." }

if (@($PreparedRows | Where-Object { [string]::IsNullOrWhiteSpace([string] $_.PSObject.Properties[$TargetColumn].Value) }).Count -gt 0) { throw "The prepared target contains missing values." }
if (@($PreparedRows | Group-Object -Property $RowIdColumn | Where-Object { $_.Count -gt 1 }).Count -gt 0) { throw "The row identifier '$RowIdColumn' is not unique." }
if (-not [string]::IsNullOrWhiteSpace($GroupColumn) -and $GroupColumn -notin @($PreparedRows[0].PSObject.Properties.Name)) { throw "Group column '$GroupColumn' does not exist." }
if (-not [string]::IsNullOrWhiteSpace($TimeColumn) -and $TimeColumn -notin @($PreparedRows[0].PSObject.Properties.Name)) { throw "Time column '$TimeColumn' does not exist." }

$PreparedDataPath = Join-Path $PreparedDirectory "steel_plates_faults_prepared_$SessionId.csv"
$PreparedRows | Export-Csv -LiteralPath $PreparedDataPath -NoTypeInformation -Encoding UTF8

& poetry run gam-app profile --data $PreparedDataPath --target $TargetColumn --output $ProfileDirectory --review-correlation $ReviewCorrelation --warn-correlation $WarnCorrelation --near-duplicate-decimals $NearDuplicateDecimals --near-duplicate-threshold $NearDuplicateThreshold --maximum-pairwise-rows $MaximumPairwiseRows
Assert-ExitCode -Operation "gam-app profile" -ExitCode $LASTEXITCODE

$DiagnosticsManifestPath = Join-Path $ProfileDirectory "diagnostics_manifest.json"
if (-not (Test-Path -LiteralPath $DiagnosticsManifestPath -PathType Leaf)) { throw "Profiling did not create diagnostics_manifest.json." }

$FeatureReviewPath = Join-Path $ProfileDirectory "feature-review.csv"
$PreparedColumns = @($PreparedRows[0].PSObject.Properties.Name)
$FeatureReview = foreach ($Column in $PreparedColumns) {
    if ($Column -eq $TargetColumn) { continue }
    $Role = if ($Column -in @($RowIdColumn, $GroupColumn, $TimeColumn)) { "exclude" } else { "review generated configuration" }
    [pscustomobject]@{ feature = $Column; proposed_role = $Role; description = ""; unit = ""; derived = "none"; derived_from = ""; derivation = ""; review_decision = ""; review_notes = "" }
}
$FeatureReview | Export-Csv -LiteralPath $FeatureReviewPath -NoTypeInformation -Encoding UTF8

$HandoffPath = Join-Path $HandoffDirectory "preprocessing_handoff.json"
$Handoff = [ordered]@{
    schema_name = "gam_preprocessing_handoff"
    schema_version = "1.0"
    created_at = [DateTimeOffset]::UtcNow.ToString("o")
    project_root = $ProjectRoot
    source_path = $SourcePath
    source_sha256 = Get-Sha256 -Path $SourcePath
    prepared_data_path = $PreparedDataPath
    prepared_data_sha256 = Get-Sha256 -Path $PreparedDataPath
    target = $TargetColumn
    row_id = $RowIdColumn
    group = if ([string]::IsNullOrWhiteSpace($GroupColumn)) { $null } else { $GroupColumn }
    time = if ([string]::IsNullOrWhiteSpace($TimeColumn)) { $null } else { $TimeColumn }
    source_row_count = $RawRows.Count
    prepared_row_count = $PreparedRows.Count
    target_construction = if ($HasOneHotTarget) { "seven_one_hot_indicators" } else { "existing_categorical_target" }
    profile_path = $ProfileDirectory
    diagnostics_manifest_path = $DiagnosticsManifestPath
    feature_review_path = $FeatureReviewPath
}
Write-JsonAtomic -Value $Handoff -Path $HandoffPath

Write-Host ""
Write-Host "Stage 1 completed"
Write-Host "Prepared data: $PreparedDataPath"
Write-Host "Profile: $ProfileDirectory"
Write-Host "Feature review: $FeatureReviewPath"
Write-Host "Handoff: $HandoffPath"
Write-Output $HandoffPath
