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
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string] $Operation,

        [Parameter(Mandatory)]
        [int] $ExitCode
    )

    if ($ExitCode -ne 0) {
        throw "$Operation failed with exit code $ExitCode."
    }
}

function Get-Sha256 {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string] $Path
    )

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-JsonAtomic {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [object] $Value,

        [Parameter(Mandatory)]
        [string] $Path
    )

    $Parent = Split-Path -Path $Path -Parent
    if (-not [string]::IsNullOrWhiteSpace($Parent)) {
        New-Item -ItemType Directory -Path $Parent -Force | Out-Null
    }

    $TemporaryPath = "$Path.tmp-$([Guid]::NewGuid().ToString('N'))"

    try {
        $Value |
            ConvertTo-Json -Depth 20 |
            Set-Content -LiteralPath $TemporaryPath -Encoding UTF8

        Move-Item -LiteralPath $TemporaryPath -Destination $Path -Force
    }
    finally {
        Remove-Item -LiteralPath $TemporaryPath -Force -ErrorAction SilentlyContinue
    }
}

function ConvertTo-BinaryIndicator {
    [CmdletBinding()]
    param(
        [AllowEmptyString()]
        [string] $Value,

        [Parameter(Mandatory)]
        [string] $Column,

        [Parameter(Mandatory)]
        [int] $LineNumber
    )

    $Parsed = 0
    if (-not [int]::TryParse($Value, [ref] $Parsed)) {
        throw "CSV line $LineNumber, column '$Column' contains noninteger value '$Value'."
    }

    if ($Parsed -notin @(0, 1)) {
        throw "CSV line $LineNumber, column '$Column' must contain 0 or 1."
    }

    return $Parsed
}

function Test-Blank {
    [CmdletBinding()]
    param(
        [AllowNull()]
        [object] $Value
    )

    return [string]::IsNullOrWhiteSpace([string] $Value)
}

# -----------------------------------------------------------------------------
# Validate parameters before performing any file-system or package operations.
# -----------------------------------------------------------------------------
if (Test-Blank $TargetColumn) {
    throw "TargetColumn cannot be empty."
}

if (Test-Blank $RowIdColumn) {
    throw "RowIdColumn cannot be empty."
}

if ($ReviewCorrelation -lt 0.0 -or $ReviewCorrelation -gt 1.0) {
    throw "ReviewCorrelation must be between 0 and 1."
}

if ($WarnCorrelation -lt 0.0 -or $WarnCorrelation -gt 1.0) {
    throw "WarnCorrelation must be between 0 and 1."
}

if ($ReviewCorrelation -gt $WarnCorrelation) {
    throw "ReviewCorrelation cannot exceed WarnCorrelation."
}

if ($NearDuplicateDecimals -lt 0) {
    throw "NearDuplicateDecimals cannot be negative."
}

if ($NearDuplicateThreshold -lt 0.0 -or $NearDuplicateThreshold -gt 1.0) {
    throw "NearDuplicateThreshold must be between 0 and 1."
}

if ($MaximumPairwiseRows -lt 1) {
    throw "MaximumPairwiseRows must be at least 1."
}

$ConfiguredRoleColumns = @(
    $TargetColumn
    $RowIdColumn
    $GroupColumn
    $TimeColumn
) | Where-Object { -not (Test-Blank $_) }

$DuplicateRoleColumns = @(
    $ConfiguredRoleColumns |
        Group-Object -CaseSensitive |
        Where-Object { $_.Count -gt 1 }
)

if ($DuplicateRoleColumns.Count -gt 0) {
    $Conflicts = ($DuplicateRoleColumns.Name -join ", ")
    throw "Target, row-ID, group, and time columns must be distinct. Conflicts: $Conflicts."
}

# -----------------------------------------------------------------------------
# Resolve the project and verify the application entry point.
# -----------------------------------------------------------------------------
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot -ErrorAction Stop).Path
Set-Location -LiteralPath $ProjectRoot

$PyprojectPath = Join-Path $ProjectRoot "pyproject.toml"
if (-not (Test-Path -LiteralPath $PyprojectPath -PathType Leaf)) {
    throw "pyproject.toml was not found. Run from the repository root or pass -ProjectRoot."
}

if (-not $SkipInstall) {
    & poetry install --extras dev
    Assert-ExitCode -Operation "poetry install" -ExitCode $LASTEXITCODE
}

& poetry run gam-app --help | Out-Null
Assert-ExitCode -Operation "gam-app --help" -ExitCode $LASTEXITCODE

# -----------------------------------------------------------------------------
# Create collision-resistant output locations.
# -----------------------------------------------------------------------------
$SessionId = "{0}-{1}" -f `
    ([DateTimeOffset]::UtcNow.ToString("yyyyMMdd-HHmmssfff")), `
    ([Guid]::NewGuid().ToString("N").Substring(0, 8))

$DataDirectory = Join-Path $ProjectRoot "data"
$PreparedDirectory = Join-Path $DataDirectory "prepared"
$ProfileDirectory = Join-Path (Join-Path $ProjectRoot "profile") "steel-$SessionId"
$HandoffDirectory = Join-Path (Join-Path $ProjectRoot "workflow") "steel-$SessionId"

foreach ($Directory in @(
    $DataDirectory,
    $PreparedDirectory,
    $ProfileDirectory,
    $HandoffDirectory
)) {
    New-Item -ItemType Directory -Path $Directory -Force | Out-Null
}

# -----------------------------------------------------------------------------
# Locate and inspect the source dataset.
# -----------------------------------------------------------------------------
$SourcePath = if ([IO.Path]::IsPathRooted($InputFile)) {
    $InputFile
}
else {
    Join-Path $DataDirectory $InputFile
}

if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
    $CsvFallback = Join-Path $DataDirectory "steel_plates_faults.csv"
    if (Test-Path -LiteralPath $CsvFallback -PathType Leaf) {
        $SourcePath = $CsvFallback
    }
    else {
        throw "Dataset not found: $SourcePath"
    }
}

$SourcePath = (Resolve-Path -LiteralPath $SourcePath).Path
$Header = Get-Content -LiteralPath $SourcePath -TotalCount 1

if ([string]::IsNullOrWhiteSpace($Header)) {
    throw "The source dataset is empty."
}

[char] $Delimiter = if ($Header.Contains(";") -and -not $Header.Contains(",")) {
    ";"
}
elseif ($Header.Contains(",")) {
    ","
}
elseif ($Header.Contains("`t")) {
    "`t"
}
else {
    throw "Could not detect comma, semicolon, or tab delimiter."
}

# This lightweight header check is suitable for the expected steel dataset.
# Import-Csv remains the authoritative parser for full records.
$RawHeaderColumns = @(
    $Header.Split($Delimiter) |
        ForEach-Object { $_.Trim().Trim('"') }
)

if (@($RawHeaderColumns | Where-Object { Test-Blank $_ }).Count -gt 0) {
    throw "The dataset contains one or more empty column names."
}

$DuplicateHeaders = @(
    $RawHeaderColumns |
        Group-Object -CaseSensitive |
        Where-Object { $_.Count -gt 1 }
)

if ($DuplicateHeaders.Count -gt 0) {
    throw "The dataset contains duplicate column names: $($DuplicateHeaders.Name -join ', ')."
}

$RawRows = @(Import-Csv -LiteralPath $SourcePath -Delimiter $Delimiter)
if ($RawRows.Count -eq 0) {
    throw "The dataset contains no records."
}

$Columns = @($RawRows[0].PSObject.Properties.Name)
if ($Columns.Count -ne (@($Columns | Select-Object -Unique)).Count) {
    throw "The imported dataset contains duplicate column names."
}

for ($Index = 0; $Index -lt $RawRows.Count; $Index++) {
    $CurrentColumns = @($RawRows[$Index].PSObject.Properties.Name)
    if (($CurrentColumns -join "`0") -ne ($Columns -join "`0")) {
        throw "CSV line $($Index + 2) has an inconsistent column schema."
    }
}

# -----------------------------------------------------------------------------
# Convert one-hot target columns when present, otherwise preserve FaultClass.
# -----------------------------------------------------------------------------
$FaultColumns = @(
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults"
)

$HasOneHotTarget = (@(
    $FaultColumns |
        Where-Object { $_ -notin $Columns }
).Count -eq 0)

$PreparedRows = [System.Collections.Generic.List[object]]::new()

if ($HasOneHotTarget) {
    for ($Index = 0; $Index -lt $RawRows.Count; $Index++) {
        $Row = $RawRows[$Index]
        $LineNumber = $Index + 2
        $ActiveFaults = [System.Collections.Generic.List[string]]::new()

        foreach ($Fault in $FaultColumns) {
            $Indicator = ConvertTo-BinaryIndicator `
                -Value ([string] $Row.PSObject.Properties[$Fault].Value) `
                -Column $Fault `
                -LineNumber $LineNumber

            if ($Indicator -eq 1) {
                $ActiveFaults.Add($Fault)
            }
        }

        if ($ActiveFaults.Count -ne 1) {
            throw "CSV line $LineNumber has $($ActiveFaults.Count) active fault indicators; exactly one is required."
        }

        $Output = [ordered]@{}
        $ExistingRowId = if ($RowIdColumn -in $Columns) {
            [string] $Row.PSObject.Properties[$RowIdColumn].Value
        }
        else {
            $null
        }

        if (Test-Blank $ExistingRowId) {
            $Output[$RowIdColumn] = "steel-{0:D6}" -f ($Index + 1)
        }
        else {
            $Output[$RowIdColumn] = $ExistingRowId.Trim()
        }

        foreach ($Property in $Row.PSObject.Properties) {
            if (
                $Property.Name -notin $FaultColumns -and
                $Property.Name -ne $RowIdColumn -and
                $Property.Name -ne $TargetColumn
            ) {
                $Output[$Property.Name] = $Property.Value
            }
        }

        $Output[$TargetColumn] = $ActiveFaults[0]
        $PreparedRows.Add([pscustomobject] $Output)
    }
}
elseif ($TargetColumn -in $Columns) {
    for ($Index = 0; $Index -lt $RawRows.Count; $Index++) {
        $Row = $RawRows[$Index]
        $LineNumber = $Index + 2
        $Output = [ordered]@{}

        $ExistingRowId = if ($RowIdColumn -in $Columns) {
            [string] $Row.PSObject.Properties[$RowIdColumn].Value
        }
        else {
            $null
        }

        if (Test-Blank $ExistingRowId) {
            $Output[$RowIdColumn] = "steel-{0:D6}" -f ($Index + 1)
        }
        else {
            $Output[$RowIdColumn] = $ExistingRowId.Trim()
        }

        foreach ($Property in $Row.PSObject.Properties) {
            if ($Property.Name -eq $RowIdColumn) {
                continue
            }

            if ($Property.Name -eq $TargetColumn) {
                $TargetValue = ([string] $Property.Value).Trim()
                if (Test-Blank $TargetValue) {
                    throw "CSV line $LineNumber has an empty target value in '$TargetColumn'."
                }
                $Output[$TargetColumn] = $TargetValue
            }
            else {
                $Output[$Property.Name] = $Property.Value
            }
        }

        $PreparedRows.Add([pscustomobject] $Output)
    }
}
else {
    throw "Neither the expected one-hot fault columns nor categorical target '$TargetColumn' were found."
}

# -----------------------------------------------------------------------------
# Validate prepared role columns and target values.
# -----------------------------------------------------------------------------
$PreparedColumns = @($PreparedRows[0].PSObject.Properties.Name)

if ($TargetColumn -notin $PreparedColumns) {
    throw "Prepared data does not contain target column '$TargetColumn'."
}

if ($RowIdColumn -notin $PreparedColumns) {
    throw "Prepared data does not contain row identifier '$RowIdColumn'."
}

$MissingTargets = @(
    $PreparedRows |
        Where-Object {
            Test-Blank $_.PSObject.Properties[$TargetColumn].Value
        }
)

if ($MissingTargets.Count -gt 0) {
    throw "The prepared target '$TargetColumn' contains missing or blank values."
}

$MissingRowIds = @(
    $PreparedRows |
        Where-Object {
            Test-Blank $_.PSObject.Properties[$RowIdColumn].Value
        }
)

if ($MissingRowIds.Count -gt 0) {
    throw "The row identifier '$RowIdColumn' contains missing or blank values."
}

$NormalizedRowIds = @(
    $PreparedRows |
        ForEach-Object {
            ([string] $_.PSObject.Properties[$RowIdColumn].Value).Trim()
        }
)

$DuplicateRowIds = @(
    $NormalizedRowIds |
        Group-Object -CaseSensitive |
        Where-Object { $_.Count -gt 1 }
)

if ($DuplicateRowIds.Count -gt 0) {
    throw "The row identifier '$RowIdColumn' is not unique."
}

if (-not (Test-Blank $GroupColumn) -and $GroupColumn -notin $PreparedColumns) {
    throw "Group column '$GroupColumn' does not exist."
}

if (-not (Test-Blank $TimeColumn) -and $TimeColumn -notin $PreparedColumns) {
    throw "Time column '$TimeColumn' does not exist."
}

$TargetClasses = @(
    $PreparedRows |
        ForEach-Object {
            ([string] $_.PSObject.Properties[$TargetColumn].Value).Trim()
        } |
        Sort-Object -Unique
)

if ($TargetClasses.Count -lt 2) {
    throw "The target '$TargetColumn' must contain at least two classes."
}

# -----------------------------------------------------------------------------
# Write prepared data and run application profiling.
# -----------------------------------------------------------------------------
$PreparedDataPath = Join-Path `
    $PreparedDirectory `
    "steel_plates_faults_prepared_$SessionId.csv"

$PreparedRows |
    Export-Csv -LiteralPath $PreparedDataPath -NoTypeInformation -Encoding UTF8

if (-not (Test-Path -LiteralPath $PreparedDataPath -PathType Leaf)) {
    throw "Prepared data was not created: $PreparedDataPath"
}

& poetry run gam-app profile `
    --data $PreparedDataPath `
    --target $TargetColumn `
    --output $ProfileDirectory `
    --review-correlation $ReviewCorrelation `
    --warn-correlation $WarnCorrelation `
    --near-duplicate-decimals $NearDuplicateDecimals `
    --near-duplicate-threshold $NearDuplicateThreshold `
    --maximum-pairwise-rows $MaximumPairwiseRows

Assert-ExitCode -Operation "gam-app profile" -ExitCode $LASTEXITCODE

$DiagnosticsManifestPath = Join-Path $ProfileDirectory "diagnostics_manifest.json"
if (-not (Test-Path -LiteralPath $DiagnosticsManifestPath -PathType Leaf)) {
    throw "Profiling did not create diagnostics_manifest.json."
}

try {
    $DiagnosticsManifest = Get-Content `
        -LiteralPath $DiagnosticsManifestPath `
        -Raw |
        ConvertFrom-Json
}
catch {
    throw "Diagnostics manifest is not valid JSON: $DiagnosticsManifestPath. $($_.Exception.Message)"
}

if (Test-Blank $DiagnosticsManifest.schema_version) {
    throw "Diagnostics manifest does not declare schema_version."
}

# -----------------------------------------------------------------------------
# Produce a feature-review worksheet. Values in proposed_role are valid roles
# or blank; review workflow state is stored separately in review_status.
# -----------------------------------------------------------------------------
$FeatureReviewPath = Join-Path $ProfileDirectory "feature-review.csv"
$ReservedColumns = @(
    $RowIdColumn
    $GroupColumn
    $TimeColumn
) | Where-Object { -not (Test-Blank $_) }

$FeatureReview = foreach ($Column in $PreparedColumns) {
    if ($Column -eq $TargetColumn) {
        continue
    }

    $IsReserved = $Column -in $ReservedColumns
    $ProposedRole = ""
    $ReviewStatus = "pending"
    $AllowedRoles = "smooth|linear|categorical|exclude"
    $MissingPolicy = ""
    $Categories = ""
    $Reference = ""

    if ($IsReserved) {
        $ProposedRole = "exclude"
        $ReviewStatus = "fixed"
        $AllowedRoles = "exclude"
        $MissingPolicy = "error"
    }
    elseif ($Column -eq "Steel_Type") {
        $ProposedRole = "categorical"
        $ReviewStatus = "proposed"
        $MissingPolicy = "most_frequent"
        $Categories = "A300|A400"
        $Reference = "A300"
    }

    [pscustomobject]@{
        feature = $Column
        proposed_role = $ProposedRole
        review_status = $ReviewStatus
        allowed_roles = $AllowedRoles
        missing_policy = $MissingPolicy
        categories = $Categories
        reference = $Reference
        description = ""
        unit = ""
        derived = "none"
        derived_from = ""
        derivation = ""
        review_decision = ""
        review_notes = ""
    }
}

$FeatureReview |
    Export-Csv -LiteralPath $FeatureReviewPath -NoTypeInformation -Encoding UTF8

# -----------------------------------------------------------------------------
# Write a preparation/profile handoff. No fitted preprocessing is claimed here.
# -----------------------------------------------------------------------------
$HandoffPath = Join-Path $HandoffDirectory "preparation_profile_handoff.json"

$Handoff = [ordered]@{
    schema_name = "gam_preparation_profile_handoff"
    schema_version = "1.1"
    stage = "data_preparation_and_profile"
    preprocessing_fitted = $false
    model_fitted = $false
    created_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
    session_id = $SessionId
    project_root = $ProjectRoot
    source_path = $SourcePath
    source_sha256 = Get-Sha256 -Path $SourcePath
    source_delimiter = [string] $Delimiter
    prepared_data_path = $PreparedDataPath
    prepared_data_sha256 = Get-Sha256 -Path $PreparedDataPath
    target = $TargetColumn
    row_id = $RowIdColumn
    group = if (Test-Blank $GroupColumn) { $null } else { $GroupColumn }
    time = if (Test-Blank $TimeColumn) { $null } else { $TimeColumn }
    source_row_count = $RawRows.Count
    prepared_row_count = $PreparedRows.Count
    prepared_column_count = $PreparedColumns.Count
    target_classes = $TargetClasses
    target_construction = if ($HasOneHotTarget) {
        "seven_one_hot_indicators"
    }
    else {
        "existing_categorical_target"
    }
    profile_path = $ProfileDirectory
    diagnostics_manifest_path = $DiagnosticsManifestPath
    diagnostics_manifest_sha256 = Get-Sha256 -Path $DiagnosticsManifestPath
    diagnostics_schema_version = [string] $DiagnosticsManifest.schema_version
    feature_review_path = $FeatureReviewPath
    feature_review_sha256 = Get-Sha256 -Path $FeatureReviewPath
}

Write-JsonAtomic -Value $Handoff -Path $HandoffPath

if (-not (Test-Path -LiteralPath $HandoffPath -PathType Leaf)) {
    throw "Handoff file was not created: $HandoffPath"
}

Write-Host ""
Write-Host "Stage 1 completed"
Write-Host "Prepared data: $PreparedDataPath"
Write-Host "Profile: $ProfileDirectory"
Write-Host "Feature review: $FeatureReviewPath"
Write-Host "Handoff: $HandoffPath"
Write-Output $HandoffPath
