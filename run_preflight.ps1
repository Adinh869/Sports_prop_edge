$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Repair = Join-Path $Root "tools\repair_utf8.ps1"
$Preflight = Join-Path $Root "tools\preflight.py"

if (-not (Test-Path $Python)) {
    Write-Host "ERROR: missing venv at $Python"
    exit 1
}

Write-Host "[preflight] fixing encoding..."
& $Repair
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$env:PYTHONPATH = Join-Path $Root "src"
Write-Host "[preflight] running offline checks..."
& $Python $Preflight
exit $LASTEXITCODE
