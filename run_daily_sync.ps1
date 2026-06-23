$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$FixEnc = Join-Path $Root "tools\fix_enc.py"
if (-not (Test-Path $Python)) { Write-Host "[sync] ERROR: missing venv"; exit 1 }
& $Python $FixEnc
$env:PYTHONPATH = Join-Path $Root "src"
Write-Host "[sync] Running daily sync..."
& $Python -m sports_prop_edge.sync_main
if ($LASTEXITCODE -ne 0) { Write-Host "[sync] errors - see data\cache\last_sync_report.json"; exit $LASTEXITCODE }
Write-Host "[sync] OK"
