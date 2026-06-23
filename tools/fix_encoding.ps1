$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$FixEnc = Join-Path $Root "tools\fix_enc.py"

if (-not (Test-Path $Python)) {
    Write-Host "[fix_encoding] ERROR: missing $Python"
    exit 1
}

$p = Start-Process -FilePath $Python -ArgumentList @($FixEnc) -WorkingDirectory $Root -Wait -PassThru -NoNewWindow
exit $p.ExitCode
