$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Streamlit = Join-Path $Root ".venv\Scripts\streamlit.exe"
$App = Join-Path $Root "app\streamlit_app.py"
$Requirements = Join-Path $Root "requirements.txt"

if (-not (Test-Path -LiteralPath $Python)) {
    python -m venv (Join-Path $Root ".venv")
}
if (-not (Test-Path -LiteralPath $Streamlit)) {
    & $Python -m pip install -r $Requirements
}

$FixEnc = Join-Path $Root "tools\fix_enc.py"
if (Test-Path -LiteralPath $FixEnc) {
    Write-Host '[run_app] Fixing UTF-16 / null-byte files (if any)...'
    & $Python $FixEnc
    if ($LASTEXITCODE -ne 0) {
        Write-Host '[run_app] ERROR: encoding repair failed. Try: .\fix_encoding.bat'
        exit $LASTEXITCODE
    }
}

Get-ChildItem -LiteralPath (Join-Path $Root "src"), (Join-Path $Root "app") -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue }

$env:PYTHONPATH = Join-Path $Root "src"
$EnvFile = Join-Path $Root ".env"
if (Test-Path -LiteralPath $EnvFile) {
    Get-Content -LiteralPath $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) { return }
        if ($line -match '^\s*export\s+(.+)$') { $line = $Matches[1] }
        $idx = $line.IndexOf("=")
        if ($idx -lt 1) { return }
        $name = $line.Substring(0, $idx).Trim()
        $value = $line.Substring($idx + 1).Trim().Trim('"').Trim("'")
        if (-not [string]::IsNullOrWhiteSpace($name) -and [string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable($name))) {
            [Environment]::SetEnvironmentVariable($name, $value, 'Process')
        }
    }
}
Write-Host '[run_app] Starting Streamlit...'
& $Streamlit run $App
exit $LASTEXITCODE
