# Daily sync — uses package module (avoids broken tools/daily_sync.py encoding).
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$env:PYTHONPATH = "src"
$py = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    throw "Missing venv. Run: python -m venv .venv; pip install -r requirements.txt"
}
& $py -m sports_prop_edge.sync_main @args
