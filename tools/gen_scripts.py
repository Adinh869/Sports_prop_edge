"""Write UTF-8 PowerShell runners (Windows PowerShell 5.1 needs BOM)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPAIR_PS1 = r"""$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
$FixEnc = Join-Path $Root "tools\fix_enc.py"
$ImportTest = Join-Path $Root "tools\_import_test.py"

Write-Host "[repair] project: $Root"

if (-not (Test-Path $Python)) {
    Write-Host "[repair] ERROR: missing $Python"
    exit 1
}

Write-Host "[repair] running fix_enc.py..."
& $Python $FixEnc
if ($LASTEXITCODE -ne 0) {
    Write-Host "[repair] fix_enc.py failed (exit $LASTEXITCODE)"
    exit $LASTEXITCODE
}

$env:PYTHONPATH = Join-Path $Root "src"
$importBody = @'
from sports_prop_edge.strategy.payouts import PayoutProfile
from sports_prop_edge.strategy.card_builder import CardRules
print("imports ok")
'@
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($ImportTest, $importBody, $utf8NoBom)

Write-Host "[repair] testing imports..."
& $Python $ImportTest
$importExit = $LASTEXITCODE
Remove-Item $ImportTest -ErrorAction SilentlyContinue
if ($importExit -ne 0) {
    Write-Host "[repair] import test FAILED (exit $importExit)"
    exit $importExit
}

Write-Host "[repair] all good"
Write-Host "Start app:  .\run_app.ps1"
Write-Host "Daily sync: .\run_daily_sync.ps1"
"""

FIX_ENCODING_PS1 = r"""$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$FixEnc = Join-Path $Root "tools\fix_enc.py"

if (-not (Test-Path $Python)) {
    Write-Host "[fix_encoding] ERROR: missing $Python"
    exit 1
}

Set-Location $Root
& $Python $FixEnc
exit $LASTEXITCODE
"""

RUN_APP_PS1 = r"""$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Streamlit = Join-Path $Root ".venv\Scripts\streamlit.exe"
$App = Join-Path $Root "app\streamlit_app.py"
$Requirements = Join-Path $Root "requirements.txt"

if (-not (Test-Path $Python)) {
    Write-Host "[run_app] Creating virtual environment..."
    python -m venv (Join-Path $Root ".venv")
}

if (-not (Test-Path $Streamlit)) {
    Write-Host "[run_app] Installing dependencies..."
    & $Python -m pip install -r $Requirements
}

$env:PYTHONPATH = Join-Path $Root "src"
Write-Host "[run_app] Starting Streamlit..."
Write-Host "[run_app] PYTHONPATH=$env:PYTHONPATH"

& $Streamlit run $App
exit $LASTEXITCODE
"""

RUN_DAILY_SYNC_PS1 = r"""$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Requirements = Join-Path $Root "requirements.txt"
if (-not (Test-Path $Python)) {
    Write-Host "[sync] Creating virtual environment..."
    python -m venv (Join-Path $Root ".venv")
    & $Python -m pip install -r $Requirements
}

$FixEnc = Join-Path $Root "tools\fix_enc.py"
if (Test-Path $FixEnc) {
    Write-Host "[sync] Checking file encoding..."
    & $Python $FixEnc
}

$env:PYTHONPATH = Join-Path $Root "src"
Write-Host "[sync] Running daily sync..."

& $Python -m sports_prop_edge.sync_main
if ($LASTEXITCODE -ne 0) {
    Write-Host "[sync] Finished with errors. See data\cache\last_sync_report.json"
    exit $LASTEXITCODE
}

Write-Host "[sync] OK. Live history: data\live\history_merged.csv"
exit 0
"""

SCHEDULE_DAILY_SYNC_PS1 = r"""$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TaskName = "SportsPropEdge-DailySync"
$RunAt = "08:00"
$SyncScript = Join-Path $ProjectRoot "run_daily_sync.ps1"

if (-not (Test-Path $SyncScript)) {
    throw "Missing run_daily_sync.ps1 at $SyncScript"
}

$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -File `"$SyncScript`"" `
    -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Daily -At $RunAt
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Refresh NBA/NFL/KBO player logs for sports_prop_edge" `
    -Force | Out-Null

Write-Host "Scheduled task '$TaskName' at $RunAt daily."
Write-Host "Project: $ProjectRoot"
Write-Host "Test now: powershell -ExecutionPolicy Bypass -File `"$SyncScript`""
"""

SCRIPTS: dict[str, str] = {
    "tools/repair_utf8.ps1": REPAIR_PS1,
    "tools/fix_encoding.ps1": FIX_ENCODING_PS1,
    "run_app.ps1": RUN_APP_PS1,
    "run_daily_sync.ps1": RUN_DAILY_SYNC_PS1,
    "tools/schedule_daily_sync.ps1": SCHEDULE_DAILY_SYNC_PS1,
}


def write_ps1(rel_path: str, content: str) -> Path:
    path = ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    text = content.strip() + "\n"
    # UTF-8 BOM: required for reliable parsing in Windows PowerShell 5.1
    path.write_bytes(b"\xef\xbb\xbf" + text.replace("\n", "\r\n").encode("utf-8"))
    return path


def fix_corrupted_files() -> int:
    fixed = 0
    patterns = ("*.py", "*.ps1", "*.csv", "*.bat", "*.mdc", "*.json")
    for pattern in patterns:
        for path in ROOT.rglob(pattern):
            if ".venv" in path.parts or "__pycache__" in path.parts:
                continue
            data = path.read_bytes()
            if b"\x00" not in data and not data.startswith(b"\xff\xfe"):
                continue
            enc = "utf-16" if data.startswith(b"\xff\xfe") else "utf-16-le"
            path.write_text(data.decode(enc), encoding="utf-8", newline="\n")
            print("fixed", path)
            fixed += 1
    return fixed


def main() -> None:
    fixed = fix_corrupted_files()
    print("encoding fixes:", fixed)
    for rel, body in SCRIPTS.items():
        out = write_ps1(rel, body)
        print("wrote", out)
    print("done - run: .\\tools\\repair_utf8.ps1")


if __name__ == "__main__":
    main()
