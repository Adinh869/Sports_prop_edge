$ErrorActionPreference = "Stop"
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
