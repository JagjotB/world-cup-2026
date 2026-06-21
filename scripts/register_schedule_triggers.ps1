<#
.SYNOPSIS
  Register the WorldCup2026-AutoUpdate scheduled task to fire 3 hours after
  the kickoff of every upcoming match (i.e. once each game should be finished).

  Re-run this whenever the schedule changes (e.g. once knockout fixtures are
  known) to refresh the triggers.
#>
param(
    [string]$TaskName = "WorldCup2026-AutoUpdate",
    [int]$HoursAfterKickoff = 3
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$schedulePath = Join-Path $root "data\processed\group_stage_schedule_2026.csv"
$batPath = Join-Path $root "scripts\auto_update.bat"

if (-not (Test-Path $schedulePath)) { throw "Schedule not found: $schedulePath" }

$now = Get-Date
$triggers = @()
$count = 0
foreach ($row in Import-Csv $schedulePath) {
    if ([string]::IsNullOrWhiteSpace($row.kickoff_utc)) { continue }
    $fire = [datetimeoffset]::Parse($row.kickoff_utc).AddHours($HoursAfterKickoff).LocalDateTime
    if ($fire -le $now.AddMinutes(1)) { continue }   # skip past / imminent
    $triggers += New-ScheduledTaskTrigger -Once -At $fire
    $count++
}

if ($count -eq 0) {
    Write-Output "No future matches to schedule. Removing existing task if present."
    schtasks /Delete /TN $TaskName /F 2>$null
    return
}

$action = New-ScheduledTaskAction -Execute $batPath
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers -Settings $settings `
    -Description "Refresh World Cup 2026 results/predictions and push, 3h after each kickoff." -Force | Out-Null

Write-Output "Registered '$TaskName' with $count trigger(s), $HoursAfterKickoff h after each kickoff."
Write-Output "Next fire times:"
(Get-ScheduledTask -TaskName $TaskName).Triggers |
    Sort-Object StartBoundary |
    Select-Object -First 5 -ExpandProperty StartBoundary
