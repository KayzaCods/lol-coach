# Registers a Windows Scheduled Task that auto-starts the lol-coach
# recorder at user logon. Runs invisibly via pythonw.exe; logs go to
# D:\projects\lol-coach\data\recorder.log.
#
# Run from an ordinary (non-admin) PowerShell:
#     .\scripts\install_autostart.ps1
#
# To remove:  .\scripts\uninstall_autostart.ps1
# To start now without logging out:  Start-ScheduledTask -TaskName 'LoLCoachRecorder'
# To stop:                            Stop-ScheduledTask  -TaskName 'LoLCoachRecorder'
# To inspect:                         Get-ScheduledTask  -TaskName 'LoLCoachRecorder'

$ErrorActionPreference = "Stop"

$projectRoot = "D:\projects\lol-coach"
$pythonw     = Join-Path $projectRoot ".venv\Scripts\pythonw.exe"
$script      = Join-Path $projectRoot "scripts\record.py"
$taskName    = "LoLCoachRecorder"

if (-not (Test-Path $pythonw)) { throw "pythonw.exe not found at $pythonw" }
if (-not (Test-Path $script))  { throw "record.py not found at $script" }

$action = New-ScheduledTaskAction `
    -Execute $pythonw `
    -Argument ('"{0}"' -f $script) `
    -WorkingDirectory $projectRoot

$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -ExecutionTimeLimit (New-TimeSpan -Days 0) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive

if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Removed existing task '$taskName'."
}

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Live Client recorder for lol-coach. Polls localhost:2999 while LoL is running. Logs at $projectRoot\data\recorder.log" | Out-Null

Write-Host ""
Write-Host "Task '$taskName' registered."
Write-Host "  Runs at:  user logon (next reboot or login)"
Write-Host "  Command:  $pythonw `"$script`""
Write-Host "  Log:      $projectRoot\data\recorder.log"
Write-Host ""
Write-Host "Starting it now so you do not need to log out..."
Start-ScheduledTask -TaskName $taskName
Start-Sleep -Seconds 2

$task = Get-ScheduledTask -TaskName $taskName
$info = Get-ScheduledTaskInfo -TaskName $taskName
Write-Host "  State:        $($task.State)"
Write-Host "  Last result:  $($info.LastTaskResult)"
Write-Host "  Last run:     $($info.LastRunTime)"
