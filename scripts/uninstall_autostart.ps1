# Removes the LoLCoachRecorder scheduled task.
$taskName = "LoLCoachRecorder"
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask  -TaskName $taskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Task '$taskName' removed."
} else {
    Write-Host "Task '$taskName' not found. Nothing to do."
}
