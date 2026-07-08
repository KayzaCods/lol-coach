@echo off
REM Launch the recorder with no visible window using pythonw.exe.
REM Used by the Windows Scheduled Task and can also be run manually.
start "" "D:\projects\lol-coach\.venv\Scripts\pythonw.exe" "D:\projects\lol-coach\scripts\record.py"
