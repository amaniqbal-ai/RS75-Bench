@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0state\main.pid" (
  echo The benchmark is not running.
  pause
  exit /b 0
)
choice /C YN /M "Stop the benchmark now? Finished work is kept and RUN.bat continues later"
if errorlevel 2 (
  echo Not stopped.
  pause
  exit /b 0
)
echo stop> "%~dp0state\STOP"
echo Stopping the running jobs - please wait up to 60 seconds...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Join-Path '%~dp0' 'state\main.pid'; for ($i = 0; $i -lt 60; $i++) { if (-not (Test-Path $p)) { exit 0 }; Start-Sleep -Seconds 1 }; $id = (Get-Content $p -Raw).Trim(); taskkill /T /F /PID $id | Out-Null; Remove-Item $p -ErrorAction SilentlyContinue"
echo Stopped. Double-click RUN.bat to continue later.
pause
